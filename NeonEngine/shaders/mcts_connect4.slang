// =============================================================================
// mcts_connect4.slang  (v2)
//
// Changes vs v1:
//  * Bitboard Connect Four (49-bit, 7 bits/column, needs shaderInt64).
//    Child record = 5 ints instead of 44. Win test = 4 shift/AND pairs.
//  * Parallel ACP expansion: 1 workgroup (8 threads) per tree, 1 thread per action.
//  * Thrifty without host round trip: expand kernels compact into blockMap with
//    one atomic per tree and write the playout grid size into dispatchArgs
//    (binding 17); host uses vkCmdDispatchIndirect.
//  * Root position is a device-local bitboard uploaded with vkCmdUpdateBuffer.
//  * New counters: SUM_A, MAX_A, PLIES_MAXSUM (warp-divergence efficiency).
//
// =============================================================================

static const int COLS = 7;
static const int ROWS = 6;
static const int CELLS = 42;
static const int MAX_ACTIONS = 7;
static const int NODE_ROW = 8;
static const int PATH_STRIDE = 44;
static const int INFO_STRIDE = 4;
static const int CB_STRIDE = 5;           // [0,1]=pos lo/hi  [2,3]=mask lo/hi  [4]=status
static const int PLAYOUTS_PER_CHILD = 32;
static const int SEL_TPB = 8;
static const int REDUCE_TPB = 64;

static const int FLAG_LEAF = 1;
static const int STATUS_ONGOING = 0;
static const int STATUS_WIN = 1;
static const int STATUS_DRAW = 2;

static const int NUM_COUNTERS = 16;
static const int CTR_DEPTH_SUM = 0;
static const int CTR_DEPTH_MAX = 1;
static const int CTR_TERMINAL_HITS = 2;
static const int CTR_CHILDREN_CREATED = 3;
static const int CTR_TREE_SIZE_MAX = 4;
static const int CTR_BLOCKS_REAL = 5;
static const int CTR_BLOCKS_VIRTUAL = 6;
static const int CTR_PLIES_SUM = 7;
static const int CTR_ANOMALIES = 8;
static const int CTR_FINAL_TIES = 9;
static const int CTR_SUM_A = 10;          // thrifty: sum over iterations of A
static const int CTR_MAX_A = 11;
static const int CTR_PLIES_MAXSUM = 12;   // sum over real blocks of max plies in the block

[[vk::binding(0, 0)]]  RWStructuredBuffer<int> rootBoard;    // [4] root bitboard: pos lo, pos hi, mask lo, mask hi (pos = player to move)
[[vk::binding(1, 0)]]  RWStructuredBuffer<int> trees;
[[vk::binding(2, 0)]]  RWStructuredBuffer<int> treeSizes;
[[vk::binding(3, 0)]]  RWStructuredBuffer<int> nodeFlags;
[[vk::binding(4, 0)]]  RWStructuredBuffer<int> nodeN;
[[vk::binding(5, 0)]]  RWStructuredBuffer<int> nodeWins;
[[vk::binding(6, 0)]]  RWStructuredBuffer<int> selNode;
[[vk::binding(7, 0)]]  RWStructuredBuffer<int> selPath;
[[vk::binding(8, 0)]]  RWStructuredBuffer<int> selActs;
[[vk::binding(9, 0)]]  RWStructuredBuffer<int> selInfo;
[[vk::binding(10, 0)]] RWStructuredBuffer<int> actionsExp;
[[vk::binding(11, 0)]] RWStructuredBuffer<int> childBoards;  // [T][MAX_ACTIONS][CB_STRIDE]
[[vk::binding(12, 0)]] RWStructuredBuffer<int> childScores;
[[vk::binding(13, 0)]] RWStructuredBuffer<int> rootStats;
[[vk::binding(14, 0)]] RWStructuredBuffer<int> treeB;        // device-local now
[[vk::binding(15, 0)]] RWStructuredBuffer<int> blockMap;     // device-local now, written by expand (thrifty)
[[vk::binding(16, 0)]] RWStructuredBuffer<int> counters;     // host-visible
[[vk::binding(17, 0)]] RWStructuredBuffer<int> dispatchArgs; // [3] indirect args {A,1,1}; needs INDIRECT_BUFFER usage

struct PushConsts
{
    int numTrees;
    int maxNodes;
    uint seed;
    int rootTurn;
    float ucbC;
    int thrifty;
    int collect;
};
[[vk::push_constant]] ConstantBuffer<PushConsts> pc;

// =============================================================================
// GAME: Connect Four bitboards. Bit index = col*7 + row (row 0 = bottom, row 6 = sentinel).
// A position is (pos, mask): mask = all stones, pos = stones of the player to move.
// =============================================================================
uint64_t bottomOf(int c) { return uint64_t(1) << (c * 7); }
uint64_t topOf(int c)    { return uint64_t(1) << (c * 7 + 5); }
uint64_t colMask(int c)  { return uint64_t(0x3F) << (c * 7); }
uint64_t topAll()
{
    uint64_t m = 0;
    for (int c = 0; c < COLS; c++) m |= topOf(c);
    return m;
}

bool alignment(uint64_t p)
{
    uint64_t m = p & (p >> 7);                       // horizontal
    if ((m & (m >> 14)) != 0) return true;
    m = p & (p >> 6);                                // diagonal
    if ((m & (m >> 12)) != 0) return true;
    m = p & (p >> 8);                                // anti-diagonal
    if ((m & (m >> 16)) != 0) return true;
    m = p & (p >> 1);                                // vertical
    if ((m & (m >> 2)) != 0) return true;
    return false;
}

bool colLegal(uint64_t mask, int c) { return (mask & topOf(c)) == 0; }

// advance a position by a (legal) move without win detection (path replay)
void applyMove(inout uint64_t pos, inout uint64_t mask, int col)
{
    uint64_t mv = (mask + bottomOf(col)) & colMask(col);
    pos ^= mask;      // pos becomes the opponent's stones = new player to move
    mask |= mv;
}

// child of (pos, mask) by playing column a. cpos = stones of the NEW player to move.
void childOf(uint64_t pos, uint64_t mask, int a, out uint64_t cpos, out uint64_t cmask, out int status)
{
    uint64_t mv = (mask + bottomOf(a)) & colMask(a);
    bool win = alignment(pos | mv);
    cmask = mask | mv;
    cpos = pos ^ mask;
    uint64_t ta = topAll();
    status = win ? STATUS_WIN : (((cmask & ta) == ta) ? STATUS_DRAW : STATUS_ONGOING);
}

uint64_t join64(int lo, int hi) { return (uint64_t(uint(hi)) << 32) | uint64_t(uint(lo)); }
int lo32(uint64_t v) { return int(uint(v & uint64_t(0xFFFFFFFF))); }
int hi32(uint64_t v) { return int(uint(v >> 32)); }

void writeRecord(int rec, uint64_t cpos, uint64_t cmask, int status)
{
    childBoards[rec + 0] = lo32(cpos);
    childBoards[rec + 1] = hi32(cpos);
    childBoards[rec + 2] = lo32(cmask);
    childBoards[rec + 3] = hi32(cmask);
    childBoards[rec + 4] = status;
}
// =============================================================================

uint pcg(uint v)
{
    uint state = v * 747796405u + 2891336245u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// reset ------------------------------------------------------------------------
[shader("compute")]
[numthreads(32, 1, 1)]
void mctsReset(uint3 dtid : SV_DispatchThreadID)
{
    int t = int(dtid.x);
    if (t >= pc.numTrees) return;

    int base = t * pc.maxNodes;
    treeSizes[t] = 1;
    nodeFlags[base] = FLAG_LEAF;
    nodeN[base] = 0;
    nodeWins[base] = 0;
    for (int j = 0; j < NODE_ROW; j++) trees[base * NODE_ROW + j] = -1;
}

// select -----------------------------------------------------------------------
groupshared float sh_ucb[SEL_TPB];
groupshared int sh_child[SEL_TPB];
groupshared int sh_act[SEL_TPB];
groupshared uint sh_prio[SEL_TPB];

[shader("compute")]
[numthreads(8, 1, 1)]
void mctsSelect(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID)
{
    int t = int(gid.x);
    int th = int(tid.x);
    int base = t * pc.maxNodes;

    // Reset the indirect playout grid for this iteration. Safe: the previous iteration's
    // backup (last reader) is behind a barrier, and expand (first writer) is behind another.
    if (t == 0 && th == 0)
    {
        dispatchArgs[0] = 0;
        dispatchArgs[1] = 1;
        dispatchArgs[2] = 1;
    }

    int node = 0;
    int depth = 0;
    if (th == 0) selPath[t * PATH_STRIDE] = 0;

    while ((nodeFlags[base + node] & FLAG_LEAF) == 0)
    {
        float ucb = -1e30;
        int child = -1;
        if (th < MAX_ACTIONS)
        {
            child = trees[(base + node) * NODE_ROW + 1 + th];
            if (child >= 0)
            {
                int cn = nodeN[base + child];
                if (cn == 0)
                {
                    ucb = 1e30;
                }
                else
                {
                    float pn = float(nodeN[base + node]);
                    ucb = float(nodeWins[base + child]) / (2.0 * float(cn))
                        + pc.ucbC * sqrt(log(pn) / float(cn));
                }
            }
        }
        sh_ucb[th] = ucb;
        sh_child[th] = child;
        sh_act[th] = th;
        sh_prio[th] = pcg(pc.seed ^ pcg(uint(t) * 4096u + uint(depth) * 8u + uint(th) + 0x2545F491u));
        GroupMemoryBarrierWithGroupSync();

        for (int stride = SEL_TPB >> 1; stride > 0; stride >>= 1)
        {
            if (th < stride)
            {
                bool take = (sh_ucb[th] < sh_ucb[th + stride])
                         || (sh_ucb[th] == sh_ucb[th + stride] && sh_prio[th] < sh_prio[th + stride]);
                if (take)
                {
                    sh_ucb[th] = sh_ucb[th + stride];
                    sh_child[th] = sh_child[th + stride];
                    sh_act[th] = sh_act[th + stride];
                    sh_prio[th] = sh_prio[th + stride];
                }
            }
            GroupMemoryBarrierWithGroupSync();
        }

        int bestChild = sh_child[0];
        int bestAct = sh_act[0];
        GroupMemoryBarrierWithGroupSync();

        if (th == 0)
        {
            selActs[t * PATH_STRIDE + depth] = bestAct;
            selPath[t * PATH_STRIDE + depth + 1] = bestChild;
        }
        node = bestChild;
        depth++;
    }

    if (th == 0)
    {
        selNode[t] = node;
        selInfo[t * INFO_STRIDE + 0] = depth + 1;
        if (pc.collect != 0)
        {
            InterlockedAdd(counters[CTR_DEPTH_SUM], depth);
            InterlockedMax(counters[CTR_DEPTH_MAX], depth);
        }
    }
}

// expand (ACP): 1 workgroup per tree, thread a creates the child for action a ------
groupshared uint64_t sh_pos;
groupshared uint64_t sh_mask;
groupshared int sh_status;
groupshared int sh_nextId;

[shader("compute")]
[numthreads(8, 1, 1)]
void mctsExpand(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID)
{
    int t = int(gid.x);
    int th = int(tid.x);
    int base = t * pc.maxNodes;
    int node = selNode[t];
    int depth = selInfo[t * INFO_STRIDE + 0] - 1;

    if (th < MAX_ACTIONS)
    {
        actionsExp[t * MAX_ACTIONS + th] = -1;
        childScores[t * MAX_ACTIONS + th] = 0;
    }

    if (th == 0)
    {
        treeB[t] = 0;
        int st = (nodeFlags[base + node] >> 1) & 3;
        sh_status = st;
        sh_nextId = treeSizes[t];
        if (st == STATUS_ONGOING)
        {
            uint64_t pos = join64(rootBoard[0], rootBoard[1]);
            uint64_t mask = join64(rootBoard[2], rootBoard[3]);
            for (int d = 0; d < depth; d++) applyMove(pos, mask, selActs[t * PATH_STRIDE + d]);
            sh_pos = pos;
            sh_mask = mask;
        }
    }
    GroupMemoryBarrierWithGroupSync();

    int status = sh_status;
    if (status != STATUS_ONGOING)   // uniform across the group
    {
        if (th == 0)
        {
            selInfo[t * INFO_STRIDE + 1] = 0;
            selInfo[t * INFO_STRIDE + 2] = (status == STATUS_WIN) ? 2 * PLAYOUTS_PER_CHILD : PLAYOUTS_PER_CHILD;
            if (pc.collect != 0) InterlockedAdd(counters[CTR_TERMINAL_HITS], 1);
        }
        return;
    }

    uint64_t pos = sh_pos;
    uint64_t mask = sh_mask;
    int nextId = sh_nextId;

    // every thread computes the same count; ord = number of legal actions below a
    int count = 0;
    int ord = 0;
    for (int a = 0; a < MAX_ACTIONS; a++)
    {
        if (colLegal(mask, a))
        {
            if (a < th) ord++;
            count++;
        }
    }

    if (th < MAX_ACTIONS && colLegal(mask, th))
    {
        int a = th;
        uint64_t cpos, cmask;
        int cstatus;
        childOf(pos, mask, a, cpos, cmask, cstatus);

        int id = nextId + ord;
        int idx = (pc.thrifty != 0) ? ord : a;

        trees[(base + node) * NODE_ROW + 1 + a] = id;
        int row = (base + id) * NODE_ROW;
        trees[row] = node;
        for (int j = 1; j < NODE_ROW; j++) trees[row + j] = -1;
        nodeN[base + id] = 0;
        nodeWins[base + id] = 0;
        nodeFlags[base + id] = FLAG_LEAF | (cstatus << 1);

        writeRecord((t * MAX_ACTIONS + idx) * CB_STRIDE, cpos, cmask, cstatus);
        actionsExp[t * MAX_ACTIONS + idx] = a;
    }

    if (th == 0)
    {
        treeSizes[t] = nextId + count;
        nodeFlags[base + node] = 0;
        selInfo[t * INFO_STRIDE + 1] = count;
        treeB[t] = count;
        if (pc.thrifty != 0 && count > 0)
        {
            int off;
            InterlockedAdd(dispatchArgs[0], count, off);   // reserve `count` playout blocks
            for (int k = 0; k < count; k++) blockMap[off + k] = t * MAX_ACTIONS + k;
        }
        if (pc.collect != 0)
        {
            InterlockedAdd(counters[CTR_CHILDREN_CREATED], count);
            InterlockedMax(counters[CTR_TREE_SIZE_MAX], nextId + count);
        }
    }
}

// expand (OCP): one child per tree; 1 thread per tree (dispatch ceil(T/32) groups) ---
[shader("compute")]
[numthreads(32, 1, 1)]
void mctsExpandOcp(uint3 dtid : SV_DispatchThreadID)
{
    int t = int(dtid.x);
    if (t >= pc.numTrees) return;

    int base = t * pc.maxNodes;
    int node = selNode[t];
    int depth = selInfo[t * INFO_STRIDE + 0] - 1;

    for (int a = 0; a < MAX_ACTIONS; a++)
    {
        actionsExp[t * MAX_ACTIONS + a] = -1;
        childScores[t * MAX_ACTIONS + a] = 0;
    }
    treeB[t] = 0;

    int status = (nodeFlags[base + node] >> 1) & 3;
    if (status != STATUS_ONGOING)
    {
        selInfo[t * INFO_STRIDE + 1] = 0;
        selInfo[t * INFO_STRIDE + 2] = (status == STATUS_WIN) ? 2 * PLAYOUTS_PER_CHILD : PLAYOUTS_PER_CHILD;
        if (pc.collect != 0) InterlockedAdd(counters[CTR_TERMINAL_HITS], 1);
        return;
    }

    uint64_t pos = join64(rootBoard[0], rootBoard[1]);
    uint64_t mask = join64(rootBoard[2], rootBoard[3]);
    for (int d = 0; d < depth; d++) applyMove(pos, mask, selActs[t * PATH_STRIDE + d]);

    int untried[MAX_ACTIONS];
    int u = 0;
    for (int a = 0; a < MAX_ACTIONS; a++)
    {
        if (colLegal(mask, a) && trees[(base + node) * NODE_ROW + 1 + a] < 0)
        {
            untried[u] = a;
            u++;
        }
    }
    if (u == 0)
    {
        selInfo[t * INFO_STRIDE + 1] = 0;
        selInfo[t * INFO_STRIDE + 2] = PLAYOUTS_PER_CHILD;
        if (pc.collect != 0) InterlockedAdd(counters[CTR_ANOMALIES], 1);
        return;
    }

    uint rng = pcg(pc.seed ^ pcg(uint(t) + 0x51ed270bu));
    int act = untried[int(pcg(rng) % uint(u))];

    uint64_t cpos, cmask;
    int cstatus;
    childOf(pos, mask, act, cpos, cmask, cstatus);

    int id = treeSizes[t];
    trees[(base + node) * NODE_ROW + 1 + act] = id;
    int row = (base + id) * NODE_ROW;
    trees[row] = node;
    for (int j = 1; j < NODE_ROW; j++) trees[row + j] = -1;
    nodeN[base + id] = 0;
    nodeWins[base + id] = 0;
    nodeFlags[base + id] = FLAG_LEAF | (cstatus << 1);

    writeRecord((t * MAX_ACTIONS) * CB_STRIDE, cpos, cmask, cstatus);
    actionsExp[t * MAX_ACTIONS] = act;
    treeSizes[t] = id + 1;
    if (u == 1) nodeFlags[base + node] = 0;
    selInfo[t * INFO_STRIDE + 1] = 1;
    treeB[t] = 1;

    if (pc.thrifty != 0)
    {
        int off;
        InterlockedAdd(dispatchArgs[0], 1, off);
        blockMap[off] = t * MAX_ACTIONS;
    }
    if (pc.collect != 0)
    {
        InterlockedAdd(counters[CTR_CHILDREN_CREATED], 1);
        InterlockedMax(counters[CTR_TREE_SIZE_MAX], id + 1);
    }
}

// playout ----------------------------------------------------------------------
groupshared int sh_score[PLAYOUTS_PER_CHILD];
groupshared int sh_plies[PLAYOUTS_PER_CHILD];
groupshared int sh_pmax[PLAYOUTS_PER_CHILD];

[shader("compute")]
[numthreads(32, 1, 1)]
void mctsPlayout(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID)
{
    int t;
    int slot;
    if (pc.thrifty != 0)
    {
        slot = blockMap[gid.x];
        t = slot / MAX_ACTIONS;
    }
    else
    {
        t = int(gid.x);
        slot = t * MAX_ACTIONS + int(gid.y);
    }
    int th = int(tid.x);

    int act = actionsExp[slot];
    if (act < 0) return;

    int rec = slot * CB_STRIDE;
    uint64_t pos = join64(childBoards[rec + 0], childBoards[rec + 1]);   // stones of player to move at the child
    uint64_t mask = join64(childBoards[rec + 2], childBoards[rec + 3]);
    int status = childBoards[rec + 4];

    // The first mover of the playout is the player to move at the child, i.e. NOT the
    // player q who moved into it. So the mover of ply k is q iff k is even.
    int score = 1;
    int plies = 0;
    if (status == STATUS_WIN)
    {
        score = 2;
    }
    else if (status == STATUS_DRAW)
    {
        score = 1;
    }
    else
    {
        uint rng = pcg(pc.seed ^ pcg(uint(slot) * 32u + uint(th) + 1u));
        while (true)
        {
            int legalBits = 0;
            for (int c = 0; c < COLS; c++)
                if (colLegal(mask, c)) legalBits |= (1 << c);
            if (legalBits == 0) { score = 1; break; }

            int n = int(countbits(uint(legalBits)));
            rng = pcg(rng);
            int k = int(rng % uint(n));
            int col = 0;
            for (int c = 0; c < COLS; c++)
            {
                if (((legalBits >> c) & 1) != 0)
                {
                    if (k == 0) { col = c; break; }
                    k--;
                }
            }

            plies++;
            uint64_t mv = (mask + bottomOf(col)) & colMask(col);
            if (alignment(pos | mv))
            {
                score = ((plies & 1) == 0) ? 2 : 0;
                break;
            }
            pos ^= mask;
            mask |= mv;
        }
    }

    sh_score[th] = score;
    sh_plies[th] = plies;
    sh_pmax[th] = plies;
    GroupMemoryBarrierWithGroupSync();
    for (int stride = PLAYOUTS_PER_CHILD >> 1; stride > 0; stride >>= 1)
    {
        if (th < stride)
        {
            sh_score[th] += sh_score[th + stride];
            sh_plies[th] += sh_plies[th + stride];
            sh_pmax[th] = max(sh_pmax[th], sh_pmax[th + stride]);
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (th == 0)
    {
        int sum = sh_score[0];
        childScores[slot] = sum;

        int node = selNode[t];
        int child = trees[(t * pc.maxNodes + node) * NODE_ROW + 1 + act];
        nodeN[t * pc.maxNodes + child] = PLAYOUTS_PER_CHILD;
        nodeWins[t * pc.maxNodes + child] = sum;

        if (pc.collect != 0)
        {
            if (status == STATUS_ONGOING)
            {
                InterlockedAdd(counters[CTR_BLOCKS_REAL], 1);
                InterlockedAdd(counters[CTR_PLIES_SUM], sh_plies[0]);
                InterlockedAdd(counters[CTR_PLIES_MAXSUM], sh_pmax[0]);
            }
            else
            {
                InterlockedAdd(counters[CTR_BLOCKS_VIRTUAL], 1);
            }
        }
    }
}

// backup -----------------------------------------------------------------------
groupshared int sh_N;
groupshared int sh_SR;

[shader("compute")]
[numthreads(64, 1, 1)]
void mctsBackup(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID)
{
    int t = int(gid.x);
    int th = int(tid.x);

    if (th == 0)
    {
        int b = selInfo[t * INFO_STRIDE + 1];
        if (b == 0)
        {
            sh_N = PLAYOUTS_PER_CHILD;
            sh_SR = selInfo[t * INFO_STRIDE + 2];
        }
        else
        {
            int sq = 0;
            for (int a = 0; a < MAX_ACTIONS; a++)
            {
                if (actionsExp[t * MAX_ACTIONS + a] >= 0) sq += childScores[t * MAX_ACTIONS + a];
            }
            int N = b * PLAYOUTS_PER_CHILD;
            sh_N = N;
            sh_SR = 2 * N - sq;
        }

        // thrifty: record this iteration's grid size (dispatchArgs is stable until next select)
        if (t == 0 && pc.thrifty != 0)
        {
            int A = dispatchArgs[0];
            InterlockedAdd(counters[CTR_SUM_A], A);
            InterlockedMax(counters[CTR_MAX_A], A);
        }
    }
    GroupMemoryBarrierWithGroupSync();

    int pathLen = selInfo[t * INFO_STRIDE + 0];
    int k = pathLen - 1;
    if (th < pathLen)
    {
        int node = selPath[t * PATH_STRIDE + th];
        int N = sh_N;
        int SR = sh_SR;
        int w = (((k - th) & 1) == 0) ? SR : (2 * N - SR);
        nodeN[t * pc.maxNodes + node] += N;
        nodeWins[t * pc.maxNodes + node] += w;
    }
}

// reduce over trees ------------------------------------------------------------
groupshared int sh_rn[REDUCE_TPB];
groupshared int sh_rw[REDUCE_TPB];

[shader("compute")]
[numthreads(64, 1, 1)]
void mctsReduceTrees(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID)
{
    int a = int(gid.x);
    int th = int(tid.x);

    int sn = 0;
    int sw = 0;
    for (int t = th; t < pc.numTrees; t += REDUCE_TPB)
    {
        int child = trees[(t * pc.maxNodes) * NODE_ROW + 1 + a];
        if (child >= 0)
        {
            sn += nodeN[t * pc.maxNodes + child];
            sw += nodeWins[t * pc.maxNodes + child];
        }
    }
    sh_rn[th] = sn;
    sh_rw[th] = sw;
    GroupMemoryBarrierWithGroupSync();

    for (int stride = REDUCE_TPB >> 1; stride > 0; stride >>= 1)
    {
        if (th < stride)
        {
            sh_rn[th] += sh_rn[th + stride];
            sh_rw[th] += sh_rw[th + stride];
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (th == 0)
    {
        rootStats[2 * a] = sh_rn[0];
        rootStats[2 * a + 1] = sh_rw[0];
    }
}

[shader("compute")]
[numthreads(1, 1, 1)]
void mctsReduceActions()
{
    int best = -1;
    int bestN = -1;
    int bestW = -1;
    int nTies = 0;
    uint rng = pcg(pc.seed ^ 0xC0FFEE11u);
    for (int a = 0; a < MAX_ACTIONS; a++)
    {
        int n = rootStats[2 * a];
        int w = rootStats[2 * a + 1];
        if (n <= 0) continue;
        if (n > bestN || (n == bestN && w > bestW))
        {
            best = a;
            bestN = n;
            bestW = w;
            nTies = 1;
        }
        else if (n == bestN && w == bestW)
        {
            nTies++;
            rng = pcg(rng);
            if (rng % uint(nTies) == 0u) best = a;
        }
    }
    rootStats[2 * MAX_ACTIONS] = best;
    if (pc.collect != 0) counters[CTR_FINAL_TIES] = nTies;
}
