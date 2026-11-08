// =============================================================================
// mcts_connect4.slang  (v3)
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
// Changes vs v2:
//  * Fix: ACP expansion could create a child for a full column (driver miscompile of a 64-bit
//    shift by a run-time amount, see bit64), corrupting trees; ACP-prodigal was not deterministic.
//    Otherwise search results are bit-identical to v2.
//  * Packed playouts: one 32-lane group serves up to slotsPerGroup children. Lanes that finish
//    a playout take the next one from a groupshared counter, so a lane no longer idles until the
//    longest playout of its child ends. Each playout keeps its own RNG stream (slot, index), so
//    results do not depend on slotsPerGroup.
//  * Playout move choice works on the "next free cell" bitmask (no per-column loop).
//  * Terminal children are scored without simulation; thrifty enqueues A slots and launches
//    ceil(A / slotsPerGroup) groups (dispatchArgs[3] = A).
//  * Counters and root statistics are device-local; the host copies them back once per submit.
//  * Command buffers are recorded once per batch size and replayed: the per-iteration seed is
//    derived here from searchParams (binding 18) and the iteration's offset in the batch, with
//    the same formula the host used before, so results are unchanged.
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
static const int MAX_SLOTS_PER_GROUP = 8;

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
[[vk::binding(16, 0)]] RWStructuredBuffer<int> counters;     // device-local, copied back by the host
[[vk::binding(17, 0)]] RWStructuredBuffer<int> dispatchArgs; // [0..2] indirect playout grid {G,1,1}, [3] = A (queued slots)
[[vk::binding(18, 0)]] RWStructuredBuffer<uint> searchParams; // [0] cfg seed, [1] call seed, [2] iterations done before this batch

struct PushConsts
{
    int numTrees;
    int maxNodes;
    int iterOffset;      // iteration index within the batch (baked into the recorded command buffer)
    int rootTurn;        // unused: the side to move is encoded in rootBoard
    float ucbC;
    int thrifty;
    int collect;
    int slotsPerGroup;   // children served by one playout group, 1..MAX_SLOTS_PER_GROUP
    int slotsPerTree;    // prodigal grid: MAX_ACTIONS (ACP) or 1 (OCP)
};
[[vk::push_constant]] ConstantBuffer<PushConsts> pc;

// =============================================================================
// GAME: Connect Four bitboards. Bit index = col*7 + row (row 0 = bottom, row 6 = sentinel).
// A position is (pos, mask): mask = all stones, pos = stones of the player to move.
// =============================================================================
//
// No 64-bit shift in this file uses a run-time shift amount. On the NVIDIA driver used for the
// paper (610.60, RTX 3050 Laptop) `uint64_t(1) << (th * 7 + 5)` evaluated wrongly in the ACP expand
// kernel: thread 3 saw a full column as legal and created an 8th child, corrupting the tree
// (ACP-prodigal became non-deterministic and diverged from ACP-thrifty). Single bits are now built
// from 32-bit halves, and legal moves are extracted with constant shifts only.
static const uint64_t TOP_ALL = 0x810204081020;   // row 5 of every column
static const uint64_t BOTTOM_ALL = 0x40810204081; // row 0 of every column
static const uint64_t BOARD_ALL = 0xfdfbf7efdfbf; // rows 0..5 of every column (no sentinels)

// 64-bit value with only bit s set (0 <= s < 64), from two 32-bit shifts
uint64_t bit64(int s)
{
    uint u = uint(s);
    uint lo = (u < 32u) ? (1u << (u & 31u)) : 0u;
    uint hi = (u < 32u) ? 0u : (1u << (u & 31u));
    return (uint64_t(hi) << 32) | uint64_t(lo);
}
uint64_t bottomOf(int c) { return bit64(c * 7); }
uint64_t colMask(int c)  { return bit64(c * 7 + 6) - bit64(c * 7); }   // rows 0..5 of column c

// bit c set iff column c is not full (constant shifts only)
int legalMoves(uint64_t mask)
{
    uint64_t f = ~mask & TOP_ALL;
    uint lo = uint(f & uint64_t(0xFFFFFFFF));
    uint hi = uint(f >> 32);
    uint b = ((lo >> 5) & 1u) | ((lo >> 11) & 2u) | ((lo >> 17) & 4u) | ((lo >> 23) & 8u)
           | ((hi << 3) & 16u) | ((hi >> 3) & 32u) | ((hi >> 9) & 64u);   // bits 33, 40, 47
    return int(b);
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
    status = win ? STATUS_WIN : (((cmask & TOP_ALL) == TOP_ALL) ? STATUS_DRAW : STATUS_ONGOING);
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

// same as hash32 in mcts_gpu.cpp
uint hash32(uint x)
{
    x ^= x >> 16; x *= 0x7feb352du;
    x ^= x >> 15; x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

// seed of global iteration (searchParams[2] + iterOffset), 1-based as on the host before
uint iterSeed()
{
    uint iter = searchParams[2] + uint(pc.iterOffset) + 1u;
    return hash32(searchParams[0] ^ hash32(searchParams[1] + 0x9E3779B9u * iter));
}

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
        dispatchArgs[3] = 0;
    }

    int node = 0;
    int depth = 0;
    uint seed = iterSeed();
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
        sh_prio[th] = pcg(seed ^ pcg(uint(t) * 4096u + uint(depth) * 8u + uint(th) + 0x2545F491u));
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
groupshared uint sh_pos[2];    // leaf position as 32-bit halves
groupshared uint sh_mask[2];
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
            sh_pos[0] = uint(pos & uint64_t(0xFFFFFFFF)); sh_pos[1] = uint(pos >> 32);
            sh_mask[0] = uint(mask & uint64_t(0xFFFFFFFF)); sh_mask[1] = uint(mask >> 32);
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

    uint64_t pos = (uint64_t(sh_pos[1]) << 32) | uint64_t(sh_pos[0]);
    uint64_t mask = (uint64_t(sh_mask[1]) << 32) | uint64_t(sh_mask[0]);
    int nextId = sh_nextId;

    // count, ord (legal actions below th) and the "create a child" test all come from one mask,
    // so they cannot disagree
    int legal = legalMoves(mask);
    int count = int(countbits(uint(legal)));
    int ord = int(countbits(uint(legal) & ((1u << uint(th)) - 1u)));

    if (th < MAX_ACTIONS && ((legal >> th) & 1) != 0)
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
            InterlockedAdd(dispatchArgs[3], count, off);   // reserve `count` playout slots
            for (int k = 0; k < count; k++) blockMap[off + k] = t * MAX_ACTIONS + k;
            InterlockedMax(dispatchArgs[0], (off + count + pc.slotsPerGroup - 1) / pc.slotsPerGroup);
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
    int legal = legalMoves(mask);
    for (int a = 0; a < MAX_ACTIONS; a++)
    {
        if (((legal >> a) & 1) != 0 && trees[(base + node) * NODE_ROW + 1 + a] < 0)
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

    uint rng = pcg(iterSeed() ^ pcg(uint(t) + 0x51ed270bu));
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
        InterlockedAdd(dispatchArgs[3], 1, off);
        blockMap[off] = t * MAX_ACTIONS;
        InterlockedMax(dispatchArgs[0], (off + 1 + pc.slotsPerGroup - 1) / pc.slotsPerGroup);
    }
    if (pc.collect != 0)
    {
        InterlockedAdd(counters[CTR_CHILDREN_CREATED], 1);
        InterlockedMax(counters[CTR_TREE_SIZE_MAX], id + 1);
    }
}

// playout ----------------------------------------------------------------------
// One 32-lane group serves up to slotsPerGroup child slots (S). Thread i < S claims candidate i:
// terminal children are scored at once (virtual playouts), ongoing ones are queued in groupshared
// memory. The group then runs nSim * 32 playouts; a lane that finishes takes the next playout
// index from sh_next. Playout w belongs to queued slot w / 32 and uses RNG stream
// (slot * 32 + w % 32), exactly as v2, so per-child sums do not depend on S or on which lane ran it.
groupshared int sh_qslot[MAX_SLOTS_PER_GROUP];
groupshared uint sh_qbb[MAX_SLOTS_PER_GROUP * 4];   // pos lo, pos hi, mask lo, mask hi
groupshared int sh_qsum[MAX_SLOTS_PER_GROUP];
groupshared int sh_nSim;
groupshared uint sh_next;
groupshared int sh_pliesSum;
groupshared int sh_pliesMax;
groupshared int sh_nVirtual;

// credit a child with its PLAYOUTS_PER_CHILD playouts (sum = total score in half-points)
void finishSlot(int slot, int sum)
{
    childScores[slot] = sum;
    int t = slot / MAX_ACTIONS;
    int act = actionsExp[slot];
    int node = selNode[t];
    int child = trees[(t * pc.maxNodes + node) * NODE_ROW + 1 + act];
    nodeN[t * pc.maxNodes + child] = PLAYOUTS_PER_CHILD;
    nodeWins[t * pc.maxNodes + child] = sum;
}

[shader("compute")]
[numthreads(32, 1, 1)]
void mctsPlayout(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID)
{
    int th = int(tid.x);
    int S = pc.slotsPerGroup;
    uint seed = iterSeed();

    if (th == 0)
    {
        sh_nSim = 0;
        sh_next = 0u;
        sh_pliesSum = 0;
        sh_pliesMax = 0;
        sh_nVirtual = 0;
    }
    GroupMemoryBarrierWithGroupSync();

    if (th < S)
    {
        int idx = int(gid.x) * S + th;
        int slot = -1;
        if (pc.thrifty != 0)
        {
            if (idx < dispatchArgs[3]) slot = blockMap[idx];
        }
        else if (idx < pc.numTrees * pc.slotsPerTree)
        {
            slot = (idx / pc.slotsPerTree) * MAX_ACTIONS + (idx % pc.slotsPerTree);
        }
        if (slot >= 0 && actionsExp[slot] >= 0)
        {
            int rec = slot * CB_STRIDE;
            int status = childBoards[rec + 4];
            if (status == STATUS_ONGOING)
            {
                int k;
                InterlockedAdd(sh_nSim, 1, k);
                sh_qslot[k] = slot;
                sh_qbb[k * 4 + 0] = uint(childBoards[rec + 0]);
                sh_qbb[k * 4 + 1] = uint(childBoards[rec + 1]);
                sh_qbb[k * 4 + 2] = uint(childBoards[rec + 2]);
                sh_qbb[k * 4 + 3] = uint(childBoards[rec + 3]);
                sh_qsum[k] = 0;
            }
            else
            {
                // the mover into the child won (2 per playout) or the board is full (1 per playout)
                finishSlot(slot, (status == STATUS_WIN) ? 2 * PLAYOUTS_PER_CHILD : PLAYOUTS_PER_CHILD);
                if (pc.collect != 0) InterlockedAdd(sh_nVirtual, 1);
            }
        }
    }
    GroupMemoryBarrierWithGroupSync();

    uint total = uint(sh_nSim) * uint(PLAYOUTS_PER_CHILD);

    // The first mover of the playout is the player to move at the child, i.e. NOT the
    // player q who moved into it. So the mover of ply k is q iff k is even.
    bool busy = false;
    int k = 0;
    uint64_t pos = 0;
    uint64_t mask = 0;
    uint rng = 0u;
    int plies = 0;
    int lanePlies = 0;
    while (true)
    {
        if (!busy)
        {
            uint w;
            InterlockedAdd(sh_next, 1u, w);
            if (w >= total) break;
            k = int(w / uint(PLAYOUTS_PER_CHILD));
            int j = int(w % uint(PLAYOUTS_PER_CHILD));
            pos = (uint64_t(sh_qbb[k * 4 + 1]) << 32) | uint64_t(sh_qbb[k * 4 + 0]);
            mask = (uint64_t(sh_qbb[k * 4 + 3]) << 32) | uint64_t(sh_qbb[k * 4 + 2]);
            rng = pcg(seed ^ pcg(uint(sh_qslot[k]) * 32u + uint(j) + 1u));
            plies = 0;
            busy = true;
        }

        int score = -1;
        // Next free cell of every non-full column. Bits are ordered by column, so the r-th set bit
        // is the r-th legal column: the same move v2 picked from its legal-column list.
        uint64_t moves = (mask + BOTTOM_ALL) & BOARD_ALL;
        uint mlo = uint(moves & uint64_t(0xFFFFFFFF));
        uint mhi = uint(moves >> 32);
        int n = int(countbits(mlo) + countbits(mhi));
        if (n == 0)
        {
            score = 1;
        }
        else
        {
            rng = pcg(rng);
            int r = int(rng % uint(n));
            for (int i = 0; i < r; i++)            // drop the r lowest candidate cells
            {
                if (mlo != 0u) mlo &= mlo - 1u;
                else mhi &= mhi - 1u;
            }
            uint64_t mv = (mlo != 0u) ? uint64_t(mlo & (0u - mlo)) : (uint64_t(mhi & (0u - mhi)) << 32);

            plies++;
            if (alignment(pos | mv))
            {
                score = ((plies & 1) == 0) ? 2 : 0;
            }
            else
            {
                pos ^= mask;
                mask |= mv;
            }
        }

        if (score >= 0)
        {
            InterlockedAdd(sh_qsum[k], score);
            lanePlies += plies;
            busy = false;
        }
    }

    if (pc.collect != 0)
    {
        InterlockedAdd(sh_pliesSum, lanePlies);
        InterlockedMax(sh_pliesMax, lanePlies);
    }
    GroupMemoryBarrierWithGroupSync();

    if (th < sh_nSim) finishSlot(sh_qslot[th], sh_qsum[th]);

    if (th == 0 && pc.collect != 0)
    {
        if (sh_nSim > 0)
        {
            InterlockedAdd(counters[CTR_BLOCKS_REAL], sh_nSim);
            InterlockedAdd(counters[CTR_PLIES_SUM], sh_pliesSum);
            InterlockedAdd(counters[CTR_PLIES_MAXSUM], sh_pliesMax);   // lane-time of the busiest lane
        }
        if (sh_nVirtual > 0) InterlockedAdd(counters[CTR_BLOCKS_VIRTUAL], sh_nVirtual);
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

        // thrifty: record this iteration's queued slots (dispatchArgs is stable until next select)
        if (t == 0 && pc.thrifty != 0 && pc.collect != 0)
        {
            int A = dispatchArgs[3];
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
    uint rng = pcg(iterSeed() ^ 0xC0FFEE11u);
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
