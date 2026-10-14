// =============================================================================
// mcts_connect4.slang
//
// GPU Monte Carlo Tree Search from MCTS-NC (Klesk 2025), specialised to
// Connect Four (7 cols x 6 rows). All four variants live in this one module:
//
//                |  prodigal                    |  thrifty
//   -------------+------------------------------+-------------------------------
//   ACP (all     |  expand: mctsExpand          |  expand: mctsExpand (compacts)
//   children)    |  playout grid (T, B)         |  playout grid (A), host readback
//   OCP (one     |  expand: mctsExpandOcp       |  expand: mctsExpandOcp
//   child)       |  playout grid (T)            |  playout grid (A), host readback
//
// The host picks the expand kernel and the playout grid. pc.thrifty tells the
// playout kernel how to find its (tree, slot).
//
// Per iteration:
//   select  -> 1 workgroup per tree, threads = actions (UCB + argmax reduction)
//   expand  -> 1 thread per tree: replays the path, creates child(ren)
//   playout -> 1 workgroup per child slot, 32 threads = 32 playouts per child
//   backup  -> 1 workgroup per tree, threads = nodes on the selected path
// After all iterations:
//   reduce over trees (workgroup per root action), argmax over actions.
//
// Compile:
//   slangc mcts_connect4.slang -target spirv -fvk-use-entrypoint-name -o mcts_connect4.spv
//
// Statistics convention (important):
//   For a node X, nodeWins[X] / (2 * nodeN[X]) is the win-rate for the player
//   who made the move INTO X.  (win = 2 points, draw = 1, loss = 0.)
//   So a parent simply picks the child with the highest value - no negation.
//
// Slot convention: slot = t * MAX_ACTIONS + k addresses actionsExp, childScores
// and childBoards.
//   ACP prodigal : k = action a   (actionsExp[slot] = a or -1)
//   ACP thrifty  : k = ordinal among expanded children (0..b_t-1), compacted
//   OCP          : k = 0 (a single child per tree)
// actionsExp[slot] always holds the real column, or -1 if the slot is unused.
//
// STATS PATCH: binding 16 `counters` (host zeroes it per search), gated by
// pc.collect. Tie-breaks (select + final argmax) are now uniformly random
// instead of "lowest column wins".
// =============================================================================

// ---- constants (keep in sync with mcts_gpu.hpp / mcts_stats.hpp) -------------
static const int COLS = 7;
static const int ROWS = 6;
static const int CELLS = 42;
static const int MAX_ACTIONS = 7;
static const int NODE_ROW = 8;            // 1 (parent) + MAX_ACTIONS (children)
static const int PATH_STRIDE = 44;        // max path nodes = 43
static const int INFO_STRIDE = 4;         // [0]=pathLen [1]=numExpanded [2]=terminal S_R
static const int CB_STRIDE = 44;          // 42 cells + turn + status
static const int PLAYOUTS_PER_CHILD = 32; // == local size of mctsPlayout
static const int SEL_TPB = 8;             // pow2 >= MAX_ACTIONS, == local size of mctsSelect
static const int REDUCE_TPB = 64;         // == local size of mctsReduceTrees
// mctsReset/mctsExpand/mctsExpandOcp use 32 threads, mctsBackup uses 64 threads

static const int FLAG_LEAF = 1;           // bit0: leaf (has unexpanded children, or terminal)
                                          // bits1-2: status 0=ongoing 1=win for mover-into-node 2=draw
static const int STATUS_ONGOING = 0;
static const int STATUS_WIN = 1;
static const int STATUS_DRAW = 2;

// ---- stats counter slots (== enum Counter in mcts_stats.hpp) -----------------
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

// ---- buffers ----------------------------------------------------------------
[[vk::binding(0, 0)]]  RWStructuredBuffer<int> rootBoard;    // [42] cells: 0 empty, +1, -1
[[vk::binding(1, 0)]]  RWStructuredBuffer<int> trees;        // [T][maxNodes][NODE_ROW] parent, child per action (-1 none)
[[vk::binding(2, 0)]]  RWStructuredBuffer<int> treeSizes;    // [T]
[[vk::binding(3, 0)]]  RWStructuredBuffer<int> nodeFlags;    // [T][maxNodes]
[[vk::binding(4, 0)]]  RWStructuredBuffer<int> nodeN;        // [T][maxNodes] visits
[[vk::binding(5, 0)]]  RWStructuredBuffer<int> nodeWins;     // [T][maxNodes] points (2/1/0)
[[vk::binding(6, 0)]]  RWStructuredBuffer<int> selNode;      // [T]
[[vk::binding(7, 0)]]  RWStructuredBuffer<int> selPath;      // [T][PATH_STRIDE] node ids
[[vk::binding(8, 0)]]  RWStructuredBuffer<int> selActs;      // [T][PATH_STRIDE] action taken at each step
[[vk::binding(9, 0)]]  RWStructuredBuffer<int> selInfo;      // [T][INFO_STRIDE]
[[vk::binding(10, 0)]] RWStructuredBuffer<int> actionsExp;   // [T][MAX_ACTIONS] column or -1 (see slot convention)
[[vk::binding(11, 0)]] RWStructuredBuffer<int> childBoards;  // [T][MAX_ACTIONS][CB_STRIDE]
[[vk::binding(12, 0)]] RWStructuredBuffer<int> childScores;  // [T][MAX_ACTIONS]
[[vk::binding(13, 0)]] RWStructuredBuffer<int> rootStats;    // [2*MAX_ACTIONS] (n, wins) per action, then [14]=best action
[[vk::binding(14, 0)]] RWStructuredBuffer<int> treeB;        // [T] b_t this iteration (host-visible; thrifty reads it back)
[[vk::binding(15, 0)]] RWStructuredBuffer<int> blockMap;     // [A] host-written (thrifty): slot id of each playout block
[[vk::binding(16, 0)]] RWStructuredBuffer<int> counters;     // [NUM_COUNTERS] stats, host-visible, zeroed per search

struct PushConsts
{
    int numTrees;
    int maxNodes;
    uint seed;
    int rootTurn;   // +1 or -1, player to move at the root
    float ucbC;
    int thrifty;    // 1: playout grid is blockMap-driven, 0: grid is (tree, slot)
    int collect;    // 1: update `counters`
};
[[vk::push_constant]] ConstantBuffer<PushConsts> pc;

// =============================================================================
// GAME: Connect Four.  Replace this block (+ the constants) to change games.
// =============================================================================
struct State
{
    int cells[CELLS];   // index = row * COLS + col, row 0 = bottom
    int heights[COLS];
    int turn;           // +1 / -1, player to move
    int moves;
};

static const int DC[4] = { 1, 0, 1, 1 };
static const int DR[4] = { 0, 1, 1, -1 };

// derive heights + move count from cells
void finishState(inout State s)
{
    s.moves = 0;
    for (int c = 0; c < COLS; c++)
    {
        int h = 0;
        while (h < ROWS && s.cells[h * COLS + c] != 0) h++;
        s.heights[c] = h;
        s.moves += h;
    }
}

bool checkWin(in State s, int col, int row, int p)
{
    for (int d = 0; d < 4; d++)
    {
        int count = 1;
        for (int sgn = -1; sgn <= 1; sgn += 2)
        {
            int c = col + sgn * DC[d];
            int r = row + sgn * DR[d];
            while (c >= 0 && c < COLS && r >= 0 && r < ROWS && s.cells[r * COLS + c] == p)
            {
                count++;
                c += sgn * DC[d];
                r += sgn * DR[d];
            }
        }
        if (count >= 4) return true;
    }
    return false;
}

// plays a legal move; returns true if it wins the game
bool playMove(inout State s, int col)
{
    int row = s.heights[col];
    int p = s.turn;
    s.cells[row * COLS + col] = p;
    s.heights[col] = row + 1;
    s.moves += 1;
    s.turn = -p;
    return checkWin(s, col, row, p);
}

State loadRootState()
{
    State s;
    for (int i = 0; i < CELLS; i++) s.cells[i] = rootBoard[i];
    finishState(s);
    s.turn = pc.rootTurn;
    return s;
}
// =============================================================================

uint pcg(uint v)
{
    uint state = v * 747796405u + 2891336245u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// -----------------------------------------------------------------------------
// reset: root node of every tree
// -----------------------------------------------------------------------------
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

// -----------------------------------------------------------------------------
// select: walk down by UCB1 until a leaf. One workgroup per tree.
// OCP note: a node stays flagged as leaf until ALL its legal children exist,
// so selection only descends through fully expanded nodes (standard MCTS).
// Ties in UCB are broken by a per-(tree, depth, iteration) random priority, so
// no column is systematically preferred.
// -----------------------------------------------------------------------------
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

    int node = 0;
    int depth = 0;
    if (th == 0) selPath[t * PATH_STRIDE] = 0;

    // loop condition reads the same memory in every thread -> uniform
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
        GroupMemoryBarrierWithGroupSync(); // everyone has read before next overwrite

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
        selInfo[t * INFO_STRIDE + 0] = depth + 1; // path length in nodes
        if (pc.collect != 0)
        {
            InterlockedAdd(counters[CTR_DEPTH_SUM], depth);
            InterlockedMax(counters[CTR_DEPTH_MAX], depth);
        }
    }
}

// -----------------------------------------------------------------------------
// expand (ACP): create ALL children of the selected leaf. One thread per tree.
// Also stores each child's board so the playout kernel doesn't have to replay.
// prodigal: child of action a goes to slot index a
// thrifty : children are compacted to slot indices 0..b_t-1
// -----------------------------------------------------------------------------
[shader("compute")]
[numthreads(32, 1, 1)]
void mctsExpand(uint3 dtid : SV_DispatchThreadID)
{
    int t = int(dtid.x);
    if (t >= pc.numTrees) return;

    int base = t * pc.maxNodes;
    int node = selNode[t];
    int depth = selInfo[t * INFO_STRIDE + 0] - 1;

    // clear first, so stale entries never trigger playouts
    for (int a = 0; a < MAX_ACTIONS; a++)
    {
        actionsExp[t * MAX_ACTIONS + a] = -1;
        childScores[t * MAX_ACTIONS + a] = 0;
    }
    treeB[t] = 0;

    int fl = nodeFlags[base + node];
    int status = (fl >> 1) & 3;
    if (status != STATUS_ONGOING)
    {
        // terminal leaf: nothing to expand. Outcome is known exactly.
        // S_R = points (from the view of the player who moved INTO this node)
        // for N = PLAYOUTS_PER_CHILD virtual playouts.
        selInfo[t * INFO_STRIDE + 1] = 0;
        selInfo[t * INFO_STRIDE + 2] = (status == STATUS_WIN) ? 2 * PLAYOUTS_PER_CHILD : PLAYOUTS_PER_CHILD;
        if (pc.collect != 0) InterlockedAdd(counters[CTR_TERMINAL_HITS], 1);
        return;
    }

    // replay path from the root to rebuild the state of the selected node
    State s = loadRootState();
    for (int d = 0; d < depth; d++) playMove(s, selActs[t * PATH_STRIDE + d]);

    int nextId = treeSizes[t];
    int count = 0;
    for (int a = 0; a < MAX_ACTIONS; a++)
    {
        if (s.heights[a] >= ROWS) continue;

        State c = s;
        bool win = playMove(c, a);
        int cstatus = win ? STATUS_WIN : ((c.moves == CELLS) ? STATUS_DRAW : STATUS_ONGOING);

        int id = nextId + count;
        int idx = (pc.thrifty != 0) ? count : a;   // slot index
        count++;

        // link + init child node
        trees[(base + node) * NODE_ROW + 1 + a] = id;
        int row = (base + id) * NODE_ROW;
        trees[row] = node;
        for (int j = 1; j < NODE_ROW; j++) trees[row + j] = -1;
        nodeN[base + id] = 0;
        nodeWins[base + id] = 0;
        nodeFlags[base + id] = FLAG_LEAF | (cstatus << 1);

        // child board record for the playout kernel
        int rec = (t * MAX_ACTIONS + idx) * CB_STRIDE;
        for (int i = 0; i < CELLS; i++) childBoards[rec + i] = c.cells[i];
        childBoards[rec + CELLS] = c.turn;
        childBoards[rec + CELLS + 1] = cstatus;

        actionsExp[t * MAX_ACTIONS + idx] = a;
    }

    treeSizes[t] = nextId + count;
    nodeFlags[base + node] = 0;                 // no longer a leaf
    selInfo[t * INFO_STRIDE + 1] = count;       // b_t
    treeB[t] = count;
    if (pc.collect != 0)
    {
        InterlockedAdd(counters[CTR_CHILDREN_CREATED], count);
        InterlockedMax(counters[CTR_TREE_SIZE_MAX], nextId + count);
    }
}

// -----------------------------------------------------------------------------
// expand (OCP): create ONE child, chosen uniformly among the untried legal
// actions of the selected leaf. The leaf stays a leaf until its last child is
// created. One thread per tree. Child always lives in slot index 0.
// -----------------------------------------------------------------------------
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

    int fl = nodeFlags[base + node];
    int status = (fl >> 1) & 3;
    if (status != STATUS_ONGOING)
    {
        selInfo[t * INFO_STRIDE + 1] = 0;
        selInfo[t * INFO_STRIDE + 2] = (status == STATUS_WIN) ? 2 * PLAYOUTS_PER_CHILD : PLAYOUTS_PER_CHILD;
        if (pc.collect != 0) InterlockedAdd(counters[CTR_TERMINAL_HITS], 1);
        return;
    }

    State s = loadRootState();
    for (int d = 0; d < depth; d++) playMove(s, selActs[t * PATH_STRIDE + d]);

    // untried = legal and no child linked yet
    int untried[MAX_ACTIONS];
    int u = 0;
    for (int a = 0; a < MAX_ACTIONS; a++)
    {
        if (s.heights[a] < ROWS && trees[(base + node) * NODE_ROW + 1 + a] < 0)
        {
            untried[u] = a;
            u++;
        }
    }
    if (u == 0)
    {
        // should not happen (fully expanded nodes are not leaves); treat as drawn leaf
        selInfo[t * INFO_STRIDE + 1] = 0;
        selInfo[t * INFO_STRIDE + 2] = PLAYOUTS_PER_CHILD;
        if (pc.collect != 0) InterlockedAdd(counters[CTR_ANOMALIES], 1);
        return;
    }

    uint rng = pcg(pc.seed ^ pcg(uint(t) + 0x51ed270bu));
    int act = untried[int(pcg(rng) % uint(u))];

    State c = s;
    bool win = playMove(c, act);
    int cstatus = win ? STATUS_WIN : ((c.moves == CELLS) ? STATUS_DRAW : STATUS_ONGOING);

    int id = treeSizes[t];
    trees[(base + node) * NODE_ROW + 1 + act] = id;
    int row = (base + id) * NODE_ROW;
    trees[row] = node;
    for (int j = 1; j < NODE_ROW; j++) trees[row + j] = -1;
    nodeN[base + id] = 0;
    nodeWins[base + id] = 0;
    nodeFlags[base + id] = FLAG_LEAF | (cstatus << 1);

    int rec = (t * MAX_ACTIONS) * CB_STRIDE;
    for (int i = 0; i < CELLS; i++) childBoards[rec + i] = c.cells[i];
    childBoards[rec + CELLS] = c.turn;
    childBoards[rec + CELLS + 1] = cstatus;

    actionsExp[t * MAX_ACTIONS] = act;
    treeSizes[t] = id + 1;
    if (u == 1) nodeFlags[base + node] = 0;     // that was the last untried child
    selInfo[t * INFO_STRIDE + 1] = 1;           // b_t
    treeB[t] = 1;
    if (pc.collect != 0)
    {
        InterlockedAdd(counters[CTR_CHILDREN_CREATED], 1);
        InterlockedMax(counters[CTR_TREE_SIZE_MAX], id + 1);
    }
}

// -----------------------------------------------------------------------------
// playout: one workgroup per child slot. 32 threads = 32 random playouts.
// Scores are from the view of the player who moved INTO the child.
// Grid layouts:
//   prodigal ACP : (T, MAX_ACTIONS)  slot = gid.x * MAX_ACTIONS + gid.y
//   prodigal OCP : (T, 1)            slot = gid.x * MAX_ACTIONS
//   thrifty      : (A)               slot = blockMap[gid.x]
// -----------------------------------------------------------------------------
groupshared int sh_score[PLAYOUTS_PER_CHILD];
groupshared int sh_plies[PLAYOUTS_PER_CHILD];

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
    if (act < 0) return;  // excess blocks bail out (uniform per group)

    int rec = slot * CB_STRIDE;
    State s;
    for (int i = 0; i < CELLS; i++) s.cells[i] = childBoards[rec + i];
    finishState(s);
    s.turn = childBoards[rec + CELLS];
    int status = childBoards[rec + CELLS + 1];
    int q = -s.turn; // player who moved into the child

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
            int legal[MAX_ACTIONS];
            int n = 0;
            for (int c = 0; c < COLS; c++)
            {
                if (s.heights[c] < ROWS) { legal[n] = c; n++; }
            }
            if (n == 0) { score = 1; break; }

            rng = pcg(rng);
            int col = legal[int(rng % uint(n))];
            int mover = s.turn;
            plies++;
            if (playMove(s, col))
            {
                score = (mover == q) ? 2 : 0;
                break;
            }
        }
    }

    sh_score[th] = score;
    sh_plies[th] = plies;
    GroupMemoryBarrierWithGroupSync();
    for (int stride = PLAYOUTS_PER_CHILD >> 1; stride > 0; stride >>= 1)
    {
        if (th < stride)
        {
            sh_score[th] += sh_score[th + stride];
            sh_plies[th] += sh_plies[th + stride];
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
            }
            else
            {
                InterlockedAdd(counters[CTR_BLOCKS_VIRTUAL], 1);
            }
        }
    }
}

// -----------------------------------------------------------------------------
// backup: sum children results, then update every node on the selected path.
// One workgroup per tree, one thread per path node. Shared by ACP and OCP
// (OCP simply has b = 1).
// -----------------------------------------------------------------------------
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
            // terminal leaf
            sh_N = PLAYOUTS_PER_CHILD;
            sh_SR = selInfo[t * INFO_STRIDE + 2];
        }
        else
        {
            int sq = 0; // points from the view of the player TO MOVE at the leaf
            for (int a = 0; a < MAX_ACTIONS; a++)
            {
                if (actionsExp[t * MAX_ACTIONS + a] >= 0) sq += childScores[t * MAX_ACTIONS + a];
            }
            int N = b * PLAYOUTS_PER_CHILD;
            sh_N = N;
            sh_SR = 2 * N - sq; // flip to the view of the player who moved INTO the leaf
        }
    }
    GroupMemoryBarrierWithGroupSync();

    int pathLen = selInfo[t * INFO_STRIDE + 0];
    int k = pathLen - 1; // depth of the leaf
    if (th < pathLen)
    {
        int node = selPath[t * PATH_STRIDE + th];
        int N = sh_N;
        int SR = sh_SR;
        // perspective alternates every level
        int w = (((k - th) & 1) == 0) ? SR : (2 * N - SR);
        nodeN[t * pc.maxNodes + node] += N;
        nodeWins[t * pc.maxNodes + node] += w;
    }
}

// -----------------------------------------------------------------------------
// reduce over trees: one workgroup per root action
// -----------------------------------------------------------------------------
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
        int child = trees[(t * pc.maxNodes) * NODE_ROW + 1 + a]; // root is node 0
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

// -----------------------------------------------------------------------------
// argmax over root actions (most visited, ties -> more wins, remaining ties ->
// uniformly random via reservoir sampling). B is tiny, so 1 thread.
// -----------------------------------------------------------------------------
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