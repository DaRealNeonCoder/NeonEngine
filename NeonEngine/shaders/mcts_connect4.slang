// =============================================================================
// mcts_connect4.slang
//
// GPU Monte Carlo Tree Search, "ACP-prodigal" variant from MCTS-NC (Klesk 2025),
// specialised to Connect Four (7 cols x 6 rows).
//
//   T independent trees (root parallelism), one iteration =
//     select  -> 1 workgroup per tree, threads = actions (UCB + argmax reduction)
//     expand  -> 1 thread per tree: replays the path, creates ALL children
//     playout -> 1 workgroup per (tree, action), 32 threads = 32 playouts/child
//     backup  -> 1 workgroup per tree, threads = nodes on the selected path
//   after all iterations:
//     reduce over trees (workgroup per root action), argmax over actions.
//
// "Prodigal" = the playout grid is always (T, MAX_ACTIONS); groups whose action
// is illegal return immediately.  No device<->host traffic during the search.
//
// Compile:
//   slangc mcts_connect4.slang -target spirv -fvk-use-entrypoint-name -o mcts_connect4.spv
//
// Statistics convention (important):
//   For a node X, nodeWins[X] / (2 * nodeN[X]) is the win-rate for the player
//   who made the move INTO X.  (win = 2 points, draw = 1, loss = 0.)
//   So a parent simply picks the child with the highest value - no negation.
// =============================================================================

// ---- constants (keep in sync with mcts_gpu.hpp) -----------------------------
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
// mctsReset/mctsExpand use 32 threads, mctsBackup uses 64 threads

static const int FLAG_LEAF = 1;           // bit0: leaf (unexpanded or terminal)
                                          // bits1-2: status 0=ongoing 1=win for mover-into-node 2=draw
static const int STATUS_ONGOING = 0;
static const int STATUS_WIN = 1;
static const int STATUS_DRAW = 2;

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
[[vk::binding(10, 0)]] RWStructuredBuffer<int> actionsExp;   // [T][MAX_ACTIONS] prodigal indexing: a or -1
[[vk::binding(11, 0)]] RWStructuredBuffer<int> childBoards;  // [T][MAX_ACTIONS][CB_STRIDE]
[[vk::binding(12, 0)]] RWStructuredBuffer<int> childScores;  // [T][MAX_ACTIONS]
[[vk::binding(13, 0)]] RWStructuredBuffer<int> rootStats;    // [2*MAX_ACTIONS] (n, wins) per action, then [14]=best action

struct PushConsts
{
    int numTrees;
    int maxNodes;
    uint seed;
    int rootTurn;   // +1 or -1, player to move at the root
    float ucbC;
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
// -----------------------------------------------------------------------------
groupshared float sh_ucb[SEL_TPB];
groupshared int sh_child[SEL_TPB];
groupshared int sh_act[SEL_TPB];

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
        GroupMemoryBarrierWithGroupSync();

        for (int stride = SEL_TPB >> 1; stride > 0; stride >>= 1)
        {
            if (th < stride && sh_ucb[th] < sh_ucb[th + stride])
            {
                sh_ucb[th] = sh_ucb[th + stride];
                sh_child[th] = sh_child[th + stride];
                sh_act[th] = sh_act[th + stride];
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
    }
}

// -----------------------------------------------------------------------------
// expand: create ALL children of the selected leaf (ACP). One thread per tree.
// Also stores each child's board so the playout kernel doesn't have to replay.
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

    // prodigal indexing: clear first, so stale entries never trigger playouts
    for (int a = 0; a < MAX_ACTIONS; a++)
    {
        actionsExp[t * MAX_ACTIONS + a] = -1;
        childScores[t * MAX_ACTIONS + a] = 0;
    }

    int fl = nodeFlags[base + node];
    int status = (fl >> 1) & 3;
    if (status != STATUS_ONGOING)
    {
        // terminal leaf: nothing to expand. Outcome is known exactly.
        // S_R = points (from the view of the player who moved INTO this node)
        // for N = PLAYOUTS_PER_CHILD virtual playouts.
        selInfo[t * INFO_STRIDE + 1] = 0;
        selInfo[t * INFO_STRIDE + 2] = (status == STATUS_WIN) ? 2 * PLAYOUTS_PER_CHILD : PLAYOUTS_PER_CHILD;
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
        int rec = (t * MAX_ACTIONS + a) * CB_STRIDE;
        for (int i = 0; i < CELLS; i++) childBoards[rec + i] = c.cells[i];
        childBoards[rec + CELLS] = c.turn;
        childBoards[rec + CELLS + 1] = cstatus;

        actionsExp[t * MAX_ACTIONS + a] = a;
    }

    treeSizes[t] = nextId + count;
    nodeFlags[base + node] = 0;                 // no longer a leaf
    selInfo[t * INFO_STRIDE + 1] = count;       // b_t
}

// -----------------------------------------------------------------------------
// playout: one workgroup per (tree, action). 32 threads = 32 random playouts.
// Scores are from the view of the player who moved INTO the child.
// -----------------------------------------------------------------------------
groupshared int sh_score[PLAYOUTS_PER_CHILD];

[shader("compute")]
[numthreads(32, 1, 1)]
void mctsPlayout(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID)
{
    int t = int(gid.x);
    int a = int(gid.y);
    int th = int(tid.x);
    int slot = t * MAX_ACTIONS + a;

    if (actionsExp[slot] < 0) return;  // prodigal: excess blocks bail out (uniform per group)

    int rec = slot * CB_STRIDE;
    State s;
    for (int i = 0; i < CELLS; i++) s.cells[i] = childBoards[rec + i];
    finishState(s);
    s.turn = childBoards[rec + CELLS];
    int status = childBoards[rec + CELLS + 1];
    int q = -s.turn; // player who moved into the child

    int score = 1;
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
            if (playMove(s, col))
            {
                score = (mover == q) ? 2 : 0;
                break;
            }
        }
    }

    sh_score[th] = score;
    GroupMemoryBarrierWithGroupSync();
    for (int stride = PLAYOUTS_PER_CHILD >> 1; stride > 0; stride >>= 1)
    {
        if (th < stride) sh_score[th] += sh_score[th + stride];
        GroupMemoryBarrierWithGroupSync();
    }

    if (th == 0)
    {
        int sum = sh_score[0];
        childScores[slot] = sum;

        int node = selNode[t];
        int child = trees[(t * pc.maxNodes + node) * NODE_ROW + 1 + a];
        nodeN[t * pc.maxNodes + child] = PLAYOUTS_PER_CHILD;
        nodeWins[t * pc.maxNodes + child] = sum;
    }
}

// -----------------------------------------------------------------------------
// backup: sum children results, then update every node on the selected path.
// One workgroup per tree, one thread per path node.
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
// argmax over root actions (most visited, ties -> more wins). B is tiny, so 1 thread.
// -----------------------------------------------------------------------------
[shader("compute")]
[numthreads(1, 1, 1)]
void mctsReduceActions()
{
    int best = -1;
    int bestN = -1;
    int bestW = -1;
    for (int a = 0; a < MAX_ACTIONS; a++)
    {
        int n = rootStats[2 * a];
        int w = rootStats[2 * a + 1];
        if (n > 0 && (n > bestN || (n == bestN && w > bestW)))
        {
            best = a;
            bestN = n;
            bestW = w;
        }
    }
    rootStats[2 * MAX_ACTIONS] = best;
}
