#pragma once
// =============================================================================
// mcts_stats.hpp  --  per-search statistics for MctsGpu (all four variants)
//
// One SearchStats is produced per MctsGpu::search() call. Serialise it with
// toJson() and append it (one line per move) to a JSONL log next to the game
// records; aggregate offline.
// =============================================================================
#include <algorithm>
#include <array>
#include <cstdint>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

namespace lve {

    constexpr uint32_t kAutoSeed = 0xFFFFFFFFu;  // "derive callSeed from the internal call counter"
    constexpr int kNumCounters = 16;             // == NUM_COUNTERS in mcts_connect4.slang
    constexpr int kPlayoutsPerChild = 32;        // == PLAYOUTS_PER_CHILD in the shader
    constexpr int kStatsKernels = 8;             // == MctsGpu::NumKernels

    // Slots of the shader-side `counters` buffer. Keep in sync with CTR_* in the .slang file.
    enum Counter : int {
        CtrDepthSum = 0,      // sum over (tree, iteration) of selected-leaf depth (edges from root)
        CtrDepthMax = 1,      // deepest selected leaf
        CtrTerminalHits = 2,  // (tree, iteration) pairs whose selected leaf was already terminal
        CtrChildrenCreated = 3,
        CtrTreeSizeMax = 4,   // largest tree (nodes) at the end of any iteration
        CtrBlocksReal = 5,    // playout workgroups that ran real random playouts
        CtrBlocksVirtual = 6, // playout workgroups for terminal children (outcome known, no simulation)
        CtrPliesSum = 7,      // total plies over all real playouts
        CtrAnomalies = 8,     // must be 0: "impossible" branches that were hit
        CtrFinalTies = 9,     // number of root actions tied for best at the final argmax
    };

    inline const char* const kKernelNames[kStatsKernels] = {
        "reset", "select", "expand", "playout", "backup", "reduce_trees", "reduce_actions", "expand_ocp" };

    // Budget for one search. Both limits may be set; whichever hits first wins.
    struct SearchLimits {
        int maxIterations = 0;  // 0 -> Config::maxIterations
        double maxMs = 0.0;     // 0 -> no wall-clock limit (checked between submits)
    };

    struct SearchStats {
        // ---- identity / config echo (so every record is self-describing) ----------
        std::string algo;
        int numTrees = 0;
        float ucbC = 0.f;
        uint32_t cfgSeed = 0;
        uint32_t callSeed = 0;
        int playoutsPerChild = kPlayoutsPerChild;
        int maxNodesPerTree = 0;
        bool countersEnabled = false;
        bool profiled = false;

        // ---- budget actually consumed ---------------------------------------------
        int iterations = 0;       // per-tree iterations performed
        int submits = 0;          // queue submissions
        int hostRoundTrips = 0;   // thrifty: mid-iteration device->host->device syncs
        double wallMs = 0.0;      // whole search() call, host clock
        double gpuMs = 0.0;       // sum of GPU kernel intervals (timestamp queries)
        double hostMapMs = 0.0;   // thrifty: host time building the block map
        double kernelMs[kStatsKernels] = {};

        // ---- raw counters (from the GPU) ------------------------------------------
        int64_t depthSum = 0, depthMax = 0, terminalHits = 0, childrenCreated = 0, treeSizeMax = 0;
        int64_t blocksReal = 0, blocksVirtual = 0, pliesSum = 0, anomalies = 0, finalTies = 0;

        // ---- raw, host side ---------------------------------------------------------
        int64_t blocksLaunched = 0;  // playout workgroups dispatched (incl. ones that bail out)
        int64_t sumA = 0;            // thrifty: sum of A over iterations
        int64_t maxA = 0;
        uint64_t treeBytes = 0;      // trees + flags + N + wins
        std::vector<int> rootVisits;
        std::vector<float> rootWinRate;  // from the root player's view, in [0,1]
        int bestAction = -1;

        // ---- derived (call finalize()) ----------------------------------------------
        int64_t treeIterations = 0;  // iterations * numTrees
        int64_t nodesTotal = 0;
        double leafDepthMean = 0, terminalHitRate = 0, childrenPerExpansion = 0;
        int64_t playoutsReal = 0, playoutsVirtual = 0, playoutsCredited = 0;
        double playoutLenMean = 0;
        double itersPerSec = 0, playoutsRealPerSec = 0, playoutsCreditedPerSec = 0;
        double overheadMs = 0, overheadFrac = 0;
        double blockUtilActive = 0, blockUtilReal = 0;
        double treeFill = 0;
        double rootVisitsTotal = 0, bestVisitShare = 0, rootValue = 0;

        static double div(double a, double b) { return b > 0.0 ? a / b : 0.0; }

        void finalize() {
            treeIterations = static_cast<int64_t>(iterations) * numTrees;
            nodesTotal = numTrees + childrenCreated;
            leafDepthMean = div(double(depthSum), double(treeIterations));
            terminalHitRate = div(double(terminalHits), double(treeIterations));
            childrenPerExpansion = div(double(childrenCreated), double(treeIterations - terminalHits));
            playoutsReal = blocksReal * playoutsPerChild;
            playoutsVirtual = blocksVirtual * playoutsPerChild;
            playoutsCredited = playoutsReal + playoutsVirtual;
            playoutLenMean = div(double(pliesSum), double(playoutsReal));
            const double s = wallMs * 1e-3;
            itersPerSec = div(iterations, s);
            playoutsRealPerSec = div(double(playoutsReal), s);
            playoutsCreditedPerSec = div(double(playoutsCredited), s);
            overheadMs = std::max(0.0, wallMs - gpuMs);
            overheadFrac = div(overheadMs, wallMs);
            blockUtilActive = div(double(blocksReal + blocksVirtual), double(blocksLaunched));
            blockUtilReal = div(double(blocksReal), double(blocksLaunched));
            treeFill = div(double(treeSizeMax), double(maxNodesPerTree));
            rootVisitsTotal = 0;
            int bestN = 0;
            for (int n : rootVisits) { rootVisitsTotal += n; }
            if (bestAction >= 0 && bestAction < static_cast<int>(rootVisits.size())) {
                bestN = rootVisits[bestAction];
                rootValue = rootWinRate[bestAction];
            }
            bestVisitShare = div(double(bestN), rootVisitsTotal);
        }

        std::string toJson() const {
            std::ostringstream o;
            o << std::setprecision(9);
            auto arrI = [&](const std::vector<int>& v) {
                o << "[";
                for (size_t i = 0; i < v.size(); i++) o << (i ? "," : "") << v[i];
                o << "]";
            };
            auto arrF = [&](const std::vector<float>& v) {
                o << "[";
                for (size_t i = 0; i < v.size(); i++) o << (i ? "," : "") << v[i];
                o << "]";
            };
            o << "{";
            o << "\"algo\":\"" << algo << "\",\"num_trees\":" << numTrees << ",\"ucb_c\":" << ucbC
                << ",\"cfg_seed\":" << cfgSeed << ",\"call_seed\":" << callSeed
                << ",\"playouts_per_child\":" << playoutsPerChild << ",\"max_nodes_per_tree\":" << maxNodesPerTree
                << ",\"counters_enabled\":" << (countersEnabled ? "true" : "false")
                << ",\"profiled\":" << (profiled ? "true" : "false");
            o << ",\"iterations\":" << iterations << ",\"submits\":" << submits
                << ",\"host_round_trips\":" << hostRoundTrips << ",\"wall_ms\":" << wallMs
                << ",\"gpu_ms\":" << gpuMs << ",\"host_map_ms\":" << hostMapMs
                << ",\"overhead_ms\":" << overheadMs << ",\"overhead_frac\":" << overheadFrac;
            o << ",\"kernel_ms\":{";
            for (int k = 0; k < kStatsKernels; k++) o << (k ? "," : "") << "\"" << kKernelNames[k] << "\":" << kernelMs[k];
            o << "}";
            o << ",\"tree\":{\"nodes_total\":" << nodesTotal << ",\"children_created\":" << childrenCreated
                << ",\"tree_size_max\":" << treeSizeMax << ",\"tree_fill\":" << treeFill
                << ",\"leaf_depth_mean\":" << leafDepthMean << ",\"leaf_depth_max\":" << depthMax
                << ",\"terminal_hits\":" << terminalHits << ",\"terminal_hit_rate\":" << terminalHitRate
                << ",\"children_per_expansion\":" << childrenPerExpansion << ",\"bytes\":" << treeBytes << "}";
            o << ",\"playouts\":{\"real\":" << playoutsReal << ",\"virtual\":" << playoutsVirtual
                << ",\"credited\":" << playoutsCredited << ",\"len_mean\":" << playoutLenMean
                << ",\"iters_per_s\":" << itersPerSec << ",\"real_per_s\":" << playoutsRealPerSec
                << ",\"credited_per_s\":" << playoutsCreditedPerSec << "}";
            o << ",\"blocks\":{\"launched\":" << blocksLaunched << ",\"real\":" << blocksReal
                << ",\"virtual\":" << blocksVirtual << ",\"util_active\":" << blockUtilActive
                << ",\"util_real\":" << blockUtilReal << ",\"A_sum\":" << sumA << ",\"A_max\":" << maxA << "}";
            o << ",\"root\":{\"visits\":";
            arrI(rootVisits);
            o << ",\"win_rate\":";
            arrF(rootWinRate);
            o << ",\"best_action\":" << bestAction << ",\"best_visit_share\":" << bestVisitShare
                << ",\"value\":" << rootValue << ",\"final_ties\":" << finalTies << "}";
            o << ",\"anomalies\":" << anomalies;
            o << "}";
            return o.str();
        }
    };

}  // namespace lve