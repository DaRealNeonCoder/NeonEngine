#pragma once
// =============================================================================
// mcts_stats.hpp -- per-search statistics for MctsGpu (all four variants)
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

    constexpr uint32_t kAutoSeed = 0xFFFFFFFFu;
    constexpr int kNumCounters = 16;  // == NUM_COUNTERS in mcts_connect4.slang
    constexpr int kPlayoutsPerChild = 32;
    constexpr int kStatsKernels = 8;

    // Slots of the shader-side counters buffer.
    // Keep in sync with CTR_* in the .slang file.
    enum Counter : int {
        CtrDepthSum = 0,
        CtrDepthMax = 1,
        CtrTerminalHits = 2,
        CtrChildrenCreated = 3,
        CtrTreeSizeMax = 4,
        CtrBlocksReal = 5,
        CtrBlocksVirtual = 6,
        CtrPliesSum = 7,
        CtrAnomalies = 8,
        CtrFinalTies = 9,
        CtrSumA = 10,
        CtrMaxA = 11,
        CtrPliesMaxSum = 12,
    };

    inline const char* const kKernelNames[kStatsKernels] = {
        "reset", "select", "expand", "playout",
        "backup", "reduce_trees", "reduce_actions", "expand_ocp"
    };

    // Budget for one search. Both limits may be set; whichever hits first wins.
    struct SearchLimits {
        int maxIterations = 0;  // 0 -> Config::maxIterations
        double maxMs = 0.0;     // 0 -> no wall-clock limit (checked between submits)
    };

    struct SearchStats {
        // ---- identity / config echo ---------------------------------------------
        std::string algo;
        int numTrees = 0;
        float ucbC = 0.f;
        uint32_t cfgSeed = 0;
        uint32_t callSeed = 0;
        int playoutsPerChild = kPlayoutsPerChild;
        int maxNodesPerTree = 0;
        bool countersEnabled = false;
        bool profiled = false;

        // ---- budget actually consumed -------------------------------------------
        int iterations = 0;
        int submits = 0;
        int hostRoundTrips = 0;
        double wallMs = 0.0;
        double gpuMs = 0.0;
        double hostMapMs = 0.0;
        double recordMs = 0.0;      // host time spent recording command buffers
        int slotsPerGroup = 1;
        double kernelMs[kStatsKernels] = {};

        // ---- raw counters from the GPU ------------------------------------------
        int64_t depthSum = 0;
        int64_t depthMax = 0;
        int64_t terminalHits = 0;
        int64_t childrenCreated = 0;
        int64_t treeSizeMax = 0;

        int64_t blocksReal = 0;
        int64_t blocksVirtual = 0;
        int64_t pliesSum = 0;
        int64_t pliesMaxSum = 0;
        int64_t anomalies = 0;
        int64_t finalTies = 0;

        // ---- raw, host-side statistics ------------------------------------------
        int64_t blocksLaunched = 0;
        int64_t sumA = 0;
        int64_t maxA = 0;
        uint64_t treeBytes = 0;

        std::vector<int> rootVisits;
        std::vector<float> rootWinRate;
        int bestAction = -1;

        // ---- derived statistics -------------------------------------------------
        int64_t treeIterations = 0;
        int64_t nodesTotal = 0;

        double leafDepthMean = 0.0;
        double terminalHitRate = 0.0;
        double childrenPerExpansion = 0.0;

        int64_t playoutsReal = 0;
        int64_t playoutsVirtual = 0;
        int64_t playoutsCredited = 0;

        double playoutLenMean = 0.0;
        double laneEff = 0.0;

        double itersPerSec = 0.0;
        double playoutsRealPerSec = 0.0;
        double playoutsCreditedPerSec = 0.0;

        double overheadMs = 0.0;
        double overheadFrac = 0.0;

        double blockUtilActive = 0.0;
        double blockUtilReal = 0.0;
        double treeFill = 0.0;

        double rootVisitsTotal = 0.0;
        double bestVisitShare = 0.0;
        double rootValue = 0.0;

        static double div(double a, double b) {
            return b > 0.0 ? a / b : 0.0;
        }

        void finalize() {
            treeIterations = static_cast<int64_t>(iterations) * numTrees;
            nodesTotal = numTrees + childrenCreated;

            leafDepthMean = div(
                static_cast<double>(depthSum),
                static_cast<double>(treeIterations));

            terminalHitRate = div(
                static_cast<double>(terminalHits),
                static_cast<double>(treeIterations));

            childrenPerExpansion = div(
                static_cast<double>(childrenCreated),
                static_cast<double>(treeIterations - terminalHits));

            playoutsReal = blocksReal * playoutsPerChild;
            playoutsVirtual = blocksVirtual * playoutsPerChild;
            playoutsCredited = playoutsReal + playoutsVirtual;

            playoutLenMean = div(
                static_cast<double>(pliesSum),
                static_cast<double>(playoutsReal));

            laneEff = div(
                static_cast<double>(pliesSum),
                static_cast<double>(kPlayoutsPerChild) *
                static_cast<double>(pliesMaxSum));

            const double s = wallMs * 1e-3;

            itersPerSec = div(iterations, s);
            playoutsRealPerSec = div(
                static_cast<double>(playoutsReal), s);
            playoutsCreditedPerSec = div(
                static_cast<double>(playoutsCredited), s);

            overheadMs = std::max(0.0, wallMs - gpuMs);
            overheadFrac = div(overheadMs, wallMs);

            blockUtilActive = div(
                static_cast<double>(blocksReal + blocksVirtual),
                static_cast<double>(blocksLaunched));

            blockUtilReal = div(
                static_cast<double>(blocksReal),
                static_cast<double>(blocksLaunched));

            treeFill = div(
                static_cast<double>(treeSizeMax),
                static_cast<double>(maxNodesPerTree));

            rootVisitsTotal = 0.0;
            int bestN = 0;

            for (int n : rootVisits) {
                rootVisitsTotal += n;
            }

            if (bestAction >= 0 &&
                bestAction < static_cast<int>(rootVisits.size())) {
                bestN = rootVisits[bestAction];
                rootValue = rootWinRate[bestAction];
            }

            bestVisitShare = div(
                static_cast<double>(bestN),
                rootVisitsTotal);
        }

        std::string toJson() const {
            std::ostringstream o;
            o << std::setprecision(9);

            auto arrI = [&](const std::vector<int>& v) {
                o << "[";
                for (size_t i = 0; i < v.size(); i++) {
                    o << (i ? "," : "") << v[i];
                }
                o << "]";
            };

            auto arrF = [&](const std::vector<float>& v) {
                o << "[";
                for (size_t i = 0; i < v.size(); i++) {
                    o << (i ? "," : "") << v[i];
                }
                o << "]";
            };

            o << "{";
            o << "\"algo\":\"" << algo
                << "\",\"num_trees\":" << numTrees
                << ",\"ucb_c\":" << ucbC
                << ",\"cfg_seed\":" << cfgSeed
                << ",\"call_seed\":" << callSeed
                << ",\"playouts_per_child\":" << playoutsPerChild
                << ",\"max_nodes_per_tree\":" << maxNodesPerTree
                << ",\"counters_enabled\":" << (countersEnabled ? "true" : "false")
                << ",\"profiled\":" << (profiled ? "true" : "false");

            o << ",\"iterations\":" << iterations
                << ",\"submits\":" << submits
                << ",\"host_round_trips\":" << hostRoundTrips
                << ",\"wall_ms\":" << wallMs
                << ",\"gpu_ms\":" << gpuMs
                << ",\"host_map_ms\":" << hostMapMs
                << ",\"record_ms\":" << recordMs
                << ",\"slots_per_group\":" << slotsPerGroup
                << ",\"overhead_ms\":" << overheadMs
                << ",\"overhead_frac\":" << overheadFrac;

            o << ",\"kernel_ms\":{";
            for (int k = 0; k < kStatsKernels; k++) {
                o << (k ? "," : "")
                    << "\"" << kKernelNames[k] << "\":" << kernelMs[k];
            }
            o << "}";

            o << ",\"tree\":{"
                << "\"nodes_total\":" << nodesTotal
                << ",\"children_created\":" << childrenCreated
                << ",\"tree_size_max\":" << treeSizeMax
                << ",\"tree_fill\":" << treeFill
                << ",\"leaf_depth_mean\":" << leafDepthMean
                << ",\"leaf_depth_max\":" << depthMax
                << ",\"terminal_hits\":" << terminalHits
                << ",\"terminal_hit_rate\":" << terminalHitRate
                << ",\"children_per_expansion\":" << childrenPerExpansion
                << ",\"bytes\":" << treeBytes
                << "}";

            o << ",\"playouts\":{"
                << "\"real\":" << playoutsReal
                << ",\"virtual\":" << playoutsVirtual
                << ",\"credited\":" << playoutsCredited
                << ",\"len_mean\":" << playoutLenMean
                << ",\"lane_eff\":" << laneEff
                << ",\"iters_per_s\":" << itersPerSec
                << ",\"real_per_s\":" << playoutsRealPerSec
                << ",\"credited_per_s\":" << playoutsCreditedPerSec
                << "}";

            o << ",\"blocks\":{"
                << "\"launched\":" << blocksLaunched
                << ",\"real\":" << blocksReal
                << ",\"virtual\":" << blocksVirtual
                << ",\"util_active\":" << blockUtilActive
                << ",\"util_real\":" << blockUtilReal
                << ",\"A_sum\":" << sumA
                << ",\"A_max\":" << maxA
                << "}";

            o << ",\"root\":{\"visits\":";
            arrI(rootVisits);

            o << ",\"win_rate\":";
            arrF(rootWinRate);

            o << ",\"best_action\":" << bestAction
                << ",\"best_visit_share\":" << bestVisitShare
                << ",\"value\":" << rootValue
                << ",\"final_ties\":" << finalTies
                << "}";

            o << ",\"anomalies\":" << anomalies;
            o << "}";

            return o.str();
        }
    };

}  // namespace lve
