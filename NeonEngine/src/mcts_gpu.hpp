#pragma once

#include "lve_buffer.hpp"
#include "lve_descriptors.hpp"
#include "lve_device.hpp"
#include "mcts_stats.hpp"

#include <array>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace lve {

    enum class Algo { AcpProdigal, AcpThrifty, OcpProdigal, OcpThrifty };

    inline const char* algoName(Algo a) {
        switch (a) {
        case Algo::AcpProdigal: return "acp_prodigal";
        case Algo::AcpThrifty:  return "acp_thrifty";
        case Algo::OcpProdigal: return "ocp_prodigal";
        case Algo::OcpThrifty:  return "ocp_thrifty";
        }
        return "unknown";
    }

    // Must match PushConsts in mcts_connect4.slang (9 x 4 bytes)
    struct MctsPush {
        int32_t numTrees;
        int32_t maxNodes;
        int32_t iterOffset;     // iteration index within the batch (baked at record time)
        int32_t rootTurn;       // unused by the shader
        float ucbC;
        int32_t thrifty;
        int32_t collect;
        int32_t slotsPerGroup;
        int32_t slotsPerTree;
    };

    // GPU MCTS for Connect Four, all four MCTS-NC variants.
    // Headless: no swapchain/renderer needed.
    class MctsGpu {
    public:
        // Must match the constants in the .slang file.
        static constexpr int COLS = 7;
        static constexpr int ROWS = 6;
        static constexpr int CELLS = COLS * ROWS;
        static constexpr int MAX_ACTIONS = COLS;
        static constexpr int NODE_ROW = 1 + MAX_ACTIONS;
        static constexpr int PATH_STRIDE = 44;
        static constexpr int INFO_STRIDE = 4;
        static constexpr int CB_STRIDE = 5;
        static constexpr int MAX_SLOTS_PER_GROUP = 8;   // == MAX_SLOTS_PER_GROUP in the .slang file
        // readback buffer: [0, 2*MAX_ACTIONS] rootStats, then the counters
        static constexpr int kReadbackCountersOffset = 2 * MAX_ACTIONS + 1;
        static constexpr int kReadbackInts = kReadbackCountersOffset + kNumCounters;

        struct Config {
            Algo algo = Algo::AcpProdigal;
            int numTrees = 128;
            int maxIterations = 1000;
            int itersPerSubmit = 50;
            float ucbC = 1.4f;
            uint32_t seed = 1234;
            // Children served by one 32-lane playout group (1..8, 0 = auto). Lanes that finish early
            // take the next playout, so S > 1 cuts warp-divergence idle time, but it also cuts the
            // number of warps; it only pays off once the playout grid oversubscribes the GPU.
            // Search results do not depend on it.
            int slotsPerGroup = 0;
            bool profile = true;            // per-kernel timestamps (total GPU time is always measured)
            bool collectCounters = true;
            std::string shaderPath = "shaders/mcts_connect4.spv";
        };

        struct Result {
            int bestAction = -1;
            std::array<int, COLS> visits{};
            std::array<float, COLS> winRate{};
            double ms = 0.0;
            SearchStats stats;
        };

        MctsGpu(LveDevice& device, const Config& config);
        ~MctsGpu();

        MctsGpu(const MctsGpu&) = delete;
        MctsGpu& operator=(const MctsGpu&) = delete;

        const Config& config() const {
            return cfg;
        }

        // board: row-major, index = row * COLS + col, row 0 = bottom.
        // 0 = empty, +1 / -1 = players.
        // rootTurn: player to move (+1 / -1).
        // Root must not already be a finished game.
        Result search(
            const std::array<int, CELLS>& board,
            int rootTurn,
            int iterations
        );

        // Budgeted search: iterations and/or wall-clock ms (whichever hits first).
        // All RNG derives from (cfg.seed, callSeed, iteration).
        Result search(
            const std::array<int, CELLS>& board,
            int rootTurn,
            const SearchLimits& limits,
            uint32_t callSeed = kAutoSeed
        );

    private:
        enum Binding {
            RootBoard = 0,
            Trees,
            TreeSizes,
            NodeFlags,
            NodeN,
            NodeWins,
            SelNode,
            SelPath,
            SelActs,
            SelInfo,
            ActionsExp,
            ChildBoards,
            ChildScores,
            RootStats,
            TreeB,
            BlockMap,
            Counters,
            DispatchArgs,
            SearchParams,   // {cfg seed, call seed, iterations done}, written per submit
            NumBindings
        };

        // Order must match entryNames[] in createPipelines().
        enum Kernel {
            KReset = 0,
            KSelect,
            KExpand,
            KPlayout,
            KBackup,
            KReduceTrees,
            KReduceActions,
            KExpandOcp,
            NumKernels
        };

        void createBuffers();
        void createDescriptors();
        void createPipelines();
        void createQueryPool();
        void recordReadback(VkCommandBuffer cb);

        // A batch of B iterations (+ root reduction + readback) is recorded once and replayed.
        struct BatchCb {
            VkCommandBuffer cb = VK_NULL_HANDLE;
            std::vector<int> kernels;   // kernel timed by query i (profile mode), queries start at 2
            uint32_t queries = 0;       // queries written by this batch
        };
        const BatchCb& batchCommands(int iterations, double& recordMs);

        LveDevice& device;
        Config cfg;
        int maxNodes = 0;
        int slotsPerGroup = 1;          // resolved cfg.slotsPerGroup
        uint32_t seedCounter = 0;

        VkCommandPool cmdPool = VK_NULL_HANDLE;
        VkCommandBuffer prologueCb = VK_NULL_HANDLE;   // re-recorded every submit (a few commands)
        VkFence fence = VK_NULL_HANDLE;
        std::map<int, BatchCb> batches;

        std::vector<std::unique_ptr<LveBuffer>> buffers;
        std::unique_ptr<LveBuffer> readback;   // host-visible copy of rootStats + counters
        std::unique_ptr<LveDescriptorSetLayout> setLayout;
        std::unique_ptr<LveDescriptorPool> pool;
        VkDescriptorSet descSet = VK_NULL_HANDLE;

        VkShaderModule shaderModule = VK_NULL_HANDLE;
        VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
        VkPipeline pipelines[NumKernels]{};

        VkQueryPool queryPool = VK_NULL_HANDLE;
        bool timingOk = false;    // start/end timestamps per submit
        bool profileOk = false;   // per-kernel timestamps
        float tsPeriodNs = 1.0f;
        int maxQueries = 0;
    };

} // namespace lve
