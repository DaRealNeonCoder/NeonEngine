#pragma once

#include "lve_buffer.hpp"
#include "lve_descriptors.hpp"
#include "lve_device.hpp"
#include "mcts_stats.hpp"

#include <array>
#include <cstdint>
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

    // Must match PushConsts in mcts_connect4.slang (7 x 4 bytes)
    struct MctsPush {
        int32_t numTrees;
        int32_t maxNodes;
        uint32_t seed;
        int32_t rootTurn;
        float ucbC;
        int32_t thrifty;
        int32_t collect;
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

        struct Config {
            Algo algo = Algo::AcpProdigal;
            int numTrees = 128;
            int maxIterations = 1000;
            int itersPerSubmit = 50;
            float ucbC = 1.4f;
            uint32_t seed = 1234;
            bool profile = true;
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

        LveDevice& device;
        Config cfg;
        int maxNodes = 0;
        uint32_t seedCounter = 0;

        std::vector<std::unique_ptr<LveBuffer>> buffers;
        std::unique_ptr<LveDescriptorSetLayout> setLayout;
        std::unique_ptr<LveDescriptorPool> pool;
        VkDescriptorSet descSet = VK_NULL_HANDLE;

        VkShaderModule shaderModule = VK_NULL_HANDLE;
        VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
        VkPipeline pipelines[NumKernels]{};

        VkQueryPool queryPool = VK_NULL_HANDLE;
        bool profileOk = false;
        float tsPeriodNs = 1.0f;
        int maxQueries = 0;
    };

} // namespace lve
