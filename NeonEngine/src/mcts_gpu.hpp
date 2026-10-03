#pragma once

#include "lve_buffer.hpp"
#include "lve_descriptors.hpp"
#include "lve_device.hpp"

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace lve {

    // Must match PushConsts in mcts_connect4.slang
    struct MctsPush {
        int32_t numTrees;
        int32_t maxNodes;
        uint32_t seed;
        int32_t rootTurn;
        float ucbC;
    };

    // GPU MCTS (ACP-prodigal) for Connect Four. Headless: no swapchain/renderer needed.
    class MctsGpu {
    public:
        // ---- must match the constants at the top of the .slang file ----
        static constexpr int COLS = 7;
        static constexpr int ROWS = 6;
        static constexpr int CELLS = COLS * ROWS;
        static constexpr int MAX_ACTIONS = COLS;
        static constexpr int NODE_ROW = 1 + MAX_ACTIONS;
        static constexpr int PATH_STRIDE = 44;
        static constexpr int INFO_STRIDE = 4;
        static constexpr int CB_STRIDE = 44;

        struct Config {
            int numTrees = 128;          // T: independent trees
            int maxIterations = 1000;    // sizes the node pool: 1 + MAX_ACTIONS * maxIterations per tree
            int itersPerSubmit = 50;     // iterations recorded per command buffer (avoids Windows TDR)
            float ucbC = 1.4f;
            uint32_t seed = 1234;
            std::string shaderPath = "shaders/mcts_connect4.spv";
        };

        struct Result {
            int bestAction = -1;
            std::array<int, COLS> visits{};
            std::array<float, COLS> winRate{};  // from the root player's point of view
            double ms = 0.0;
        };

        MctsGpu(LveDevice& device, const Config& config);
        ~MctsGpu();
        MctsGpu(const MctsGpu&) = delete;
        MctsGpu& operator=(const MctsGpu&) = delete;

        // board: row-major, index = row * COLS + col, row 0 = bottom. 0 empty, +1, -1.
        // rootTurn: player to move (+1 / -1). Root must not already be a finished game.
        Result search(const std::array<int, CELLS>& board, int rootTurn, int iterations);

    private:
        enum Binding {
            RootBoard = 0, Trees, TreeSizes, NodeFlags, NodeN, NodeWins, SelNode, SelPath,
            SelActs, SelInfo, ActionsExp, ChildBoards, ChildScores, RootStats,
            NumBindings
        };
        enum Kernel {
            KReset = 0, KSelect, KExpand, KPlayout, KBackup, KReduceTrees, KReduceActions,
            NumKernels
        };

        void createBuffers();
        void createDescriptors();
        void createPipelines();

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
    };

}  // namespace lve