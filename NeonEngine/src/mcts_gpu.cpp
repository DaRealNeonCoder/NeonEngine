#include "mcts_gpu.hpp"

#include "lve_pipeline.hpp"

#include <algorithm>
#include <chrono>
#include <stdexcept>

namespace lve {

    MctsGpu::MctsGpu(LveDevice& dev, const Config& config) : device{ dev }, cfg{ config } {
        maxNodes = 1 + MAX_ACTIONS * cfg.maxIterations;
        createBuffers();
        createDescriptors();
        createPipelines();
    }

    MctsGpu::~MctsGpu() {
        for (int k = 0; k < NumKernels; k++) {
            if (pipelines[k] != VK_NULL_HANDLE) vkDestroyPipeline(device.device(), pipelines[k], nullptr);
        }
        vkDestroyPipelineLayout(device.device(), pipelineLayout, nullptr);
        vkDestroyShaderModule(device.device(), shaderModule, nullptr);
    }

    // -----------------------------------------------------------------------------
    void MctsGpu::createBuffers() {
        // every buffer is a plain array of 32-bit ints
        auto deviceLocal = [&](size_t numInts) {
            return std::make_unique<LveBuffer>(
                device,
                sizeof(int32_t),
                static_cast<uint32_t>(numInts),
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        };
        auto hostVisible = [&](size_t numInts) {
            auto b = std::make_unique<LveBuffer>(
                device,
                sizeof(int32_t),
                static_cast<uint32_t>(numInts),
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            b->map();
            return b;
        };

        const size_t T = static_cast<size_t>(cfg.numTrees);
        const size_t nodes = T * static_cast<size_t>(maxNodes);

        buffers.resize(NumBindings);
        buffers[RootBoard] = hostVisible(CELLS);
        buffers[Trees] = deviceLocal(nodes * NODE_ROW);
        buffers[TreeSizes] = deviceLocal(T);
        buffers[NodeFlags] = deviceLocal(nodes);
        buffers[NodeN] = deviceLocal(nodes);
        buffers[NodeWins] = deviceLocal(nodes);
        buffers[SelNode] = deviceLocal(T);
        buffers[SelPath] = deviceLocal(T * PATH_STRIDE);
        buffers[SelActs] = deviceLocal(T * PATH_STRIDE);
        buffers[SelInfo] = deviceLocal(T * INFO_STRIDE);
        buffers[ActionsExp] = deviceLocal(T * MAX_ACTIONS);
        buffers[ChildBoards] = deviceLocal(T * MAX_ACTIONS * CB_STRIDE);
        buffers[ChildScores] = deviceLocal(T * MAX_ACTIONS);
        buffers[RootStats] = hostVisible(2 * MAX_ACTIONS + 1);
    }

    // -----------------------------------------------------------------------------
    void MctsGpu::createDescriptors() {
        LveDescriptorSetLayout::Builder lb(device);
        for (int i = 0; i < NumBindings; i++) {
            lb.addBinding(i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT);
        }
        setLayout = lb.build();

        pool = LveDescriptorPool::Builder(device)
            .setMaxSets(1)
            .addPoolSize(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, NumBindings)
            .build();

        std::vector<VkDescriptorBufferInfo> infos(NumBindings);
        for (int i = 0; i < NumBindings; i++) infos[i] = buffers[i]->descriptorInfo();

        LveDescriptorWriter writer(*setLayout, *pool);
        for (int i = 0; i < NumBindings; i++) writer.writeBuffer(i, &infos[i]);
        if (!writer.build(descSet)) throw std::runtime_error("MctsGpu: failed to build descriptor set");
    }

    // -----------------------------------------------------------------------------
    void MctsGpu::createPipelines() {
        VkPushConstantRange pcr{};
        pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pcr.offset = 0;
        pcr.size = sizeof(MctsPush);

        VkDescriptorSetLayout dsl = setLayout->getDescriptorSetLayout();
        VkPipelineLayoutCreateInfo li{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
        li.setLayoutCount = 1;
        li.pSetLayouts = &dsl;
        li.pushConstantRangeCount = 1;
        li.pPushConstantRanges = &pcr;
        if (vkCreatePipelineLayout(device.device(), &li, nullptr, &pipelineLayout) != VK_SUCCESS) {
            throw std::runtime_error("MctsGpu: failed to create pipeline layout");
        }

        shaderModule = LvePipeline::loadShaderModule(cfg.shaderPath, device.device());

        static const char* entryNames[NumKernels] = {
            "mctsReset", "mctsSelect", "mctsExpand", "mctsPlayout",
            "mctsBackup", "mctsReduceTrees", "mctsReduceActions" };

        for (int k = 0; k < NumKernels; k++) {
            VkComputePipelineCreateInfo ci{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
            ci.stage = VkPipelineShaderStageCreateInfo{ VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO };
            ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            ci.stage.module = shaderModule;
            ci.stage.pName = entryNames[k];
            ci.layout = pipelineLayout;
            if (vkCreateComputePipelines(device.device(), VK_NULL_HANDLE, 1, &ci, nullptr, &pipelines[k]) !=
                VK_SUCCESS) {
                throw std::runtime_error(std::string("MctsGpu: failed to create pipeline ") + entryNames[k]);
            }
        }
    }

    // -----------------------------------------------------------------------------
    // Compute->compute global memory barrier (every pass depends on the previous one)
    static void computeBarrier(VkCommandBuffer cb) {
        VkMemoryBarrier mb{ VK_STRUCTURE_TYPE_MEMORY_BARRIER };
        mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(
            cb,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 1, &mb, 0, nullptr, 0, nullptr);
    }

    static void hostReadBarrier(VkCommandBuffer cb) {
        VkMemoryBarrier mb{ VK_STRUCTURE_TYPE_MEMORY_BARRIER };
        mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        vkCmdPipelineBarrier(
            cb,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_HOST_BIT,
            0, 1, &mb, 0, nullptr, 0, nullptr);
    }

    // -----------------------------------------------------------------------------
    MctsGpu::Result MctsGpu::search(
        const std::array<int, CELLS>& board, int rootTurn, int iterations) {
        if (iterations > cfg.maxIterations) {
            throw std::runtime_error("MctsGpu::search: iterations > Config::maxIterations");
        }
        auto t0 = std::chrono::steady_clock::now();

        buffers[RootBoard]->writeToBuffer(const_cast<int*>(board.data()));
        buffers[RootBoard]->flush();

        MctsPush pc{ cfg.numTrees, maxNodes, 0u, rootTurn, cfg.ucbC };
        const uint32_t T = static_cast<uint32_t>(cfg.numTrees);
        const uint32_t groups32 = (T + 31) / 32;

        int done = 0;
        while (done < iterations) {
            const int batch = std::min(cfg.itersPerSubmit, iterations - done);

            VkCommandBuffer cb = device.beginSingleTimeCommands();
            vkCmdBindDescriptorSets(
                cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &descSet, 0, nullptr);

            auto dispatch = [&](Kernel k, uint32_t x, uint32_t y = 1, uint32_t z = 1) {
                vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipelines[k]);
                vkCmdPushConstants(cb, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
                vkCmdDispatch(cb, x, y, z);
                computeBarrier(cb);
            };

            if (done == 0) dispatch(KReset, groups32);

            for (int i = 0; i < batch; i++) {
                pc.seed = cfg.seed * 2654435761u + (++seedCounter) * 0x9E3779B9u;
                dispatch(KSelect, T);
                dispatch(KExpand, groups32);
                dispatch(KPlayout, T, MAX_ACTIONS);
                dispatch(KBackup, T);
            }

            done += batch;
            if (done == iterations) {
                dispatch(KReduceTrees, MAX_ACTIONS);
                dispatch(KReduceActions, 1);
                hostReadBarrier(cb);
            }

            device.endSingleTimeCommands(cb);  // submits + waits for the queue
        }

        const int32_t* stats = static_cast<const int32_t*>(buffers[RootStats]->getMappedMemory());
        Result r;
        for (int a = 0; a < COLS; a++) {
            int n = stats[2 * a];
            int w = stats[2 * a + 1];
            r.visits[a] = n;
            r.winRate[a] = (n > 0) ? static_cast<float>(w) / (2.0f * static_cast<float>(n)) : 0.0f;
        }
        r.bestAction = stats[2 * MAX_ACTIONS];
        r.ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        return r;
    }

}  // namespace lve