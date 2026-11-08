#include "mcts_gpu.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace lve {

    static inline bool algoIsOcp(Algo a) { return a == Algo::OcpProdigal || a == Algo::OcpThrifty; }
    static inline bool algoIsThrifty(Algo a) { return a == Algo::AcpThrifty || a == Algo::OcpThrifty; }

    static VkShaderModule loadSpirv(const std::string& path, VkDevice dev) {
        std::ifstream f(path, std::ios::ate | std::ios::binary);
        if (!f.is_open()) throw std::runtime_error("MctsGpu: cannot open shader " + path);
        const size_t n = static_cast<size_t>(f.tellg());
        if (n == 0 || n % 4 != 0) throw std::runtime_error("MctsGpu: bad SPIR-V file " + path);
        std::vector<uint32_t> code(n / 4);
        f.seekg(0);
        f.read(reinterpret_cast<char*>(code.data()), static_cast<std::streamsize>(n));
        VkShaderModuleCreateInfo ci{ VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
        ci.codeSize = n;
        ci.pCode = code.data();
        VkShaderModule m = VK_NULL_HANDLE;
        if (vkCreateShaderModule(dev, &ci, nullptr, &m) != VK_SUCCESS)
            throw std::runtime_error("MctsGpu: vkCreateShaderModule failed for " + path);
        return m;
    }

    MctsGpu::MctsGpu(LveDevice& dev, const Config& config) : device{ dev }, cfg{ config } {
        if (NumKernels != kStatsKernels) throw std::runtime_error("MctsGpu: kStatsKernels out of sync with Kernel enum");

        if (cfg.slotsPerGroup < 0 || cfg.slotsPerGroup > MAX_SLOTS_PER_GROUP) {
            throw std::runtime_error("MctsGpu: slotsPerGroup must be in [0, " + std::to_string(MAX_SLOTS_PER_GROUP) + "]");
        }
        if (cfg.itersPerSubmit < 1) throw std::runtime_error("MctsGpu: itersPerSubmit must be >= 1");
        // auto: one child per group until the grid has more than ~448 child slots (14 warps per SM
        // on the 16-SM RTX 3050 Laptop); beyond that, packing wins (measured: T = 256 and 1024)
        {
            const int slots = cfg.numTrees * (algoIsOcp(cfg.algo) ? 1 : MAX_ACTIONS);
            slotsPerGroup = cfg.slotsPerGroup > 0 ? cfg.slotsPerGroup : std::clamp(slots / 448, 1, 4);
        }
        maxNodes = algoIsOcp(cfg.algo) ? (1 + cfg.maxIterations)
            : (1 + MAX_ACTIONS * cfg.maxIterations);

        const uint64_t treeBytes = uint64_t(cfg.numTrees) * uint64_t(maxNodes) * NODE_ROW * sizeof(int32_t);
        const uint64_t rangeLimit = device.properties.limits.maxStorageBufferRange;
        if (treeBytes > rangeLimit) {
            throw std::runtime_error("MctsGpu: tree buffer (" + std::to_string(treeBytes >> 20) +
                " MiB) exceeds maxStorageBufferRange (" + std::to_string(rangeLimit >> 20) +
                " MiB); lower numTrees or maxIterations");
        }

        createBuffers();
        createDescriptors();
        createPipelines();
        createQueryPool();

        VkCommandPoolCreateInfo pci{ VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
        pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        pci.queueFamilyIndex = device.findPhysicalQueueFamilies().graphicsFamily;
        if (vkCreateCommandPool(device.device(), &pci, nullptr, &cmdPool) != VK_SUCCESS)
            throw std::runtime_error("MctsGpu: failed to create command pool");
        VkCommandBufferAllocateInfo ai{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
        ai.commandPool = cmdPool;
        ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ai.commandBufferCount = 1;
        if (vkAllocateCommandBuffers(device.device(), &ai, &prologueCb) != VK_SUCCESS)
            throw std::runtime_error("MctsGpu: failed to allocate command buffer");
        VkFenceCreateInfo fci{ VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
        if (vkCreateFence(device.device(), &fci, nullptr, &fence) != VK_SUCCESS)
            throw std::runtime_error("MctsGpu: failed to create fence");
    }

    MctsGpu::~MctsGpu() {
        if (fence != VK_NULL_HANDLE) vkDestroyFence(device.device(), fence, nullptr);
        if (cmdPool != VK_NULL_HANDLE) vkDestroyCommandPool(device.device(), cmdPool, nullptr);   // frees all command buffers
        if (queryPool != VK_NULL_HANDLE) vkDestroyQueryPool(device.device(), queryPool, nullptr);
        for (int k = 0; k < NumKernels; k++) {
            if (pipelines[k] != VK_NULL_HANDLE) vkDestroyPipeline(device.device(), pipelines[k], nullptr);
        }
        vkDestroyPipelineLayout(device.device(), pipelineLayout, nullptr);
        vkDestroyShaderModule(device.device(), shaderModule, nullptr);
    }

    // -----------------------------------------------------------------------------
    void MctsGpu::createBuffers() {
        auto deviceLocal = [&](size_t numInts, VkBufferUsageFlags extra = 0) {
            return std::make_unique<LveBuffer>(
                device,
                sizeof(int32_t),
                static_cast<uint32_t>(numInts),
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | extra,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        };

        const size_t T = static_cast<size_t>(cfg.numTrees);
        const size_t nodes = T * static_cast<size_t>(maxNodes);

        buffers.resize(NumBindings);
        // root bitboard (4 ints), device-local, filled with vkCmdUpdateBuffer
        buffers[RootBoard] = deviceLocal(4, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
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
        buffers[ChildBoards] = deviceLocal(T * MAX_ACTIONS * CB_STRIDE);   // CB_STRIDE == 5 now
        buffers[ChildScores] = deviceLocal(T * MAX_ACTIONS);
        // Nothing the kernels touch is host-visible. rootStats and counters are copied into
        // `readback` once at the end of every submit; counters are zeroed with vkCmdFillBuffer.
        const VkBufferUsageFlags xfer = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        buffers[RootStats] = deviceLocal(2 * MAX_ACTIONS + 1, xfer);
        buffers[TreeB] = deviceLocal(T);                  // no longer read by the host
        buffers[BlockMap] = deviceLocal(T * MAX_ACTIONS); // compacted on the GPU
        buffers[Counters] = deviceLocal(kNumCounters, xfer);
        buffers[DispatchArgs] = deviceLocal(4, VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT);
        buffers[SearchParams] = deviceLocal(4, VK_BUFFER_USAGE_TRANSFER_DST_BIT);

        readback = std::make_unique<LveBuffer>(
            device, sizeof(int32_t), static_cast<uint32_t>(kReadbackInts),
            VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        readback->map();
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

        shaderModule = loadSpirv(cfg.shaderPath, device.device());

        static const char* entryNames[NumKernels] = {
            "mctsReset", "mctsSelect", "mctsExpand", "mctsPlayout",
            "mctsBackup", "mctsReduceTrees", "mctsReduceActions", "mctsExpandOcp" };

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
    // Two timestamps per submit are always written so gpuMs (and overhead = wall - gpu) is known
    // without per-kernel profiling; cfg.profile adds one timestamp after every dispatch.
    void MctsGpu::createQueryPool() {
        timingOk = profileOk = false;
        const VkPhysicalDeviceLimits& lim = device.properties.limits;
        if (!lim.timestampComputeAndGraphics || lim.timestampPeriod <= 0.f) return;
        tsPeriodNs = lim.timestampPeriod;
        // query 0/1: start / end of the prologue (reset); batch queries start at 2:
        // per iteration select, expand, playout, backup, then the two reduces (profile), or one end stamp
        maxQueries = cfg.profile ? (2 + 4 * cfg.itersPerSubmit + 2) : 3;
        VkQueryPoolCreateInfo ci{ VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO };
        ci.queryType = VK_QUERY_TYPE_TIMESTAMP;
        ci.queryCount = static_cast<uint32_t>(maxQueries);
        timingOk = (vkCreateQueryPool(device.device(), &ci, nullptr, &queryPool) == VK_SUCCESS);
        profileOk = timingOk && cfg.profile;
    }

    // rootStats + counters -> host-visible readback buffer
    void MctsGpu::recordReadback(VkCommandBuffer cb) {
        VkMemoryBarrier mb{ VK_STRUCTURE_TYPE_MEMORY_BARRIER };
        mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
            0, 1, &mb, 0, nullptr, 0, nullptr);
        VkBufferCopy regions[2]{};
        regions[0].size = (2 * MAX_ACTIONS + 1) * sizeof(int32_t);
        vkCmdCopyBuffer(cb, buffers[RootStats]->getBuffer(), readback->getBuffer(), 1, &regions[0]);
        regions[1].dstOffset = kReadbackCountersOffset * sizeof(int32_t);
        regions[1].size = kNumCounters * sizeof(int32_t);
        vkCmdCopyBuffer(cb, buffers[Counters]->getBuffer(), readback->getBuffer(), 1, &regions[1]);
        VkMemoryBarrier hb{ VK_STRUCTURE_TYPE_MEMORY_BARRIER };
        hb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        hb.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
            0, 1, &hb, 0, nullptr, 0, nullptr);
    }

    // -----------------------------------------------------------------------------
    // forIndirect: the next command is vkCmdDispatchIndirect reading a buffer this pass wrote
    static void computeBarrier(VkCommandBuffer cb, bool forIndirect = false) {
        VkMemoryBarrier mb{ VK_STRUCTURE_TYPE_MEMORY_BARRIER };
        mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        VkPipelineStageFlags dst = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
        if (forIndirect) {
            mb.dstAccessMask |= VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
            dst |= VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT;   // also covers vkCmdDispatchIndirect
        }
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, dst, 0, 1, &mb, 0, nullptr, 0, nullptr);
    }


    // -----------------------------------------------------------------------------
    MctsGpu::Result MctsGpu::search(
        const std::array<int, CELLS>& board, int rootTurn, int iterations) {
        SearchLimits lim;
        lim.maxIterations = iterations;
        return search(board, rootTurn, lim, kAutoSeed);
    }

    // -----------------------------------------------------------------------------
    // Records (once) the command buffer for a batch of `iterations` iterations followed by the
    // root reduction and the readback copy. Every per-search value lives in device buffers
    // (rootBoard, searchParams), and each dispatch's push constants hold only engine constants
    // plus the iteration's offset in the batch, so the same command buffer serves every search.
    const MctsGpu::BatchCb& MctsGpu::batchCommands(int iterations, double& recordMs) {
        auto it = batches.find(iterations);
        if (it != batches.end()) return it->second;

        const auto t0 = std::chrono::steady_clock::now();
        const bool ocp = algoIsOcp(cfg.algo);
        const bool thrifty = algoIsThrifty(cfg.algo);
        const Kernel expandKernel = ocp ? KExpandOcp : KExpand;
        const uint32_t T = static_cast<uint32_t>(cfg.numTrees);
        const uint32_t expandGroups = ocp ? (T + 31) / 32 : T;   // OCP: 1 thread per tree; ACP: 8 per tree
        const uint32_t slotsPerTree = ocp ? 1u : static_cast<uint32_t>(MAX_ACTIONS);
        const uint32_t S = static_cast<uint32_t>(slotsPerGroup);
        const uint32_t prodigalPlayoutGroups = (T * slotsPerTree + S - 1) / S;

        BatchCb b;
        VkCommandBufferAllocateInfo ai{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
        ai.commandPool = cmdPool;
        ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ai.commandBufferCount = 1;
        if (vkAllocateCommandBuffers(device.device(), &ai, &b.cb) != VK_SUCCESS)
            throw std::runtime_error("MctsGpu: failed to allocate command buffer");
        VkCommandBufferBeginInfo bi{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
        vkBeginCommandBuffer(b.cb, &bi);
        vkCmdBindDescriptorSets(b.cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &descSet, 0, nullptr);

        MctsPush pc{ cfg.numTrees, maxNodes, 0, 0, cfg.ucbC, thrifty ? 1 : 0, cfg.collectCounters ? 1 : 0,
            slotsPerGroup, static_cast<int32_t>(slotsPerTree) };
        uint32_t q = 2;
        auto stamp = [&](Kernel k) {
            if (!profileOk) return;
            vkCmdWriteTimestamp(b.cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queryPool, q++);
            b.kernels.push_back(static_cast<int>(k));
        };
        auto dispatch = [&](Kernel k, uint32_t x) {
            vkCmdBindPipeline(b.cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipelines[k]);
            vkCmdPushConstants(b.cb, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
            vkCmdDispatch(b.cb, x, 1, 1);
            // after expand in thrifty mode the next command is the indirect playout dispatch
            computeBarrier(b.cb, thrifty && k == expandKernel);
            stamp(k);
        };

        for (int i = 0; i < iterations; i++) {
            pc.iterOffset = i;
            dispatch(KSelect, T);
            dispatch(expandKernel, expandGroups);
            if (thrifty) {
                vkCmdBindPipeline(b.cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipelines[KPlayout]);
                vkCmdPushConstants(b.cb, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
                vkCmdDispatchIndirect(b.cb, buffers[DispatchArgs]->getBuffer(), 0);   // grid written by expand
                computeBarrier(b.cb);
                stamp(KPlayout);
            }
            else {
                dispatch(KPlayout, prodigalPlayoutGroups);
            }
            dispatch(KBackup, T);
        }
        pc.iterOffset = iterations - 1;   // the final-move tie-break uses the last iteration's seed
        dispatch(KReduceTrees, MAX_ACTIONS);
        dispatch(KReduceActions, 1);
        recordReadback(b.cb);
        if (timingOk && !profileOk) vkCmdWriteTimestamp(b.cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queryPool, q++);
        if (vkEndCommandBuffer(b.cb) != VK_SUCCESS) throw std::runtime_error("MctsGpu: failed to record batch");
        b.queries = timingOk ? q - 2 : 0;

        recordMs += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        return batches.emplace(iterations, std::move(b)).first->second;
    }

    // -----------------------------------------------------------------------------
    // All four variants run fully on the device; thrifty compacts with an atomic in the expand
    // kernel and launches playouts with vkCmdDispatchIndirect.
    //
    // Host work per submit: record a prologue of a few commands (search parameters, and on the
    // first submit the root position, counter reset and tree reset), then submit it together with
    // a cached batch command buffer and wait on a fence.
    //
    // Reproducibility: every RNG stream depends only on (cfg.seed, callSeed, iteration index),
    // never on batch size or timing. A time-limited search is therefore bit-reproducible by
    // replaying with maxIterations = stats.iterations and maxMs = 0 (see runReplayCheck).
    // -----------------------------------------------------------------------------
    MctsGpu::Result MctsGpu::search(
        const std::array<int, CELLS>& board, int rootTurn, const SearchLimits& lim, uint32_t callSeed) {
        using clock = std::chrono::steady_clock;
        const auto t0 = clock::now();
        auto elapsedMs = [&]() { return std::chrono::duration<double, std::milli>(clock::now() - t0).count(); };

        const int maxIters = lim.maxIterations > 0 ? lim.maxIterations : cfg.maxIterations;
        if (maxIters > cfg.maxIterations) {
            throw std::runtime_error("MctsGpu::search: iterations > Config::maxIterations");
        }

        const bool ocp = algoIsOcp(cfg.algo);
        const bool thrifty = algoIsThrifty(cfg.algo);
        const uint32_t T = static_cast<uint32_t>(cfg.numTrees);
        const uint32_t slotsPerTree = ocp ? 1u : static_cast<uint32_t>(MAX_ACTIONS);

        {
            const int64_t perIter = (ocp ? 1 : MAX_ACTIONS) * int64_t(kPlayoutsPerChild);
            if (2 * perIter * int64_t(maxIters) * int64_t(T) > int64_t(INT32_MAX)) {
                throw std::runtime_error("MctsGpu::search: visit/win counters would overflow int32");
            }
        }

        if (callSeed == kAutoSeed) callSeed = static_cast<uint32_t>(seedCounter++);

        SearchStats st;
        st.algo = algoName(cfg.algo);
        st.numTrees = cfg.numTrees;
        st.ucbC = cfg.ucbC;
        st.cfgSeed = static_cast<uint32_t>(cfg.seed);
        st.callSeed = callSeed;
        st.maxNodesPerTree = maxNodes;
        st.countersEnabled = cfg.collectCounters;
        st.profiled = profileOk;
        st.slotsPerGroup = slotsPerGroup;
        st.treeBytes = uint64_t(T) * uint64_t(maxNodes) * (NODE_ROW + 3) * sizeof(int32_t);

        // ---- root bitboard: bit = col*7 + row, pos = stones of the player to move ----
        int32_t rootBB[4];
        {
            uint64_t pos = 0, mask = 0;
            for (int r = 0; r < ROWS; r++)
                for (int c = 0; c < COLS; c++) {
                    const int v = board[r * COLS + c];
                    if (v == 0) continue;
                    const uint64_t bit = uint64_t(1) << (c * (ROWS + 1) + r);
                    mask |= bit;
                    if (v == rootTurn) pos |= bit;
                }
            rootBB[0] = static_cast<int32_t>(static_cast<uint32_t>(pos));
            rootBB[1] = static_cast<int32_t>(static_cast<uint32_t>(pos >> 32));
            rootBB[2] = static_cast<int32_t>(static_cast<uint32_t>(mask));
            rootBB[3] = static_cast<int32_t>(static_cast<uint32_t>(mask >> 32));
        }

        // prologue: search parameters (+ root, counters, tree reset on the first submit)
        auto recordPrologue = [&](bool first, int done) {
            const auto tr = clock::now();
            vkResetCommandBuffer(prologueCb, 0);
            VkCommandBufferBeginInfo bi{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
            bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            vkBeginCommandBuffer(prologueCb, &bi);
            if (timingOk) {
                vkCmdResetQueryPool(prologueCb, queryPool, 0, static_cast<uint32_t>(maxQueries));
                vkCmdWriteTimestamp(prologueCb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queryPool, 0);
            }
            const uint32_t params[4] = { static_cast<uint32_t>(cfg.seed), callSeed, static_cast<uint32_t>(done), 0u };
            vkCmdUpdateBuffer(prologueCb, buffers[SearchParams]->getBuffer(), 0, sizeof(params), params);
            if (first) {
                vkCmdUpdateBuffer(prologueCb, buffers[RootBoard]->getBuffer(), 0, sizeof(rootBB), rootBB);
                vkCmdFillBuffer(prologueCb, buffers[Counters]->getBuffer(), 0, VK_WHOLE_SIZE, 0u);
            }
            VkMemoryBarrier mb{ VK_STRUCTURE_TYPE_MEMORY_BARRIER };
            mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            vkCmdPipelineBarrier(prologueCb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0, 1, &mb, 0, nullptr, 0, nullptr);
            if (first) {
                MctsPush pc{ cfg.numTrees, maxNodes, 0, 0, cfg.ucbC, thrifty ? 1 : 0, cfg.collectCounters ? 1 : 0,
                    slotsPerGroup, static_cast<int32_t>(slotsPerTree) };
                vkCmdBindDescriptorSets(prologueCb, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &descSet, 0, nullptr);
                vkCmdBindPipeline(prologueCb, VK_PIPELINE_BIND_POINT_COMPUTE, pipelines[KReset]);
                vkCmdPushConstants(prologueCb, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
                vkCmdDispatch(prologueCb, (T + 31) / 32, 1, 1);
                computeBarrier(prologueCb);
            }
            if (timingOk) vkCmdWriteTimestamp(prologueCb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queryPool, 1);
            vkEndCommandBuffer(prologueCb);
            st.recordMs += std::chrono::duration<double, std::milli>(clock::now() - tr).count();
        };

        auto harvest = [&](const BatchCb& b) {
            if (!timingOk) return;
            const uint32_t n = 2 + b.queries;
            std::vector<uint64_t> ts(n);
            if (vkGetQueryPoolResults(device.device(), queryPool, 0, n, ts.size() * sizeof(uint64_t),
                ts.data(), sizeof(uint64_t), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT) != VK_SUCCESS) {
                return;
            }
            const double toMs = double(tsPeriodNs) * 1e-6;
            st.gpuMs += double(ts[n - 1] - ts[0]) * toMs;
            if (profileOk) {
                st.kernelMs[KReset] += double(ts[1] - ts[0]) * toMs;
                for (uint32_t i = 2; i < n; i++) st.kernelMs[b.kernels[i - 2]] += double(ts[i] - ts[i - 1]) * toMs;
            }
        };

        int done = 0;
        double msPerIter = 0.0;
        while (done < maxIters) {
            double remaining = 1e300;
            if (lim.maxMs > 0.0) {
                remaining = lim.maxMs - elapsedMs();
                if (remaining <= 0.0 && done > 0) break;
            }

            int batch = std::min(cfg.itersPerSubmit, maxIters - done);
            if (lim.maxMs > 0.0 && msPerIter > 0.0) {
                // time-limited: largest power of two that fits, so few batch sizes get recorded
                int fit = std::max(1, static_cast<int>(remaining / msPerIter));
                int p2 = 1;
                while (p2 * 2 <= fit) p2 *= 2;
                batch = std::min(batch, p2);
            }

            const double tb = elapsedMs();
            const BatchCb& b = batchCommands(batch, st.recordMs);
            recordPrologue(done == 0, done);

            VkCommandBuffer cbs[2] = { prologueCb, b.cb };
            VkSubmitInfo si{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
            si.commandBufferCount = 2;
            si.pCommandBuffers = cbs;
            if (vkQueueSubmit(device.graphicsQueue(), 1, &si, fence) != VK_SUCCESS)
                throw std::runtime_error("MctsGpu: vkQueueSubmit failed");
            vkWaitForFences(device.device(), 1, &fence, VK_TRUE, UINT64_MAX);
            vkResetFences(device.device(), 1, &fence);
            ++st.submits;
            harvest(b);

            if (!thrifty) st.blocksLaunched += int64_t(batch) * int64_t(T) * slotsPerTree;   // slots covered by the grid
            done += batch;
            msPerIter = (elapsedMs() - tb) / batch;
        }

        // ---- results (copied back at the end of the last submit) -----------------
        const int32_t* stats = static_cast<const int32_t*>(readback->getMappedMemory());
        Result r;
        st.rootVisits.assign(COLS, 0);
        st.rootWinRate.assign(COLS, 0.0f);
        for (int a = 0; a < COLS; a++) {
            int n = stats[2 * a];
            int w = stats[2 * a + 1];
            r.visits[a] = n;
            r.winRate[a] = (n > 0) ? static_cast<float>(w) / (2.0f * static_cast<float>(n)) : 0.0f;
            st.rootVisits[a] = n;
            st.rootWinRate[a] = r.winRate[a];
        }
        r.bestAction = stats[2 * MAX_ACTIONS];
        st.bestAction = r.bestAction;

        const int32_t* ctr = stats + kReadbackCountersOffset;
        st.depthSum = ctr[CtrDepthSum];
        st.depthMax = ctr[CtrDepthMax];
        st.terminalHits = ctr[CtrTerminalHits];
        st.childrenCreated = ctr[CtrChildrenCreated];
        st.treeSizeMax = ctr[CtrTreeSizeMax];
        st.blocksReal = ctr[CtrBlocksReal];
        st.blocksVirtual = ctr[CtrBlocksVirtual];
        st.pliesSum = ctr[CtrPliesSum];
        st.anomalies = ctr[CtrAnomalies];
        st.finalTies = ctr[CtrFinalTies];
        st.pliesMaxSum = ctr[CtrPliesMaxSum];          // laneEff = pliesSum / (32 * pliesMaxSum)
        if (thrifty) {
            st.sumA = ctr[CtrSumA];                    // device-side (counters on); hostRoundTrips/hostMapMs stay 0
            st.maxA = ctr[CtrMaxA];
            st.blocksLaunched = st.sumA;               // queued slots
        }

        st.iterations = done;
        st.wallMs = elapsedMs();
        st.finalize();

        r.ms = st.wallMs;
        r.stats = st;
        return r;
    }

}  // namespace lve