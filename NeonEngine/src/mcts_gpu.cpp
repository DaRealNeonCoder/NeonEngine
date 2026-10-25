#include "mcts_gpu.hpp"

#include "lve_pipeline.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace lve {

    static inline bool algoIsOcp(Algo a) { return a == Algo::OcpProdigal || a == Algo::OcpThrifty; }
    static inline bool algoIsThrifty(Algo a) { return a == Algo::AcpThrifty || a == Algo::OcpThrifty; }

    static inline uint32_t hash32(uint32_t x) {
        x ^= x >> 16; x *= 0x7feb352dU;
        x ^= x >> 15; x *= 0x846ca68bU;
        x ^= x >> 16;
        return x;
    }

    MctsGpu::MctsGpu(LveDevice& dev, const Config& config) : device{ dev }, cfg{ config } {
        if (NumKernels != kStatsKernels) throw std::runtime_error("MctsGpu: kStatsKernels out of sync with Kernel enum");

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
    }

    MctsGpu::~MctsGpu() {
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
        buffers[RootStats] = hostVisible(2 * MAX_ACTIONS + 1);
        buffers[TreeB] = deviceLocal(T);                  // no longer read by the host
        buffers[BlockMap] = deviceLocal(T * MAX_ACTIONS); // compacted on the GPU
        buffers[Counters] = hostVisible(kNumCounters);    // must be >= 13 ints
        buffers[DispatchArgs] = deviceLocal(4, VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT);
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
    void MctsGpu::createQueryPool() {
        profileOk = false;
        if (!cfg.profile) return;
        const VkPhysicalDeviceLimits& lim = device.properties.limits;
        if (!lim.timestampComputeAndGraphics || lim.timestampPeriod <= 0.f) return;
        tsPeriodNs = lim.timestampPeriod;
        maxQueries = 4 * std::max(cfg.itersPerSubmit, 1) + 16;
        VkQueryPoolCreateInfo ci{ VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO };
        ci.queryType = VK_QUERY_TYPE_TIMESTAMP;
        ci.queryCount = static_cast<uint32_t>(maxQueries);
        profileOk = (vkCreateQueryPool(device.device(), &ci, nullptr, &queryPool) == VK_SUCCESS);
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

    static void hostReadBarrier(VkCommandBuffer cb) {
        VkMemoryBarrier mb{ VK_STRUCTURE_TYPE_MEMORY_BARRIER };
        mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
            0, 1, &mb, 0, nullptr, 0, nullptr);
    }

    // -----------------------------------------------------------------------------
    MctsGpu::Result MctsGpu::search(
        const std::array<int, CELLS>& board, int rootTurn, int iterations) {
        SearchLimits lim;
        lim.maxIterations = iterations;
        return search(board, rootTurn, lim, kAutoSeed);
    }

    // -----------------------------------------------------------------------------
    // All four variants now run fully on the device; thrifty compacts with an atomic in the
    // expand kernel and launches playouts with vkCmdDispatchIndirect, so thrifty batches too.
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
        const Kernel expandKernel = ocp ? KExpandOcp : KExpand;
        // ACP expand: one 8-thread workgroup per tree. OCP expand: one thread per tree.
        const uint32_t T = static_cast<uint32_t>(cfg.numTrees);
        const uint32_t groups32 = (T + 31) / 32;
        const uint32_t expandGroups = ocp ? groups32 : T;
        const uint32_t playoutY = ocp ? 1u : static_cast<uint32_t>(MAX_ACTIONS);

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
        std::memset(buffers[Counters]->getMappedMemory(), 0, kNumCounters * sizeof(int32_t));

        MctsPush pc{ cfg.numTrees, maxNodes, 0u, rootTurn, cfg.ucbC, thrifty ? 1 : 0, cfg.collectCounters ? 1 : 0 };

        VkCommandBuffer cb = VK_NULL_HANDLE;
        std::vector<int> cbKernels;
        uint32_t qCount = 0;

        auto begin = [&](bool uploadRoot) {
            cb = device.beginSingleTimeCommands();
            vkCmdBindDescriptorSets(
                cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &descSet, 0, nullptr);
            cbKernels.clear();
            qCount = 0;
            if (profileOk) {
                vkCmdResetQueryPool(cb, queryPool, 0, static_cast<uint32_t>(maxQueries));
                vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queryPool, qCount++);
            }
            if (uploadRoot) {
                vkCmdUpdateBuffer(cb, buffers[RootBoard]->getBuffer(), 0, sizeof(rootBB), rootBB);
                VkMemoryBarrier mb{ VK_STRUCTURE_TYPE_MEMORY_BARRIER };
                mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
                vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    0, 1, &mb, 0, nullptr, 0, nullptr);
            }
        };
        auto dispatch = [&](Kernel k, uint32_t x, uint32_t y = 1, uint32_t z = 1) {
            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipelines[k]);
            vkCmdPushConstants(cb, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
            vkCmdDispatch(cb, x, y, z);
            // after expand in thrifty mode the next command is the indirect playout dispatch
            computeBarrier(cb, thrifty && k == expandKernel);
            if (profileOk && qCount < static_cast<uint32_t>(maxQueries)) {
                vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queryPool, qCount++);
                cbKernels.push_back(static_cast<int>(k));
            }
        };
        auto dispatchPlayoutIndirect = [&]() {
            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipelines[KPlayout]);
            vkCmdPushConstants(cb, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
            vkCmdDispatchIndirect(cb, buffers[DispatchArgs]->getBuffer(), 0);
            computeBarrier(cb);
            if (profileOk && qCount < static_cast<uint32_t>(maxQueries)) {
                vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queryPool, qCount++);
                cbKernels.push_back(static_cast<int>(KPlayout));
            }
        };
        auto harvest = [&]() {
            if (!profileOk || qCount < 2) return;
            std::vector<uint64_t> ts(qCount);
            if (vkGetQueryPoolResults(device.device(), queryPool, 0, qCount, ts.size() * sizeof(uint64_t),
                ts.data(), sizeof(uint64_t), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT) != VK_SUCCESS) {
                return;
            }
            for (uint32_t i = 1; i < qCount; i++) {
                const double ms = double(ts[i] - ts[i - 1]) * double(tsPeriodNs) * 1e-6;
                st.kernelMs[cbKernels[i - 1]] += ms;
                st.gpuMs += ms;
            }
        };
        auto submit = [&]() {
            device.endSingleTimeCommands(cb);
            ++st.submits;
            harvest();
        };

        int done = 0;
        double msPerIter = 0.0;
        while (done < maxIters) {
            double remaining = 1e300;
            if (lim.maxMs > 0.0) {
                remaining = lim.maxMs - elapsedMs();
                if (remaining <= 0.0 && done > 0) break;
            }

            int batch = std::min(cfg.itersPerSubmit, maxIters - done);   // thrifty batches too now
            if (lim.maxMs > 0.0 && msPerIter > 0.0) {
                batch = std::max(1, std::min(batch, static_cast<int>(remaining / msPerIter)));
            }

            const double tb = elapsedMs();
            begin(done == 0);
            if (done == 0) dispatch(KReset, groups32);

            for (int i = 0; i < batch; i++) {
                pc.seed = hash32(static_cast<uint32_t>(cfg.seed) ^
                    hash32(callSeed + 0x9E3779B9u * static_cast<uint32_t>(done + i + 1)));
                dispatch(KSelect, T);
                dispatch(expandKernel, expandGroups);

                if (thrifty) {
                    dispatchPlayoutIndirect();                 // A workgroups, A written by expand
                }
                else {
                    st.blocksLaunched += int64_t(T) * playoutY;
                    dispatch(KPlayout, T, playoutY);
                }
                dispatch(KBackup, T);
            }

            done += batch;
            submit();
            msPerIter = (elapsedMs() - tb) / batch;
        }

        begin(false);
        dispatch(KReduceTrees, MAX_ACTIONS);
        dispatch(KReduceActions, 1);
        hostReadBarrier(cb);
        submit();

        // ---- results -------------------------------------------------------------
        const int32_t* stats = static_cast<const int32_t*>(buffers[RootStats]->getMappedMemory());
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

        const int32_t* ctr = static_cast<const int32_t*>(buffers[Counters]->getMappedMemory());
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
        // new (add the fields to SearchStats):
        st.pliesMaxSum = ctr[CtrPliesMaxSum];          // laneEff = pliesSum / (32 * pliesMaxSum)
        if (thrifty) {
            st.sumA = ctr[CtrSumA];                    // now device-side; hostRoundTrips/hostMapMs stay 0
            st.maxA = ctr[CtrMaxA];
            st.blocksLaunched = st.sumA;
        }

        st.iterations = done;
        st.wallMs = elapsedMs();
        st.finalize();

        r.ms = st.wallMs;
        r.stats = st;
        return r;
    }

}  // namespace lve