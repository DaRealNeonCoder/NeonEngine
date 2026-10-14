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

        // ACP creates up to MAX_ACTIONS nodes per iteration, OCP exactly one.
        maxNodes = algoIsOcp(cfg.algo) ? (1 + cfg.maxIterations)
            : (1 + MAX_ACTIONS * cfg.maxIterations);

        // Fail loudly instead of silently reading/writing past the binding range.
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
        // thrifty only: GPU writes b_t per tree, host reads it and writes the block map
        buffers[TreeB] = hostVisible(T);
        buffers[BlockMap] = hostVisible(T * MAX_ACTIONS);
        // GPU-side statistics counters (zeroed by the host at the start of every search)
        buffers[Counters] = hostVisible(kNumCounters);
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

        // order must match the Kernel enum
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
    // Timestamp queries: one per dispatch (taken after its barrier) + one at the start
    // of each command buffer. Consecutive differences = per-kernel GPU time.
    void MctsGpu::createQueryPool() {
        profileOk = false;
        if (!cfg.profile) return;
        const VkPhysicalDeviceLimits& lim = device.properties.limits;
        if (!lim.timestampComputeAndGraphics || lim.timestampPeriod <= 0.f) return;
        tsPeriodNs = lim.timestampPeriod;
        // worst case per command buffer: 1 start + reset + 4 kernels per batched iteration
        maxQueries = 4 * std::max(cfg.itersPerSubmit, 1) + 16;
        VkQueryPoolCreateInfo ci{ VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO };
        ci.queryType = VK_QUERY_TYPE_TIMESTAMP;
        ci.queryCount = static_cast<uint32_t>(maxQueries);
        profileOk = (vkCreateQueryPool(device.device(), &ci, nullptr, &queryPool) == VK_SUCCESS);
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
        SearchLimits lim;
        lim.maxIterations = iterations;
        return search(board, rootTurn, lim, kAutoSeed);
    }

    // -----------------------------------------------------------------------------
    //   AcpProdigal : all children expanded, playout grid (T, B), no host traffic
    //   AcpThrifty  : all children expanded, compacted, playout grid (A), host readback each iteration
    //   OcpProdigal : one random untried child, playout grid (T), no host traffic
    //   OcpThrifty  : one random untried child, playout grid (A), host readback each iteration
    //
    // Budget: SearchLimits (iterations and/or wall-clock ms). All RNG streams derive from
    // (cfg.seed, callSeed, iteration index), so a search is reproducible given its callSeed.
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
        const uint32_t playoutY = ocp ? 1u : static_cast<uint32_t>(MAX_ACTIONS);
        const uint32_t T = static_cast<uint32_t>(cfg.numTrees);
        const uint32_t groups32 = (T + 31) / 32;

        // int32 overflow guard for nodeN / nodeWins (root wins <= 2N; ReduceTrees sums over T trees)
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

        buffers[RootBoard]->writeToBuffer(const_cast<int*>(board.data()));
        buffers[RootBoard]->flush();
        std::memset(buffers[Counters]->getMappedMemory(), 0, kNumCounters * sizeof(int32_t));

        MctsPush pc{ cfg.numTrees, maxNodes, 0u, rootTurn, cfg.ucbC, thrifty ? 1 : 0, cfg.collectCounters ? 1 : 0 };

        VkCommandBuffer cb = VK_NULL_HANDLE;
        std::vector<int> cbKernels;  // kernel id of each timestamp interval in the current command buffer
        uint32_t qCount = 0;

        auto begin = [&]() {
            cb = device.beginSingleTimeCommands();
            vkCmdBindDescriptorSets(
                cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &descSet, 0, nullptr);
            cbKernels.clear();
            qCount = 0;
            if (profileOk) {
                vkCmdResetQueryPool(cb, queryPool, 0, static_cast<uint32_t>(maxQueries));
                vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queryPool, qCount++);
            }
        };
        auto dispatch = [&](Kernel k, uint32_t x, uint32_t y = 1, uint32_t z = 1) {
            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipelines[k]);
            vkCmdPushConstants(cb, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
            vkCmdDispatch(cb, x, y, z);
            computeBarrier(cb);
            if (profileOk && qCount < static_cast<uint32_t>(maxQueries)) {
                vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queryPool, qCount++);
                cbKernels.push_back(static_cast<int>(k));
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
            device.endSingleTimeCommands(cb);  // submits + waits for the queue
            ++st.submits;
            harvest();
        };

        int done = 0;
        double msPerIter = 0.0;  // measured on the previous batch; used to avoid overshooting maxMs
        while (done < maxIters) {
            double remaining = 1e300;
            if (lim.maxMs > 0.0) {
                remaining = lim.maxMs - elapsedMs();
                if (remaining <= 0.0 && done > 0) break;  // always do at least one iteration
            }

            // thrifty needs a host round trip inside every iteration, so batching is moot
            int batch = thrifty ? 1 : std::min(cfg.itersPerSubmit, maxIters - done);
            if (!thrifty && lim.maxMs > 0.0 && msPerIter > 0.0) {
                batch = std::max(1, std::min(batch, static_cast<int>(remaining / msPerIter)));
            }

            const double tb = elapsedMs();
            begin();
            if (done == 0) dispatch(KReset, groups32);

            for (int i = 0; i < batch; i++) {
                pc.seed = hash32(static_cast<uint32_t>(cfg.seed) ^
                    hash32(callSeed + 0x9E3779B9u * static_cast<uint32_t>(done + i + 1)));
                dispatch(KSelect, T);
                dispatch(expandKernel, groups32);

                if (thrifty) {
                    // ---- device -> host: how many children did each tree expand? ----
                    hostReadBarrier(cb);
                    submit();
                    ++st.hostRoundTrips;

                    const auto h0 = clock::now();
                    const int32_t* b = static_cast<const int32_t*>(buffers[TreeB]->getMappedMemory());
                    int32_t* map = static_cast<int32_t*>(buffers[BlockMap]->getMappedMemory());
                    uint32_t A = 0;  // total number of expanded children over all trees
                    for (int t = 0; t < cfg.numTrees; t++) {
                        if (b[t] < 0 || b[t] > MAX_ACTIONS) {
                            throw std::runtime_error("MctsGpu::search: corrupt b_t read back from device");
                        }
                        for (int k = 0; k < b[t]; k++) map[A++] = t * MAX_ACTIONS + k;  // slot id
                    }
                    st.hostMapMs += std::chrono::duration<double, std::milli>(clock::now() - h0).count();
                    st.blocksLaunched += A;
                    st.sumA += A;
                    st.maxA = std::max<int64_t>(st.maxA, A);

                    // ---- host -> device: playouts get exactly A workgroups ----
                    begin();
                    dispatch(KPlayout, A);  // A == 0 is a legal no-op dispatch
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

        // final reduction in its own submit (the loop length is not known in advance when time-limited)
        begin();
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

        st.iterations = done;
        st.wallMs = elapsedMs();
        st.finalize();

        r.ms = st.wallMs;
        r.stats = st;
        return r;
    }

}  // namespace lve