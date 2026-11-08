#include "digit_recognizer_train_gpu.hpp"

#include "slang_runtime_compiler.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <stdexcept>

namespace lve {

namespace {

// Must match the constants in digit_recognizer_train.slang.
constexpr uint32_t IN = 27u * 27u;  // 729
constexpr uint32_t HID = 128u;
constexpr uint32_t OUT = 10u;
constexpr uint32_t OFF_W1 = 0u;
constexpr uint32_t OFF_B1 = IN * HID;               // 93312
constexpr uint32_t OFF_W2 = OFF_B1 + HID;           // 93440
constexpr uint32_t OFF_B2 = OFF_W2 + HID * OUT;     // 94720
constexpr uint32_t PARAM_COUNT = OFF_B2 + OUT;      // 94730
constexpr uint32_t THREADS = 256u;

uint32_t ceilDiv(uint32_t a, uint32_t b) { return (a + b - 1u) / b; }

// Copies host bytes into a device-local buffer through a temporary staging buffer.
void uploadBytes(LveDevice& device, LveBuffer& dst, const void* src, size_t size) {
    LveBuffer staging(device, size, 1, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    staging.map();
    staging.writeToBuffer(const_cast<void*>(src), size);

    VkCommandBuffer cb = device.beginSingleTimeCommands();
    VkBufferCopy copy{};
    copy.size = size;
    vkCmdCopyBuffer(cb, staging.getBuffer(), dst.getBuffer(), 1, &copy);
    device.endSingleTimeCommands(cb);
}

// Compute-to-compute barrier between dependent training dispatches.
void computeBarrier(VkCommandBuffer cb) {
    VkMemoryBarrier mb{ VK_STRUCTURE_TYPE_MEMORY_BARRIER };
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 1, &mb, 0, nullptr, 0, nullptr);
}

}  // namespace

DigitRecognizerTrainer::DigitRecognizerTrainer(
    LveDevice& dev,
    const DigitTrainerConfig& config,
    const MnistSplit& split)
    : device{dev}, cfg{config}, trainCount{split.trainCount()} {
    if (cfg.train.batchSize == 0) throw std::runtime_error("DigitRecognizerTrainer: batch size must be > 0");
    if (trainCount == 0) throw std::runtime_error("DigitRecognizerTrainer: empty training split");

    stepsPerEpoch = ceilDiv(static_cast<uint32_t>(trainCount), cfg.train.batchSize);
    createBuffers();
    createDescriptors();
    createPipelines();
    uploadTrainingSet(split);
}

DigitRecognizerTrainer::~DigitRecognizerTrainer() {
    for (VkPipeline p : pipelines) vkDestroyPipeline(device.device(), p, nullptr);
    for (VkShaderModule m : shaderModules) vkDestroyShaderModule(device.device(), m, nullptr);
    if (pipelineLayout != VK_NULL_HANDLE) vkDestroyPipelineLayout(device.device(), pipelineLayout, nullptr);
}

void DigitRecognizerTrainer::createBuffers() {
    const VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
        | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;

    auto deviceLocal = [&](size_t byteSize) {
        return std::make_unique<LveBuffer>(device, byteSize, 1, usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    };
    auto hostVisible = [&](size_t byteSize) {
        auto b = std::make_unique<LveBuffer>(
            device, byteSize, 1, usage,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        b->map();
        return b;
    };

    const size_t n = trainCount;
    const size_t batch = cfg.train.batchSize;

    buffers.resize(NumBindings);
    buffers[TrainImages] = deviceLocal(n * IN * sizeof(float));
    buffers[TrainLabels] = deviceLocal(n * sizeof(uint32_t));
    buffers[Order] = hostVisible(n * sizeof(uint32_t));
    buffers[Params] = deviceLocal(PARAM_COUNT * sizeof(float));
    buffers[Grads] = deviceLocal(PARAM_COUNT * sizeof(float));
    buffers[HiddenAct] = deviceLocal(batch * HID * sizeof(float));
    buffers[HiddenGrad] = deviceLocal(batch * HID * sizeof(float));
    buffers[OutputGrad] = deviceLocal(batch * OUT * sizeof(float));
    // Two floats (loss, correct) per sample, for every minibatch of one epoch.
    buffers[Stats] = hostVisible(static_cast<size_t>(stepsPerEpoch) * batch * 2 * sizeof(float));
}

void DigitRecognizerTrainer::createDescriptors() {
    LveDescriptorSetLayout::Builder lb(device);
    for (uint32_t i = 0; i < NumBindings; ++i) {
        lb.addBinding(i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT);
    }
    setLayout = lb.build();

    pool = LveDescriptorPool::Builder(device)
        .setMaxSets(1)
        .addPoolSize(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, NumBindings)
        .build();

    std::vector<VkDescriptorBufferInfo> infos(NumBindings);
    for (uint32_t i = 0; i < NumBindings; ++i) infos[i] = buffers[i]->descriptorInfo();

    LveDescriptorWriter writer(*setLayout, *pool);
    for (uint32_t i = 0; i < NumBindings; ++i) writer.writeBuffer(i, &infos[i]);
    if (!writer.build(descSet)) throw std::runtime_error("DigitRecognizerTrainer: failed to build descriptor set");
}

void DigitRecognizerTrainer::createPipelines() {
    VkPushConstantRange pcr{};
    pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcr.offset = 0;
    pcr.size = sizeof(DigitTrainPushConstants);

    VkDescriptorSetLayout dsl = setLayout->getDescriptorSetLayout();
    VkPipelineLayoutCreateInfo li{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    li.setLayoutCount = 1;
    li.pSetLayouts = &dsl;
    li.pushConstantRangeCount = 1;
    li.pPushConstantRanges = &pcr;
    if (vkCreatePipelineLayout(device.device(), &li, nullptr, &pipelineLayout) != VK_SUCCESS) {
        throw std::runtime_error("DigitRecognizerTrainer: failed to create pipeline layout");
    }

    static const char* entryPoints[NumStages] = {
        "sml_model_digit_recognizer_train_hidden_fwd",
        "sml_model_digit_recognizer_train_output_fwd",
        "sml_model_digit_recognizer_train_hidden_bwd",
        "sml_model_digit_recognizer_train_grad_output",
        "sml_model_digit_recognizer_train_grad_hidden",
        "sml_model_digit_recognizer_train_sgd",
    };

    SlangRuntimeCompiler compiler;
    for (int s = 0; s < NumStages; ++s) {
        std::vector<uint32_t> spirv = compiler.compileEntryPoint(cfg.slangSource, cfg.slangIncludeDir, entryPoints[s]);

        VkShaderModuleCreateInfo smi{ VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
        smi.codeSize = spirv.size() * sizeof(uint32_t);
        smi.pCode = spirv.data();
        VkShaderModule module = VK_NULL_HANDLE;
        if (vkCreateShaderModule(device.device(), &smi, nullptr, &module) != VK_SUCCESS) {
            throw std::runtime_error(std::string("DigitRecognizerTrainer: shader module failed for ") + entryPoints[s]);
        }
        shaderModules.push_back(module);

        VkComputePipelineCreateInfo ci{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
        ci.stage = VkPipelineShaderStageCreateInfo{ VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO };
        ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        ci.stage.module = module;
        ci.stage.pName = entryPoints[s];
        ci.layout = pipelineLayout;

        VkPipeline pipeline = VK_NULL_HANDLE;
        if (vkCreateComputePipelines(device.device(), VK_NULL_HANDLE, 1, &ci, nullptr, &pipeline) != VK_SUCCESS) {
            throw std::runtime_error(std::string("DigitRecognizerTrainer: failed to create pipeline ") + entryPoints[s]);
        }
        pipelines.push_back(pipeline);
    }
}

// Copies the training split and labels to the device once. Staging is freed after each copy.
void DigitRecognizerTrainer::uploadTrainingSet(const MnistSplit& split) {
    uploadBytes(device, *buffers[TrainImages], split.trainX.data(), split.trainX.size() * sizeof(float));

    std::vector<uint32_t> labels(split.trainY.begin(), split.trainY.end());
    uploadBytes(device, *buffers[TrainLabels], labels.data(), labels.size() * sizeof(uint32_t));
}

// Weights and biases drawn from N(0, 1), like Nielsen's network.py; gradients start at zero.
void DigitRecognizerTrainer::initializeParameters(std::mt19937& rng) {
    std::vector<float> params(PARAM_COUNT);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (float& p : params) p = dist(rng);

    const std::vector<float> zeros(PARAM_COUNT, 0.0f);
    const size_t bytes = PARAM_COUNT * sizeof(float);
    uploadBytes(device, *buffers[Params], params.data(), bytes);
    uploadBytes(device, *buffers[Grads], zeros.data(), bytes);
}

void DigitRecognizerTrainer::dispatchStage(
    VkCommandBuffer cb,
    Stage stage,
    const DigitTrainPushConstants& pc,
    uint32_t threads) {
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipelines[stage]);
    vkCmdPushConstants(cb, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cb, ceilDiv(threads, THREADS), 1, 1);
    computeBarrier(cb);
}

// One minibatch: forward, loss, backward, gradient reductions, SGD.
void DigitRecognizerTrainer::recordStep(VkCommandBuffer cb, const DigitTrainPushConstants& pc) {
    dispatchStage(cb, HiddenFwd, pc, pc.batchCount * HID);
    dispatchStage(cb, OutputFwd, pc, pc.batchCount);
    dispatchStage(cb, HiddenBwd, pc, pc.batchCount * HID);
    dispatchStage(cb, GradOutput, pc, HID * OUT + OUT);
    dispatchStage(cb, GradHidden, pc, IN * HID + HID);
    dispatchStage(cb, Sgd, pc, PARAM_COUNT);
}

MlpParams DigitRecognizerTrainer::train() {
    std::mt19937 rng(cfg.train.seed);
    initializeParameters(rng);

    const uint32_t batchSize = cfg.train.batchSize;
    std::vector<uint32_t> order(trainCount);
    std::iota(order.begin(), order.end(), 0u);

    const auto startTime = std::chrono::steady_clock::now();

    for (uint32_t epoch = 0; epoch < cfg.train.epochs; ++epoch) {
        std::shuffle(order.begin(), order.end(), rng);
        std::memcpy(buffers[Order]->getMappedMemory(), order.data(), order.size() * sizeof(uint32_t));

        // Every minibatch of the epoch goes into one command buffer; the GPU runs them in order.
        VkCommandBuffer cb = device.beginSingleTimeCommands();
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &descSet, 0, nullptr);

        for (uint32_t step = 0; step < stepsPerEpoch; ++step) {
            const uint32_t batchStart = step * batchSize;
            const uint32_t batchCount = std::min<uint32_t>(batchSize, static_cast<uint32_t>(trainCount) - batchStart);

            DigitTrainPushConstants pc{};
            pc.batchStart = batchStart;
            pc.batchCount = batchCount;
            pc.statsOffset = step * batchSize;
            pc.learningRate = cfg.train.learningRate;
            pc.invBatch = 1.0f / float(batchCount);
            recordStep(cb, pc);
        }
        device.endSingleTimeCommands(cb);

        // Per-sample stats were written for this epoch; reduce them on the host.
        const float* stats = static_cast<const float*>(buffers[Stats]->getMappedMemory());
        double epochLoss = 0.0;
        size_t correct = 0;
        for (uint32_t step = 0; step < stepsPerEpoch; ++step) {
            const uint32_t batchStart = step * batchSize;
            const uint32_t batchCount = std::min<uint32_t>(batchSize, static_cast<uint32_t>(trainCount) - batchStart);
            for (uint32_t b = 0; b < batchCount; ++b) {
                const size_t row = size_t(step) * batchSize + b;
                epochLoss += stats[row * 2 + 0];
                correct += stats[row * 2 + 1] > 0.5f ? 1u : 0u;
            }
        }

        std::printf("epoch %2u/%u  train_quadratic_cost=%.4f  train_acc(running)=%.4f\n", epoch + 1, cfg.train.epochs,
            epochLoss / double(trainCount), double(correct) / double(trainCount));
    }

    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - startTime).count();
    std::printf("GPU training: %u epochs in %.2f s\n", cfg.train.epochs, seconds);

    // Read the trained flat parameters back to the host.
    const size_t bytes = PARAM_COUNT * sizeof(float);
    LveBuffer readback(device, bytes, 1, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    readback.map();

    VkCommandBuffer cb = device.beginSingleTimeCommands();
    VkBufferCopy copy{};
    copy.size = bytes;
    vkCmdCopyBuffer(cb, buffers[Params]->getBuffer(), readback.getBuffer(), 1, &copy);
    device.endSingleTimeCommands(cb);

    const float* p = static_cast<const float*>(readback.getMappedMemory());
    MlpParams out;
    out.w1.assign(p + OFF_W1, p + OFF_B1);
    out.b1.assign(p + OFF_B1, p + OFF_W2);
    out.w2.assign(p + OFF_W2, p + OFF_B2);
    out.b2.assign(p + OFF_B2, p + PARAM_COUNT);
    return out;
}

}  // namespace lve
