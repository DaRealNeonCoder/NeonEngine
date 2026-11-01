#include "digit_recognizer_gpu.hpp"

#include "lve_pipeline.hpp"
#include "slang_runtime_compiler.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace lve {

// Helper to get element size for a data type
static inline uint32_t elementSize(DataType dt) {
    switch (dt) {
        case DataType::FP32: return 4;
        case DataType::FP16: return 2;
        case DataType::BF16: return 2;
        case DataType::INT8: return 1;
        case DataType::UINT8: return 1;
        case DataType::INT32: return 4;
        case DataType::INT64: return 8;
        case DataType::BOOL: return 1;
        case DataType::FP8_E4M3: return 1;
        case DataType::FP8_E5M2: return 1;
        case DataType::PACKED_INT4: return 1; // packed, 2 elements per byte
        default: return 0;
    }
}

static inline VkBufferUsageFlags defaultStorageUsage() {
    return VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
}

DigitRecognizerGpu::DigitRecognizerGpu(LveDevice& dev, const DigitRecognizerConfig& config)
    : device{dev}, cfg{config} {
    createBuffers(cfg.maxBatchSize);
    createDescriptors();
    createPipeline();
    createQueryPool();
}

DigitRecognizerGpu::~DigitRecognizerGpu() {
    if (queryPool != VK_NULL_HANDLE) vkDestroyQueryPool(device.device(), queryPool, nullptr);
    if (pipeline != VK_NULL_HANDLE) vkDestroyPipeline(device.device(), pipeline, nullptr);
    vkDestroyPipelineLayout(device.device(), pipelineLayout, nullptr);
    vkDestroyShaderModule(device.device(), shaderModule, nullptr);
}

void DigitRecognizerGpu::createBuffers(uint32_t maxBatchSize) {
    auto deviceLocal = [&](size_t byteSize, VkBufferUsageFlags extra = 0) {
        return std::make_unique<LveBuffer>(
            device,
            byteSize,
            1,
            defaultStorageUsage() | extra,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    };

    auto hostVisible = [&](size_t byteSize) {
        auto b = std::make_unique<LveBuffer>(
            device,
            byteSize,
            1,
            defaultStorageUsage(),
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        b->map();
        return b;
    };

    // Weight buffers (persistent, device-local)
    const size_t weight1Size = DIGIT_INPUT_FEATURES * DIGIT_HIDDEN_FEATURES * elementSize(cfg.weightDtype);
    const size_t bias1Size = DIGIT_HIDDEN_FEATURES * sizeof(float);
    const size_t weight2Size = DIGIT_HIDDEN_FEATURES * DIGIT_CLASSES * elementSize(cfg.weightDtype);
    const size_t bias2Size = DIGIT_CLASSES * sizeof(float);

    weight1Buffer = deviceLocal(weight1Size);
    bias1Buffer = deviceLocal(bias1Size);
    weight2Buffer = deviceLocal(weight2Size);
    bias2Buffer = deviceLocal(bias2Size);

    // Tensor table: 7 tensors * 96 bytes each
    tensorTableBuffer = hostVisible(7 * TENSOR_DESC_BYTES);

    // Kernel header: 48 bytes
    kernelHeaderBuffer = hostVisible(KERNEL_HEADER_BYTES);

    // Per-inference buffers (max batch size)
    const size_t inputSize = maxBatchSize * DIGIT_INPUT_FEATURES * sizeof(float);
    const size_t probSize = maxBatchSize * DIGIT_CLASSES * sizeof(float);
    const size_t predSize = maxBatchSize * sizeof(uint32_t);

    inputBuffer = hostVisible(inputSize);
    probabilitiesBuffer = hostVisible(probSize);
    predictionBuffer = hostVisible(predSize);

    // Descriptor set buffers array
    buffers.resize(NumBindings);
    buffers[TensorTable] = std::move(tensorTableBuffer);
    buffers[KernelHeader] = std::move(kernelHeaderBuffer);
}

void DigitRecognizerGpu::createDescriptors() {
    LveDescriptorSetLayout::Builder lb(device);
    lb.addBinding(TensorTable, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT);
    lb.addBinding(KernelHeader, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT);
    setLayout = lb.build();

    pool = LveDescriptorPool::Builder(device)
        .setMaxSets(1)
        .addPoolSize(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, NumBindings)
        .build();

    std::vector<VkDescriptorBufferInfo> infos(NumBindings);
    for (int i = 0; i < NumBindings; i++) infos[i] = buffers[i]->descriptorInfo();

    LveDescriptorWriter writer(*setLayout, *pool);
    for (int i = 0; i < NumBindings; i++) writer.writeBuffer(i, &infos[i]);
    if (!writer.build(descSet)) throw std::runtime_error("DigitRecognizerGpu: failed to build descriptor set");
}

void DigitRecognizerGpu::createPipeline() {
    VkPushConstantRange pcr{};
    pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcr.offset = 0;
    pcr.size = sizeof(DigitRecognizerPushConstants);

    VkDescriptorSetLayout dsl = setLayout->getDescriptorSetLayout();
    VkPipelineLayoutCreateInfo li{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    li.setLayoutCount = 1;
    li.pSetLayouts = &dsl;
    li.pushConstantRangeCount = 1;
    li.pPushConstantRanges = &pcr;
    if (vkCreatePipelineLayout(device.device(), &li, nullptr, &pipelineLayout) != VK_SUCCESS) {
        throw std::runtime_error("DigitRecognizerGpu: failed to create pipeline layout");
    }

    // Choose entry point based on weight data type
    const char* entryPoint = (cfg.weightDtype == DataType::BF16)
        ? "sml_model_digit_recognizer_fwd_bf16_f32"
        : "sml_model_digit_recognizer_fwd_f16_f32";

    if (!cfg.slangSource.empty()) {
        SlangRuntimeCompiler compiler;
        std::vector<uint32_t> spirv = compiler.compileEntryPoint(cfg.slangSource, cfg.slangIncludeDir, entryPoint);

        VkShaderModuleCreateInfo smi{ VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
        smi.codeSize = spirv.size() * sizeof(uint32_t);
        smi.pCode = spirv.data();
        if (vkCreateShaderModule(device.device(), &smi, nullptr, &shaderModule) != VK_SUCCESS) {
            throw std::runtime_error("DigitRecognizerGpu: failed to create shader module from Slang source");
        }
    } else {
        shaderModule = LvePipeline::loadShaderModule(cfg.shaderPath, device.device());
    }

    VkComputePipelineCreateInfo ci{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
    ci.stage = VkPipelineShaderStageCreateInfo{ VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO };
    ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    ci.stage.module = shaderModule;
    ci.stage.pName = entryPoint;
    ci.layout = pipelineLayout;
    if (vkCreateComputePipelines(device.device(), VK_NULL_HANDLE, 1, &ci, nullptr, &pipeline) != VK_SUCCESS) {
        throw std::runtime_error(std::string("DigitRecognizerGpu: failed to create pipeline ") + entryPoint);
    }
}

void DigitRecognizerGpu::createQueryPool() {
    profileOk = false;
    if (!cfg.profile) return;
    const VkPhysicalDeviceLimits& lim = device.properties.limits;
    if (!lim.timestampComputeAndGraphics || lim.timestampPeriod <= 0.f) return;
    tsPeriodNs = lim.timestampPeriod;
    maxQueries = 4;
    VkQueryPoolCreateInfo ci{ VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO };
    ci.queryType = VK_QUERY_TYPE_TIMESTAMP;
    ci.queryCount = maxQueries;
    profileOk = (vkCreateQueryPool(device.device(), &ci, nullptr, &queryPool) == VK_SUCCESS);
}

static void computeBarrier(VkCommandBuffer cb) {
    VkMemoryBarrier mb{ VK_STRUCTURE_TYPE_MEMORY_BARRIER };
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 1, &mb, 0, nullptr, 0, nullptr);
}

static void hostReadBarrier(VkCommandBuffer cb) {
    VkMemoryBarrier mb{ VK_STRUCTURE_TYPE_MEMORY_BARRIER };
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
        0, 1, &mb, 0, nullptr, 0, nullptr);
}

void DigitRecognizerGpu::loadWeights(
    const void* weight1, size_t weight1Size,
    const float* bias1, size_t bias1Size,
    const void* weight2, size_t weight2Size,
    const float* bias2, size_t bias2Size) {

    // Validate sizes
    const size_t expectedWeight1Size = DIGIT_INPUT_FEATURES * DIGIT_HIDDEN_FEATURES * elementSize(cfg.weightDtype);
    const size_t expectedBias1Size = DIGIT_HIDDEN_FEATURES * sizeof(float);
    const size_t expectedWeight2Size = DIGIT_HIDDEN_FEATURES * DIGIT_CLASSES * elementSize(cfg.weightDtype);
    const size_t expectedBias2Size = DIGIT_CLASSES * sizeof(float);

    if (weight1Size != expectedWeight1Size) throw std::runtime_error("weight1 size mismatch");
    if (bias1Size != expectedBias1Size) throw std::runtime_error("bias1 size mismatch");
    if (weight2Size != expectedWeight2Size) throw std::runtime_error("weight2 size mismatch");
    if (bias2Size != expectedBias2Size) throw std::runtime_error("bias2 size mismatch");

    // Upload to device-local buffers via staging
    auto upload = [&](LveBuffer* dst, const void* src, size_t size) {
        auto staging = std::make_unique<LveBuffer>(
            device, size, 1, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        staging->map();
        staging->writeToBuffer(const_cast<void*>(src), size);

        VkCommandBuffer cb = device.beginSingleTimeCommands();
        VkBufferCopy copy{};
        copy.size = size;
        vkCmdCopyBuffer(cb, staging->getBuffer(), dst->getBuffer(), 1, &copy);
        device.endSingleTimeCommands(cb);
    };

    upload(weight1Buffer.get(), weight1, weight1Size);
    upload(bias1Buffer.get(), bias1, bias1Size);
    upload(weight2Buffer.get(), weight2, weight2Size);
    upload(bias2Buffer.get(), bias2, bias2Size);

    weightsLoaded = true;
}

void DigitRecognizerGpu::updateTensorTable(uint32_t batchSize, float inputScale, float inputBias) {
    if (!weightsLoaded) throw std::runtime_error("DigitRecognizerGpu: weights not loaded");

    // Get device addresses for all buffers
    VkBufferDeviceAddressInfo addrInfo{ VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO };

    addrInfo.buffer = weight1Buffer->getBuffer();
    uint64_t weight1Addr = vkGetBufferDeviceAddress(device.device(), &addrInfo);

    addrInfo.buffer = bias1Buffer->getBuffer();
    uint64_t bias1Addr = vkGetBufferDeviceAddress(device.device(), &addrInfo);

    addrInfo.buffer = weight2Buffer->getBuffer();
    uint64_t weight2Addr = vkGetBufferDeviceAddress(device.device(), &addrInfo);

    addrInfo.buffer = bias2Buffer->getBuffer();
    uint64_t bias2Addr = vkGetBufferDeviceAddress(device.device(), &addrInfo);

    addrInfo.buffer = inputBuffer->getBuffer();
    uint64_t inputAddr = vkGetBufferDeviceAddress(device.device(), &addrInfo);

    addrInfo.buffer = probabilitiesBuffer->getBuffer();
    uint64_t probAddr = vkGetBufferDeviceAddress(device.device(), &addrInfo);

    addrInfo.buffer = predictionBuffer->getBuffer();
    uint64_t predAddr = vkGetBufferDeviceAddress(device.device(), &addrInfo);

    // Tensor table address
    addrInfo.buffer = buffers[TensorTable]->getBuffer();
    uint64_t tensorTableAddr = vkGetBufferDeviceAddress(device.device(), &addrInfo);

    // Build tensor descriptors (7 tensors)
    TensorDesc tensors[7] = {};

    // Tensor 0: input (batch x 729, f32)
    tensors[0].shape[0] = batchSize;
    tensors[0].shape[1] = DIGIT_INPUT_FEATURES;
    tensors[0].stride[0] = static_cast<int32_t>(DIGIT_INPUT_FEATURES);
    tensors[0].stride[1] = 1;
    tensors[0].baseOffset = 0;
    tensors[0].dataAddress = inputAddr;
    tensors[0].dtype = static_cast<uint32_t>(DataType::FP32);
    tensors[0].rank = 2;
    tensors[0].flags = CONTIGUOUS | CANONICAL | RANK_LE3 | ALIGNED16;
    tensors[0].bufferIndex = 0xFFFFFFFF;

    // Tensor 1: weight1 (729 x 128, fp16/bf16)
    tensors[1].shape[0] = DIGIT_INPUT_FEATURES;
    tensors[1].shape[1] = DIGIT_HIDDEN_FEATURES;
    tensors[1].stride[0] = static_cast<int32_t>(DIGIT_HIDDEN_FEATURES);
    tensors[1].stride[1] = 1;
    tensors[1].baseOffset = 0;
    tensors[1].dataAddress = weight1Addr;
    tensors[1].dtype = static_cast<uint32_t>(cfg.weightDtype);
    tensors[1].rank = 2;
    tensors[1].flags = CONTIGUOUS | READ_ONLY | CANONICAL | RANK_LE3;
    tensors[1].bufferIndex = 0xFFFFFFFF;

    // Tensor 2: bias1 (128, f32)
    tensors[2].shape[0] = DIGIT_HIDDEN_FEATURES;
    tensors[2].stride[0] = 1;
    tensors[2].baseOffset = 0;
    tensors[2].dataAddress = bias1Addr;
    tensors[2].dtype = static_cast<uint32_t>(DataType::FP32);
    tensors[2].rank = 1;
    tensors[2].flags = CONTIGUOUS | READ_ONLY | CANONICAL | RANK_LE3 | INNER_STRIDE1 | ALIGNED16;
    tensors[2].bufferIndex = 0xFFFFFFFF;

    // Tensor 3: weight2 (128 x 10, fp16/bf16)
    tensors[3].shape[0] = DIGIT_HIDDEN_FEATURES;
    tensors[3].shape[1] = DIGIT_CLASSES;
    tensors[3].stride[0] = static_cast<int32_t>(DIGIT_CLASSES);
    tensors[3].stride[1] = 1;
    tensors[3].baseOffset = 0;
    tensors[3].dataAddress = weight2Addr;
    tensors[3].dtype = static_cast<uint32_t>(cfg.weightDtype);
    tensors[3].rank = 2;
    tensors[3].flags = CONTIGUOUS | READ_ONLY | CANONICAL | RANK_LE3;
    tensors[3].bufferIndex = 0xFFFFFFFF;

    // Tensor 4: bias2 (10, f32)
    tensors[4].shape[0] = DIGIT_CLASSES;
    tensors[4].stride[0] = 1;
    tensors[4].baseOffset = 0;
    tensors[4].dataAddress = bias2Addr;
    tensors[4].dtype = static_cast<uint32_t>(DataType::FP32);
    tensors[4].rank = 1;
    tensors[4].flags = CONTIGUOUS | READ_ONLY | CANONICAL | RANK_LE3 | INNER_STRIDE1 | ALIGNED16;
    tensors[4].bufferIndex = 0xFFFFFFFF;

    // Tensor 5: probabilities output (batch x 10, f32)
    tensors[5].shape[0] = batchSize;
    tensors[5].shape[1] = DIGIT_CLASSES;
    tensors[5].stride[0] = static_cast<int32_t>(DIGIT_CLASSES);
    tensors[5].stride[1] = 1;
    tensors[5].baseOffset = 0;
    tensors[5].dataAddress = probAddr;
    tensors[5].dtype = static_cast<uint32_t>(DataType::FP32);
    tensors[5].rank = 2;
    tensors[5].flags = CONTIGUOUS | WRITE_ONLY | CANONICAL | RANK_LE3 | ALIGNED16;
    tensors[5].bufferIndex = 0xFFFFFFFF;

    // Tensor 6: prediction output (batch, u32)
    tensors[6].shape[0] = batchSize;
    tensors[6].stride[0] = 1;
    tensors[6].baseOffset = 0;
    tensors[6].dataAddress = predAddr;
    tensors[6].dtype = static_cast<uint32_t>(DataType::INT32); // Using INT32 for uint32_t
    tensors[6].rank = 1;
    tensors[6].flags = CONTIGUOUS | WRITE_ONLY | CANONICAL | RANK_LE3 | INNER_STRIDE1 | ALIGNED16;
    tensors[6].bufferIndex = 0xFFFFFFFF;

    // Copy tensor table to host-visible buffer
    std::memcpy(buffers[TensorTable]->getMappedMemory(), tensors, sizeof(tensors));

    // Build the parameter block: KernelHeader followed by batch/stride/scale fields.
    // Strides are in elements, matching the row-major layout of the host weight arrays.
    // lve::DigitRecognizerParamBlock (not the enum member KernelHeader) is the struct type here.
    lve::DigitRecognizerParamBlock params{};
    params.header.dispatchBase[0] = 0;
    params.header.dispatchBase[1] = 0;
    params.header.dispatchBase[2] = 0;
    params.header.numel = batchSize;
    params.header.flags = 0;
    params.header.tensorTableOffset = 0;
    params.header.tensorCount = 7;
    params.header.reserved0 = 0;
    params.header.tensorTableAddress = tensorTableAddr;
    params.header.reserved1 = 0;
    params.header.reserved2 = 0;

    params.batchCount = batchSize;
    params.inputStrideBatch = static_cast<int32_t>(DIGIT_INPUT_FEATURES);  // [batch, 729]
    params.inputStrideFeature = 1;
    params.weight1StrideFeature = static_cast<int32_t>(DIGIT_HIDDEN_FEATURES);  // [729, 128]
    params.weight1StrideHidden = 1;
    params.bias1Stride = 1;
    params.weight2StrideHidden = static_cast<int32_t>(DIGIT_CLASSES);  // [128, 10]
    params.weight2StrideClass = 1;
    params.bias2Stride = 1;
    params.outputStrideBatch = static_cast<int32_t>(DIGIT_CLASSES);  // [batch, 10]
    params.outputStrideClass = 1;
    params.predictionStrideBatch = 1;  // [batch]
    params.inputScale = inputScale;
    params.inputBias = inputBias;

    // Copy parameter block to host-visible buffer
    std::memcpy(buffers[KernelHeader]->getMappedMemory(), &params, sizeof(params));
}

DigitRecognizerResult DigitRecognizerGpu::infer(
    const std::vector<float>& input,
    uint32_t batchSize,
    float inputScale,
    float inputBias) {

    if (batchSize == 0 || batchSize > cfg.maxBatchSize) {
        throw std::runtime_error("DigitRecognizerGpu: invalid batch size");
    }
    if (input.size() < batchSize * DIGIT_INPUT_FEATURES) {
        throw std::runtime_error("DigitRecognizerGpu: input buffer too small");
    }

    // Copy input data to host-visible buffer
    // Apply scale and bias on the host side (shader also applies it, but we can do it here or in shader)
    // The shader does: value * inputScale + inputBias
    // We'll just copy raw input and let shader handle normalization
    std::memcpy(inputBuffer->getMappedMemory(), input.data(), batchSize * DIGIT_INPUT_FEATURES * sizeof(float));

    // Update tensor table and kernel header
    updateTensorTable(batchSize, inputScale, inputBias);

    // Prepare push constants
    VkBufferDeviceAddressInfo addrInfo{ VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO };
    addrInfo.buffer = buffers[KernelHeader]->getBuffer();
    uint64_t headerAddr = vkGetBufferDeviceAddress(device.device(), &addrInfo);

    DigitRecognizerPushConstants pc{};
    pc.headerAddress = headerAddr;
    pc.workspaceAddress = 0;  // No workspace used
    pc.flags = 0;
    pc.reserved = 0;

    using clock = std::chrono::steady_clock;
    const auto t0 = clock::now();
    uint32_t qCount = 0;

    VkCommandBuffer cb = device.beginSingleTimeCommands();
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &descSet, 0, nullptr);

    if (profileOk) {
        vkCmdResetQueryPool(cb, queryPool, 0, maxQueries);
        vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queryPool, qCount++);
    }

    // Dispatch: 1 workgroup per batch item, 256 threads per workgroup
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdPushConstants(cb, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cb, batchSize, 1, 1);

    computeBarrier(cb);

    if (profileOk && qCount < maxQueries) {
        vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queryPool, qCount++);
    }

    hostReadBarrier(cb);
    device.endSingleTimeCommands(cb);

    double ms = 0.0;
    if (profileOk && qCount >= 2) {
        std::vector<uint64_t> ts(qCount);
        if (vkGetQueryPoolResults(device.device(), queryPool, 0, qCount, ts.size() * sizeof(uint64_t),
            ts.data(), sizeof(uint64_t), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT) == VK_SUCCESS) {
            ms = double(ts[1] - ts[0]) * double(tsPeriodNs) * 1e-6;
        }
    } else {
        ms = std::chrono::duration<double, std::milli>(clock::now() - t0).count();
    }

    // Read results
    const float* probData = static_cast<const float*>(probabilitiesBuffer->getMappedMemory());
    const uint32_t* predData = static_cast<const uint32_t*>(predictionBuffer->getMappedMemory());

    DigitRecognizerResult result;
    result.probabilities.assign(probData, probData + batchSize * DIGIT_CLASSES);
    result.predictions.assign(predData, predData + batchSize);
    result.ms = ms;

    return result;
}

}  // namespace lve