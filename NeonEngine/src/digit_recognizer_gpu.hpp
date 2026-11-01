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

// Must match DigitRecognizerParams in digit_recognizer.slang
struct DigitRecognizerPushConstants {
    uint64_t headerAddress;
    uint64_t workspaceAddress;
    uint32_t flags;
    uint32_t reserved;
};

// Must match KernelHeader in config.slang
struct KernelHeader {
    uint32_t dispatchBase[3];
    uint32_t numel;
    uint32_t flags;
    uint32_t tensorTableOffset;
    uint32_t tensorCount;
    uint32_t reserved0;
    uint64_t tensorTableAddress;
    uint32_t reserved1;
    uint32_t reserved2;
};

// Must match DigitRecognizerParams in digit_recognizer.slang (112 bytes).
// The shader reads the strides and input scale from this block, so the whole
// block must be written, not just the KernelHeader at offset 0.
struct DigitRecognizerParamBlock {
    KernelHeader header;
    uint32_t batchCount;
    int32_t inputStrideBatch;
    int32_t inputStrideFeature;
    int32_t weight1StrideFeature;
    int32_t weight1StrideHidden;
    int32_t bias1Stride;
    int32_t weight2StrideHidden;
    int32_t weight2StrideClass;
    int32_t bias2Stride;
    int32_t outputStrideBatch;
    int32_t outputStrideClass;
    int32_t predictionStrideBatch;
    float inputScale;
    float inputBias;
    uint32_t reserved0;
    uint32_t reserved1;
};

// Must match TensorDesc in tensor.slang (96 bytes)
struct TensorDesc {
    uint32_t shape[8];
    int32_t stride[8];
    int64_t baseOffset;
    uint64_t dataAddress;
    uint32_t dtype;
    uint32_t rank;
    uint32_t flags;
    uint32_t bufferIndex;
};

// Data type tags (must match dtypes.slang)
enum class DataType : uint32_t {
    FP32 = 0,
    FP16 = 1,
    BF16 = 2,
    INT8 = 3,
    UINT8 = 4,
    INT32 = 5,
    INT64 = 6,
    BOOL = 7,
    FP8_E4M3 = 8,
    FP8_E5M2 = 9,
    PACKED_INT4 = 10
};

// Tensor descriptor flags (must match config.slang)
enum TensorFlags : uint32_t {
    CONTIGUOUS = 1u << 0,
    READ_ONLY = 1u << 1,
    WRITE_ONLY = 1u << 2,
    BROADCAST_VIEW = 1u << 3,
    ALIGNED16 = 1u << 4,
    RANK_LE3 = 1u << 5,
    INNER_STRIDE1 = 1u << 6,
    CANONICAL = 1u << 7
};

// Kernel flags (must match config.slang)
enum KernelFlags : uint32_t {
    GUARDED_BOUNDS = 1u << 0,
    DETERMINISTIC = 1u << 1,
    HAS_WORKSPACE = 1u << 2
};

// Fixed architecture constants (must match digit_recognizer.slang)
static constexpr uint32_t DIGIT_INPUT_FEATURES = 27u * 27u;  // 729
static constexpr uint32_t DIGIT_HIDDEN_FEATURES = 128u;
static constexpr uint32_t DIGIT_CLASSES = 10u;
static constexpr uint32_t TENSOR_DESC_BYTES = 96;
static constexpr uint32_t KERNEL_HEADER_BYTES = 112;  // sizeof(DigitRecognizerParamBlock)
static constexpr uint32_t MAX_RANK = 8;

struct DigitRecognizerConfig {
    std::string shaderPath = "shaders/ML/slangml/models/digit_recognizer.slang.spv";
    // If slangSource is set, the shader is compiled from source at construction and
    // shaderPath is ignored. slangIncludeDir is searched for imported Slang modules.
    std::string slangSource;
    std::string slangIncludeDir;
    DataType weightDtype = DataType::FP16;  // FP16 or BF16
    uint32_t maxBatchSize = 1;
    bool profile = false;
};

struct DigitRecognizerResult {
    std::vector<float> probabilities;  // batch * 10
    std::vector<uint32_t> predictions; // batch
    double ms = 0.0;
};

class DigitRecognizerGpu {
public:
    DigitRecognizerGpu(LveDevice& device, const DigitRecognizerConfig& config);
    ~DigitRecognizerGpu();

    DigitRecognizerGpu(const DigitRecognizerGpu&) = delete;
    DigitRecognizerGpu& operator=(const DigitRecognizerGpu&) = delete;

    const DigitRecognizerConfig& config() const { return cfg; }

    // Run inference on a batch of 27x27 images (flattened to 729 floats each, row-major)
    // input: batch * 729 floats, values typically in [0, 1] or [-1, 1]
    // inputScale/inputBias: applied as value * scale + bias (for normalization)
    DigitRecognizerResult infer(
        const std::vector<float>& input,
        uint32_t batchSize,
        float inputScale = 1.0f,
        float inputBias = 0.0f
    );

    // Load model weights from host memory
    // weight1: 729 * 128 elements (fp16 or bf16)
    // bias1: 128 elements (fp32)
    // weight2: 128 * 10 elements (fp16 or bf16)
    // bias2: 10 elements (fp32)
    void loadWeights(
        const void* weight1, size_t weight1Size,
        const float* bias1, size_t bias1Size,
        const void* weight2, size_t weight2Size,
        const float* bias2, size_t bias2Size
    );

private:
    enum Binding {
        TensorTable = 0,
        KernelHeader,
        NumBindings
    };

    void createBuffers(uint32_t maxBatchSize);
    void createDescriptors();
    void createPipeline();
    void createQueryPool();
    void updateTensorTable(uint32_t batchSize, float inputScale, float inputBias);

    LveDevice& device;
    DigitRecognizerConfig cfg;

    std::vector<std::unique_ptr<LveBuffer>> buffers;
    std::unique_ptr<LveDescriptorSetLayout> setLayout;
    std::unique_ptr<LveDescriptorPool> pool;
    VkDescriptorSet descSet = VK_NULL_HANDLE;

    VkShaderModule shaderModule = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;

    VkQueryPool queryPool = VK_NULL_HANDLE;
    bool profileOk = false;
    float tsPeriodNs = 1.0f;
    uint32_t maxQueries = 0;

    // Weight buffers (persistent)
    std::unique_ptr<LveBuffer> weight1Buffer;
    std::unique_ptr<LveBuffer> bias1Buffer;
    std::unique_ptr<LveBuffer> weight2Buffer;
    std::unique_ptr<LveBuffer> bias2Buffer;

    // Tensor table and kernel header (updated per inference)
    std::unique_ptr<LveBuffer> tensorTableBuffer;
    std::unique_ptr<LveBuffer> kernelHeaderBuffer;

    // Per-inference buffers
    std::unique_ptr<LveBuffer> inputBuffer;
    std::unique_ptr<LveBuffer> probabilitiesBuffer;
    std::unique_ptr<LveBuffer> predictionBuffer;

    bool weightsLoaded = false;
};

}  // namespace lve