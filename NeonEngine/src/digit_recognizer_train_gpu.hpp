#pragma once

#include "digit_recognizer_train.hpp"
#include "lve_buffer.hpp"
#include "lve_descriptors.hpp"
#include "lve_device.hpp"

#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace lve {

// Push constants for digit_recognizer_train.slang (DigitTrainPush). Field order and
// types must match the shader; all fields are 4-byte scalars.
struct DigitTrainPushConstants {
    uint32_t batchStart;
    uint32_t batchCount;
    uint32_t statsOffset;
    uint32_t reserved;
    float learningRate;
    float invBatch;
};

struct DigitTrainerConfig {
    std::string slangSource;     // path to digit_recognizer_train.slang
    std::string slangIncludeDir; // directory searched for `import`ed slangml modules
    TrainConfig train;           // epochs, batch size, learning rate, seed
};

// Trains the 729 -> 128 -> 10 sigmoid MLP entirely on the GPU with Vulkan compute,
// using the same algorithm as Nielsen's network.py (quadratic cost, plain mini-batch SGD).
// The training split is uploaded once; each epoch the host only shuffles an index
// array and records one command buffer covering every minibatch of that epoch.
class DigitRecognizerTrainer {
public:
    DigitRecognizerTrainer(LveDevice& device, const DigitTrainerConfig& config, const MnistSplit& split);
    ~DigitRecognizerTrainer();

    DigitRecognizerTrainer(const DigitRecognizerTrainer&) = delete;
    DigitRecognizerTrainer& operator=(const DigitRecognizerTrainer&) = delete;

    // Runs all epochs and returns the trained fp32 parameters.
    MlpParams train();

private:
    enum Binding {
        TrainImages = 0,
        TrainLabels,
        Order,
        Params,
        Grads,
        HiddenAct,
        HiddenGrad,
        OutputGrad,
        Stats,
        NumBindings
    };

    enum Stage {
        HiddenFwd = 0,
        OutputFwd,
        HiddenBwd,
        GradOutput,
        GradHidden,
        Sgd,
        NumStages
    };

    void createBuffers();
    void createDescriptors();
    void createPipelines();
    void uploadTrainingSet(const MnistSplit& split);
    void initializeParameters(std::mt19937& rng);
    void recordStep(VkCommandBuffer cb, const DigitTrainPushConstants& pc);
    void dispatchStage(VkCommandBuffer cb, Stage stage, const DigitTrainPushConstants& pc, uint32_t threads);

    LveDevice& device;
    DigitTrainerConfig cfg;
    size_t trainCount = 0;
    uint32_t stepsPerEpoch = 0;

    std::vector<std::unique_ptr<LveBuffer>> buffers;
    std::unique_ptr<LveDescriptorSetLayout> setLayout;
    std::unique_ptr<LveDescriptorPool> pool;
    VkDescriptorSet descSet = VK_NULL_HANDLE;

    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    std::vector<VkShaderModule> shaderModules;
    std::vector<VkPipeline> pipelines;  // indexed by Stage
};

}  // namespace lve
