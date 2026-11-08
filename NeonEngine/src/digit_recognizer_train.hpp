#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace lve {

// MNIST resized to 27x27 (729 features, row-major, values in [0,1]).
struct MnistSplit {
    std::vector<float> trainX;    // [trainCount, 729]
    std::vector<uint8_t> trainY;  // [trainCount]
    std::vector<float> testX;     // [testCount, 729]
    std::vector<uint8_t> testY;   // [testCount]

    size_t trainCount() const { return trainY.size(); }
    size_t testCount() const { return testY.size(); }
};

// Layout matches what DigitRecognizerGpu and tryLoadWeights expect.
struct MlpParams {
    std::vector<float> w1;  // [729, 128]
    std::vector<float> b1;  // [128]
    std::vector<float> w2;  // [128, 10]
    std::vector<float> b2;  // [10]
};

// Defaults are the hyperparameters from Nielsen's book: net.SGD(training_data, 30, 10, 3.0).
struct TrainConfig {
    uint32_t epochs = 30;
    uint32_t batchSize = 10;
    float learningRate = 3.0f;  // eta
    // With sigmoid outputs + quadratic cost, some seeds leave a digit's output neuron saturated and it
    // never learns (~77-90% test accuracy). Seed 9 trains cleanly (97.4%); 20241 gets stuck at 87.7%.
    uint32_t seed = 9;
};

// Reads the official MNIST IDX files from dir (60k train, 10k test) and resizes them to 27x27.
// Expects: train-images-idx3-ubyte, train-labels-idx1-ubyte, t10k-images-idx3-ubyte, t10k-labels-idx1-ubyte.
// Returns false if any file is missing or malformed.
bool loadMnistSplit(const std::string& dir, MnistSplit& out);

// Writes weight1_f16.bin, bias1_f32.bin, weight2_f16.bin, bias2_f32.bin into dir.
bool saveWeights(const std::string& dir, const MlpParams& params);

}  // namespace lve
