#include "digit_recognizer_app.hpp"

#include "digit_recognizer_gpu.hpp"
#include "digit_recognizer_train.hpp"
#include "digit_recognizer_train_gpu.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include <fcntl.h>
#include <io.h>

namespace lve {

namespace {

constexpr uint32_t INPUT_FEATURES = DIGIT_INPUT_FEATURES;  // 729
constexpr uint32_t HIDDEN_FEATURES = DIGIT_HIDDEN_FEATURES;  // 128
constexpr uint32_t CLASSES = DIGIT_CLASSES;  // 10

// Experiment settings
constexpr bool RUN_SANITY = true;
constexpr bool RUN_BENCH = true;
constexpr bool RUN_RANDOM_TEST = true;
constexpr bool RUN_TRAIN = true;       // train on MNIST train split on the GPU, save weights to WEIGHTS_DIR
constexpr bool RUN_MNIST_TEST = true;  // score the MNIST test split through the Vulkan inference path

constexpr uint32_t MAX_BATCH_SIZE = 32;
constexpr uint32_t BENCH_ITERS = 100;
constexpr uint32_t BENCH_WARMUP = 10;
constexpr uint32_t RANDOM_TEST_SAMPLES = 100;

const char* SLANG_SOURCE = "C:\\Users\\ZyBros\\Downloads\\NeonEngine\\NeonEngine\\shaders\\ML\\slangml\\models\\digit_recognizer.slang";
const char* SLANG_TRAIN_SOURCE = "C:\\Users\\ZyBros\\Downloads\\NeonEngine\\NeonEngine\\shaders\\ML\\slangml\\models\\digit_recognizer_train.slang";
const char* SLANG_INCLUDE_DIR ="C:\\Users\\ZyBros\\Downloads\\NeonEngine\\NeonEngine\\shaders\\ML\\slangml";
const char* WEIGHTS_DIR = "C:\\Users\\ZyBros\\Downloads\\NeonEngine\\weights\\digit_recognizer";  // Optional: directory with weight files
const char* MNIST_DIR = "C:\\Users\\ZyBros\\Downloads\\NeonEngine\\data\\mnist";  // raw IDX files (not gzipped)

// Simple struct to hold model weights
struct ModelWeights {
    std::vector<uint16_t> weight1_f16;  // 729 * 128 FP16
    std::vector<float> bias1;           // 128 FP32
    std::vector<uint16_t> weight2_f16;  // 128 * 10 FP16
    std::vector<float> bias2;           // 10 FP32
    bool loaded = false;
};

// Try to load weights from binary files
ModelWeights tryLoadWeights(const std::string& baseDir) {
    ModelWeights w;
    namespace fs = std::filesystem;

    auto loadFile = [](const fs::path& path, std::vector<char>& out) -> bool {
        if (!fs::exists(path)) return false;
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        if (!file) return false;
        size_t size = file.tellg();
        file.seekg(0);
        out.resize(size);
        file.read(out.data(), size);
        return true;
    };

    std::vector<char> data;

    // Try FP16 weights
    if (loadFile(fs::path(baseDir) / "weight1_f16.bin", data)) {
        w.weight1_f16.assign(reinterpret_cast<uint16_t*>(data.data()),
                            reinterpret_cast<uint16_t*>(data.data() + data.size()));
    }
    if (loadFile(fs::path(baseDir) / "bias1_f32.bin", data)) {
        w.bias1.assign(reinterpret_cast<float*>(data.data()),
                      reinterpret_cast<float*>(data.data() + data.size()));
    }
    if (loadFile(fs::path(baseDir) / "weight2_f16.bin", data)) {
        w.weight2_f16.assign(reinterpret_cast<uint16_t*>(data.data()),
                            reinterpret_cast<uint16_t*>(data.data() + data.size()));
    }
    if (loadFile(fs::path(baseDir) / "bias2_f32.bin", data)) {
        w.bias2.assign(reinterpret_cast<float*>(data.data()),
                      reinterpret_cast<float*>(data.data() + data.size()));
    }

    w.loaded = !w.weight1_f16.empty() && !w.bias1.empty() &&
               !w.weight2_f16.empty() && !w.bias2.empty();

    if (w.loaded) {
        std::cout << "Loaded weights from " << baseDir << "\n";
        std::cout << "  weight1: " << w.weight1_f16.size() << " elements (FP16)\n";
        std::cout << "  bias1: " << w.bias1.size() << " elements (FP32)\n";
        std::cout << "  weight2: " << w.weight2_f16.size() << " elements (FP16)\n";
        std::cout << "  bias2: " << w.bias2.size() << " elements (FP32)\n";
    }

    return w;
}

// Generate random weights for testing (Xavier/Glorot initialization)
ModelWeights generateRandomWeights(uint32_t seed = 42) {
    ModelWeights w;
    std::mt19937 rng(seed);

    // Xavier initialization for weight1: 729 -> 128
    // scale = sqrt(6 / (fan_in + fan_out))
    float scale1 = std::sqrt(6.0f / (INPUT_FEATURES + HIDDEN_FEATURES));
    std::uniform_real_distribution<float> dist1(-scale1, scale1);

    w.weight1_f16.resize(INPUT_FEATURES * HIDDEN_FEATURES);
    for (size_t i = 0; i < w.weight1_f16.size(); ++i) {
        float val = dist1(rng);
        // Convert to FP16
        uint32_t bits = *reinterpret_cast<uint32_t*>(&val);
        uint32_t sign = bits & 0x80000000u;
        int32_t exponent = ((bits >> 23) & 0xFF) - 127 + 15;
        uint32_t mantissa = (bits >> 13) & 0x3FFu;
        if (exponent <= 0) {
            w.weight1_f16[i] = static_cast<uint16_t>(sign >> 16);
        } else if (exponent >= 31) {
            w.weight1_f16[i] = static_cast<uint16_t>((sign >> 16) | 0x7C00u);
        } else {
            w.weight1_f16[i] = static_cast<uint16_t>((sign >> 16) | (exponent << 10) | mantissa);
        }
    }

    // bias1: zeros
    w.bias1.assign(HIDDEN_FEATURES, 0.0f);

    // Xavier initialization for weight2: 128 -> 10
    float scale2 = std::sqrt(6.0f / (HIDDEN_FEATURES + CLASSES));
    std::uniform_real_distribution<float> dist2(-scale2, scale2);

    w.weight2_f16.resize(HIDDEN_FEATURES * CLASSES);
    for (size_t i = 0; i < w.weight2_f16.size(); ++i) {
        float val = dist2(rng);
        uint32_t bits = *reinterpret_cast<uint32_t*>(&val);
        uint32_t sign = bits & 0x80000000u;
        int32_t exponent = ((bits >> 23) & 0xFF) - 127 + 15;
        uint32_t mantissa = (bits >> 13) & 0x3FFu;
        if (exponent <= 0) {
            w.weight2_f16[i] = static_cast<uint16_t>(sign >> 16);
        } else if (exponent >= 31) {
            w.weight2_f16[i] = static_cast<uint16_t>((sign >> 16) | 0x7C00u);
        } else {
            w.weight2_f16[i] = static_cast<uint16_t>((sign >> 16) | (exponent << 10) | mantissa);
        }
    }

    // bias2: zeros
    w.bias2.assign(CLASSES, 0.0f);

    w.loaded = true;
    std::cout << "Generated random weights (Xavier init)\n";
    return w;
}

// Generate a simple synthetic digit pattern for testing
// Creates a pattern that resembles a specific digit (0-9)
std::vector<float> generateSyntheticDigit(int digit, float noise = 0.1f) {
    std::vector<float> img(INPUT_FEATURES, 0.0f);
    std::mt19937 rng(digit * 12345);
    std::uniform_real_distribution<float> noiseDist(-noise, noise);

    const int W = 27, H = 27;
    auto setPixel = [&](int x, int y, float v) {
        if (x >= 0 && x < W && y >= 0 && y < H) {
            img[y * W + x] = std::clamp(v + noiseDist(rng), 0.0f, 1.0f);
        }
    };

    // Simple digit templates (very basic)
    switch (digit) {
        case 0: // Circle/oval
            for (int y = 5; y < 22; ++y) {
                for (int x = 8; x < 19; ++x) {
                    float dx = (x - 13.5f) / 5.5f;
                    float dy = (y - 13.5f) / 8.5f;
                    if (dx*dx + dy*dy > 0.7f && dx*dx + dy*dy < 1.1f) setPixel(x, y, 1.0f);
                }
            }
            break;
        case 1: // Vertical line
            for (int y = 3; y < 24; ++y) setPixel(13, y, 1.0f);
            for (int y = 3; y < 8; ++y) setPixel(12, y, 1.0f); // base
            break;
        case 2: // 2 shape
            for (int x = 7; x < 20; ++x) setPixel(x, 4, 1.0f); // top
            for (int y = 4; y < 13; ++y) setPixel(19, y, 1.0f); // right top
            for (int x = 7; x < 20; ++x) setPixel(x, 13, 1.0f); // middle
            for (int y = 13; y < 22; ++y) setPixel(7, y, 1.0f); // left bottom
            for (int x = 7; x < 20; ++x) setPixel(x, 22, 1.0f); // bottom
            break;
        case 3: // 3 shape
            for (int x = 7; x < 20; ++x) setPixel(x, 4, 1.0f); // top
            for (int y = 4; y < 22; ++y) setPixel(19, y, 1.0f); // right
            for (int x = 7; x < 20; ++x) setPixel(x, 13, 1.0f); // middle
            for (int x = 7; x < 20; ++x) setPixel(x, 22, 1.0f); // bottom
            break;
        case 4: // 4 shape
            for (int y = 4; y < 14; ++y) setPixel(7, y, 1.0f); // left top
            for (int y = 4; y < 23; ++y) setPixel(19, y, 1.0f); // right
            for (int x = 7; x < 20; ++x) setPixel(x, 13, 1.0f); // middle
            break;
        case 5: // 5 shape
            for (int x = 7; x < 20; ++x) setPixel(x, 4, 1.0f); // top
            for (int y = 4; y < 13; ++y) setPixel(7, y, 1.0f); // left top
            for (int x = 7; x < 20; ++x) setPixel(x, 13, 1.0f); // middle
            for (int y = 13; y < 22; ++y) setPixel(19, y, 1.0f); // right bottom
            for (int x = 7; x < 20; ++x) setPixel(x, 22, 1.0f); // bottom
            break;
        case 6: // 6 shape
            for (int x = 7; x < 20; ++x) setPixel(x, 4, 1.0f); // top
            for (int y = 4; y < 22; ++y) setPixel(7, y, 1.0f); // left
            for (int x = 7; x < 20; ++x) setPixel(x, 13, 1.0f); // middle
            for (int y = 13; y < 22; ++y) setPixel(19, y, 1.0f); // right bottom
            for (int x = 7; x < 20; ++x) setPixel(x, 22, 1.0f); // bottom
            break;
        case 7: // 7 shape
            for (int x = 7; x < 20; ++x) setPixel(x, 4, 1.0f); // top
            for (int y = 4; y < 22; ++y) setPixel(19, y, 1.0f); // right
            break;
        case 8: // 8 shape
            for (int y = 4; y < 22; ++y) { setPixel(7, y, 1.0f); setPixel(19, y, 1.0f); } // sides
            for (int x = 7; x < 20; ++x) { setPixel(x, 4, 1.0f); setPixel(x, 13, 1.0f); setPixel(x, 22, 1.0f); } // horizontals
            break;
        case 9: // 9 shape
            for (int x = 7; x < 20; ++x) setPixel(x, 4, 1.0f); // top
            for (int y = 4; y < 13; ++y) { setPixel(7, y, 1.0f); setPixel(19, y, 1.0f); } // sides top
            for (int x = 7; x < 20; ++x) setPixel(x, 13, 1.0f); // middle
            for (int y = 13; y < 22; ++y) setPixel(19, y, 1.0f); // right bottom
            break;
    }

    return img;
}

float sigmoid(float z) { return 1.0f / (1.0f + std::exp(-z)); }

// Simple CPU reference implementation for verification
std::vector<float> cpuForward(const std::vector<float>& input,
                              const std::vector<uint16_t>& weight1_f16,
                              const std::vector<float>& bias1,
                              const std::vector<uint16_t>& weight2_f16,
                              const std::vector<float>& bias2) {
    // Convert FP16 weights to FP32 for CPU computation
    auto f16tof32 = [](uint16_t h) -> float {
        uint32_t sign = (h >> 15) & 1;
        uint32_t exponent = (h >> 10) & 0x1F;
        uint32_t mantissa = h & 0x3FF;
        if (exponent == 0) {
            if (mantissa == 0) return sign ? -0.0f : 0.0f;
            return (sign ? -1.0f : 1.0f) * std::ldexp(static_cast<float>(mantissa), -24);
        }
        if (exponent == 31) {
            return sign ? -INFINITY : INFINITY;
        }
        return (sign ? -1.0f : 1.0f) * std::ldexp(1.0f + mantissa / 1024.0f, exponent - 15);
    };

    // Layer 1: input (729) -> hidden (128) with sigmoid
    std::vector<float> hidden(HIDDEN_FEATURES);
    for (uint32_t h = 0; h < HIDDEN_FEATURES; ++h) {
        float val = bias1[h];
        for (uint32_t i = 0; i < INPUT_FEATURES; ++i) {
            uint16_t w = weight1_f16[i * HIDDEN_FEATURES + h];
            val += input[i] * f16tof32(w);
        }
        hidden[h] = sigmoid(val);
    }

    // Layer 2: hidden (128) -> output (10) with sigmoid
    std::vector<float> outputs(CLASSES);
    for (uint32_t c = 0; c < CLASSES; ++c) {
        float val = bias2[c];
        for (uint32_t h = 0; h < HIDDEN_FEATURES; ++h) {
            uint16_t w = weight2_f16[h * CLASSES + c];
            val += hidden[h] * f16tof32(w);
        }
        outputs[c] = sigmoid(val);
    }

    return outputs;
}

// Print probabilities nicely
void printProbabilities(const std::vector<float>& probs, const char* label) {
    std::printf("%s: ", label);
    for (int c = 0; c < CLASSES; ++c) {
        std::printf(" %d:%.3f", c, probs[c]);
    }
    std::printf("\n");
}

// Find predicted class
int argmax(const std::vector<float>& probs) {
    return static_cast<int>(std::distance(probs.begin(), std::max_element(probs.begin(), probs.end())));
}

}  // namespace

void DigitRecognizerApp::run() {
    std::cout << "=== Digit Recognizer Demo ===\n\n";

    // Create GPU inference engine
    DigitRecognizerConfig cfg;
    cfg.slangSource = SLANG_SOURCE;  // compiled to SPIR-V at startup
    cfg.slangIncludeDir = SLANG_INCLUDE_DIR;
    cfg.weightDtype = DataType::FP16;
    cfg.maxBatchSize = MAX_BATCH_SIZE;
    cfg.profile = true;

    DigitRecognizerGpu recognizer(lveDevice, cfg);

    // Load the MNIST split once; it is used for training and for the test-set score.
    MnistSplit mnist;
    const bool haveMnist = loadMnistSplit(MNIST_DIR, mnist);
    if (haveMnist) {
        std::printf("MNIST: %zu train, %zu test (27x27)\n", mnist.trainCount(), mnist.testCount());
    } else {
        std::cout << "MNIST not found in " << MNIST_DIR << "\n";
    }

    // Train on the GPU with the Slang training kernels and write the weights the loader below reads back.
    if (RUN_TRAIN && haveMnist) {
        std::cout << "######## TRAINING (Vulkan/Slang, 729-128-10 sigmoid MLP, quadratic cost, SGD) ########\n";
        DigitTrainerConfig trainCfg;
        trainCfg.slangSource = SLANG_TRAIN_SOURCE;
        trainCfg.slangIncludeDir = SLANG_INCLUDE_DIR;
        std::printf("epochs=%u  batch=%u  eta=%.3f  seed=%u\n", trainCfg.train.epochs, trainCfg.train.batchSize,
            trainCfg.train.learningRate, trainCfg.train.seed);
        const auto setupStart = std::chrono::steady_clock::now();
        DigitRecognizerTrainer trainer(lveDevice, trainCfg, mnist);  // compiles shaders, uploads the train split
        const double setupSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - setupStart).count();
        const MlpParams trained = trainer.train();
        const double totalSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - setupStart).count();
        std::printf("GPU trainer setup (shader compile + upload): %.2f s; setup + training: %.2f s\n",
            setupSeconds, totalSeconds);
        if (!saveWeights(WEIGHTS_DIR, trained)) {
            throw std::runtime_error("failed to write weights to WEIGHTS_DIR");
        }
        std::cout << "Saved trained weights to " << WEIGHTS_DIR << "\n\n";
    }

    // Load or generate weights
    ModelWeights weights = tryLoadWeights(WEIGHTS_DIR);
    if (!weights.loaded) {
        std::cout << "No pre-trained weights found, generating random weights for testing\n";
        weights = generateRandomWeights(20241);
    }

    // Upload weights to GPU
    recognizer.loadWeights(
        weights.weight1_f16.data(), weights.weight1_f16.size() * sizeof(uint16_t),
        weights.bias1.data(), weights.bias1.size() * sizeof(float),
        weights.weight2_f16.data(), weights.weight2_f16.size() * sizeof(uint16_t),
        weights.bias2.data(), weights.bias2.size() * sizeof(float)
    );

    std::cout << "Weights uploaded to GPU\n\n";

    // ============================================================
    // 0. MNIST TEST SET: score the held-out split through the GPU
    // ============================================================
    if (RUN_MNIST_TEST && haveMnist) {
        std::cout << "######## MNIST TEST SET (GPU) ########\n";
        const size_t total = mnist.testCount();
        size_t correct = 0;
        for (size_t start = 0; start < total; start += MAX_BATCH_SIZE) {
            const uint32_t bs = static_cast<uint32_t>(std::min<size_t>(MAX_BATCH_SIZE, total - start));
            std::vector<float> batch(mnist.testX.begin() + start * INPUT_FEATURES,
                                     mnist.testX.begin() + (start + bs) * INPUT_FEATURES);
            const auto result = recognizer.infer(batch, bs);
            for (uint32_t b = 0; b < bs; ++b) {
                if (static_cast<uint8_t>(result.predictions[b]) == mnist.testY[start + b]) ++correct;
            }
        }
        std::printf("MNIST test accuracy: %zu/%zu = %.4f\n\n", correct, total,
                    double(correct) / double(total));
    }

    // ============================================================
    // 1. SANITY TEST: Synthetic digits
    // ============================================================
    if (RUN_SANITY) {
        std::cout << "######## 1. SANITY TEST (synthetic digits) ########\n";

        int correct = 0;
        for (int digit = 0; digit < 10; ++digit) {
            // Generate synthetic digit
            std::vector<float> input = generateSyntheticDigit(digit, 0.05f);

            // Run inference
            auto result = recognizer.infer(input, 1);

            int predicted = result.predictions[0];
            float confidence = result.probabilities[predicted];

            std::printf("Digit %d: predicted=%d, confidence=%.3f, latency=%.2f ms %s\n",
                digit, predicted, confidence, result.ms,
                predicted == digit ? "[OK]" : "[MISMATCH]");

            if (predicted == digit) correct++;

            // Print top 3 probabilities
            std::vector<std::pair<int, float>> top;
            for (int c = 0; c < 10; ++c) top.emplace_back(c, result.probabilities[c]);
            std::sort(top.begin(), top.end(), [](auto& a, auto& b) { return a.second > b.second; });
            std::printf("  Top-3: ");
            for (int i = 0; i < 3; ++i) {
                std::printf("%d(%.3f) ", top[i].first, top[i].second);
            }
            std::printf("\n");
        }
        std::printf("Accuracy: %d/10\n\n", correct);
    }

    // ============================================================
    // 2. BENCHMARK
    // ============================================================
    if (RUN_BENCH) {
        std::cout << "######## 2. BENCHMARK ########\n";

        // Generate test batch
        std::vector<std::vector<float>> testInputs;
        for (int i = 0; i < MAX_BATCH_SIZE; ++i) {
            testInputs.push_back(generateSyntheticDigit(i % 10, 0.1f));
        }

        // Warmup
        for (uint32_t i = 0; i < BENCH_WARMUP; ++i) {
            recognizer.infer(testInputs[i % MAX_BATCH_SIZE], 1);
        }

        // Benchmark batch sizes
        for (uint32_t batchSize : {1u, 4u, 8u, 16u, 32u}) {
            if (batchSize > MAX_BATCH_SIZE) continue;

            // Flatten inputs
            std::vector<float> batchInput(batchSize * INPUT_FEATURES);
            for (uint32_t b = 0; b < batchSize; ++b) {
                std::copy(testInputs[b].begin(), testInputs[b].end(),
                         batchInput.begin() + b * INPUT_FEATURES);
            }

            double totalMs = 0.0;
            for (uint32_t i = 0; i < BENCH_ITERS; ++i) {
                auto result = recognizer.infer(batchInput, batchSize);
                totalMs += result.ms;
            }

            double avgMs = totalMs / BENCH_ITERS;
            double throughput = batchSize / (avgMs / 1000.0);
            std::printf("Batch %2d: %.3f ms/iter (%.1f images/sec, %.3f ms/image)\n",
                batchSize, avgMs, throughput, avgMs / batchSize);
        }
        std::printf("\n");
    }

    // ============================================================
    // 3. RANDOM TEST: Compare GPU vs CPU
    // ============================================================
    if (RUN_RANDOM_TEST && weights.loaded) {
        std::cout << "######## 3. GPU vs CPU VERIFICATION ########\n";

        std::mt19937 rng(12345);
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);

        int mismatches = 0;
        double maxDiff = 0.0;

        for (uint32_t i = 0; i < RANDOM_TEST_SAMPLES; ++i) {
            // Generate random input
            std::vector<float> input(INPUT_FEATURES);
            for (auto& v : input) v = dist(rng);

            // GPU inference
            auto gpuResult = recognizer.infer(input, 1);

            // CPU reference
            std::vector<float> cpuProbs = cpuForward(input, weights.weight1_f16, weights.bias1,
                                                     weights.weight2_f16, weights.bias2);

            // Compare
            int gpuPred = gpuResult.predictions[0];
            int cpuPred = argmax(cpuProbs);

            if (gpuPred != cpuPred) {
                mismatches++;
                std::printf("Sample %d: GPU pred=%d, CPU pred=%d [MISMATCH]\n", i, gpuPred, cpuPred);
            }

            for (int c = 0; c < CLASSES; ++c) {
                double diff = std::abs(gpuResult.probabilities[c] - cpuProbs[c]);
                maxDiff = std::max(maxDiff, diff);
            }
        }

        std::printf("Mismatches: %d/%d\n", mismatches, RANDOM_TEST_SAMPLES);
        std::printf("Max probability difference: %.6f\n", maxDiff);
        std::printf("%s\n\n", mismatches == 0 && maxDiff < 1e-3 ? "[VERIFICATION PASSED]" : "[VERIFICATION FAILED]");
    }

    // ============================================================
    // 4. BATCH INFERENCE DEMO
    // ============================================================
    std::cout << "######## 4. BATCH INFERENCE DEMO ########\n";

    std::vector<std::vector<float>> batchInputs;
    std::vector<int> trueLabels;
    for (int digit = 0; digit < 10; ++digit) {
        batchInputs.push_back(generateSyntheticDigit(digit, 0.0f)); // no noise for clean test
        trueLabels.push_back(digit);
    }

    // Flatten batch
    std::vector<float> batchInput(batchInputs.size() * INPUT_FEATURES);
    for (size_t b = 0; b < batchInputs.size(); ++b) {
        std::copy(batchInputs[b].begin(), batchInputs[b].end(),
                 batchInput.begin() + b * INPUT_FEATURES);
    }

    auto batchResult = recognizer.infer(batchInput, static_cast<uint32_t>(batchInputs.size()));

    std::printf("Batch size: %zu, Latency: %.2f ms (%.2f ms/image)\n",
        batchInputs.size(), batchResult.ms, batchResult.ms / batchInputs.size());

    int batchCorrect = 0;
    for (size_t b = 0; b < batchInputs.size(); ++b) {
        int pred = batchResult.predictions[b];
        float conf = batchResult.probabilities[b * CLASSES + pred];
        bool ok = (pred == trueLabels[b]);
        if (ok) batchCorrect++;
        std::printf("  Sample %zu: true=%d, pred=%d, conf=%.3f %s\n",
            b, trueLabels[b], pred, conf, ok ? "[OK]" : "[WRONG]");
    }
    std::printf("Batch accuracy: %d/%zu\n\n", batchCorrect, batchInputs.size());

    std::cout << "=== Demo Complete ===\n";
}

// Serve mode: a long-lived Vulkan inference process for external clients (the Python drawing app).
//
// Protocol over the process's stdin/stdout, all little-endian:
//   startup:  the 5 bytes "READY" once the weights are on the GPU
//   request:  729 float32 values, a 27x27 row-major image in [0, 1]
//   response: 10 float32 probabilities, then one uint32 predicted digit
//
// Every other console message from the engine (device enumeration, validation) is
// redirected to stderr at the file-descriptor level so it cannot corrupt the stream.
namespace {
int gServeProtocolFd = -1;
}

void beginServeProtocol() {
    std::fflush(stdout);
    gServeProtocolFd = _dup(_fileno(stdout));
    _dup2(_fileno(stderr), _fileno(stdout));
    _setmode(gServeProtocolFd, _O_BINARY);
    _setmode(_fileno(stdin), _O_BINARY);
}

void DigitRecognizerApp::serve() {
    if (gServeProtocolFd < 0) throw std::runtime_error("serve: beginServeProtocol() was not called");
    const int protoFd = gServeProtocolFd;

    auto writeAll = [protoFd](const void* data, size_t bytes) {
        const char* p = static_cast<const char*>(data);
        while (bytes > 0) {
            const int n = _write(protoFd, p, static_cast<unsigned int>(bytes));
            if (n <= 0) throw std::runtime_error("serve: failed to write to client");
            p += n;
            bytes -= static_cast<size_t>(n);
        }
    };
    auto readAll = [](void* data, size_t bytes) {
        char* p = static_cast<char*>(data);
        while (bytes > 0) {
            const size_t n = std::fread(p, 1, bytes, stdin);
            if (n == 0) return false;  // client closed the pipe
            p += n;
            bytes -= n;
        }
        return true;
    };

    DigitRecognizerConfig cfg;
    cfg.slangSource = SLANG_SOURCE;
    cfg.slangIncludeDir = SLANG_INCLUDE_DIR;
    cfg.weightDtype = DataType::FP16;
    cfg.maxBatchSize = 1;
    cfg.profile = false;
    DigitRecognizerGpu recognizer(lveDevice, cfg);

    ModelWeights weights = tryLoadWeights(WEIGHTS_DIR);
    if (!weights.loaded) throw std::runtime_error("serve: no weights found in WEIGHTS_DIR");
    recognizer.loadWeights(
        weights.weight1_f16.data(), weights.weight1_f16.size() * sizeof(uint16_t),
        weights.bias1.data(), weights.bias1.size() * sizeof(float),
        weights.weight2_f16.data(), weights.weight2_f16.size() * sizeof(uint16_t),
        weights.bias2.data(), weights.bias2.size() * sizeof(float));

    writeAll("READY", 5);
    std::cerr << "[serve] ready\n";

    std::vector<float> image(INPUT_FEATURES);
    while (readAll(image.data(), image.size() * sizeof(float))) {
        const DigitRecognizerResult result = recognizer.infer(image, 1);
        writeAll(result.probabilities.data(), CLASSES * sizeof(float));
        const uint32_t prediction = result.predictions[0];
        writeAll(&prediction, sizeof(prediction));
    }
    std::cerr << "[serve] client closed the pipe, exiting\n";
}

}  // namespace lve