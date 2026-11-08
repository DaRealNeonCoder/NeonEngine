#pragma once

// gpt-oss (OpenAI open-weight MoE transformer) on the SlangML kernels.
//
// Translated from the official reference implementation
// https://github.com/openai/gpt-oss  gpt_oss/torch/model.py + weights.py
// and loads the official checkpoint layout (the `original/` folder of
// openai/gpt-oss-20b or -120b on Hugging Face: config.json + *.safetensors with
// names like block.N.attn.qkv.weight, block.N.mlp.mlp1_weight.blocks).
//
// Per layer (official TransformerBlock):
//   t = rmsnorm(x); qkv = t Wqkv^T + b; q,k = rope_yarn(q,k); a = sdpa(q,k,v,sinks, window if layer even)
//   x = x + a Wo^T + bo
//   t = rmsnorm(x); g = t Wg^T + bg; top-4 experts, softmax over their logits
//   x = x + sum_e w_e (swiglu_oai(t W1_e^T + b1_e) W2_e^T + b2_e)
// then logits = rmsnorm(x) Wu^T.
//
// Decode runs one token per step (prefill feeds the prompt token by token for
// now). Expert weights stay MXFP4 (4.25 bits); each layer's experts live in
// VRAM or in host memory that the GPU reads over PCIe (see ExpertPlacement).

#include "ml_compute.hpp"

#include <cstdint>
#include <iosfwd>
#include <string>
#include <vector>

namespace lve::ml {

class SafeTensors;

struct GptOssConfig {
    uint32_t numLayers = 24;
    uint32_t numExperts = 32;
    uint32_t expertsPerToken = 4;
    uint32_t vocabSize = 201088;
    uint32_t hidden = 2880;
    uint32_t intermediate = 2880;
    float swigluLimit = 7.0f;
    uint32_t headDim = 64;
    uint32_t numHeads = 64;
    uint32_t numKVHeads = 8;
    uint32_t slidingWindow = 128;
    uint32_t initialContext = 4096;
    double ropeTheta = 150000.0;
    double ropeScaling = 32.0;
    double ropeNtkAlpha = 1.0;
    double ropeNtkBeta = 32.0;

    static GptOssConfig load(const std::string& configJsonPath);
    uint32_t qkvDim() const { return headDim * (numHeads + 2 * numKVHeads); }
};

enum class DenseFormat { BF16, Q8 };            // attention / LM-head weights
enum class ExpertPlacement { Device, Host, Auto };

struct GptOssOptions {
    uint32_t maxContext = 4096;
    DenseFormat dense = DenseFormat::BF16;
    ExpertPlacement experts = ExpertPlacement::Auto;
    // Auto placement keeps this much VRAM free for the driver/other apps.
    uint64_t vramReserveBytes = 512ull << 20;
    bool verbose = true;
};

class GptOssModel {
public:
    GptOssModel(ComputeContext& ctx, const std::string& modelDir, const GptOssOptions& options);
    ~GptOssModel();

    GptOssModel(const GptOssModel&) = delete;
    GptOssModel& operator=(const GptOssModel&) = delete;

    const GptOssConfig& config() const { return cfg; }
    uint32_t position() const { return pos; }
    void reset() { pos = 0; }

    // Feeds `token` at the current position and advances it. With wantLogits the
    // LM head runs and the greedy (argmax) next token is returned; logitsOut, if
    // given, receives the vocabulary logits. Without wantLogits returns UINT32_MAX
    // (prompt tokens skip the 1.2 GB LM-head read).
    // hiddenOut, if given, receives rmsnorm(x) after the last layer (the Memorilla self-encoder state).
    uint32_t forward(uint32_t token, bool wantLogits, std::vector<float>* logitsOut = nullptr, double* gpuMs = nullptr,
                     std::vector<float>* hiddenOut = nullptr);
    // Same, but the input at this position is the vector `embedding` (hidden floats) instead of a
    // token's embedding row: Memorilla soft memory tokens. Only this path compiles the copy kernel.
    uint32_t forwardEmbedding(const float* vec, bool wantLogits, std::vector<float>* logitsOut = nullptr,
                              double* gpuMs = nullptr, std::vector<float>* hiddenOut = nullptr);

    void printMemoryReport(std::ostream& os) const;

private:
    struct Dense {
        GpuBuffer codes;  // BF16 [N][K] or Q8 int8 [N][K]
        GpuBuffer meta;   // Q8: half2 (d, 0) per 32-block
        DenseFormat fmt = DenseFormat::BF16;
        uint32_t N = 0, K = 0;
    };
    struct Layer {
        GpuBuffer attnNorm, qkvBias, sinks, outBias, mlpNorm, gateBias;
        Dense qkv, out, gate;
        // Experts, MXFP4: codes [E][rows][K/2] split-nibble, scales [E][rows][K/32] (padded strides).
        GpuBuffer gateCodes, gateScales, upCodes, upScales, downCodes, downScales;
        GpuBuffer gateBiasE, upBiasE, downBiasE;  // fp32 [E][I], [E][I], [E][H]
        GpuBuffer table1, table2;                 // uint64 [E][4] expert address tables
        GpuBuffer kCache, vCache;                 // fp16 [Hkv][maxCtx][D]
        bool expertsOnHost = false;
    };

    void loadWeights(const std::string& modelDir);
    Dense loadDense(SafeTensors& st, const std::string& name, DenseFormat fmt, uint32_t N, uint32_t K);
    GpuBuffer loadF32(SafeTensors& st, const std::string& name, size_t expectCount);
    void loadExperts(SafeTensors& st, uint32_t l, Layer& L, bool onHost);
    void createActivations();
    std::vector<float> yarnInvFreq(float& concentration) const;

    // Dispatch helpers (record into `rec`, parameters into `arena`).
    void gemv(ComputeContext::Recorder& rec, const Dense& w, uint64_t xF32, uint64_t xQ, uint64_t xS, uint64_t y,
              uint64_t bias, uint64_t residual, uint32_t flags);
    void rmsnorm(ComputeContext::Recorder& rec, uint64_t x, const GpuBuffer& w, bool outF32, bool outQ8);
    void quantizeAct(ComputeContext::Recorder& rec, uint64_t x, uint64_t xq, uint64_t xs, uint32_t rows, uint32_t cols);
    uint32_t step(uint32_t token, const float* softVec, bool wantLogits, std::vector<float>* logitsOut, double* gpuMs,
                  std::vector<float>* hiddenOut);

    ComputeContext& ctx;
    GptOssOptions opts;
    GptOssConfig cfg;
    std::vector<Layer> layers;
    GpuBuffer embedding;        // bf16 [V][H], host memory (one row read per token)
    GpuBuffer finalNorm;
    Dense unembed;
    GpuBuffer invFreq;          // YaRN inv_freq [D/2]
    float concentration = 1.0f;

    // Activations (one token).
    GpuBuffer x, hNorm, hQ, hS, qkv, qRot, attn, attnQ, attnS, wsO, wsML;
    GpuBuffer routerLogits, expertIds, expertWeights, moeH, moeQ, moeS;
    GpuBuffer logits, argmaxV, argmaxI;
    GpuBuffer tokenBuf, posBuf, outToken;  // host-visible
    GpuBuffer softIn;                      // host-visible soft-token input (created on first use)
    uint32_t maxChunks = 16;

    std::unique_ptr<ParamArena> arena;
    uint32_t pos = 0;
    uint64_t deviceBytes = 0, hostBytes = 0;
    uint32_t hostExpertLayers = 0;
};

}  // namespace lve::ml
