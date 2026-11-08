#pragma once

// gpt-oss training on the SlangML kernels (docs: shaders/ML/docs/gpt_oss_training.md).
//
// Trains a gpt-oss-architecture model from scratch: the same blocks as
// gpt_oss_model.hpp (RMSNorm, QKV with bias, YaRN RoPE, attention with sinks and
// alternating sliding window, top-k MoE with clamped SwiGLU, untied LM head).
// Master weights, gradients and AdamW state are fp32 on the GPU. MoE experts can
// be trained quantization-aware (MXFP4 fake quantization with a straight-through
// gradient) so the exported checkpoint, whose experts are MXFP4 like the official
// one, loses little accuracy.
//
// A step: per micro-batch, one command buffer with the whole forward, the
// chunked LM head + cross-entropy, and the backward; then one with gradient-norm
// clipping + AdamW. Checkpoints (params + Adam moments + step) are written
// atomically and can be mirrored to a second directory (e.g. Google Drive).
// `export` turns a checkpoint into the official layout that `--gpt-oss` loads.

#include "gpt_oss_model.hpp"
#include "ml_compute.hpp"

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace lve::ml {

struct TrainOptions {
    std::string dataDir;          // train.bin / val.bin (uint16 tokens) + meta.json
    std::string outDir = "run";   // checkpoints, log, export
    std::string mirrorDir;        // optional second copy of every checkpoint (Google Drive)
    std::string tokenizer = "gpt2";  // tiktoken encoding name, or "neon_bpe" (tokenizer.tiktoken from the data dir)
    std::string tokenizerFile;       // data-dir tokenizer.tiktoken (neon_bpe), copied next to checkpoints and exports
    double bytesPerToken = 0.0;      // validation text bytes per token (meta.json) -> bits-per-byte in the log
    uint32_t batch = 16;          // sequences per micro-batch
    uint32_t seq = 512;           // tokens per sequence (multiple of 64)
    uint32_t accum = 1;           // micro-batches per optimizer step
    uint64_t maxSteps = 20000;
    uint64_t warmup = 500;
    float lr = 1.0e-3f, minLr = 1.0e-4f;
    // After warmup: "cosine" to minLr, "linear" to minLr, or "wsd" (warmup-stable-decay:
    // constant lr, then a linear cooldown to minLr over the last decayFrac of the steps;
    // a run can be extended before its cooldown starts without restarting the schedule).
    std::string schedule = "wsd";
    float decayFrac = 0.2f;
    float beta1 = 0.9f, beta2 = 0.95f, eps = 1.0e-8f, weightDecay = 0.1f, clip = 1.0f;
    float auxCoef = 0.01f;        // MoE load-balancing loss
    int64_t qatStart = -1;        // step where MXFP4 fake quantization of experts starts (-1: 80% of maxSteps, -2: never)
    uint32_t evalEvery = 500, evalBatches = 20, logEvery = 10;
    double ckptMinutes = 30.0;    // wall-clock checkpoint interval
    uint64_t seed = 1234;
    uint32_t lmChunk = 2048;      // rows per LM-head chunk (bounds the logits buffer)
    bool fresh = false;           // ignore existing checkpoints
    bool exportAtEnd = true;
    // GEMM precision: "fp16" = tensor cores (fp16 inputs, fp32 accumulate) with
    // dynamic loss scaling, "fp32" = SIMT fp32, "auto" = fp16 when the device has
    // cooperative matrices. Everything outside the GEMMs stays fp32.
    std::string precision = "auto";
    float lossScale = 65536.0f;   // initial loss scale for fp16 (halved on overflow, doubled after growthInterval clean steps)
    uint32_t growthInterval = 1000;
    uint32_t profileSteps = 0;    // > 0: time every kernel for this many steps (after 2 warm-up steps), print, exit
    // Frozen decoder (Memorilla training and encoding; off for normal training):
    // no weight gradients, no optimizer state for the LLM, only the input gradient dR.
    bool frozen = false;
    // Memorilla: rows [0, memRows) of every sequence take their input vector from
    // the memory source buffer (setMemorySource) instead of the token embedding.
    uint32_t memRows = 0;
};

// One trainable tensor inside the flat fp32 parameter buffer.
struct ParamTensor {
    std::string name;             // official checkpoint name
    std::vector<uint64_t> shape;
    uint64_t offset = 0;          // in floats
    uint64_t count = 0;
    bool decay = false;
    enum class Export { F32, BF16, MXFP4 } exportAs = Export::BF16;
};

struct ParamLayout {
    std::vector<ParamTensor> tensors;
    uint64_t total = 0;           // floats (each tensor 16-byte aligned)
    uint64_t decayEnd = 0;        // tensors with decay come first
    static ParamLayout build(const GptOssConfig& c);
    const ParamTensor& at(const std::string& name) const;
};

// Writes config.json + model.safetensors in the official gpt-oss layout from
// fp32 parameters (dense -> bf16, experts -> MXFP4, norms -> fp32). A BPE
// tokenizer file (if given and present) is copied in as tokenizer.tiktoken.
void exportGptOssCheckpoint(const GptOssConfig& cfg, const ParamLayout& layout, const std::vector<float>& params,
                            const std::string& outDir, const std::string& tokenizer,
                            const std::string& tokenizerFile = "");

// Checkpoint file: header JSON + params + Adam m + Adam v.
struct CheckpointInfo {
    std::string path;
    uint64_t step = 0;
    bool valid = false;
};
CheckpointInfo probeCheckpoint(const std::string& path);
// Reads only the parameters (for export); returns the config.
GptOssConfig readCheckpointParams(const std::string& path, std::vector<float>& params, std::string* tokenizer = nullptr);

class GptOssTrainer {
public:
    GptOssTrainer(ComputeContext& ctx, const GptOssConfig& cfg, const TrainOptions& opts);
    ~GptOssTrainer();

    GptOssTrainer(const GptOssTrainer&) = delete;
    GptOssTrainer& operator=(const GptOssTrainer&) = delete;

    // Full training loop (resumes from the newest checkpoint in outDir / mirrorDir).
    void train();

    // One forward/backward on the given tokens (batch * (seq + 1) ids) with the
    // current parameters; returns the mean loss and leaves gradients in `grads`.
    float gradientCheck(const std::vector<uint32_t>& tokens, std::vector<float>& params, std::vector<float>& grads);

    const ParamLayout& layout() const { return lay; }
    bool usesTensorCores() const { return tensorCores; }
    static std::atomic<bool>& stopRequested();

    // ---- Lower-level access for the frozen-decoder (Memorilla) path ----
    const GptOssConfig& config() const { return cfg; }
    const TrainOptions& options() const { return opts; }
    void setParams(const std::vector<float>& host);      // full parameter vector (layout order)
    void initRandom() { initParams(); }                  // the trainer's from-scratch init (seeded by --seed)
    std::vector<float> paramsHost() const;
    // Expert weights as the MXFP4 export rounds them (forward and dX use them from now on).
    void useMxfp4Experts();
    // inputs / targets: batch * seq ids each; a target of 0xFFFFFFFF carries no loss.
    void setBatch(const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& targets);
    void setMemorySource(uint64_t address) { memSource = address; }  // [batch][memRows][hidden]
    void setLossDenominator(float d) { lossDenom = d; }               // 0 = batch * seq
    void beginRecording() { arena->reset(); }
    void recordStep(ComputeContext::Recorder& rec, bool backward);   // forward + head (+ backward)
    void recordForwardOnly(ComputeContext::Recorder& rec) { recordForward(rec); }
    double lossSum() const;                                           // sum of per-row losses of the last step
    uint64_t inputGradAddress() const { return dR.address; }          // dLoss/d(input rows) after a backward
    const GpuBuffer& finalHidden() const { return tf; }               // rmsnorm(x_L) [batch*seq][hidden]
    const GpuBuffer& residual(uint32_t i) const { return resid[i]; } // i = 2l + 2: after block l
    float lossScaleValue() const { return lossScale; }
    void setLossScale(float s) { lossScale = s; }

private:
    struct LayerAct {
        GpuBuffer t1, rstd1, qkv, O, L, t2, rstd2, logits, ids, w, slotPos, rowEntry, tileGroup, seg, counts, H1, Y2;
    };
    struct GemmArgs {
        uint64_t A = 0, B = 0, C = 0, bias = 0, R = 0, table = 0;
        uint32_t M = 0, N = 0, K = 0, sAm = 0, sAk = 0, sBk = 0, sBn = 0, ldC = 0;
        uint32_t flags = 0, mode = 0, strideBG = 0, strideBiasG = 0, strideCG = 0, groups = 1;
        float alpha = 1.0f;
        const char* tag = "";     // profile label
    };

    void allocate();
    void initParams();
    void uploadRopeTables();

    // Recording helpers.
    void gemm(ComputeContext::Recorder& rec, GemmArgs a);
    void linearFwd(ComputeContext::Recorder& rec, uint64_t X, uint64_t W, uint64_t b, uint64_t Y, uint32_t rows,
                   uint32_t in, uint32_t out, uint64_t resid = 0);
    void linearDX(ComputeContext::Recorder& rec, uint64_t dY, uint64_t W, uint64_t dX, uint32_t rows, uint32_t in,
                  uint32_t out, bool accum);
    void linearDW(ComputeContext::Recorder& rec, uint64_t dY, uint64_t X, uint64_t dW, uint32_t rows, uint32_t in,
                  uint32_t out);
    void colsum(ComputeContext::Recorder& rec, uint64_t in, uint64_t out, uint32_t rows, uint32_t cols, uint32_t ld,
                bool accum, uint64_t seg = 0, uint32_t groups = 1, uint32_t strideOut = 0);
    void rmsnormFwd(ComputeContext::Recorder& rec, uint64_t x, uint64_t w, uint64_t y, uint64_t rstd);
    void rmsnormBwd(ComputeContext::Recorder& rec, uint64_t x, uint64_t w, uint64_t rstd, uint64_t dy, uint64_t dX,
                    bool overwrite);
    void rope(ComputeContext::Recorder& rec, uint64_t qkv, bool inverse);
    void moeElt(ComputeContext::Recorder& rec, const char* entry, std::vector<ParamArena::TensorRef> t, uint32_t rows,
                uint32_t cols, uint32_t flags, uint64_t elements);

    void recordForward(ComputeContext::Recorder& rec);
    void recordHead(ComputeContext::Recorder& rec, bool backward);
    void recordBackward(ComputeContext::Recorder& rec);
    void recordOptimizer(ComputeContext::Recorder& rec, float lr, uint64_t step);

    // Batches.
    void loadBatch(const std::string& file, uint64_t fileTokens, uint64_t streamSeed, uint64_t index);
    void setBatchTokens(const std::vector<uint32_t>& tokens);  // batch * (seq + 1) ids
    float runMicroStep(bool backward);                         // returns mean loss of the micro-batch
    float evaluate();

    // Parameter addresses.
    uint64_t pAddr(const std::string& name) const;
    uint64_t gAddr(const std::string& name) const;
    uint64_t expertWeightAddr(const std::string& name) const;  // QAT shadow when active, else master

    // Checkpoints.
    void saveCheckpoint(uint64_t step, double bestVal);
    bool loadNewestCheckpoint(uint64_t& step, double& bestVal);
    void mirrorAsync(const std::string& localPath);
    float lrAt(uint64_t step) const;
    bool updateLossScale();  // after an optimizer step: false if it was skipped (overflow)

    ComputeContext& ctx;
    GptOssConfig cfg;
    TrainOptions opts;
    ParamLayout lay;
    uint32_t N = 0, Q = 0, HD = 0, maxRows = 0, maxTiles = 0, splitCap = 0;
    bool qatActive = false;
    bool tensorCores = false;     // fp16 cooperative-matrix GEMMs
    float lossScale = 1.0f;       // gradients in G are lossScale times the true ones
    uint64_t goodSteps = 0, skippedSteps = 0;

    GpuBuffer P, G, Mo, Ve, Pq;   // params, grads, Adam m, Adam v, QAT expert shadow
    std::vector<std::pair<uint64_t, uint64_t>> qatMap;  // (param offset, shadow offset) per expert tensor, floats
    std::vector<uint64_t> qatCounts;
    std::vector<LayerAct> acts;
    std::vector<GpuBuffer> resid;  // 2L + 1 residual-stream snapshots [N][H]
    GpuBuffer tf, rstdf;
    GpuBuffer dR, dT, dO, dqkv, Dbuf, sinkTerm, Xe, hbuf, dY2, dw, dLogits, aux, logitsChunk, splitBuf;
    GpuBuffer cosT, sinT;
    GpuBuffer tokens, targets, uniqTok, uniqOff, uniqPos, lossRows, normPartial, normSq;  // host-visible
    uint32_t uniqueCount = 0;
    uint64_t memSource = 0;
    float lossDenom = 0.0f;
    std::unique_ptr<ParamArena> arena;

    std::thread mirrorThread;
    std::string logPath;
};

int runGptOssTrainCli(int argc, char** argv);

}  // namespace lve::ml
