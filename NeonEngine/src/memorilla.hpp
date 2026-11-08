#pragma once

// Memorilla: latent semantic memory for a frozen LLM (Fallahpour et al., Stanford,
// arXiv 2610.21375). Plan and design notes: shaders/ML/docs/memorilla_plan.md.
//
// A frozen encoder turns each document into one vector (the store Z_D, N x h_e).
// For a query embedding z_q, the memory module writes K soft tokens
// M (K x h_L) that go in front of the prompt of a frozen gpt-oss decoder:
//
//   u  = W_o (W_v z_q + b_v) + b_o           (Stage 2: one key, so attention weight 1)
//   R  = LN_c(Q0 + u)                        (Q0: K learned slots)
//   Z^ = LN_d(Z_D)
//   per block l < L_ref:
//     R = R + W_o( MHA(Q = W_q LN_1(R), K = W_k Z^, V = W_v Z^) )   (8 heads, padded docs masked)
//     R = R + W_down GELU(W_up LN_2(R))
//   M  = LN_o(R) W_proj + b_proj
//
// Memorilla is OFF unless asked for: nothing here is constructed, compiled or
// allocated by the gpt-oss host or trainer unless a Memorilla option is given.

#include "gpt_oss_train.hpp"
#include "ml_compute.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace lve::ml {

// Inference setting (gpt_oss_app.cpp). Disabled by default.
struct MemorillaOptions {
    bool enabled = false;          // set only by --memorilla
    std::string moduleDir;         // memorilla.json + memorilla.safetensors
    std::string storePath;         // .nmem document store
    uint32_t slotsToUse = 0;       // --mem-k (0 = all trained slots)
    std::vector<uint32_t> queryTokens;  // --mem-query(-file); empty = the prompt
};

struct MemorillaConfig {
    uint32_t slots = 16;           // K
    uint32_t encoderDim = 0;       // h_e (must equal the store's width)
    uint32_t llmDim = 0;           // h_L (the decoder's hidden size)
    uint32_t heads = 8;
    uint32_t refineLayers = 2;     // L_ref
    uint32_t ffnMult = 4;
    float eps = 1.0e-5f;
    std::string encoder = "self";  // "self" = the frozen decoder, mean-pooled; else an external model name
    int32_t encoderLayer = -1;     // self encoder: -1 = after the final RMSNorm, l = residual after block l
    std::string decoder;           // fingerprint of the decoder the module was trained for

    uint32_t headDim() const { return encoderDim / heads; }
    uint32_t ffnDim() const { return encoderDim * ffnMult; }
    static MemorillaConfig fromJson(const std::string& text);
    std::string toJson() const;
};

// Parameter tensors in the module's flat fp32 buffer (decayed tensors first).
ParamLayout buildMemorillaLayout(const MemorillaConfig& c);
// Parameter count of the paper's architecture, which also has W_q / W_k in the
// query-conditioning attention (unused: one key). Paper config -> 194,123,776.
uint64_t memorillaPaperParamCount(const MemorillaConfig& c);

// Fingerprint of a decoder config, written into stores and modules.
std::string decoderFingerprint(const GptOssConfig& c);

// ---------------------------------------------------------------------------
// Document store (.nmem): header JSON, count x dim fp32 vectors, per-document
// table (id, token count, flags; flag 1 = deleted). Small, so it is read and
// rewritten whole (temp file + rename).
// ---------------------------------------------------------------------------
struct MemoryStore {
    std::string encoder = "self";
    int32_t encoderLayer = -1;
    std::string pooling = "mean";
    std::string decoder;           // decoder fingerprint for self-encoded stores
    uint32_t dim = 0;
    std::vector<float> vectors;    // count x dim
    struct Doc {
        uint64_t id = 0;
        uint32_t tokens = 0;
        uint32_t flags = 0;        // 1 = deleted (tombstone)
    };
    std::vector<Doc> docs;

    size_t count() const { return docs.size(); }
    size_t liveCount() const;
    const float* vector(size_t i) const { return vectors.data() + i * dim; }
    void add(const float* v, uint64_t id, uint32_t tokens);
    bool remove(uint64_t id);      // tombstone; returns false if not found
    void compact();                // drop tombstoned documents
    std::vector<float> liveVectors() const;

    static MemoryStore load(const std::string& path);
    void save(const std::string& path) const;
};

// Token sequences file (input to the encoder): magic, count, then per sequence
// a uint32 length and that many uint32 token ids.
std::vector<std::vector<uint32_t>> readSequencesFile(const std::string& path);

// ---------------------------------------------------------------------------
// GPU memory module. Forward for inference; forward + backward + AdamW for training.
// ---------------------------------------------------------------------------
class MemorillaModule {
public:
    MemorillaModule(ComputeContext& ctx, const MemorillaConfig& cfg, bool tensorCores, bool trainable);
    ~MemorillaModule();
    MemorillaModule(const MemorillaModule&) = delete;
    MemorillaModule& operator=(const MemorillaModule&) = delete;

    const MemorillaConfig& config() const { return cfg; }
    const ParamLayout& layout() const { return lay; }
    bool usesTensorCores() const { return tensorCores; }

    // Activation buffers for up to maxBatch queries of up to maxDocs documents each.
    void allocate(uint32_t maxBatch, uint32_t maxDocs);
    uint32_t maxBatch() const { return capB; }
    uint32_t maxDocs() const { return capN; }

    // Init: slots Xavier-uniform, linears U(+-1/sqrt(in)), norms 1 / 0, projection N(0, projStd).
    void initParams(uint64_t seed, float projStd);
    void setParams(const std::vector<float>& host);
    std::vector<float> params() const;
    std::vector<float> grads() const;  // raw (still carry the loss scale)

    // Host-visible inputs: Z_D [maxBatch][maxDocs][h_e] (padded), counts [maxBatch], z_q [maxBatch][h_e].
    float* docsHost() { return static_cast<float*>(docsIn.mapped); }
    uint32_t* countsHost() { return static_cast<uint32_t*>(countsIn.mapped); }
    float* queryHost() { return static_cast<float*>(queryIn.mapped); }

    // Must be called before recording the module into a new command buffer.
    void beginRecording() { arena->reset(); }
    // M [B][K][h_L] from the host inputs, for B queries padded to Nd documents.
    void recordForward(ComputeContext::Recorder& rec, uint32_t B, uint32_t Nd);
    uint64_t outputAddress() const { return M.address; }
    const GpuBuffer& outputBuffer() const { return M; }
    // dM [B][K][h_L] (filled by the caller) -> parameter gradients (accumulated).
    uint64_t outputGradAddress() const { return dM.address; }
    void recordBackward(ComputeContext::Recorder& rec);
    // out[(b*outT + outOff + k)*H + c] (+)= in[(b*inT + inOff + k)*H + c] for b < B, k < K.
    void recordRowsCopy(ComputeContext::Recorder& rec, uint64_t in, uint64_t out, uint32_t B, uint32_t K, uint32_t H,
                        uint32_t inT, uint32_t inOff, uint32_t outT, uint32_t outOff, bool accum);
    // Gradient-norm clipping + AdamW on the module parameters; zeroes the gradients.
    // gradScale multiplies the raw gradients (1 / (accum * lossScale)).
    void recordOptimizer(ComputeContext::Recorder& rec, float lr, uint64_t step, float gradScale, float beta1,
                         float beta2, float eps, float weightDecay, float clip);
    float lastNormSq() const { return static_cast<const float*>(normSq.mapped)[0]; }
    void zeroGrads();

    // Module directory: memorilla.json + memorilla.safetensors (fp32).
    void save(const std::string& dir) const;
    static MemorillaConfig loadConfig(const std::string& dir);
    void load(const std::string& dir);
    // Training state (params + Adam moments).
    void saveState(std::FILE* f, const std::string& path) const;
    void loadState(std::FILE* f, const std::string& path);

private:
    struct Block {
        GpuBuffer n1, m1, r1, q, k, v, P, O, n2, m2, r2, hu, ha;
    };
    struct GemmArgs {
        uint64_t A = 0, B = 0, C = 0, bias = 0, R = 0;
        uint32_t M = 0, N = 0, K = 0, sAm = 0, sAk = 0, sBk = 0, sBn = 0, ldC = 0, flags = 0;
    };
    void gemm(ComputeContext::Recorder& rec, const GemmArgs& a);
    void linear(ComputeContext::Recorder& rec, uint64_t X, const std::string& w, uint64_t Y, uint32_t rows,
                uint32_t in, uint32_t out, uint64_t resid = 0);
    void linearDX(ComputeContext::Recorder& rec, uint64_t dY, const std::string& w, uint64_t dX, uint32_t rows,
                  uint32_t in, uint32_t out, bool accum);
    void linearDW(ComputeContext::Recorder& rec, uint64_t dY, uint64_t X, const std::string& w, uint32_t rows,
                  uint32_t in, uint32_t out);  // dW += dY^T X and db += colsum dY
    void colsum(ComputeContext::Recorder& rec, uint64_t in, uint64_t out, uint32_t rows, uint32_t cols);
    void layerNorm(ComputeContext::Recorder& rec, uint64_t x, const std::string& name, uint64_t y, uint64_t mean,
                   uint64_t rstd, uint32_t rows);
    void layerNormBwd(ComputeContext::Recorder& rec, uint64_t x, const std::string& name, uint64_t mean, uint64_t rstd,
                      uint64_t dy, uint64_t dx, uint32_t rows, bool accum, bool noDx);
    void attnDispatch(ComputeContext::Recorder& rec, const char* entry, uint64_t t0, uint64_t t1, uint64_t t2,
                      uint64_t t3, uint32_t I, uint32_t J, uint32_t ldA, uint32_t ldB, float scale, uint32_t flags,
                      uint64_t work);
    uint64_t pAddr(const std::string& name) const;
    uint64_t gAddr(const std::string& name) const;

    ComputeContext& ctx;
    MemorillaConfig cfg;
    ParamLayout lay;
    bool tensorCores = false;
    bool trainable = false;
    uint32_t capB = 0, capN = 0;
    uint32_t curB = 0, curN = 0;

    GpuBuffer P, G, Mo, Ve;
    GpuBuffer docsIn, countsIn, queryIn;                    // host-visible inputs
    GpuBuffer u1, u, pre, cm, cr, zn, dmn, drs;             // conditioning, document norm
    std::vector<GpuBuffer> res;                             // residual stream: R0, Rmid_0, Rout_0, ...
    std::vector<Block> blocks;
    GpuBuffer rf, fm, fr, M;                                // final norm, output
    GpuBuffer dM, dR, dN, dQ, dK, dV, dZn, dP, dO, dH, gx, dPre, du, du1;
    GpuBuffer normPartial, normSq, splitBuf;
    uint32_t splitCap = 0;
    std::unique_ptr<ParamArena> arena;
};

// NeonML / LveEngine --memorilla encode|train|eval|gradcheck|params (memorilla_train.cpp).
int runMemorillaCli(int argc, char** argv);

}  // namespace lve::ml
