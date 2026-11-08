#include "gpt_oss_model.hpp"

#include "ml_json.hpp"
#include "ml_safetensors.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace lve::ml {

namespace {

// ---- kernel sources and parameter payloads (offsets after the 48-byte KernelHeader) ----
const char* kGemvSrc = "slangml/quant/gemv_quant.slang";
const char* kMoeSrc = "slangml/quant/gemv_moe.slang";
const char* kQuantSrc = "slangml/quant/quantize_f32.slang";
const char* kEmbedSrc = "slangml/quant/embedding.slang";
const char* kNormSrc = "slangml/llm/rmsnorm_fused_f32.slang";
const char* kRopeSrc = "slangml/llm/rope_qknorm_f32.slang";
const char* kAttnSrc = "slangml/llm/attention_flash_f32.slang";
const char* kRouterSrc = "slangml/llm/moe_router_f32.slang";
const char* kSampleSrc = "slangml/llm/sampling_f32.slang";
const char* kMemSrc = "slangml/memorilla/memorilla_f32.slang";  // soft-token input only

constexpr uint32_t DT_FP32 = 0, DT_FP16 = 1, DT_BF16 = 2, DT_INT8 = 3, DT_UINT8 = 4, DT_INT32 = 5;

struct GemvPayload {  // GemvParams (gemv_quant.slang)
    uint32_t N, K, M, groupShift, flags;
    float alpha;
    uint32_t ldY, ldR, actMode;
    float actLimit, actAlpha;
    uint32_t reserved;
};
static_assert(sizeof(GemvPayload) == 48, "GemvParams");

struct MoePayload {  // MoeGemvParams (gemv_moe.slang)
    uint32_t N, K, numSlots, groupShift, flags;
    float alpha;
    uint32_t ldY, actMode;
    float actLimit, actAlpha;
    uint32_t codeStride, metaStride, biasStride, slotsPerToken;
};
static_assert(sizeof(MoePayload) == 56, "MoeGemvParams");

struct NormPayload {  // RMSNormFusedParams
    uint32_t rows, cols;
    float eps;
    uint32_t flags, ldX, ldR, ldY, ldZ;
};
static_assert(sizeof(NormPayload) == 32, "RMSNormFusedParams");

struct RopePayload {  // RopeParams
    uint32_t T, numQHeads, numKVHeads, headDim, rotaryDim, flags;
    float eps, log2Theta, attnFactor;
    uint32_t qTokenStride, qHeadStride, kTokenStride, kHeadStride, vTokenStride, vHeadStride;
    uint32_t maxCtx, slotBase, posStride, section0, section1, section2, reserved;
};
static_assert(sizeof(RopePayload) == 88, "RopeParams");

struct AttnPayload {  // AttnParams
    uint32_t numQHeads, numKVHeads, T, tokensPerGroup, qTokenStride, maxCtx, kvLenBase, chunkSize, numChunks;
    float scale, softcap;
    uint32_t window, flags, gateTokenStride, gateHeadStride, outTokenStride;
};
static_assert(sizeof(AttnPayload) == 64, "AttnParams");

struct AttnCombinePayload {
    AttnPayload attn;
    uint32_t headDim;
};

struct RouterPayload {
    uint32_t T, E, topK, flags;
    float scaleOut;
    uint32_t ldLogits, reserved0, reserved1;
};

struct QuantPayload {
    uint32_t rows, cols, ldX, reserved;
};

struct EmbedPayload {
    uint32_t T, K, groupShift, ldOut;
    float scale;
    uint32_t reserved0, reserved1, reserved2;
};

struct SamplePayload {
    uint32_t R, V, ldLogits, K, numChunks;
    float temperature, topP, minP;
    uint32_t seed, step, flags, reserved;
};

// Kernel flag values (see the .slang sources).
constexpr uint32_t GEMV_BIAS = 1, GEMV_RESIDUAL = 2;
constexpr uint32_t MOE_BIAS = 1, MOE_RESIDUAL = 2, MOE_EXPERT_TABLE = 4;
constexpr uint32_t NORM_OUT_F32 = 16, NORM_OUT_Q8 = 32;
constexpr uint32_t ROPE_FREQ_TABLE = 64, ROPE_WRITE_V = 128;
constexpr uint32_t ATTN_CAUSAL = 1, ATTN_SINKS = 2;
constexpr uint32_t ROUTER_NORM_TOPK = 2;
constexpr uint32_t ACT_SWIGLU_OAI = 2;

constexpr uint32_t kSampleChunk = 4096;  // logits per argmax workgroup (sampling_f32.slang)
constexpr uint32_t kTargetChunks = 16;   // split-KV chunks per KV head at decode

uint32_t ceilDiv(uint32_t a, uint32_t b) { return (a + b - 1) / b; }
size_t roundUp(size_t v, size_t a) { return (v + a - 1) / a * a; }

std::string mib(uint64_t bytes) {
    std::ostringstream os;
    os << std::fixed << std::setprecision(1) << double(bytes) / double(1 << 20) << " MiB";
    return os.str();
}

// Checkpoint MXFP4 block (byte j = element 2j low, 2j+1 high) -> SlangML split-nibble
// order (byte j = element j low, element j + 16 high). See docs/quant.md.
void repackMxfp4Block(const uint8_t* in, uint8_t* out) {
    uint8_t codes[32];
    for (int j = 0; j < 16; ++j) {
        codes[2 * j] = in[j] & 0x0F;
        codes[2 * j + 1] = in[j] >> 4;
    }
    for (int j = 0; j < 16; ++j) out[j] = uint8_t(codes[j] | (codes[j + 16] << 4));
}

// bf16 [N][K] -> Q8 symmetric per 32-block (int8 codes, half2 (d, 0) meta); matches llm_formats.pack_q8.
void quantizeQ8(const uint16_t* w, size_t N, size_t K, std::vector<int8_t>& codes, std::vector<uint32_t>& meta) {
    const size_t nb = K / 32;
    codes.resize(N * K);
    meta.resize(N * nb);
    for (size_t n = 0; n < N; ++n) {
        for (size_t b = 0; b < nb; ++b) {
            const uint16_t* src = w + n * K + b * 32;
            float amax = 0.0f;
            float vals[32];
            for (int i = 0; i < 32; ++i) {
                vals[i] = bf16ToF32(src[i]);
                amax = std::max(amax, std::fabs(vals[i]));
            }
            uint16_t dh = f32ToF16(amax / 127.0f);
            float d = f16ToF32(dh);
            for (int i = 0; i < 32; ++i) {
                float q = d > 0.0f ? std::nearbyint(vals[i] / d) : 0.0f;
                codes[n * K + b * 32 + size_t(i)] = int8_t(std::clamp(q, -127.0f, 127.0f));
            }
            meta[n * nb + b] = uint32_t(dh);  // m = 0 in the high half
        }
    }
}

}  // namespace

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------
GptOssConfig GptOssConfig::load(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("gpt-oss: cannot open " + path);
    std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    JsonValue j = JsonValue::parse(text);
    if (!j.has("num_experts") || !j.has("experts_per_token"))
        throw std::runtime_error("gpt-oss: " + path +
                                 " is not an official gpt-oss config. Point --gpt-oss at the checkpoint's original/ folder.");
    GptOssConfig c;
    c.numLayers = uint32_t(j["num_hidden_layers"].asInt());
    c.numExperts = uint32_t(j["num_experts"].asInt());
    c.expertsPerToken = uint32_t(j["experts_per_token"].asInt());
    c.vocabSize = uint32_t(j["vocab_size"].asInt());
    c.hidden = uint32_t(j["hidden_size"].asInt());
    c.intermediate = uint32_t(j["intermediate_size"].asInt());
    c.swigluLimit = float(j.numberOr("swiglu_limit", 7.0));
    c.headDim = uint32_t(j.numberOr("head_dim", 64));
    c.numHeads = uint32_t(j["num_attention_heads"].asInt());
    c.numKVHeads = uint32_t(j["num_key_value_heads"].asInt());
    c.slidingWindow = uint32_t(j.numberOr("sliding_window", 128));
    c.initialContext = uint32_t(j.numberOr("initial_context_length", 4096));
    c.ropeTheta = j.numberOr("rope_theta", 150000.0);
    c.ropeScaling = j.numberOr("rope_scaling_factor", 32.0);
    c.ropeNtkAlpha = j.numberOr("rope_ntk_alpha", 1.0);
    c.ropeNtkBeta = j.numberOr("rope_ntk_beta", 32.0);
    return c;
}

// ---------------------------------------------------------------------------
// Construction / weight loading
// ---------------------------------------------------------------------------
GptOssModel::GptOssModel(ComputeContext& c, const std::string& modelDir, const GptOssOptions& options)
    : ctx(c), opts(options) {
    namespace fs = std::filesystem;
    cfg = GptOssConfig::load((fs::path(modelDir) / "config.json").string());

    // Kernel limits (docs/llm_attention.md, docs/quant.md).
    if (cfg.hidden % 32 || cfg.intermediate % 32) throw std::runtime_error("gpt-oss: hidden/intermediate must be multiples of 32");
    if (cfg.headDim != 64 && cfg.headDim != 128 && cfg.headDim != 256) throw std::runtime_error("gpt-oss: head_dim must be 64/128/256");
    if (cfg.numHeads % cfg.numKVHeads || cfg.numHeads / cfg.numKVHeads > 8)
        throw std::runtime_error("gpt-oss: attention kernel supports GQA groups of at most 8 query heads");
    if (cfg.hidden > 8192) throw std::runtime_error("gpt-oss: rmsnorm row kernel supports hidden <= 8192");
    if (cfg.expertsPerToken > 32 || cfg.numExperts > 512) throw std::runtime_error("gpt-oss: router supports E <= 512, k <= 32");
    if (cfg.vocabSize > 4096u * 256u * 64u) throw std::runtime_error("gpt-oss: vocabulary too large for argmax");

    arena = std::make_unique<ParamArena>(ctx, 4u << 20);
    createActivations();
    loadWeights(modelDir);
}

GptOssModel::~GptOssModel() {
    auto free = [&](GpuBuffer& b) { ctx.destroyBuffer(b); };
    auto freeDense = [&](Dense& d) {
        free(d.codes);
        free(d.meta);
    };
    for (Layer& L : layers) {
        for (GpuBuffer* b : { &L.attnNorm, &L.qkvBias, &L.sinks, &L.outBias, &L.mlpNorm, &L.gateBias, &L.gateCodes,
                              &L.gateScales, &L.upCodes, &L.upScales, &L.downCodes, &L.downScales, &L.gateBiasE,
                              &L.upBiasE, &L.downBiasE, &L.table1, &L.table2, &L.kCache, &L.vCache })
            free(*b);
        freeDense(L.qkv);
        freeDense(L.out);
        freeDense(L.gate);
    }
    for (GpuBuffer* b : { &embedding, &finalNorm, &invFreq, &x, &hNorm, &hQ, &hS, &qkv, &qRot, &attn, &attnQ, &attnS,
                          &wsO, &wsML, &routerLogits, &expertIds, &expertWeights, &moeH, &moeQ, &moeS, &logits,
                          &argmaxV, &argmaxI, &tokenBuf, &posBuf, &outToken, &softIn })
        free(*b);
    freeDense(unembed);
}

std::vector<float> GptOssModel::yarnInvFreq(float& conc) const {
    // Official RotaryEmbedding._compute_concentration_and_inv_freq (YaRN, NTK by parts).
    const uint32_t half = cfg.headDim / 2;
    const double base = cfg.ropeTheta;
    std::vector<float> inv(half);
    if (cfg.ropeScaling > 1.0) {
        conc = float(0.1 * std::log(cfg.ropeScaling) + 1.0);
        const double dHalf = double(half);
        const double twoPi = 2.0 * 3.14159265358979323846;
        const double low = dHalf * std::log(cfg.initialContext / (cfg.ropeNtkBeta * twoPi)) / std::log(base);
        const double high = dHalf * std::log(cfg.initialContext / (cfg.ropeNtkAlpha * twoPi)) / std::log(base);
        if (!(0.0 < low && low < high && high < dHalf - 1.0)) throw std::runtime_error("gpt-oss: invalid YaRN parameters");
        for (uint32_t i = 0; i < half; ++i) {
            double freq = std::pow(base, double(2 * i) / double(cfg.headDim));
            double interpolation = 1.0 / (cfg.ropeScaling * freq);
            double extrapolation = 1.0 / freq;
            double ramp = (double(i) - low) / (high - low);
            double mask = 1.0 - std::clamp(ramp, 0.0, 1.0);
            inv[i] = float(interpolation * (1.0 - mask) + extrapolation * mask);
        }
    } else {
        conc = 1.0f;
        for (uint32_t i = 0; i < half; ++i) inv[i] = float(1.0 / std::pow(base, double(2 * i) / double(cfg.headDim)));
    }
    return inv;
}

void GptOssModel::createActivations() {
    const uint32_t H = cfg.hidden, Hq = cfg.numHeads, D = cfg.headDim, E = cfg.numExperts, k = cfg.expertsPerToken;
    const uint32_t I = cfg.intermediate, V = cfg.vocabSize;
    auto dev = [&](size_t bytes) {
        GpuBuffer b = ctx.createBuffer(bytes, MemKind::Device);
        deviceBytes += b.size;
        return b;
    };
    x = dev(H * 4);
    hNorm = dev(H * 4);
    hQ = dev(H);
    hS = dev(H / 32 * 8);
    qkv = dev(cfg.qkvDim() * 4);
    qRot = dev(Hq * D * 4);
    attn = dev(Hq * D * 4);
    attnQ = dev(Hq * D);
    attnS = dev(Hq * D / 32 * 8);
    maxChunks = kTargetChunks;
    wsO = dev(size_t(maxChunks) * Hq * D * 4);
    wsML = dev(size_t(maxChunks) * Hq * 8);
    routerLogits = dev(E * 4);
    expertIds = dev(k * 4);
    expertWeights = dev(k * 4);
    moeH = dev(size_t(k) * I * 4);
    moeQ = dev(size_t(k) * I);
    moeS = dev(size_t(k) * I / 32 * 8);
    logits = dev(size_t(V) * 4);
    argmaxV = dev(ceilDiv(V, kSampleChunk) * 4);
    argmaxI = dev(ceilDiv(V, kSampleChunk) * 4);
    tokenBuf = ctx.createBuffer(16, MemKind::HostDevice);
    posBuf = ctx.createBuffer(16, MemKind::HostDevice);
    outToken = ctx.createBuffer(16, MemKind::HostDevice);

    std::vector<float> inv = yarnInvFreq(concentration);
    invFreq = dev(inv.size() * 4);
    ctx.upload(invFreq, 0, inv.data(), inv.size() * 4);
}

GpuBuffer GptOssModel::loadF32(SafeTensors& st, const std::string& name, size_t expectCount) {
    std::vector<float> v = st.readF32(name);
    if (v.size() != expectCount)
        throw std::runtime_error("gpt-oss: " + name + " has " + std::to_string(v.size()) + " elements, expected " +
                                 std::to_string(expectCount));
    GpuBuffer b = ctx.createBuffer(v.size() * 4, MemKind::Device);
    deviceBytes += b.size;
    ctx.upload(b, 0, v.data(), v.size() * 4);
    return b;
}

GptOssModel::Dense GptOssModel::loadDense(SafeTensors& st, const std::string& name, DenseFormat fmt, uint32_t N, uint32_t K) {
    const SafeTensorInfo& info = st.info(name);
    if (info.dtype != "BF16" || info.numel() != uint64_t(N) * K)
        throw std::runtime_error("gpt-oss: " + name + " must be BF16 [" + std::to_string(N) + ", " + std::to_string(K) + "]");
    Dense d;
    d.fmt = fmt;
    d.N = N;
    d.K = K;
    std::vector<uint8_t> raw = st.readBytes(name);
    if (fmt == DenseFormat::BF16) {
        d.codes = ctx.createBuffer(raw.size(), MemKind::Device);
        ctx.upload(d.codes, 0, raw.data(), raw.size());
        deviceBytes += d.codes.size;
    } else {
        std::vector<int8_t> codes;
        std::vector<uint32_t> meta;
        quantizeQ8(reinterpret_cast<const uint16_t*>(raw.data()), N, K, codes, meta);
        d.codes = ctx.createBuffer(codes.size(), MemKind::Device);
        d.meta = ctx.createBuffer(meta.size() * 4, MemKind::Device);
        ctx.upload(d.codes, 0, codes.data(), codes.size());
        ctx.upload(d.meta, 0, meta.data(), meta.size() * 4);
        deviceBytes += d.codes.size + d.meta.size;
    }
    return d;
}

void GptOssModel::loadExperts(SafeTensors& st, uint32_t l, Layer& L, bool onHost) {
    const uint32_t E = cfg.numExperts, H = cfg.hidden, I = cfg.intermediate;
    const std::string p = "block." + std::to_string(l) + ".mlp.";
    const MemKind kind = onHost ? MemKind::Host : MemKind::Device;
    L.expertsOnHost = onHost;

    // Per-expert strides, padded to 16 bytes (codes) and 16 bytes (scales) so table addresses stay aligned.
    const size_t gCodeStride = size_t(I) * H / 2, gScaleStride = roundUp(size_t(I) * H / 32, 16);
    const size_t dCodeStride = size_t(H) * I / 2, dScaleStride = roundUp(size_t(H) * I / 32, 16);

    auto alloc = [&](size_t bytes) {
        GpuBuffer b = ctx.createBuffer(bytes, kind);
        (onHost ? hostBytes : deviceBytes) += b.size;
        return b;
    };
    L.gateCodes = alloc(E * gCodeStride);
    L.upCodes = alloc(E * gCodeStride);
    L.gateScales = alloc(E * gScaleStride);
    L.upScales = alloc(E * gScaleStride);
    L.downCodes = alloc(E * dCodeStride);
    L.downScales = alloc(E * dScaleStride);

    // mlp1: blocks [E][2I][H/32][16], scales [E][2I][H/32]. Rows alternate gate (even) / up (odd):
    // official swiglu takes x[..., ::2] as the gate and x[..., 1::2] as the linear part.
    {
        const std::string bn = p + "mlp1_weight.blocks", sn = p + "mlp1_weight.scales";
        const uint32_t nbK = H / 32;
        if (st.info(bn).bytes != uint64_t(E) * 2 * I * nbK * 16 || st.info(sn).bytes != uint64_t(E) * 2 * I * nbK)
            throw std::runtime_error("gpt-oss: unexpected shape for " + bn);
        std::vector<uint8_t> blocks = st.readBytes(bn);
        std::vector<uint8_t> scales = st.readBytes(sn);
        std::vector<uint8_t> gc(E * gCodeStride), uc(E * gCodeStride), gs(E * gScaleStride, 0), us(E * gScaleStride, 0);
        for (uint32_t e = 0; e < E; ++e) {
            for (uint32_t r = 0; r < 2 * I; ++r) {
                const size_t srcRow = size_t(e) * 2 * I + r;
                const bool isGate = (r & 1u) == 0;
                const size_t dstRow = r / 2;
                uint8_t* codesOut = (isGate ? gc.data() : uc.data()) + e * gCodeStride + dstRow * nbK * 16;
                uint8_t* scalesOut = (isGate ? gs.data() : us.data()) + e * gScaleStride + dstRow * nbK;
                for (uint32_t b = 0; b < nbK; ++b)
                    repackMxfp4Block(&blocks[(srcRow * nbK + b) * 16], codesOut + b * 16);
                std::memcpy(scalesOut, &scales[srcRow * nbK], nbK);
            }
        }
        ctx.upload(L.gateCodes, 0, gc.data(), gc.size());
        ctx.upload(L.upCodes, 0, uc.data(), uc.size());
        ctx.upload(L.gateScales, 0, gs.data(), gs.size());
        ctx.upload(L.upScales, 0, us.data(), us.size());
    }
    // mlp2: blocks [E][H][I/32][16], scales [E][H][I/32].
    {
        const std::string bn = p + "mlp2_weight.blocks", sn = p + "mlp2_weight.scales";
        const uint32_t nbK = I / 32;
        if (st.info(bn).bytes != uint64_t(E) * H * nbK * 16 || st.info(sn).bytes != uint64_t(E) * H * nbK)
            throw std::runtime_error("gpt-oss: unexpected shape for " + bn);
        std::vector<uint8_t> blocks = st.readBytes(bn);
        std::vector<uint8_t> scales = st.readBytes(sn);
        std::vector<uint8_t> dc(E * dCodeStride), ds(E * dScaleStride, 0);
        for (uint32_t e = 0; e < E; ++e) {
            for (uint32_t r = 0; r < H; ++r) {
                const size_t srcRow = size_t(e) * H + r;
                for (uint32_t b = 0; b < nbK; ++b)
                    repackMxfp4Block(&blocks[(srcRow * nbK + b) * 16], dc.data() + e * dCodeStride + (size_t(r) * nbK + b) * 16);
                std::memcpy(ds.data() + e * dScaleStride + size_t(r) * nbK, &scales[srcRow * nbK], nbK);
            }
        }
        ctx.upload(L.downCodes, 0, dc.data(), dc.size());
        ctx.upload(L.downScales, 0, ds.data(), ds.size());
    }
    // Biases: mlp1_bias [E][2I] interleaved like the rows; mlp2_bias [E][H].
    {
        std::vector<float> b1 = st.readF32(p + "mlp1_bias");
        std::vector<float> b2 = st.readF32(p + "mlp2_bias");
        if (b1.size() != size_t(E) * 2 * I || b2.size() != size_t(E) * H) throw std::runtime_error("gpt-oss: bad expert bias shape");
        std::vector<float> bg(size_t(E) * I), bu(size_t(E) * I);
        for (size_t e = 0; e < E; ++e)
            for (size_t i = 0; i < I; ++i) {
                bg[e * I + i] = b1[e * 2 * I + 2 * i];
                bu[e * I + i] = b1[e * 2 * I + 2 * i + 1];
            }
        auto upF32 = [&](const std::vector<float>& v) {
            GpuBuffer b = ctx.createBuffer(v.size() * 4, MemKind::Device);
            deviceBytes += b.size;
            ctx.upload(b, 0, v.data(), v.size() * 4);
            return b;
        };
        L.gateBiasE = upF32(bg);
        L.upBiasE = upF32(bu);
        L.downBiasE = upF32(b2);
    }
    // Expert address tables: {codes, scales, codes2, scales2} per expert.
    std::vector<uint64_t> t1(size_t(E) * 4), t2(size_t(E) * 4, 0);
    for (uint32_t e = 0; e < E; ++e) {
        t1[e * 4 + 0] = L.gateCodes.address + e * gCodeStride;
        t1[e * 4 + 1] = L.gateScales.address + e * gScaleStride;
        t1[e * 4 + 2] = L.upCodes.address + e * gCodeStride;
        t1[e * 4 + 3] = L.upScales.address + e * gScaleStride;
        t2[e * 4 + 0] = L.downCodes.address + e * dCodeStride;
        t2[e * 4 + 1] = L.downScales.address + e * dScaleStride;
    }
    L.table1 = ctx.createBuffer(t1.size() * 8, MemKind::Device);
    L.table2 = ctx.createBuffer(t2.size() * 8, MemKind::Device);
    deviceBytes += L.table1.size + L.table2.size;
    ctx.upload(L.table1, 0, t1.data(), t1.size() * 8);
    ctx.upload(L.table2, 0, t2.data(), t2.size() * 8);
}

void GptOssModel::loadWeights(const std::string& modelDir) {
    using clock = std::chrono::steady_clock;
    const auto t0 = clock::now();
    SafeTensors st(modelDir);
    const uint32_t H = cfg.hidden, Hq = cfg.numHeads, Hkv = cfg.numKVHeads, D = cfg.headDim;
    const uint32_t E = cfg.numExperts, I = cfg.intermediate, V = cfg.vocabSize;

    // Embedding table stays bf16 in host memory: the GPU reads one row per token.
    {
        const SafeTensorInfo& info = st.info("embedding.weight");
        if (info.dtype != "BF16" || info.numel() != uint64_t(V) * H) throw std::runtime_error("gpt-oss: bad embedding.weight");
        embedding = ctx.createBuffer(info.bytes, MemKind::Host);
        hostBytes += embedding.size;
        st.read("embedding.weight", embedding.mapped, info.bytes);
    }
    finalNorm = loadF32(st, "norm.scale", H);
    unembed = loadDense(st, "unembedding.weight", opts.dense, V, H);

    // Decide which layers keep their experts in VRAM.
    const uint64_t expertBytesPerLayer =
        uint64_t(E) * (2 * (uint64_t(I) * H / 2 + roundUp(uint64_t(I) * H / 32, 16)) + uint64_t(H) * I / 2 + roundUp(uint64_t(H) * I / 32, 16));
    const double denseBytesPerWeight = opts.dense == DenseFormat::BF16 ? 2.0 : 1.0 + 4.0 / 32.0;
    const uint64_t perLayerOther = uint64_t(denseBytesPerWeight * double(uint64_t(cfg.qkvDim()) * H + uint64_t(H) * Hq * D)) +
                                   uint64_t(E) * H * 2 + 2ull * Hkv * opts.maxContext * D * 2 + uint64_t(E) * (2 * I + H) * 4;
    uint32_t deviceExpertLayers = cfg.numLayers;
    if (opts.experts == ExpertPlacement::Host) {
        deviceExpertLayers = 0;
    } else if (opts.experts == ExpertPlacement::Auto) {
        const uint64_t total = ctx.deviceLocalBytes();
        const uint64_t fixed = deviceBytes + uint64_t(perLayerOther) * cfg.numLayers + opts.vramReserveBytes;
        const uint64_t avail = total > fixed ? total - fixed : 0;
        deviceExpertLayers = uint32_t(std::min<uint64_t>(cfg.numLayers, avail / std::max<uint64_t>(expertBytesPerLayer, 1)));
    }
    hostExpertLayers = cfg.numLayers - deviceExpertLayers;
    if (opts.verbose) {
        std::cout << "gpt-oss: " << cfg.numLayers << " layers, " << E << " experts (top " << cfg.expertsPerToken
                  << "), hidden " << H << ", vocab " << V << "\n"
                  << "gpt-oss: experts per layer " << mib(expertBytesPerLayer) << "; " << deviceExpertLayers
                  << " layer(s) in VRAM, " << hostExpertLayers << " in host memory\n";
    }

    layers.resize(cfg.numLayers);
    for (uint32_t l = 0; l < cfg.numLayers; ++l) {
        Layer& L = layers[l];
        const std::string a = "block." + std::to_string(l) + ".attn.";
        const std::string m = "block." + std::to_string(l) + ".mlp.";
        L.attnNorm = loadF32(st, a + "norm.scale", H);
        L.qkv = loadDense(st, a + "qkv.weight", opts.dense, cfg.qkvDim(), H);
        L.qkvBias = loadF32(st, a + "qkv.bias", cfg.qkvDim());
        L.sinks = loadF32(st, a + "sinks", Hq);
        L.out = loadDense(st, a + "out.weight", opts.dense, H, Hq * D);
        L.outBias = loadF32(st, a + "out.bias", H);
        L.mlpNorm = loadF32(st, m + "norm.scale", H);
        L.gate = loadDense(st, m + "gate.weight", DenseFormat::BF16, E, H);  // tiny; keeps routing exact
        L.gateBias = loadF32(st, m + "gate.bias", E);
        loadExperts(st, l, L, l >= deviceExpertLayers);

        const size_t kvBytes = size_t(Hkv) * opts.maxContext * D * 2;
        L.kCache = ctx.createBuffer(kvBytes, MemKind::Device);
        L.vCache = ctx.createBuffer(kvBytes, MemKind::Device);
        deviceBytes += L.kCache.size + L.vCache.size;
        if (opts.verbose) {
            std::cout << "\rgpt-oss: loaded layer " << (l + 1) << "/" << cfg.numLayers << std::flush;
        }
    }
    if (opts.verbose) {
        double s = std::chrono::duration<double>(clock::now() - t0).count();
        std::cout << "\ngpt-oss: weights loaded in " << std::fixed << std::setprecision(1) << s << " s\n";
        printMemoryReport(std::cout);
    }
}

void GptOssModel::printMemoryReport(std::ostream& os) const {
    os << "gpt-oss: VRAM " << mib(deviceBytes) << " (device has " << mib(ctx.deviceLocalBytes()) << "), host-visible "
       << mib(hostBytes) << ", context " << opts.maxContext << " tokens, dense weights "
       << (opts.dense == DenseFormat::BF16 ? "bf16" : "q8") << "\n";
}

// ---------------------------------------------------------------------------
// Dispatch helpers
// ---------------------------------------------------------------------------
void GptOssModel::gemv(ComputeContext::Recorder& rec, const Dense& w, uint64_t xF32, uint64_t xQ, uint64_t xS, uint64_t y,
                       uint64_t bias, uint64_t residual, uint32_t flags) {
    GemvPayload p{};
    p.N = w.N;
    p.K = w.K;
    p.M = 1;
    p.flags = flags;
    p.alpha = 1.0f;
    p.ldY = w.N;
    p.ldR = w.N;
    const bool q8 = w.fmt == DenseFormat::Q8;
    const Pipeline& pipe = ctx.pipeline(kGemvSrc, q8 ? "sml_quant_gemv_fwd_q8_m1" : "sml_quant_gemv_fwd_bf16_m1");
    std::vector<ParamArena::TensorRef> t = {
        { w.codes.address, q8 ? DT_INT8 : DT_BF16 }, { q8 ? w.meta.address : 0, DT_UINT8 },
        { q8 ? xQ : xF32, q8 ? DT_INT8 : DT_FP32 }, { q8 ? xS : 0, DT_FP32 }, { y, DT_FP32 },
        { bias, DT_FP32 }, { residual, DT_FP32 },
    };
    const uint32_t rowsPerGroup = 2 * 128 / ctx.subgroupSize();  // _m1: 2 rows per subgroup
    rec.dispatch(pipe, arena->params(t, p), ceilDiv(w.N, rowsPerGroup));
}

void GptOssModel::rmsnorm(ComputeContext::Recorder& rec, uint64_t xa, const GpuBuffer& w, bool outF32, bool outQ8) {
    NormPayload p{};
    p.rows = 1;
    p.cols = cfg.hidden;
    p.eps = 1e-5f;  // official RMSNorm eps
    p.flags = (outF32 ? NORM_OUT_F32 : 0) | (outQ8 ? NORM_OUT_Q8 : 0);
    p.ldX = p.ldR = p.ldY = p.ldZ = cfg.hidden;
    std::vector<ParamArena::TensorRef> t = {
        { xa, DT_FP32 }, { 0, DT_FP32 }, { w.address, DT_FP32 }, { hNorm.address, DT_FP32 },
        { 0, DT_FP32 }, { 0, DT_FP32 }, { hQ.address, DT_INT8 }, { hS.address, DT_FP32 },
    };
    rec.dispatch(ctx.pipeline(kNormSrc, "sml_llm_rmsnorm_fwd_f32_row"), arena->params(t, p), 1);
}

void GptOssModel::quantizeAct(ComputeContext::Recorder& rec, uint64_t xa, uint64_t xq, uint64_t xs, uint32_t rows, uint32_t cols) {
    QuantPayload p{ rows, cols, cols, 0 };
    std::vector<ParamArena::TensorRef> t = { { xa, DT_FP32 }, { xq, DT_INT8 }, { xs, DT_FP32 } };
    rec.dispatch(ctx.pipeline(kQuantSrc, "sml_quant_quantize_q8act_fwd_f32"), arena->params(t, p),
                 ceilDiv(cols / 4, 256), rows);
}

// ---------------------------------------------------------------------------
// Forward (one token)
// ---------------------------------------------------------------------------
uint32_t GptOssModel::forward(uint32_t token, bool wantLogits, std::vector<float>* logitsOut, double* gpuMs,
                              std::vector<float>* hiddenOut) {
    if (token >= cfg.vocabSize) throw std::runtime_error("gpt-oss: token id out of range");
    return step(token, nullptr, wantLogits, logitsOut, gpuMs, hiddenOut);
}

uint32_t GptOssModel::forwardEmbedding(const float* vec, bool wantLogits, std::vector<float>* logitsOut,
                                       double* gpuMs, std::vector<float>* hiddenOut) {
    if (!vec) throw std::runtime_error("gpt-oss: null soft-token input");
    if (!softIn) softIn = ctx.createBuffer(size_t(cfg.hidden) * 4, MemKind::HostDevice);
    std::memcpy(softIn.mapped, vec, size_t(cfg.hidden) * 4);
    return step(0, vec, wantLogits, logitsOut, gpuMs, hiddenOut);
}

uint32_t GptOssModel::step(uint32_t token, const float* softVec, bool wantLogits, std::vector<float>* logitsOut,
                           double* gpuMs, std::vector<float>* hiddenOut) {
    if (pos >= opts.maxContext) throw std::runtime_error("gpt-oss: context full (" + std::to_string(opts.maxContext) + " tokens)");
    const uint32_t H = cfg.hidden, Hq = cfg.numHeads, Hkv = cfg.numKVHeads, D = cfg.headDim;
    const uint32_t E = cfg.numExperts, k = cfg.expertsPerToken, I = cfg.intermediate, V = cfg.vocabSize;
    const uint32_t sg = ctx.subgroupSize();
    const bool denseQ8 = opts.dense == DenseFormat::Q8;

    static_cast<int32_t*>(tokenBuf.mapped)[0] = int32_t(token);
    static_cast<int32_t*>(posBuf.mapped)[0] = int32_t(pos);
    arena->reset();
    ComputeContext::Recorder rec(ctx);
    rec.timestamp(0);

    // x = embedding[token], or the soft-token vector (already copied to softIn)
    if (softVec) {
        struct { uint32_t B, K, H, inT, inOff, outT, outOff, flags, stride, r0, r1, r2; } p{
            1, 1, H, 1, 0, 1, 0, 0, ceilDiv(H, 256) * 256u, 0, 0, 0 };
        std::vector<ParamArena::TensorRef> t = { { softIn.address, DT_FP32 }, { x.address, DT_FP32 } };
        rec.dispatch(ctx.pipeline(kMemSrc, "sml_mem_rows_copy_f32"), arena->params(t, p), ceilDiv(H, 256));
    } else {
        EmbedPayload p{ 1, H, 0, H, 1.0f, 0, 0, 0 };
        std::vector<ParamArena::TensorRef> t = { { embedding.address, DT_BF16 }, { 0, DT_UINT8 },
                                                 { tokenBuf.address, DT_INT32 }, { x.address, DT_FP32 } };
        rec.dispatch(ctx.pipeline(kEmbedSrc, "sml_quant_embedding_fwd_bf16"), arena->params(t, p), ceilDiv(H / 4, 256));
    }

    const Pipeline& ropePipe = ctx.pipeline(kRopeSrc, "sml_llm_rope_qknorm_fwd_f32");
    const std::string attnEntry = "sml_llm_attention_fwd_f16kv_d" + std::to_string(D);
    const Pipeline& attnPipe = ctx.pipeline(kAttnSrc, attnEntry);
    const Pipeline& combinePipe = ctx.pipeline(kAttnSrc, "sml_llm_attention_combine_fwd_f32");
    const Pipeline& routerPipe = ctx.pipeline(kRouterSrc, "sml_llm_moe_router_fwd_f32");
    const Pipeline& moeUpPipe = ctx.pipeline(kMoeSrc, "sml_quant_gemv_moe_swiglu_fwd_mxfp4");
    const Pipeline& moeDownPipe = ctx.pipeline(kMoeSrc, "sml_quant_gemv_moe_down_fwd_mxfp4");

    for (uint32_t l = 0; l < cfg.numLayers; ++l) {
        const Layer& L = layers[l];
        // ---- attention block
        rmsnorm(rec, x.address, L.attnNorm, true, denseQ8);
        gemv(rec, L.qkv, hNorm.address, hQ.address, hS.address, qkv.address, L.qkvBias.address, 0, GEMV_BIAS);
        {
            RopePayload p{};
            p.T = 1;
            p.numQHeads = Hq;
            p.numKVHeads = Hkv;
            p.headDim = D;
            p.rotaryDim = D;
            p.flags = ROPE_FREQ_TABLE | ROPE_WRITE_V;
            p.eps = 1e-6f;
            p.log2Theta = float(std::log2(cfg.ropeTheta));
            p.attnFactor = concentration;  // official: cos/sin scaled by the YaRN concentration for q and k
            p.qTokenStride = p.kTokenStride = p.vTokenStride = cfg.qkvDim();
            p.qHeadStride = p.kHeadStride = p.vHeadStride = D;
            p.maxCtx = opts.maxContext;
            p.slotBase = pos;
            std::vector<ParamArena::TensorRef> t = {
                { qkv.address, DT_FP32 }, { qkv.address + uint64_t(Hq) * D * 4, DT_FP32 },
                { qkv.address + uint64_t(Hq + Hkv) * D * 4, DT_FP32 }, { 0, DT_FP32 }, { 0, DT_FP32 },
                { posBuf.address, DT_INT32 }, { qRot.address, DT_FP32 }, { L.kCache.address, DT_FP16 },
                { L.vCache.address, DT_FP16 }, { invFreq.address, DT_FP32 }, { 0, DT_INT32 },
            };
            rec.dispatch(ropePipe, arena->params(t, p), Hq + 2 * Hkv, 1);
        }
        {
            // Sliding-window layers (even index, as in the official AttentionBlock) attend to the
            // last `slidingWindow` positions: shift the cache view so the kernel only walks those.
            const bool sliding = (l % 2) == 0 && cfg.slidingWindow > 0;
            const uint32_t start = (sliding && pos + 1 > cfg.slidingWindow) ? pos + 1 - cfg.slidingWindow : 0;
            const uint32_t len = pos + 1 - start;
            const uint32_t chunk = std::max<uint32_t>(32, uint32_t(roundUp(ceilDiv(len, kTargetChunks), 32)));
            const uint32_t chunks = ceilDiv(len, chunk);
            AttnPayload p{};
            p.numQHeads = Hq;
            p.numKVHeads = Hkv;
            p.T = 1;
            p.tokensPerGroup = 1;
            p.qTokenStride = Hq * D;
            p.maxCtx = opts.maxContext;
            p.kvLenBase = pos - start;
            p.chunkSize = chunk;
            p.numChunks = chunks;
            p.scale = 1.0f / std::sqrt(float(D));
            p.softcap = 0.0f;
            p.window = sliding ? cfg.slidingWindow : 0;
            p.flags = ATTN_CAUSAL | ATTN_SINKS;
            p.outTokenStride = Hq * D;
            const uint64_t viewOffset = uint64_t(start) * D * 2;
            std::vector<ParamArena::TensorRef> t = {
                { qRot.address, DT_FP32 }, { L.kCache.address + viewOffset, DT_FP16 },
                { L.vCache.address + viewOffset, DT_FP16 }, { attn.address, DT_FP32 }, { L.sinks.address, DT_FP32 },
                { 0, DT_FP32 }, { wsO.address, DT_FP32 }, { wsML.address, DT_FP32 },
            };
            rec.dispatch(attnPipe, arena->params(t, p), chunks, 1, Hkv);
            if (chunks > 1) {
                AttnCombinePayload cp{ p, D };
                rec.dispatch(combinePipe, arena->params(t, cp), Hq);
            }
        }
        if (denseQ8) quantizeAct(rec, attn.address, attnQ.address, attnS.address, 1, Hq * D);
        gemv(rec, L.out, attn.address, attnQ.address, attnS.address, x.address, L.outBias.address, x.address,
             GEMV_BIAS | GEMV_RESIDUAL);

        // ---- MoE block
        rmsnorm(rec, x.address, L.mlpNorm, true, true);
        gemv(rec, L.gate, hNorm.address, 0, 0, routerLogits.address, L.gateBias.address, 0, GEMV_BIAS);
        {
            RouterPayload p{ 1, E, k, ROUTER_NORM_TOPK, 1.0f, E, 0, 0 };
            std::vector<ParamArena::TensorRef> t = { { routerLogits.address, DT_FP32 }, { 0, DT_FP32 },
                                                     { expertIds.address, DT_INT32 }, { expertWeights.address, DT_FP32 } };
            rec.dispatch(routerPipe, arena->params(t, p), 1);
        }
        const uint32_t rowsPerGroup = 2 * 128 / sg;
        {
            MoePayload p{};
            p.N = I;
            p.K = H;
            p.numSlots = k;
            p.flags = MOE_BIAS | MOE_EXPERT_TABLE;
            p.alpha = 1.0f;
            p.ldY = I;
            p.actMode = ACT_SWIGLU_OAI;
            p.actLimit = cfg.swigluLimit;
            p.actAlpha = 1.702f;
            p.biasStride = I;
            p.slotsPerToken = k;
            std::vector<ParamArena::TensorRef> t = {
                { 0, DT_UINT8 }, { 0, DT_UINT8 }, { hQ.address, DT_INT8 }, { hS.address, DT_FP32 },
                { moeH.address, DT_FP32 }, { L.gateBiasE.address, DT_FP32 }, { L.upBiasE.address, DT_FP32 },
                { 0, DT_UINT8 }, { 0, DT_UINT8 }, { expertIds.address, DT_INT32 }, { expertWeights.address, DT_FP32 },
                { L.table1.address, DT_UINT8 },
            };
            rec.dispatch(moeUpPipe, arena->params(t, p), ceilDiv(I, rowsPerGroup), k);
        }
        quantizeAct(rec, moeH.address, moeQ.address, moeS.address, k, I);
        {
            MoePayload p{};
            p.N = H;
            p.K = I;
            p.numSlots = k;
            p.flags = MOE_BIAS | MOE_RESIDUAL | MOE_EXPERT_TABLE;
            p.alpha = 1.0f;
            p.ldY = H;
            p.actLimit = cfg.swigluLimit;
            p.actAlpha = 1.702f;
            p.biasStride = H;
            p.slotsPerToken = k;
            std::vector<ParamArena::TensorRef> t = {
                { 0, DT_UINT8 }, { 0, DT_UINT8 }, { moeQ.address, DT_INT8 }, { moeS.address, DT_FP32 },
                { x.address, DT_FP32 }, { L.downBiasE.address, DT_FP32 }, { x.address, DT_FP32 },
                { 0, DT_UINT8 }, { 0, DT_UINT8 }, { expertIds.address, DT_INT32 }, { expertWeights.address, DT_FP32 },
                { L.table2.address, DT_UINT8 },
            };
            rec.dispatch(moeDownPipe, arena->params(t, p), ceilDiv(H, rowsPerGroup), 1);
        }
    }

    uint32_t result = UINT32_MAX;
    if (hiddenOut && !wantLogits) rmsnorm(rec, x.address, finalNorm, true, denseQ8);
    if (wantLogits) {
        rmsnorm(rec, x.address, finalNorm, true, denseQ8);
        gemv(rec, unembed, hNorm.address, hQ.address, hS.address, logits.address, 0, 0, 0);
        const uint32_t chunks = ceilDiv(V, kSampleChunk);
        SamplePayload p{};
        p.R = 1;
        p.V = V;
        p.ldLogits = V;
        p.numChunks = chunks;
        std::vector<ParamArena::TensorRef> t = { { logits.address, DT_FP32 }, { argmaxV.address, DT_FP32 },
                                                 { argmaxI.address, DT_INT32 }, { outToken.address, DT_INT32 } };
        uint64_t pa = arena->params(t, p);
        rec.dispatch(ctx.pipeline(kSampleSrc, "sml_llm_argmax_partial_fwd_f32"), pa, chunks, 1);
        rec.dispatch(ctx.pipeline(kSampleSrc, "sml_llm_argmax_final_fwd_f32"), pa, 1);
    }
    rec.timestamp(1);
    double ms = rec.submitAndWait();
    if (gpuMs) *gpuMs = ms;
    ++pos;

    if (hiddenOut) {
        hiddenOut->resize(H);
        ctx.download(hNorm, 0, hiddenOut->data(), size_t(H) * 4);
    }
    if (wantLogits) {
        result = static_cast<const uint32_t*>(outToken.mapped)[0];
        if (logitsOut) {
            logitsOut->resize(V);
            ctx.download(logits, 0, logitsOut->data(), size_t(V) * 4);
        }
    }
    return result;
}

}  // namespace lve::ml
