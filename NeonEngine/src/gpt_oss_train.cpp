#include "gpt_oss_train.hpp"

#include "ml_json.hpp"
#include "ml_safetensors.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>

namespace lve::ml {

namespace fs = std::filesystem;

namespace {

const char* kGemmSrc = "slangml/train/gemm_train_f32.slang";
const char* kGemmTcSrc = "slangml/train/gemm_train_coopmat.slang";
const char* kOpsSrc = "slangml/train/train_ops_f32.slang";
const char* kAttnSrc = "slangml/train/attention_train_f32.slang";
const char* kAttnTcSrc = "slangml/train/attention_train_coopmat.slang";
const char* kMoeSrc = "slangml/train/moe_train_f32.slang";
const char* kRouterSrc = "slangml/llm/moe_router_f32.slang";
const char* kMemSrc = "slangml/memorilla/memorilla_f32.slang";

constexpr uint32_t DT_FP32 = 0, DT_I32 = 5;
constexpr uint32_t GEMM_BIAS = 1, GEMM_RESID = 2, GEMM_ACCUM = 4;
constexpr uint32_t TILE = 128;  // GEMM tile height = MoE segment padding
constexpr uint32_t kMaxGroups = 65535;
constexpr double kPi = 3.14159265358979323846;
constexpr char kMagic[8] = { 'N', 'E', 'O', 'N', 'G', 'P', 'T', '1' };
constexpr char kEndMagic[8] = { 'N', 'E', 'O', 'N', 'E', 'N', 'D', '1' };

uint32_t ceilDiv(uint64_t a, uint64_t b) { return uint32_t((a + b - 1) / b); }
uint64_t roundUp(uint64_t a, uint64_t b) { return (a + b - 1) / b * b; }

// Attention kernels: tensor cores (fp16 inputs, fp32 accumulate) when the GEMMs use them, else fp32 SIMT.
// Both variants take the same parameters, tensor tables and grids.
const Pipeline& attnPipeline(ComputeContext& ctx, bool tensorCores, const char* kind) {
    return tensorCores ? ctx.pipeline(kAttnTcSrc, std::string("sml_train_attn_") + kind + "_tc")
                       : ctx.pipeline(kAttnSrc, std::string("sml_train_attn_") + kind + "_f32");
}

// ---- deterministic RNG (splitmix64) ----
uint64_t splitmix(uint64_t& s) {
    uint64_t z = (s += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}
uint64_t hashMix(uint64_t a, uint64_t b) {
    uint64_t s = a * 0x9E3779B97F4A7C15ull ^ (b + 0x632BE59BD9B4E019ull);
    return splitmix(s);
}
uint64_t hashName(const std::string& s) {
    uint64_t h = 1469598103934665603ull;
    for (char c : s) h = (h ^ uint8_t(c)) * 1099511628211ull;
    return h;
}

uint16_t f32ToBf16(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    if ((u & 0x7F800000u) == 0x7F800000u && (u & 0x7FFFFFu)) return uint16_t((u >> 16) | 0x40);  // NaN
    u += 0x7FFFu + ((u >> 16) & 1u);
    return uint16_t(u >> 16);
}

// MXFP4 rule shared with slangml-train.slang and tests/llm_formats.py.
int mxfp4Exponent(float amax) {
    if (!(amax > 0.0f)) return -127;
    int e = std::ilogb(double(amax)) - 2;
    while (double(amax) > 6.0 * std::ldexp(1.0, e)) ++e;
    return std::clamp(e, -127, 127);
}
uint8_t e2m1Code(float scaled) {
    const float a = std::fabs(scaled);
    uint8_t m;
    if (a <= 0.25f) m = 0;
    else if (a <= 0.75f) m = 1;
    else if (a <= 1.25f) m = 2;
    else if (a <= 1.75f) m = 3;
    else if (a <= 2.5f) m = 4;
    else if (a <= 3.5f) m = 5;
    else if (a <= 5.0f) m = 6;
    else m = 7;
    return uint8_t(m | (scaled < 0.0f ? 8 : 0));
}

std::string jsonEscape(const std::string& s) {
    std::string o;
    for (char c : s) {
        if (c == '"' || c == '\\') o += '\\';
        o += c;
    }
    return o;
}

std::string configJson(const GptOssConfig& c, const std::string& tokenizer, const std::string& extra = "") {
    std::ostringstream o;
    o << std::setprecision(17);
    o << "{\n  \"num_hidden_layers\": " << c.numLayers << ",\n  \"num_experts\": " << c.numExperts
      << ",\n  \"experts_per_token\": " << c.expertsPerToken << ",\n  \"vocab_size\": " << c.vocabSize
      << ",\n  \"hidden_size\": " << c.hidden << ",\n  \"intermediate_size\": " << c.intermediate
      << ",\n  \"swiglu_limit\": " << c.swigluLimit << ",\n  \"head_dim\": " << c.headDim
      << ",\n  \"num_attention_heads\": " << c.numHeads << ",\n  \"num_key_value_heads\": " << c.numKVHeads
      << ",\n  \"sliding_window\": " << c.slidingWindow << ",\n  \"initial_context_length\": " << c.initialContext
      << ",\n  \"rope_theta\": " << c.ropeTheta << ",\n  \"rope_scaling_factor\": " << c.ropeScaling
      << ",\n  \"rope_ntk_alpha\": " << c.ropeNtkAlpha << ",\n  \"rope_ntk_beta\": " << c.ropeNtkBeta
      << ",\n  \"neon_tokenizer\": \"" << jsonEscape(tokenizer) << "\"" << extra << "\n}\n";
    return o.str();
}

GptOssConfig configFromJson(const JsonValue& j) {
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

// Official YaRN (gpt_oss/torch/model.py RotaryEmbedding), in double.
void yarnTables(const GptOssConfig& c, uint32_t T, std::vector<float>& cosT, std::vector<float>& sinT) {
    const uint32_t half = c.headDim / 2;
    std::vector<double> invFreq(half);
    double concentration = 1.0;
    for (uint32_t i = 0; i < half; ++i) {
        const double freq = std::pow(c.ropeTheta, double(2 * i) / double(c.headDim));
        invFreq[i] = 1.0 / freq;
    }
    if (c.ropeScaling > 1.0) {
        concentration = 0.1 * std::log(c.ropeScaling) + 1.0;
        const double dHalf = double(half);
        const double low = dHalf * std::log(c.initialContext / (c.ropeNtkBeta * 2.0 * kPi)) / std::log(c.ropeTheta);
        const double high = dHalf * std::log(c.initialContext / (c.ropeNtkAlpha * 2.0 * kPi)) / std::log(c.ropeTheta);
        for (uint32_t i = 0; i < half; ++i) {
            const double freq = std::pow(c.ropeTheta, double(2 * i) / double(c.headDim));
            const double interp = 1.0 / (c.ropeScaling * freq), extrap = 1.0 / freq;
            const double ramp = (double(i) - low) / (high - low);
            const double mask = 1.0 - std::clamp(ramp, 0.0, 1.0);
            invFreq[i] = interp * (1.0 - mask) + extrap * mask;
        }
    }
    cosT.resize(size_t(T) * half);
    sinT.resize(size_t(T) * half);
    for (uint32_t t = 0; t < T; ++t)
        for (uint32_t i = 0; i < half; ++i) {
            // The reference computes the angle in fp32; match it so long positions agree.
            const float ang = float(t) * float(invFreq[i]);
            cosT[size_t(t) * half + i] = float(std::cos(double(ang)) * concentration);
            sinT[size_t(t) * half + i] = float(std::sin(double(ang)) * concentration);
        }
}

void writeAll(std::FILE* f, const void* p, size_t n, const std::string& path) {
    if (n && std::fwrite(p, 1, n, f) != n) throw std::runtime_error("write failed: " + path);
}
void readAll(std::FILE* f, void* p, size_t n, const std::string& path) {
    if (n && std::fread(p, 1, n, f) != n) throw std::runtime_error("read failed (truncated?): " + path);
}

struct CkptHeader {
    std::string json;
    uint64_t count = 0;
    uint64_t dataOffset = 0;
};

bool readHeader(std::FILE* f, const std::string& path, CkptHeader& h) {
    char magic[8];
    if (std::fread(magic, 1, 8, f) != 8 || std::memcmp(magic, kMagic, 8) != 0) return false;
    uint32_t version = 0, len = 0;
    if (std::fread(&version, 4, 1, f) != 1 || version != 1) return false;
    if (std::fread(&len, 4, 1, f) != 1 || len > (64u << 20)) return false;
    h.json.resize(len);
    if (len && std::fread(h.json.data(), 1, len, f) != len) return false;
    if (std::fread(&h.count, 8, 1, f) != 1) return false;
    h.dataOffset = 8 + 4 + 4 + uint64_t(len) + 8;
    (void)path;
    return true;
}

std::atomic<bool> gStop{ false };
void onSignal(int) { gStop = true; }

}  // namespace

std::atomic<bool>& GptOssTrainer::stopRequested() { return gStop; }

// ---------------------------------------------------------------------------
// Parameter layout
// ---------------------------------------------------------------------------
ParamLayout ParamLayout::build(const GptOssConfig& c) {
    ParamLayout L;
    const uint64_t H = c.hidden, I = c.intermediate, E = c.numExperts, V = c.vocabSize;
    const uint64_t Q = uint64_t(c.qkvDim()), HQD = uint64_t(c.numHeads) * c.headDim;
    auto add = [&](const std::string& name, std::vector<uint64_t> shape, bool decay, ParamTensor::Export ex) {
        ParamTensor t;
        t.name = name;
        t.shape = std::move(shape);
        t.count = 1;
        for (uint64_t d : t.shape) t.count *= d;
        t.offset = L.total;
        t.decay = decay;
        t.exportAs = ex;
        L.total = roundUp(L.total + t.count, 4);
        L.tensors.push_back(std::move(t));
    };
    using X = ParamTensor::Export;
    add("embedding.weight", { V, H }, true, X::BF16);
    for (uint32_t l = 0; l < c.numLayers; ++l) {
        const std::string a = "block." + std::to_string(l) + ".attn.", m = "block." + std::to_string(l) + ".mlp.";
        add(a + "qkv.weight", { Q, H }, true, X::BF16);
        add(a + "out.weight", { H, HQD }, true, X::BF16);
        add(m + "gate.weight", { E, H }, true, X::BF16);
        add(m + "mlp1_weight", { E, 2 * I, H }, true, X::MXFP4);
        add(m + "mlp2_weight", { E, H, I }, true, X::MXFP4);
    }
    add("unembedding.weight", { V, H }, true, X::BF16);
    L.decayEnd = L.total;
    for (uint32_t l = 0; l < c.numLayers; ++l) {
        const std::string a = "block." + std::to_string(l) + ".attn.", m = "block." + std::to_string(l) + ".mlp.";
        add(a + "norm.scale", { H }, false, X::F32);
        add(a + "qkv.bias", { Q }, false, X::BF16);
        add(a + "sinks", { uint64_t(c.numHeads) }, false, X::BF16);
        add(a + "out.bias", { H }, false, X::BF16);
        add(m + "norm.scale", { H }, false, X::F32);
        add(m + "gate.bias", { E }, false, X::BF16);
        add(m + "mlp1_bias", { E, 2 * I }, false, X::BF16);
        add(m + "mlp2_bias", { E, H }, false, X::BF16);
    }
    add("norm.scale", { H }, false, X::F32);
    return L;
}

const ParamTensor& ParamLayout::at(const std::string& name) const {
    for (const auto& t : tensors)
        if (t.name == name) return t;
    throw std::runtime_error("gpt-oss train: no parameter " + name);
}

// ---------------------------------------------------------------------------
// Export to the official checkpoint layout
// ---------------------------------------------------------------------------
void exportGptOssCheckpoint(const GptOssConfig& cfg, const ParamLayout& layout, const std::vector<float>& params,
                            const std::string& outDir, const std::string& tokenizer, const std::string& tokenizerFile) {
    fs::create_directories(outDir);
    std::string tokName = tokenizer;
    std::error_code tec;
    if (!tokenizerFile.empty() && fs::exists(tokenizerFile, tec)) {
        const fs::path dst = fs::path(outDir) / "tokenizer.tiktoken";
        if (!fs::exists(dst, tec) || !fs::equivalent(tokenizerFile, dst, tec))
            fs::copy_file(tokenizerFile, dst, fs::copy_options::overwrite_existing);
        tokName = "tokenizer.tiktoken";  // gpt_oss_tokens.py --model <dir> resolves it relative to config.json
    }
    struct Blob {
        std::string name, dtype;
        std::vector<uint64_t> shape;
        std::vector<uint8_t> data;
    };
    std::vector<Blob> blobs;
    for (const auto& t : layout.tensors) {
        const float* src = params.data() + t.offset;
        if (t.exportAs == ParamTensor::Export::F32) {
            Blob b{ t.name, "F32", t.shape, {} };
            b.data.resize(t.count * 4);
            std::memcpy(b.data.data(), src, t.count * 4);
            blobs.push_back(std::move(b));
        } else if (t.exportAs == ParamTensor::Export::BF16) {
            Blob b{ t.name, "BF16", t.shape, {} };
            b.data.resize(t.count * 2);
            for (uint64_t i = 0; i < t.count; ++i) {
                const uint16_t v = f32ToBf16(src[i]);
                std::memcpy(&b.data[i * 2], &v, 2);
            }
            blobs.push_back(std::move(b));
        } else {
            // [E][rows][K] -> blocks U8 [E][rows][K/32][16] + scales U8 [E][rows][K/32].
            const uint64_t K = t.shape.back();
            const uint64_t rowsAll = t.count / K, nb = K / 32;
            Blob bl{ t.name + ".blocks", "U8", { t.shape[0], t.shape[1], nb, 16 }, {} };
            Blob sc{ t.name + ".scales", "U8", { t.shape[0], t.shape[1], nb }, {} };
            bl.data.resize(rowsAll * nb * 16);
            sc.data.resize(rowsAll * nb);
            for (uint64_t r = 0; r < rowsAll; ++r)
                for (uint64_t b = 0; b < nb; ++b) {
                    const float* x = src + r * K + b * 32;
                    float amax = 0.0f;
                    for (int i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(x[i]));
                    const int e = mxfp4Exponent(amax);
                    const float inv = float(std::ldexp(1.0, -e));
                    uint8_t codes[32];
                    for (int i = 0; i < 32; ++i) codes[i] = e <= -127 ? 0 : e2m1Code(x[i] * inv);
                    uint8_t* dst = &bl.data[(r * nb + b) * 16];
                    for (int j = 0; j < 16; ++j) dst[j] = uint8_t(codes[2 * j] | (codes[2 * j + 1] << 4));
                    sc.data[r * nb + b] = uint8_t(e + 127);
                }
            blobs.push_back(std::move(bl));
            blobs.push_back(std::move(sc));
        }
    }
    std::ostringstream h;
    h << "{";
    uint64_t off = 0;
    for (size_t i = 0; i < blobs.size(); ++i) {
        const Blob& b = blobs[i];
        h << (i ? "," : "") << "\"" << b.name << "\":{\"dtype\":\"" << b.dtype << "\",\"shape\":[";
        for (size_t d = 0; d < b.shape.size(); ++d) h << (d ? "," : "") << b.shape[d];
        h << "],\"data_offsets\":[" << off << "," << off + b.data.size() << "]}";
        off += b.data.size();
    }
    h << "}";
    std::string header = h.str();
    while (header.size() % 8) header += ' ';
    const std::string path = (fs::path(outDir) / "model.safetensors").string();
    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) throw std::runtime_error("cannot write " + path);
    const uint64_t hl = header.size();
    writeAll(f, &hl, 8, path);
    writeAll(f, header.data(), header.size(), path);
    for (const Blob& b : blobs) writeAll(f, b.data.data(), b.data.size(), path);
    std::fclose(f);
    std::ofstream(fs::path(outDir) / "config.json") << configJson(cfg, tokName);
}

// ---------------------------------------------------------------------------
// Checkpoint files
// ---------------------------------------------------------------------------
CheckpointInfo probeCheckpoint(const std::string& path) {
    CheckpointInfo info;
    info.path = path;
    std::error_code ec;
    if (!fs::exists(path, ec)) return info;
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return info;
    CkptHeader h;
    bool ok = readHeader(f, path, h);
    std::fclose(f);
    if (!ok) return info;
    const uint64_t expect = h.dataOffset + 3 * h.count * 4 + 8;
    if (fs::file_size(path, ec) != expect) return info;
    f = std::fopen(path.c_str(), "rb");
    char end[8] = {};
#ifdef _WIN32
    _fseeki64(f, (long long)(expect - 8), SEEK_SET);
#else
    fseeko(f, off_t(expect - 8), SEEK_SET);
#endif
    ok = std::fread(end, 1, 8, f) == 8 && std::memcmp(end, kEndMagic, 8) == 0;
    std::fclose(f);
    if (!ok) return info;
    JsonValue j = JsonValue::parse(h.json);
    info.step = uint64_t(j["step"].asInt());
    info.valid = true;
    return info;
}

GptOssConfig readCheckpointParams(const std::string& path, std::vector<float>& params, std::string* tokenizer) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("cannot open " + path);
    CkptHeader h;
    if (!readHeader(f, path, h)) {
        std::fclose(f);
        throw std::runtime_error(path + " is not a NeonEngine gpt-oss checkpoint");
    }
    JsonValue j = JsonValue::parse(h.json);
    GptOssConfig cfg = configFromJson(j);
    if (tokenizer) *tokenizer = j.has("neon_tokenizer") ? j["neon_tokenizer"].asString() : "gpt2";
    params.resize(h.count);
    readAll(f, params.data(), h.count * 4, path);
    std::fclose(f);
    return cfg;
}

// ---------------------------------------------------------------------------
// Trainer
// ---------------------------------------------------------------------------
GptOssTrainer::GptOssTrainer(ComputeContext& c, const GptOssConfig& config, const TrainOptions& o)
    : ctx(c), cfg(config), opts(o) {
    if (cfg.headDim != 64) throw std::runtime_error("gpt-oss train: head_dim must be 64");
    if (opts.seq % 64) throw std::runtime_error("gpt-oss train: --seq must be a multiple of 64");
    if (cfg.hidden % 32 || cfg.intermediate % 32) throw std::runtime_error("gpt-oss train: hidden/intermediate must be multiples of 32");
    if (cfg.numHeads % cfg.numKVHeads) throw std::runtime_error("gpt-oss train: heads must be a multiple of kv heads");
    if (cfg.numExperts > 512 || cfg.expertsPerToken > 32 || cfg.expertsPerToken > cfg.numExperts)
        throw std::runtime_error("gpt-oss train: need E <= 512, k <= 32, k <= E");
    lay = ParamLayout::build(cfg);
    N = opts.batch * opts.seq;
    Q = cfg.qkvDim();
    HD = cfg.numHeads * cfg.headDim;
    maxRows = uint32_t(roundUp(uint64_t(N) * cfg.expertsPerToken + uint64_t(cfg.numExperts) * (TILE - 1), TILE));
    maxTiles = maxRows / TILE;
    opts.lmChunk = std::min(opts.lmChunk, N);
    const bool tcAvailable = ctx.hasCoopMat() && ctx.subgroupSize() == 32;
    if (opts.precision == "fp16") {
        if (!tcAvailable) throw std::runtime_error("gpt-oss train: --precision fp16 needs cooperative matrices (tensor cores)");
        tensorCores = true;
    } else if (opts.precision == "auto") {
        tensorCores = tcAvailable;
    } else if (opts.precision != "fp32") {
        throw std::runtime_error("gpt-oss train: --precision must be fp16, fp32 or auto");
    }
    lossScale = tensorCores ? std::max(1.0f, opts.lossScale) : 1.0f;
    std::cout << "gpt-oss train: GEMMs and attention " << (tensorCores ? "fp16 tensor cores (fp32 accumulate), loss scale " : "fp32")
              << (tensorCores ? std::to_string(int64_t(lossScale)) : std::string()) << "\n";
    arena = std::make_unique<ParamArena>(ctx, 16u << 20);
    allocate();
    uploadRopeTables();
}

GptOssTrainer::~GptOssTrainer() {
    if (mirrorThread.joinable()) mirrorThread.join();
    std::vector<GpuBuffer*> all = { &P, &G, &Mo, &Ve, &Pq, &tf, &rstdf, &dR, &dT, &dO, &dqkv, &Dbuf, &sinkTerm, &Xe,
                                    &hbuf, &dY2, &dw, &dLogits, &aux, &logitsChunk, &splitBuf, &cosT, &sinT, &tokens,
                                    &targets, &uniqTok, &uniqOff, &uniqPos, &lossRows, &normPartial, &normSq };
    for (GpuBuffer* b : all) ctx.destroyBuffer(*b);
    for (auto& r : resid) ctx.destroyBuffer(r);
    for (auto& a : acts)
        for (GpuBuffer* b : { &a.t1, &a.rstd1, &a.qkv, &a.O, &a.L, &a.t2, &a.rstd2, &a.logits, &a.ids, &a.w,
                              &a.slotPos, &a.rowEntry, &a.tileGroup, &a.seg, &a.counts, &a.H1, &a.Y2 })
            ctx.destroyBuffer(*b);
}

void GptOssTrainer::allocate() {
    const uint64_t H = cfg.hidden, I = cfg.intermediate, E = cfg.numExperts, K = cfg.expertsPerToken;
    uint64_t bytes = 0;
    auto dev = [&](uint64_t floats) {
        GpuBuffer b = ctx.createBuffer(std::max<uint64_t>(floats, 4) * 4, MemKind::Device);
        bytes += b.size;
        return b;
    };
    auto host = [&](uint64_t floats) { return ctx.createBuffer(std::max<uint64_t>(floats, 4) * 4, MemKind::Host); };

    P = dev(lay.total);
    const uint64_t optFloats = opts.frozen ? 4 : lay.total;  // frozen decoder: no gradients or Adam state
    G = dev(optFloats);
    Mo = dev(optFloats);
    Ve = dev(optFloats);
    ctx.fillZero(G);
    ctx.fillZero(Mo);
    ctx.fillZero(Ve);
    if (opts.qatStart != -2 || opts.frozen) {
        uint64_t q = 0;
        for (const auto& t : lay.tensors)
            if (t.exportAs == ParamTensor::Export::MXFP4) {
                qatMap.push_back({ t.offset, q });
                qatCounts.push_back(t.count);
                q = roundUp(q + t.count, 4);
            }
        Pq = dev(q);
    }

    acts.resize(cfg.numLayers);
    for (auto& a : acts) {
        a.t1 = dev(uint64_t(N) * H);
        a.rstd1 = dev(N);
        a.qkv = dev(uint64_t(N) * Q);
        a.O = dev(uint64_t(N) * HD);
        a.L = dev(uint64_t(N) * cfg.numHeads);
        a.t2 = dev(uint64_t(N) * H);
        a.rstd2 = dev(N);
        a.logits = dev(uint64_t(N) * E);
        a.ids = dev(uint64_t(N) * K);
        a.w = dev(uint64_t(N) * K);
        a.slotPos = dev(uint64_t(N) * K);
        a.rowEntry = dev(maxRows);
        a.tileGroup = dev(maxTiles);
        a.seg = dev(E + 1);
        a.counts = dev(E);
        a.H1 = dev(uint64_t(maxRows) * 2 * I);
        a.Y2 = dev(uint64_t(maxRows) * H);
    }
    resid.resize(2 * cfg.numLayers + 1);
    for (auto& r : resid) r = dev(uint64_t(N) * H);
    tf = dev(uint64_t(N) * H);
    rstdf = dev(N);

    dR = dev(uint64_t(N) * H);
    dT = dev(uint64_t(N) * H);
    dO = dev(uint64_t(N) * HD);
    dqkv = dev(uint64_t(N) * Q);
    Dbuf = dev(uint64_t(N) * cfg.numHeads);
    sinkTerm = dev(uint64_t(N) * cfg.numHeads);
    Xe = dev(uint64_t(maxRows) * H);
    hbuf = dev(uint64_t(maxRows) * I);
    dY2 = dev(uint64_t(maxRows) * H);
    dw = dev(uint64_t(N) * K);
    dLogits = dev(uint64_t(N) * E);
    aux = dev(uint64_t(N) * cfg.numLayers);
    logitsChunk = dev(uint64_t(opts.lmChunk) * cfg.vocabSize);
    splitCap = uint32_t(8 * std::max({ uint64_t(Q) * H, H * HD, E * H, H * H }));
    splitBuf = dev(splitCap);
    cosT = dev(uint64_t(opts.seq) * 32);
    sinT = dev(uint64_t(opts.seq) * 32);

    tokens = host(N);
    targets = host(N);
    uniqTok = host(N);
    uniqOff = host(N + 1);
    uniqPos = host(N);
    lossRows = host(N);
    normPartial = dev(kMaxGroups + 1);
    normSq = host(4);

    const uint64_t paramBytes = lay.total * 4;
    std::cout << "gpt-oss train: " << std::fixed << std::setprecision(2) << double(lay.total) / 1e6
              << " M parameters (" << double(lay.total - lay.at("embedding.weight").count - lay.at("unembedding.weight").count) / 1e6
              << " M without embeddings); GPU memory " << double(bytes) / double(1ull << 30) << " GiB ("
              << double(4 * paramBytes + (Pq ? Pq.size : 0)) / double(1ull << 30) << " GiB weights + optimizer)\n";
}

void GptOssTrainer::initParams() {
    std::vector<float> host(lay.total, 0.0f);
    const double residScale = 1.0 / std::sqrt(2.0 * cfg.numLayers);
    for (const auto& t : lay.tensors) {
        float* p = host.data() + t.offset;
        const std::string& n = t.name;
        auto endsWith = [&](const char* s) {
            const size_t k = std::strlen(s);
            return n.size() >= k && n.compare(n.size() - k, k, s) == 0;
        };
        if (endsWith("norm.scale")) {
            std::fill(p, p + t.count, 1.0f);
            continue;
        }
        if (!t.decay) continue;  // biases and sinks start at 0
        double std = 0.02;
        if (endsWith("out.weight") || endsWith("mlp2_weight")) std *= residScale;
        uint64_t s = hashMix(opts.seed, hashName(n));
        for (uint64_t i = 0; i < t.count; i += 2) {
            const double u1 = (double(splitmix(s) >> 11) + 0.5) * (1.0 / 9007199254740992.0);
            const double u2 = double(splitmix(s) >> 11) * (1.0 / 9007199254740992.0);
            const double r = std::sqrt(-2.0 * std::log(u1));
            p[i] = float(std * r * std::cos(2.0 * kPi * u2));
            if (i + 1 < t.count) p[i + 1] = float(std * r * std::sin(2.0 * kPi * u2));
        }
    }
    ctx.upload(P, 0, host.data(), host.size() * 4);
}

void GptOssTrainer::uploadRopeTables() {
    std::vector<float> c, s;
    yarnTables(cfg, opts.seq, c, s);
    ctx.upload(cosT, 0, c.data(), c.size() * 4);
    ctx.upload(sinT, 0, s.data(), s.size() * 4);
}

uint64_t GptOssTrainer::pAddr(const std::string& name) const { return P.address + lay.at(name).offset * 4; }
uint64_t GptOssTrainer::gAddr(const std::string& name) const { return G.address + lay.at(name).offset * 4; }

uint64_t GptOssTrainer::expertWeightAddr(const std::string& name) const {
    const ParamTensor& t = lay.at(name);
    if (qatActive)
        for (const auto& [src, dst] : qatMap)
            if (src == t.offset) return Pq.address + dst * 4;
    return P.address + t.offset * 4;
}

// ---------------------------------------------------------------------------
// Recording helpers
// ---------------------------------------------------------------------------
struct TrainGemmParams {
    uint32_t M, N, K, flags, sAm, sAk, sBk, sBn, ldC, mode, kChunk, strideBG, strideBiasG, strideCG;
    float alpha;
    uint32_t reserved;
};

void GptOssTrainer::gemm(ComputeContext::Recorder& rec, GemmArgs a) {
    // Tensor-core kernel: one operand of each pair must be contiguous along k or
    // along the output dimension, with 16-byte aligned rows (see gemm_train_coopmat.slang).
    bool tc = false;
    std::string entry = "sml_train_gemm_fwd_f32";
    if (tensorCores) {
        const bool aK = a.sAk == 1, aM = !aK && a.sAm == 1, bK = a.sBk == 1, bN = !bK && a.sBn == 1;
        const uint32_t ldA = aK ? a.sAm : a.sAk, ldB = bK ? a.sBn : a.sBk;
        tc = (aK || aM) && (bK || bN) && ldA % 4 == 0 && ldB % 4 == 0 && a.ldC % 4 == 0 && a.strideBG % 4 == 0 &&
             a.strideCG % 4 == 0 && a.strideBiasG % 4 == 0 && a.A % 16 == 0 && a.B % 16 == 0 && a.C % 16 == 0 &&
             a.bias % 16 == 0 && a.R % 16 == 0 && (a.M * a.N) % 4 == 0;
        if (tc) entry = std::string("sml_train_gemm_tc_") + (aK ? "k" : "m") + (bK ? "k" : "n");
    }
    const Pipeline& pipe = tc ? ctx.pipeline(kGemmTcSrc, entry) : ctx.pipeline(kGemmSrc, entry);
    TrainGemmParams p{};
    p.M = a.M; p.N = a.N; p.K = a.K; p.flags = a.flags;
    p.sAm = a.sAm; p.sAk = a.sAk; p.sBk = a.sBk; p.sBn = a.sBn; p.ldC = a.ldC;
    p.mode = a.mode; p.strideBG = a.strideBG; p.strideBiasG = a.strideBiasG; p.strideCG = a.strideCG;
    p.alpha = a.alpha;
    const uint32_t gx = ceilDiv(a.N, TILE), gy = ceilDiv(a.M, TILE);
    uint32_t gz = a.mode == 2 ? a.groups : 1;
    uint64_t C = a.C;
    uint32_t splits = 1;
    if (a.mode == 0) {
        const uint32_t tiles = gx * gy;
        if (!(a.flags & (GEMM_BIAS | GEMM_RESID)) && a.ldC == a.N && tiles < 64 && a.K >= 1024) {
            splits = std::min<uint32_t>({ 8u, ceilDiv(128, tiles), a.K / 256 });
            while (splits > 1 && uint64_t(a.M) * a.N * splits > splitCap) --splits;
        }
        p.kChunk = uint32_t(roundUp(ceilDiv(a.K, splits), tc ? 32 : 8));
        splits = ceilDiv(a.K, p.kChunk);
        if (splits > 1) {
            C = splitBuf.address;
            p.flags = 0;
            p.strideCG = a.M * a.N;
        }
        gz = splits;
    }
    std::vector<ParamArena::TensorRef> t = { { a.A, DT_FP32 }, { a.B, DT_FP32 }, { C, DT_FP32 },
                                             { a.bias, DT_FP32 }, { a.R, DT_FP32 }, { a.table, DT_I32 } };
    if (ctx.profiling()) {
        std::ostringstream l;
        l << (tc ? "gemm tc  " : "gemm f32 ") << a.tag << " " << a.M << "x" << a.N << "x" << a.K;
        rec.label(l.str());
    }
    rec.dispatch(pipe, arena->params(t, p), gx, gy, gz);
    if (splits > 1) {
        struct { uint32_t n, splits, stride, accum; } r{ a.M * a.N, splits, a.M * a.N, (a.flags & GEMM_ACCUM) ? 1u : 0u };
        const Pipeline& red = ctx.pipeline(kGemmSrc, "sml_train_splitk_reduce_f32");
        std::vector<ParamArena::TensorRef> rt = { { splitBuf.address, DT_FP32 }, { a.C, DT_FP32 } };
        rec.dispatch(red, arena->params(rt, r), ceilDiv(r.n, 256));
    }
}

void GptOssTrainer::linearFwd(ComputeContext::Recorder& rec, uint64_t X, uint64_t W, uint64_t b, uint64_t Y,
                              uint32_t rows, uint32_t in, uint32_t out, uint64_t residIn) {
    GemmArgs a;
    a.A = X; a.B = W; a.C = Y; a.bias = b; a.R = residIn;
    a.M = rows; a.N = out; a.K = in;
    a.sAm = in; a.sAk = 1; a.sBk = 1; a.sBn = in; a.ldC = out;
    a.flags = (b ? GEMM_BIAS : 0) | (residIn ? GEMM_RESID : 0);
    a.tag = "fwd";
    gemm(rec, a);
}

void GptOssTrainer::linearDX(ComputeContext::Recorder& rec, uint64_t dY, uint64_t W, uint64_t dX, uint32_t rows,
                             uint32_t in, uint32_t out, bool accum) {
    GemmArgs a;
    a.A = dY; a.B = W; a.C = dX;
    a.M = rows; a.N = in; a.K = out;
    a.sAm = out; a.sAk = 1; a.sBk = in; a.sBn = 1; a.ldC = in;
    a.flags = accum ? GEMM_ACCUM : 0;
    a.tag = "dX";
    gemm(rec, a);
}

void GptOssTrainer::linearDW(ComputeContext::Recorder& rec, uint64_t dY, uint64_t X, uint64_t dW, uint32_t rows,
                             uint32_t in, uint32_t out) {
    GemmArgs a;
    a.A = dY; a.B = X; a.C = dW;
    a.M = out; a.N = in; a.K = rows;
    a.sAm = 1; a.sAk = out; a.sBk = in; a.sBn = 1; a.ldC = in;
    a.flags = GEMM_ACCUM;
    a.tag = "dW";
    gemm(rec, a);
}

void GptOssTrainer::colsum(ComputeContext::Recorder& rec, uint64_t in, uint64_t out, uint32_t rows, uint32_t cols,
                           uint32_t ld, bool accum, uint64_t seg, uint32_t groups, uint32_t strideOut) {
    struct { uint32_t rows, cols, ld, flags, strideOut, r0; } p{ rows, cols, ld, (accum ? 1u : 0u) | (seg ? 2u : 0u), strideOut, 0 };
    std::vector<ParamArena::TensorRef> t = { { in, DT_FP32 }, { out, DT_FP32 }, { seg, DT_I32 } };
    rec.dispatch(ctx.pipeline(kOpsSrc, "sml_train_colsum_f32"), arena->params(t, p), ceilDiv(cols, 64), 1, groups);
}

void GptOssTrainer::rmsnormFwd(ComputeContext::Recorder& rec, uint64_t x, uint64_t w, uint64_t y, uint64_t rstd) {
    struct { uint32_t rows, H; float eps; uint32_t flags; } p{ N, cfg.hidden, 1.0e-5f, 0 };
    std::vector<ParamArena::TensorRef> t = { { x, DT_FP32 }, { w, DT_FP32 }, { y, DT_FP32 }, { rstd, DT_FP32 } };
    rec.dispatch(ctx.pipeline(kOpsSrc, "sml_train_rmsnorm_fwd_f32"), arena->params(t, p), N);
}

void GptOssTrainer::rmsnormBwd(ComputeContext::Recorder& rec, uint64_t x, uint64_t w, uint64_t rstd, uint64_t dy,
                               uint64_t dX, bool overwrite) {
    struct { uint32_t rows, H; float eps; uint32_t flags; } p{ N, cfg.hidden, 1.0e-5f, overwrite ? 1u : 0u };
    std::vector<ParamArena::TensorRef> t = { { x, DT_FP32 }, { w, DT_FP32 }, { rstd, DT_FP32 }, { dy, DT_FP32 }, { dX, DT_FP32 } };
    rec.dispatch(ctx.pipeline(kOpsSrc, "sml_train_rmsnorm_bwd_f32"), arena->params(t, p), N);
}

void GptOssTrainer::rope(ComputeContext::Recorder& rec, uint64_t qkv, bool inverse) {
    const uint64_t total = uint64_t(N) * (cfg.numHeads + cfg.numKVHeads) * (cfg.headDim / 2);
    const uint32_t groups = std::min<uint32_t>(ceilDiv(total, 256), kMaxGroups);
    struct { uint32_t rows, T, heads, D, ld, inverse, stride, r0; } p{
        N, opts.seq, cfg.numHeads + cfg.numKVHeads, cfg.headDim, Q, inverse ? 1u : 0u, groups * 256u, 0 };
    std::vector<ParamArena::TensorRef> t = { { qkv, DT_FP32 }, { cosT.address, DT_FP32 }, { sinT.address, DT_FP32 } };
    rec.dispatch(ctx.pipeline(kOpsSrc, "sml_train_rope_f32"), arena->params(t, p), groups);
}

void GptOssTrainer::moeElt(ComputeContext::Recorder& rec, const char* entry, std::vector<ParamArena::TensorRef> t,
                           uint32_t rows, uint32_t cols, uint32_t flags, uint64_t elements) {
    const uint32_t groups = std::min<uint32_t>(ceilDiv(elements, 256), kMaxGroups);
    struct { uint32_t rows, cols, K, stride; float limit, alpha; uint32_t flags, r0; } p{
        rows, cols, cfg.expertsPerToken, groups * 256u, cfg.swigluLimit, 1.702f, flags, 0 };
    rec.dispatch(ctx.pipeline(kMoeSrc, entry), arena->params(t, p), groups);
}

// ---------------------------------------------------------------------------
// Forward / head / backward
// ---------------------------------------------------------------------------
struct AttnParams {
    uint32_t T, Hq, Hkv, window, ldQKV, ldO;
    float scale;
    uint32_t r0;
};

void GptOssTrainer::recordForward(ComputeContext::Recorder& rec) {
    const uint32_t H = cfg.hidden, I = cfg.intermediate, E = cfg.numExperts, K = cfg.expertsPerToken;
    {
        struct { uint32_t rows, H, r0, r1; } p{ N, H, 0, 0 };
        std::vector<ParamArena::TensorRef> t = { { tokens.address, DT_I32 }, { pAddr("embedding.weight"), DT_FP32 },
                                                 { resid[0].address, DT_FP32 } };
        rec.dispatch(ctx.pipeline(kOpsSrc, "sml_train_embed_fwd_f32"), arena->params(t, p), N);
    }
    if (opts.memRows && memSource) {  // Memorilla: memory vectors replace the first memRows inputs of each sequence
        const uint64_t n = uint64_t(opts.batch) * opts.memRows * H;
        const uint32_t g = std::min<uint32_t>(ceilDiv(n, 256), kMaxGroups);
        struct { uint32_t B, K, H, inT, inOff, outT, outOff, flags, stride, r0, r1, r2; } p{
            opts.batch, opts.memRows, H, opts.memRows, 0, opts.seq, 0, 0, g * 256u, 0, 0, 0 };
        std::vector<ParamArena::TensorRef> t = { { memSource, DT_FP32 }, { resid[0].address, DT_FP32 } };
        rec.dispatch(ctx.pipeline(kMemSrc, "sml_mem_rows_copy_f32"), arena->params(t, p), g);
    }
    for (uint32_t l = 0; l < cfg.numLayers; ++l) {
        LayerAct& A = acts[l];
        const std::string a = "block." + std::to_string(l) + ".attn.", m = "block." + std::to_string(l) + ".mlp.";
        rmsnormFwd(rec, resid[2 * l].address, pAddr(a + "norm.scale"), A.t1.address, A.rstd1.address);
        linearFwd(rec, A.t1.address, pAddr(a + "qkv.weight"), pAddr(a + "qkv.bias"), A.qkv.address, N, H, Q);
        rope(rec, A.qkv.address, false);
        {
            AttnParams p{ opts.seq, cfg.numHeads, cfg.numKVHeads, (l % 2 == 0) ? cfg.slidingWindow : 0u, Q, HD,
                          1.0f / std::sqrt(float(cfg.headDim)), 0 };
            std::vector<ParamArena::TensorRef> t = { { A.qkv.address, DT_FP32 }, { pAddr(a + "sinks"), DT_FP32 },
                                                     { A.O.address, DT_FP32 }, { A.L.address, DT_FP32 } };
            rec.dispatch(attnPipeline(ctx, tensorCores, "fwd"), arena->params(t, p), opts.seq / 64,
                         cfg.numHeads, opts.batch);
        }
        linearFwd(rec, A.O.address, pAddr(a + "out.weight"), pAddr(a + "out.bias"), resid[2 * l + 1].address, N, HD, H,
                  resid[2 * l].address);

        rmsnormFwd(rec, resid[2 * l + 1].address, pAddr(m + "norm.scale"), A.t2.address, A.rstd2.address);
        linearFwd(rec, A.t2.address, pAddr(m + "gate.weight"), pAddr(m + "gate.bias"), A.logits.address, N, H, E);
        {
            struct { uint32_t T, E, topK, flags; float scaleOut; uint32_t ld, r0, r1; } p{ N, E, K, 2u, 1.0f, E, 0, 0 };
            std::vector<ParamArena::TensorRef> t = { { A.logits.address, DT_FP32 }, { 0, DT_FP32 },
                                                     { A.ids.address, DT_I32 }, { A.w.address, DT_FP32 } };
            rec.dispatch(ctx.pipeline(kRouterSrc, "sml_llm_moe_router_fwd_f32"), arena->params(t, p), ceilDiv(N, 4));
        }
        {
            struct { uint32_t N, K, E, pad, maxRows, maxTiles, r0, r1; } p{ N, K, E, TILE, maxRows, maxTiles, 0, 0 };
            std::vector<ParamArena::TensorRef> t = { { A.ids.address, DT_I32 },      { A.slotPos.address, DT_I32 },
                                                     { A.rowEntry.address, DT_I32 }, { A.tileGroup.address, DT_I32 },
                                                     { A.seg.address, DT_I32 },      { A.counts.address, DT_I32 } };
            rec.dispatch(ctx.pipeline(kMoeSrc, "sml_train_moe_route_f32"), arena->params(t, p), 1);
        }
        moeElt(rec, "sml_train_moe_gather_f32",
               { { A.rowEntry.address, DT_I32 }, { A.t2.address, DT_FP32 }, { Xe.address, DT_FP32 } }, maxRows, H, 0,
               uint64_t(maxRows) * H / 4);
        {
            GemmArgs g;
            g.A = Xe.address; g.B = expertWeightAddr(m + "mlp1_weight"); g.C = A.H1.address;
            g.bias = pAddr(m + "mlp1_bias"); g.table = A.tileGroup.address;
            g.M = maxRows; g.N = 2 * I; g.K = H;
            g.sAm = H; g.sAk = 1; g.sBk = 1; g.sBn = H; g.ldC = 2 * I;
            g.mode = 1; g.flags = GEMM_BIAS; g.strideBG = 2 * I * H; g.strideBiasG = 2 * I;
            g.tag = "moe mlp1 fwd";
            gemm(rec, g);
        }
        moeElt(rec, "sml_train_moe_swiglu_fwd_f32", { { A.H1.address, DT_FP32 }, { hbuf.address, DT_FP32 } }, maxRows,
               I, 0, uint64_t(maxRows) * I);
        {
            GemmArgs g;
            g.A = hbuf.address; g.B = expertWeightAddr(m + "mlp2_weight"); g.C = A.Y2.address;
            g.bias = pAddr(m + "mlp2_bias"); g.table = A.tileGroup.address;
            g.M = maxRows; g.N = H; g.K = I;
            g.sAm = I; g.sAk = 1; g.sBk = 1; g.sBn = I; g.ldC = H;
            g.mode = 1; g.flags = GEMM_BIAS; g.strideBG = H * I; g.strideBiasG = H;
            g.tag = "moe mlp2 fwd";
            gemm(rec, g);
        }
        moeElt(rec, "sml_train_moe_combine_fwd_f32",
               { { resid[2 * l + 1].address, DT_FP32 }, { A.Y2.address, DT_FP32 }, { A.slotPos.address, DT_I32 },
                 { A.w.address, DT_FP32 }, { resid[2 * l + 2].address, DT_FP32 } },
               N, H, 0, uint64_t(N) * H / 4);
    }
    rmsnormFwd(rec, resid[2 * cfg.numLayers].address, pAddr("norm.scale"), tf.address, rstdf.address);
}

void GptOssTrainer::recordHead(ComputeContext::Recorder& rec, bool backward) {
    const uint32_t H = cfg.hidden, V = cfg.vocabSize;
    const float scale = lossScale / (lossDenom > 0.0f ? lossDenom : float(N));  // dlogits carry the loss scale
    const bool dW = !opts.frozen;
    for (uint32_t c0 = 0; c0 < N; c0 += opts.lmChunk) {
        const uint32_t rows = std::min(opts.lmChunk, N - c0);
        const uint64_t tfc = tf.address + uint64_t(c0) * H * 4;
        linearFwd(rec, tfc, pAddr("unembedding.weight"), 0, logitsChunk.address, rows, H, V);
        {
            struct { uint32_t rows, V; float scale; uint32_t r0; } p{ rows, V, scale, 0 };
            std::vector<ParamArena::TensorRef> t = { { logitsChunk.address, DT_FP32 },
                                                     { targets.address + uint64_t(c0) * 4, DT_I32 },
                                                     { lossRows.address + uint64_t(c0) * 4, DT_FP32 } };
            rec.dispatch(ctx.pipeline(kOpsSrc, "sml_train_xent_f32"), arena->params(t, p), rows);
        }
        if (backward) {
            if (dW) linearDW(rec, logitsChunk.address, tfc, gAddr("unembedding.weight"), rows, H, V);
            linearDX(rec, logitsChunk.address, pAddr("unembedding.weight"), dT.address + uint64_t(c0) * H * 4, rows, H,
                     V, false);
        }
    }
    if (backward) {
        rmsnormBwd(rec, resid[2 * cfg.numLayers].address, pAddr("norm.scale"), rstdf.address, dT.address, dR.address, true);
        if (dW) colsum(rec, dT.address, gAddr("norm.scale"), N, H, H, true);
    }
}

void GptOssTrainer::recordBackward(ComputeContext::Recorder& rec) {
    const uint32_t H = cfg.hidden, I = cfg.intermediate, E = cfg.numExperts, K = cfg.expertsPerToken;
    const bool dW = !opts.frozen;  // frozen decoder: input gradients only
    for (int li = int(cfg.numLayers) - 1; li >= 0; --li) {
        const uint32_t l = uint32_t(li);
        LayerAct& A = acts[l];
        const std::string a = "block." + std::to_string(l) + ".attn.", m = "block." + std::to_string(l) + ".mlp.";

        // ---- MoE block: x_out = x_mid + sum_s w_s expert_s(t2)
        moeElt(rec, "sml_train_moe_combine_bwd_f32",
               { { A.rowEntry.address, DT_I32 }, { A.w.address, DT_FP32 }, { dR.address, DT_FP32 }, { dY2.address, DT_FP32 } },
               maxRows, H, 0, uint64_t(maxRows) * H / 4);
        {
            const uint32_t entries = N * K;
            const uint32_t groups = std::min<uint32_t>(ceilDiv(entries, 8), kMaxGroups);
            struct { uint32_t entries, H, K, groups; } p{ entries, H, K, groups };
            std::vector<ParamArena::TensorRef> t = { { A.slotPos.address, DT_I32 }, { dR.address, DT_FP32 },
                                                     { A.Y2.address, DT_FP32 }, { dw.address, DT_FP32 } };
            rec.dispatch(ctx.pipeline(kMoeSrc, "sml_train_moe_dweight_f32"), arena->params(t, p), groups);
        }
        if (dW) colsum(rec, dY2.address, gAddr(m + "mlp2_bias"), maxRows, H, H, true, A.seg.address, E, H);
        if (dW) moeElt(rec, "sml_train_moe_swiglu_fwd_f32", { { A.H1.address, DT_FP32 }, { hbuf.address, DT_FP32 } }, maxRows,
               I, 0, uint64_t(maxRows) * I);
        if (dW) {
            GemmArgs g;  // dW2_e += dY2_e^T h_e
            g.A = dY2.address; g.B = hbuf.address; g.C = gAddr(m + "mlp2_weight"); g.table = A.seg.address;
            g.M = H; g.N = I; g.sAm = 1; g.sAk = H; g.sBk = I; g.sBn = 1; g.ldC = I;
            g.mode = 2; g.groups = E; g.strideCG = H * I; g.flags = GEMM_ACCUM;
            g.tag = "moe mlp2 dW";
            gemm(rec, g);
        }
        {
            GemmArgs g;  // dh = dY2 W2_e
            g.A = dY2.address; g.B = expertWeightAddr(m + "mlp2_weight"); g.C = hbuf.address; g.table = A.tileGroup.address;
            g.M = maxRows; g.N = I; g.K = H; g.sAm = H; g.sAk = 1; g.sBk = I; g.sBn = 1; g.ldC = I;
            g.mode = 1; g.strideBG = H * I;
            g.tag = "moe mlp2 dX";
            gemm(rec, g);
        }
        moeElt(rec, "sml_train_moe_swiglu_bwd_f32", { { A.H1.address, DT_FP32 }, { hbuf.address, DT_FP32 } }, maxRows,
               I, 0, uint64_t(maxRows) * I);
        if (dW) colsum(rec, A.H1.address, gAddr(m + "mlp1_bias"), maxRows, 2 * I, 2 * I, true, A.seg.address, E, 2 * I);
        if (dW) moeElt(rec, "sml_train_moe_gather_f32",
               { { A.rowEntry.address, DT_I32 }, { A.t2.address, DT_FP32 }, { Xe.address, DT_FP32 } }, maxRows, H, 0,
               uint64_t(maxRows) * H / 4);
        if (dW) {
            GemmArgs g;  // dW1_e += dH1_e^T Xe_e
            g.A = A.H1.address; g.B = Xe.address; g.C = gAddr(m + "mlp1_weight"); g.table = A.seg.address;
            g.M = 2 * I; g.N = H; g.sAm = 1; g.sAk = 2 * I; g.sBk = H; g.sBn = 1; g.ldC = H;
            g.mode = 2; g.groups = E; g.strideCG = 2 * I * H; g.flags = GEMM_ACCUM;
            g.tag = "moe mlp1 dW";
            gemm(rec, g);
        }
        {
            GemmArgs g;  // dXe = dH1 W1_e
            g.A = A.H1.address; g.B = expertWeightAddr(m + "mlp1_weight"); g.C = Xe.address; g.table = A.tileGroup.address;
            g.M = maxRows; g.N = H; g.K = 2 * I; g.sAm = 2 * I; g.sAk = 1; g.sBk = H; g.sBn = 1; g.ldC = H;
            g.mode = 1; g.strideBG = 2 * I * H;
            g.tag = "moe mlp1 dX";
            gemm(rec, g);
        }
        moeElt(rec, "sml_train_moe_gather_add_f32",
               { { A.slotPos.address, DT_I32 }, { Xe.address, DT_FP32 }, { dT.address, DT_FP32 } }, N, H, 0,
               uint64_t(N) * H / 4);
        {
            struct { uint32_t N, E, K; float auxScale; } p{ N, E, K, lossScale * opts.auxCoef * float(E) / float(N) };
            std::vector<ParamArena::TensorRef> t = { { A.logits.address, DT_FP32 }, { A.ids.address, DT_I32 },
                                                     { A.w.address, DT_FP32 },      { dw.address, DT_FP32 },
                                                     { A.counts.address, DT_I32 },  { dLogits.address, DT_FP32 },
                                                     { aux.address + uint64_t(l) * N * 4, DT_FP32 } };
            rec.dispatch(ctx.pipeline(kMoeSrc, "sml_train_moe_router_bwd_f32"), arena->params(t, p), ceilDiv(N, 8));
        }
        if (dW) linearDW(rec, dLogits.address, A.t2.address, gAddr(m + "gate.weight"), N, H, E);
        if (dW) colsum(rec, dLogits.address, gAddr(m + "gate.bias"), N, E, E, true);
        linearDX(rec, dLogits.address, pAddr(m + "gate.weight"), dT.address, N, H, E, true);
        rmsnormBwd(rec, resid[2 * l + 1].address, pAddr(m + "norm.scale"), A.rstd2.address, dT.address, dR.address, false);
        if (dW) colsum(rec, dT.address, gAddr(m + "norm.scale"), N, H, H, true);

        // ---- attention block: x_mid = x_in + out(attn(rope(qkv(norm(x_in)))))
        linearDX(rec, dR.address, pAddr(a + "out.weight"), dO.address, N, HD, H, false);
        if (dW) linearDW(rec, dR.address, A.O.address, gAddr(a + "out.weight"), N, HD, H);
        if (dW) colsum(rec, dR.address, gAddr(a + "out.bias"), N, H, H, true);
        {
            AttnParams p{ opts.seq, cfg.numHeads, cfg.numKVHeads, (l % 2 == 0) ? cfg.slidingWindow : 0u, Q, HD,
                          1.0f / std::sqrt(float(cfg.headDim)), 0 };
            std::vector<ParamArena::TensorRef> t = { { A.qkv.address, DT_FP32 }, { pAddr(a + "sinks"), DT_FP32 },
                                                     { A.O.address, DT_FP32 },   { dO.address, DT_FP32 },
                                                     { A.L.address, DT_FP32 },   { dqkv.address, DT_FP32 },
                                                     { Dbuf.address, DT_FP32 },  { sinkTerm.address, DT_FP32 } };
            rec.dispatch(attnPipeline(ctx, tensorCores, "bwd_dq"), arena->params(t, p), opts.seq / 64,
                         cfg.numHeads, opts.batch);
            std::vector<ParamArena::TensorRef> t2 = { { A.qkv.address, DT_FP32 }, { dO.address, DT_FP32 },
                                                      { A.L.address, DT_FP32 },   { Dbuf.address, DT_FP32 },
                                                      { dqkv.address, DT_FP32 } };
            rec.dispatch(attnPipeline(ctx, tensorCores, "bwd_dkv"), arena->params(t2, p), opts.seq / 64,
                         cfg.numKVHeads, opts.batch);
        }
        if (dW) colsum(rec, sinkTerm.address, gAddr(a + "sinks"), N, cfg.numHeads, cfg.numHeads, true);
        rope(rec, dqkv.address, true);
        if (dW) linearDW(rec, dqkv.address, A.t1.address, gAddr(a + "qkv.weight"), N, H, Q);
        if (dW) colsum(rec, dqkv.address, gAddr(a + "qkv.bias"), N, Q, Q, true);
        linearDX(rec, dqkv.address, pAddr(a + "qkv.weight"), dT.address, N, H, Q, false);
        rmsnormBwd(rec, resid[2 * l].address, pAddr(a + "norm.scale"), A.rstd1.address, dT.address, dR.address, false);
        if (dW) colsum(rec, dT.address, gAddr(a + "norm.scale"), N, H, H, true);
    }
    if (dW) {
        struct { uint32_t rows, H, r0, r1; } p{ uniqueCount, H, 0, 0 };
        std::vector<ParamArena::TensorRef> t = { { uniqTok.address, DT_I32 }, { uniqOff.address, DT_I32 },
                                                 { uniqPos.address, DT_I32 }, { dR.address, DT_FP32 },
                                                 { gAddr("embedding.weight"), DT_FP32 } };
        rec.dispatch(ctx.pipeline(kOpsSrc, "sml_train_embed_bwd_f32"), arena->params(t, p), uniqueCount);
    }
}

void GptOssTrainer::recordOptimizer(ComputeContext::Recorder& rec, float lr, uint64_t step) {
    const uint64_t n = lay.total;
    uint32_t items = 16;
    while (ceilDiv(n, 256ull * items) > kMaxGroups) items *= 2;
    const uint32_t groups = ceilDiv(n, 256ull * items);
    {
        struct { uint32_t n, items, square, accum; } p{ uint32_t(n), items, 1, 0 };
        std::vector<ParamArena::TensorRef> t = { { G.address, DT_FP32 }, { normPartial.address, DT_FP32 } };
        rec.dispatch(ctx.pipeline(kOpsSrc, "sml_train_sumsq_f32"), arena->params(t, p), groups);
        struct { uint32_t n, items, square, accum; } p2{ groups, ceilDiv(groups, 256), 0, 0 };
        std::vector<ParamArena::TensorRef> t2 = { { normPartial.address, DT_FP32 }, { normSq.address, DT_FP32 } };
        rec.dispatch(ctx.pipeline(kOpsSrc, "sml_train_sumsq_f32"), arena->params(t2, p2), 1);
    }
    {
        const double t = double(step + 1);
        const uint32_t g = std::min<uint32_t>(ceilDiv(n, 256), kMaxGroups);
        struct { uint32_t n, decayEnd; float lr, b1, b2, eps, wd, bc1, bc2, clip, gradScale; uint32_t stride; } p{
            uint32_t(n), uint32_t(lay.decayEnd), lr, opts.beta1, opts.beta2, opts.eps, opts.weightDecay,
            float(1.0 - std::pow(double(opts.beta1), t)), float(1.0 - std::pow(double(opts.beta2), t)), opts.clip,
            1.0f / (float(opts.accum) * lossScale), g * 256u };
        std::vector<ParamArena::TensorRef> tt = { { P.address, DT_FP32 }, { G.address, DT_FP32 }, { Mo.address, DT_FP32 },
                                                  { Ve.address, DT_FP32 }, { normSq.address, DT_FP32 } };
        rec.dispatch(ctx.pipeline(kOpsSrc, "sml_train_adamw_f32"), arena->params(tt, p), g);
    }
}

// ---------------------------------------------------------------------------
// Batches and steps
// ---------------------------------------------------------------------------
void GptOssTrainer::setBatchTokens(const std::vector<uint32_t>& ids) {
    const uint32_t T = opts.seq;
    if (ids.size() != size_t(opts.batch) * (T + 1)) throw std::runtime_error("gpt-oss train: bad batch size");
    uint32_t* tok = static_cast<uint32_t*>(tokens.mapped);
    uint32_t* tgt = static_cast<uint32_t*>(targets.mapped);
    for (uint32_t b = 0; b < opts.batch; ++b)
        for (uint32_t t = 0; t < T; ++t) {
            const uint32_t x = ids[size_t(b) * (T + 1) + t], y = ids[size_t(b) * (T + 1) + t + 1];
            if (x >= cfg.vocabSize || y >= cfg.vocabSize) throw std::runtime_error("gpt-oss train: token id >= vocab_size");
            tok[b * T + t] = x;
            tgt[b * T + t] = y;
        }
    // CSR of positions per distinct token for the embedding backward.
    std::vector<std::pair<uint32_t, uint32_t>> order(N);
    for (uint32_t i = 0; i < N; ++i) order[i] = { tok[i], i };
    std::sort(order.begin(), order.end());
    uint32_t* ut = static_cast<uint32_t*>(uniqTok.mapped);
    uint32_t* uo = static_cast<uint32_t*>(uniqOff.mapped);
    uint32_t* up = static_cast<uint32_t*>(uniqPos.mapped);
    uniqueCount = 0;
    for (uint32_t i = 0; i < N; ++i) {
        if (i == 0 || order[i].first != order[i - 1].first) {
            ut[uniqueCount] = order[i].first;
            uo[uniqueCount] = i;
            ++uniqueCount;
        }
        up[i] = order[i].second;
    }
    uo[uniqueCount] = N;
}

void GptOssTrainer::loadBatch(const std::string& file, uint64_t fileTokens, uint64_t streamSeed, uint64_t index) {
    const uint32_t T = opts.seq;
    if (fileTokens < uint64_t(T) + 2) throw std::runtime_error(file + " has fewer tokens than one sequence");
    std::vector<uint32_t> ids(size_t(opts.batch) * (T + 1));
    std::vector<uint16_t> buf(T + 1);
    std::FILE* f = std::fopen(file.c_str(), "rb");
    if (!f) throw std::runtime_error("cannot open " + file);
    for (uint32_t b = 0; b < opts.batch; ++b) {
        const uint64_t start = hashMix(hashMix(opts.seed, streamSeed), index * 4096 + b) % (fileTokens - T - 1);
#ifdef _WIN32
        _fseeki64(f, (long long)(start * 2), SEEK_SET);
#else
        fseeko(f, off_t(start * 2), SEEK_SET);
#endif
        readAll(f, buf.data(), (T + 1) * 2, file);
        for (uint32_t t = 0; t <= T; ++t) ids[size_t(b) * (T + 1) + t] = buf[t];
    }
    std::fclose(f);
    setBatchTokens(ids);
}

float GptOssTrainer::runMicroStep(bool backward) {
    arena->reset();
    ComputeContext::Recorder rec(ctx);
    recordForward(rec);
    recordHead(rec, backward);
    if (backward) recordBackward(rec);
    rec.submitAndWait();
    const float* lr = static_cast<const float*>(lossRows.mapped);
    double s = 0.0;
    for (uint32_t i = 0; i < N; ++i) s += lr[i];
    return float(s / N);
}

float GptOssTrainer::gradientCheck(const std::vector<uint32_t>& ids, std::vector<float>& params, std::vector<float>& grads) {
    initParams();
    setBatchTokens(ids);
    const float loss = runMicroStep(true);
    params.resize(lay.total);
    grads.resize(lay.total);
    ctx.download(P, 0, params.data(), lay.total * 4);
    ctx.download(G, 0, grads.data(), lay.total * 4);
    for (float& g : grads) g /= lossScale;
    return loss;
}

// ---------------------------------------------------------------------------
// Frozen-decoder access (Memorilla)
// ---------------------------------------------------------------------------
void GptOssTrainer::setParams(const std::vector<float>& host) {
    if (host.size() != lay.total) throw std::runtime_error("gpt-oss train: parameter count mismatch");
    ctx.upload(P, 0, host.data(), host.size() * 4);
}

std::vector<float> GptOssTrainer::paramsHost() const {
    std::vector<float> h(lay.total);
    ctx.download(P, 0, h.data(), h.size() * 4);
    return h;
}

void GptOssTrainer::useMxfp4Experts() {
    if (!Pq) throw std::runtime_error("gpt-oss train: no MXFP4 shadow buffer (QAT disabled)");
    arena->reset();
    ComputeContext::Recorder rec(ctx);
    for (size_t i = 0; i < qatMap.size(); ++i) {
        const uint32_t blocks = uint32_t(qatCounts[i] / 32);
        const uint32_t g = std::min<uint32_t>(ceilDiv(blocks, 256), kMaxGroups);
        struct { uint32_t blocks, stride, r1, r2; } p{ blocks, g * 256u, 0, 0 };
        std::vector<ParamArena::TensorRef> t = { { P.address + qatMap[i].first * 4, DT_FP32 },
                                                 { Pq.address + qatMap[i].second * 4, DT_FP32 } };
        rec.dispatch(ctx.pipeline(kOpsSrc, "sml_train_mxfp4_fakequant_f32"), arena->params(t, p), g);
    }
    rec.submitAndWait();
    qatActive = true;
}

void GptOssTrainer::setBatch(const std::vector<uint32_t>& in, const std::vector<uint32_t>& tg) {
    if (in.size() != N || tg.size() != N) throw std::runtime_error("gpt-oss train: bad batch size");
    uint32_t* tok = static_cast<uint32_t*>(tokens.mapped);
    uint32_t* tgt = static_cast<uint32_t*>(targets.mapped);
    for (uint32_t i = 0; i < N; ++i) {
        if (in[i] >= cfg.vocabSize || (tg[i] != 0xFFFFFFFFu && tg[i] >= cfg.vocabSize))
            throw std::runtime_error("gpt-oss train: token id >= vocab_size");
        tok[i] = in[i];
        tgt[i] = tg[i];
    }
    std::vector<std::pair<uint32_t, uint32_t>> order(N);
    for (uint32_t i = 0; i < N; ++i) order[i] = { tok[i], i };
    std::sort(order.begin(), order.end());
    uint32_t* ut = static_cast<uint32_t*>(uniqTok.mapped);
    uint32_t* uo = static_cast<uint32_t*>(uniqOff.mapped);
    uint32_t* up = static_cast<uint32_t*>(uniqPos.mapped);
    uniqueCount = 0;
    for (uint32_t i = 0; i < N; ++i) {
        if (i == 0 || order[i].first != order[i - 1].first) {
            ut[uniqueCount] = order[i].first;
            uo[uniqueCount] = i;
            ++uniqueCount;
        }
        up[i] = order[i].second;
    }
    uo[uniqueCount] = N;
}

void GptOssTrainer::recordStep(ComputeContext::Recorder& rec, bool backward) {
    recordForward(rec);
    recordHead(rec, backward);
    if (backward) recordBackward(rec);
}

double GptOssTrainer::lossSum() const {
    const float* lr = static_cast<const float*>(lossRows.mapped);
    double s = 0.0;
    for (uint32_t i = 0; i < N; ++i) s += lr[i];
    return s;
}

float GptOssTrainer::evaluate() {
    const std::string val = (fs::path(opts.dataDir) / "val.bin").string();
    const uint64_t valTokens = fs::file_size(val) / 2;
    double s = 0.0;
    for (uint32_t i = 0; i < opts.evalBatches; ++i) {
        loadBatch(val, valTokens, 0xE7A1ull, i);
        s += runMicroStep(false);
    }
    return float(s / std::max(1u, opts.evalBatches));
}

bool GptOssTrainer::updateLossScale() {
    // The AdamW kernel skips the update (and only clears the gradients) when the
    // squared gradient norm is not finite; mirror that decision here.
    const float ns = static_cast<const float*>(normSq.mapped)[0];
    if (!std::isfinite(ns)) {
        ++skippedSteps;
        goodSteps = 0;
        if (tensorCores && lossScale > 1.0f) {
            lossScale = std::max(1.0f, lossScale * 0.5f);
            std::cout << "fp16 gradient overflow: step skipped, loss scale -> " << lossScale << "\n";
        } else {
            std::cout << "warning: non-finite gradient, step skipped\n";
        }
        return false;
    }
    if (tensorCores && ++goodSteps >= opts.growthInterval) {
        goodSteps = 0;
        if (lossScale < 16777216.0f) lossScale *= 2.0f;
    }
    return true;
}

float GptOssTrainer::lrAt(uint64_t step) const {
    if (step < opts.warmup) return opts.lr * float(step + 1) / float(opts.warmup);
    if (step >= opts.maxSteps) return opts.minLr;
    const double lr = opts.lr, lo = opts.minLr;
    if (opts.schedule == "wsd") {
        const double decaySteps = std::max(1.0, double(opts.maxSteps - opts.warmup) * double(opts.decayFrac));
        const double start = double(opts.maxSteps) - decaySteps;
        if (double(step) < start) return float(lr);
        return float(lr + (lo - lr) * (double(step) - start + 1.0) / decaySteps);
    }
    const double t = double(step - opts.warmup) / double(std::max<uint64_t>(1, opts.maxSteps - opts.warmup));
    if (opts.schedule == "linear") return float(lr + (lo - lr) * t);
    return float(lo + 0.5 * (lr - lo) * (1.0 + std::cos(kPi * t)));
}

// ---------------------------------------------------------------------------
// Checkpoints
// ---------------------------------------------------------------------------
void GptOssTrainer::saveCheckpoint(uint64_t step, double bestVal) {
    if (mirrorThread.joinable()) mirrorThread.join();  // never rewrite a file that is still being copied
    fs::create_directories(opts.outDir);
    const fs::path dir(opts.outDir);
    const std::string tmp = (dir / "latest.nckpt.tmp").string();
    std::ostringstream extra;
    extra << std::setprecision(17) << ",\n  \"step\": " << step << ",\n  \"best_val\": " << bestVal
          << ",\n  \"tokens_seen\": " << uint64_t(step) * N * opts.accum << ",\n  \"loss_scale\": " << lossScale
          << ",\n  \"version\": 1";
    const std::string json = configJson(cfg, opts.tokenizer, extra.str());
    std::FILE* f = std::fopen(tmp.c_str(), "wb");
    if (!f) throw std::runtime_error("cannot write " + tmp);
    const uint32_t version = 1, len = uint32_t(json.size());
    const uint64_t count = lay.total;
    writeAll(f, kMagic, 8, tmp);
    writeAll(f, &version, 4, tmp);
    writeAll(f, &len, 4, tmp);
    writeAll(f, json.data(), json.size(), tmp);
    writeAll(f, &count, 8, tmp);
    std::vector<float> host(count);
    for (GpuBuffer* b : { &P, &Mo, &Ve }) {
        ctx.download(*b, 0, host.data(), count * 4);
        writeAll(f, host.data(), count * 4, tmp);
    }
    writeAll(f, kEndMagic, 8, tmp);
    if (std::fflush(f) != 0 || std::fclose(f) != 0) throw std::runtime_error("write failed: " + tmp);
    std::error_code ec;
    if (fs::exists(dir / "latest.nckpt")) fs::rename(dir / "latest.nckpt", dir / "prev.nckpt", ec);
    fs::rename(tmp, dir / "latest.nckpt");
    std::cout << "checkpoint: step " << step << " -> " << (dir / "latest.nckpt").string() << "\n";
    if (!opts.mirrorDir.empty()) mirrorAsync((dir / "latest.nckpt").string());
}

void GptOssTrainer::mirrorAsync(const std::string& localPath) {
    const std::string mirror = opts.mirrorDir;
    mirrorThread = std::thread([localPath, mirror] {
        try {
            const auto t0 = std::chrono::steady_clock::now();
            const fs::path dir(mirror);
            fs::create_directories(dir);
            const fs::path tmp = dir / "latest.nckpt.tmp";
            fs::copy_file(localPath, tmp, fs::copy_options::overwrite_existing);
            std::error_code ec;
            if (fs::exists(dir / "latest.nckpt")) {
                fs::remove(dir / "prev.nckpt", ec);
                fs::rename(dir / "latest.nckpt", dir / "prev.nckpt", ec);
            }
            fs::rename(tmp, dir / "latest.nckpt");
            const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            std::cout << "mirror: copied checkpoint to " << (dir / "latest.nckpt").string() << " in " << std::fixed
                      << std::setprecision(1) << s << " s\n"
                      << std::flush;
        } catch (const std::exception& e) {
            std::cerr << "mirror: copy failed (" << e.what() << "); training continues\n";
        }
    });
}

bool GptOssTrainer::loadNewestCheckpoint(uint64_t& step, double& bestVal) {
    std::vector<std::string> cands;
    for (const std::string& d : { opts.outDir, opts.mirrorDir }) {
        if (d.empty()) continue;
        cands.push_back((fs::path(d) / "latest.nckpt").string());
        cands.push_back((fs::path(d) / "prev.nckpt").string());
    }
    CheckpointInfo best;
    for (const auto& c : cands) {
        CheckpointInfo i = probeCheckpoint(c);
        if (i.valid && (!best.valid || i.step > best.step)) best = i;
    }
    if (!best.valid) return false;
    std::FILE* f = std::fopen(best.path.c_str(), "rb");
    CkptHeader h;
    if (!f || !readHeader(f, best.path, h)) throw std::runtime_error("cannot read " + best.path);
    JsonValue j = JsonValue::parse(h.json);
    const GptOssConfig c = configFromJson(j);
    if (c.numLayers != cfg.numLayers || c.hidden != cfg.hidden || c.numExperts != cfg.numExperts ||
        c.vocabSize != cfg.vocabSize || c.intermediate != cfg.intermediate || h.count != lay.total)
        throw std::runtime_error("checkpoint " + best.path + " has a different model config");
    std::vector<float> host(h.count);
    for (GpuBuffer* b : { &P, &Mo, &Ve }) {
        readAll(f, host.data(), h.count * 4, best.path);
        ctx.upload(*b, 0, host.data(), h.count * 4);
    }
    std::fclose(f);
    step = uint64_t(j["step"].asInt());
    bestVal = j.numberOr("best_val", 1e30);
    if (tensorCores) lossScale = std::max(1.0f, float(j.numberOr("loss_scale", lossScale)));
    std::cout << "resumed from " << best.path << " at step " << step << "\n";
    return true;
}

// ---------------------------------------------------------------------------
// Training loop
// ---------------------------------------------------------------------------
void GptOssTrainer::train() {
    using clock = std::chrono::steady_clock;
    const fs::path data(opts.dataDir);
    const std::string trainFile = (data / "train.bin").string();
    if (!fs::exists(trainFile)) throw std::runtime_error("missing " + trainFile + " (run train/prepare_data.py)");
    const uint64_t trainTokens = fs::file_size(trainFile) / 2;
    const bool haveVal = fs::exists(data / "val.bin");

    fs::create_directories(opts.outDir);
    logPath = (fs::path(opts.outDir) / "train_log.csv").string();
    const std::string runTokenizer = (fs::path(opts.outDir) / "tokenizer.tiktoken").string();
    if (!opts.tokenizerFile.empty()) {
        std::error_code ec;
        for (const std::string& d : { opts.outDir, opts.mirrorDir }) {
            if (d.empty()) continue;
            const fs::path dst = fs::path(d) / "tokenizer.tiktoken";
            fs::create_directories(d, ec);
            if (!fs::exists(dst, ec) || !fs::equivalent(opts.tokenizerFile, dst, ec))
                fs::copy_file(opts.tokenizerFile, dst, fs::copy_options::overwrite_existing, ec);
        }
    }
    uint64_t step = 0;
    double bestVal = 1e30;
    const char* logHeader = "step,loss,val_loss,lr,grad_norm,tokens_per_s,aux,val_bpb,loss_scale\n";
    if (opts.fresh || !loadNewestCheckpoint(step, bestVal)) {
        initParams();
        std::ofstream(logPath) << logHeader;
        std::cout << "starting from scratch (seed " << opts.seed << ")\n";
    } else if (!fs::exists(logPath)) {
        std::ofstream(logPath) << logHeader;  // resumed from the mirror into an empty --out
    }
    std::signal(SIGINT, onSignal);
    std::signal(SIGTERM, onSignal);

    const int64_t qatStart = opts.qatStart == -1 ? int64_t(opts.maxSteps * 8 / 10) : opts.qatStart;
    const uint64_t tokensPerStep = uint64_t(N) * opts.accum;
    std::cout << "training: " << opts.batch << " x " << opts.seq << " tokens x " << opts.accum << " micro-batches = "
              << tokensPerStep << " tokens/step, steps " << step << " -> " << opts.maxSteps << ", "
              << trainTokens / 1000000 << " M training tokens";
    if (opts.qatStart != -2) std::cout << ", MXFP4 QAT from step " << qatStart;
    std::cout << "\n";

    auto lastCkpt = clock::now();
    auto tLog = clock::now();
    double lossAcc = 0.0;
    uint32_t lossN = 0;
    const uint64_t profBegin = step + 2, profEnd = profBegin + opts.profileSteps;
    auto tProf = clock::now();
    uint32_t badInARow = 0;
    while (step < opts.maxSteps && !gStop) {
        if (opts.profileSteps && step == profBegin) {
            ctx.resetProfile();
            ctx.setProfiling(true);
            tProf = clock::now();
        }
        qatActive = opts.qatStart != -2 && int64_t(step) >= qatStart && Pq;
        if (qatActive) {
            arena->reset();
            ComputeContext::Recorder rec(ctx);
            for (size_t i = 0; i < qatMap.size(); ++i) {
                const uint32_t blocks = uint32_t(qatCounts[i] / 32);
                const uint32_t g = std::min<uint32_t>(ceilDiv(blocks, 256), kMaxGroups);
                struct { uint32_t blocks, stride, r1, r2; } p{ blocks, g * 256u, 0, 0 };
                std::vector<ParamArena::TensorRef> t = { { P.address + qatMap[i].first * 4, DT_FP32 },
                                                         { Pq.address + qatMap[i].second * 4, DT_FP32 } };
                rec.dispatch(ctx.pipeline(kOpsSrc, "sml_train_mxfp4_fakequant_f32"), arena->params(t, p), g);
            }
            rec.submitAndWait();
        }
        double stepLoss = 0.0;
        for (uint32_t micro = 0; micro < opts.accum; ++micro) {
            loadBatch(trainFile, trainTokens, 0x7121ull, step * opts.accum + micro);
            stepLoss += runMicroStep(true);
        }
        stepLoss /= opts.accum;
        const float lr = lrAt(step);
        const float stepScale = lossScale;
        {
            arena->reset();
            ComputeContext::Recorder rec(ctx);
            recordOptimizer(rec, lr, step);
            rec.submitAndWait();
        }
        badInARow = updateLossScale() ? 0 : badInARow + 1;
        if (badInARow >= 40) throw std::runtime_error("gradients not finite for 40 steps in a row");
        ++step;
        if (opts.profileSteps && step == profEnd) {
            const double secs = std::chrono::duration<double>(clock::now() - tProf).count();
            ctx.setProfiling(false);
            ctx.printProfile(std::cout, double(opts.profileSteps));
            std::cout << "wall clock: " << std::fixed << std::setprecision(1) << 1e3 * secs / opts.profileSteps
                      << " ms/step, " << std::setprecision(0) << double(opts.profileSteps) * tokensPerStep / secs
                      << " tok/s (with profiling)\n";
            return;
        }
        lossAcc += stepLoss;
        ++lossN;
        if (!std::isfinite(stepLoss)) throw std::runtime_error("loss is not finite at step " + std::to_string(step));

        double valLoss = -1.0;
        const bool doEval = haveVal && opts.evalEvery && !opts.profileSteps &&
                            (step % opts.evalEvery == 0 || step == opts.maxSteps);
        if (doEval) {
            valLoss = evaluate();
            bestVal = std::min(bestVal, valLoss);
        }
        if (step % opts.logEvery == 0 || doEval || step == opts.maxSteps) {
            const double secs = std::chrono::duration<double>(clock::now() - tLog).count();
            tLog = clock::now();
            const float gnorm = std::sqrt(static_cast<const float*>(normSq.mapped)[0]) / (float(opts.accum) * stepScale);
            std::vector<float> auxHost(size_t(N) * cfg.numLayers);
            ctx.download(aux, 0, auxHost.data(), auxHost.size() * 4);
            double auxSum = 0.0;
            for (float v : auxHost) auxSum += v;
            auxSum /= std::max(1u, cfg.numLayers) * std::max(opts.auxCoef, 1e-12f) * stepScale;  // ~1.0 = perfectly balanced
            const double tps = double(lossN) * tokensPerStep / std::max(secs, 1e-9);
            const double bpb = valLoss >= 0 && opts.bytesPerToken > 0 ? valLoss / (std::log(2.0) * opts.bytesPerToken) : -1.0;
            std::cout << "step " << step << "  loss " << std::fixed << std::setprecision(4) << lossAcc / lossN;
            if (valLoss >= 0) std::cout << "  val " << valLoss;
            if (bpb >= 0) std::cout << " (" << bpb << " bpb)";
            std::cout << "  lr " << std::scientific << std::setprecision(2) << lr << std::fixed << "  gnorm "
                      << std::setprecision(3) << gnorm << "  balance " << std::setprecision(3) << auxSum << "  "
                      << std::setprecision(0) << tps << " tok/s" << (qatActive ? "  [qat]" : "") << "\n"
                      << std::flush;
            std::ofstream(logPath, std::ios::app) << step << "," << lossAcc / lossN << "," << valLoss << "," << lr << ","
                                                  << gnorm << "," << tps << "," << auxSum << "," << bpb << ","
                                                  << stepScale << "\n";
            lossAcc = 0.0;
            lossN = 0;
        }
        const double sinceCkpt = std::chrono::duration<double>(clock::now() - lastCkpt).count();
        if (sinceCkpt >= opts.ckptMinutes * 60.0 && !opts.profileSteps) {
            saveCheckpoint(step, bestVal);
            lastCkpt = clock::now();
        }
    }
    if (gStop) std::cout << "stop requested: saving checkpoint\n";
    saveCheckpoint(step, bestVal);
    if (mirrorThread.joinable()) mirrorThread.join();
    if (step >= opts.maxSteps && opts.exportAtEnd) {
        std::vector<float> host(lay.total);
        ctx.download(P, 0, host.data(), lay.total * 4);
        const std::string exp = (fs::path(opts.outDir) / "export").string();
        exportGptOssCheckpoint(cfg, lay, host, exp, opts.tokenizer, opts.tokenizerFile.empty() ? "" : runTokenizer);
        std::cout << "exported inference checkpoint to " << exp << "\n";
        if (!opts.mirrorDir.empty()) {
            const fs::path m = fs::path(opts.mirrorDir) / "export";
            fs::create_directories(m);
            fs::copy(exp, m, fs::copy_options::overwrite_existing | fs::copy_options::recursive);
            std::cout << "copied export to " << m.string() << "\n";
        }
    }
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------
namespace {

std::string readText(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + path);
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

std::vector<uint32_t> parseIdList(const std::string& text) {
    std::vector<uint32_t> ids;
    std::string cur;
    for (char c : text) {
        if (c >= '0' && c <= '9') cur += c;
        else if (!cur.empty()) {
            ids.push_back(uint32_t(std::stoul(cur)));
            cur.clear();
        }
    }
    if (!cur.empty()) ids.push_back(uint32_t(std::stoul(cur)));
    return ids;
}

std::string findCheckpoint(const std::string& p) {
    if (fs::is_directory(p)) {
        CheckpointInfo best;
        for (const char* n : { "latest.nckpt", "prev.nckpt" }) {
            CheckpointInfo i = probeCheckpoint((fs::path(p) / n).string());
            if (i.valid && (!best.valid || i.step > best.step)) best = i;
        }
        if (!best.valid) throw std::runtime_error("no valid checkpoint in " + p);
        return best.path;
    }
    return p;
}

void writeFloats(const std::string& path, const std::vector<float>& v) {
    std::ofstream o(path, std::ios::binary);
    o.write(reinterpret_cast<const char*>(v.data()), std::streamsize(v.size() * 4));
}

const char* kUsage =
    "usage:\n"
    "  NeonML --gpt-oss-train train --config <model.json> --data <dir> [--out run] [--mirror <dir>]\n"
    "         [--batch 16] [--seq 512] [--accum 1] [--steps 20000] [--warmup 500] [--lr 1e-3] [--min-lr 1e-4]\n"
    "         [--wd 0.1] [--clip 1.0] [--aux 0.01] [--qat-start <step>|-1 (auto)|-2 (off)] [--eval-every 500]\n"
    "         [--eval-batches 20] [--log-every 10] [--ckpt-minutes 30] [--seed 1234] [--lm-chunk 2048] [--fresh]\n"
    "         [--precision auto|fp16|fp32] [--loss-scale 65536] [--profile <steps>]\n"
    "         [--schedule wsd|cosine|linear] [--decay-frac 0.2]\n"
    "  NeonML --gpt-oss-train export --ckpt <file or run dir> --out <dir>\n"
    "  NeonML --gpt-oss-train gradcheck --config <model.json> --tokens-file <ids> --batch B --seq T --out <dir> [--aux 0]\n"
    "         [--precision fp32 (default here)|fp16]\n"
    "  common: --shaders <shaders/ML dir> --device <name substring>\n";

}  // namespace

int runGptOssTrainCli(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << kUsage;
        return 2;
    }
    const std::string mode = argv[2];
    TrainOptions o;
    std::string configPath, shaders = "shaders/ML", device, ckpt, out, tokensFile;
    for (int i = 3; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::runtime_error("missing value after " + a);
            return argv[++i];
        };
        if (a == "--config") configPath = next();
        else if (a == "--data") o.dataDir = next();
        else if (a == "--out") out = next();
        else if (a == "--mirror") o.mirrorDir = next();
        else if (a == "--batch") o.batch = uint32_t(std::stoul(next()));
        else if (a == "--seq") o.seq = uint32_t(std::stoul(next()));
        else if (a == "--accum") o.accum = uint32_t(std::stoul(next()));
        else if (a == "--steps") o.maxSteps = std::stoull(next());
        else if (a == "--warmup") o.warmup = std::stoull(next());
        else if (a == "--lr") o.lr = std::stof(next());
        else if (a == "--min-lr") o.minLr = std::stof(next());
        else if (a == "--schedule") o.schedule = next();
        else if (a == "--decay-frac") o.decayFrac = std::stof(next());
        else if (a == "--wd") o.weightDecay = std::stof(next());
        else if (a == "--clip") o.clip = std::stof(next());
        else if (a == "--aux") o.auxCoef = std::stof(next());
        else if (a == "--qat-start") o.qatStart = std::stoll(next());
        else if (a == "--eval-every") o.evalEvery = uint32_t(std::stoul(next()));
        else if (a == "--eval-batches") o.evalBatches = uint32_t(std::stoul(next()));
        else if (a == "--log-every") o.logEvery = std::max(1u, uint32_t(std::stoul(next())));
        else if (a == "--ckpt-minutes") o.ckptMinutes = std::stod(next());
        else if (a == "--seed") o.seed = std::stoull(next());
        else if (a == "--lm-chunk") o.lmChunk = uint32_t(std::stoul(next()));
        else if (a == "--fresh") o.fresh = true;
        else if (a == "--no-export") o.exportAtEnd = false;
        else if (a == "--precision") o.precision = next();
        else if (a == "--loss-scale") o.lossScale = std::stof(next());
        else if (a == "--profile") o.profileSteps = uint32_t(std::stoul(next()));
        else if (a == "--ckpt") ckpt = next();
        else if (a == "--tokens-file") tokensFile = next();
        else if (a == "--shaders") shaders = next();
        else if (a == "--device") device = next();
        else throw std::runtime_error("unknown option " + a + "\n" + kUsage);
    }
    if (!out.empty()) o.outDir = out;

    if (mode == "export") {
        if (ckpt.empty() || out.empty()) throw std::runtime_error("export needs --ckpt and --out");
        const std::string path = findCheckpoint(ckpt);
        std::vector<float> params;
        std::string tok;
        const GptOssConfig cfg = readCheckpointParams(path, params, &tok);
        exportGptOssCheckpoint(cfg, ParamLayout::build(cfg), params, out, tok,
                               (fs::path(path).parent_path() / "tokenizer.tiktoken").string());
        std::cout << "exported " << path << " -> " << out << "\n";
        return 0;
    }

    GptOssConfig cfg;
    if (mode == "train") {
        if (o.dataDir.empty()) throw std::runtime_error("train needs --data");
        if (o.schedule != "cosine" && o.schedule != "linear" && o.schedule != "wsd")
            throw std::runtime_error("--schedule must be cosine, linear or wsd");
        const fs::path meta = fs::path(o.dataDir) / "meta.json";
        uint32_t dataVocab = 0;
        if (fs::exists(meta)) {
            JsonValue j = JsonValue::parse(readText(meta.string()));
            if (j.has("tokenizer")) o.tokenizer = j["tokenizer"].asString();
            if (j.has("tokenizer_file")) o.tokenizerFile = (fs::path(o.dataDir) / j["tokenizer_file"].asString()).string();
            if (j.has("vocab_size")) dataVocab = uint32_t(j["vocab_size"].asInt());
            o.bytesPerToken = j.numberOr("bytes_per_token", 0.0);
        }
        // A run that already has a checkpoint keeps its model config.
        std::string resumeFrom;
        if (!o.fresh)
            for (const std::string& d : { o.outDir, o.mirrorDir }) {
                if (d.empty()) continue;
                for (const char* n : { "latest.nckpt", "prev.nckpt" }) {
                    CheckpointInfo i = probeCheckpoint((fs::path(d) / n).string());
                    if (i.valid) resumeFrom = i.path;
                }
            }
        if (!resumeFrom.empty()) {
            std::FILE* f = std::fopen(resumeFrom.c_str(), "rb");
            CkptHeader h;
            readHeader(f, resumeFrom, h);
            std::fclose(f);
            cfg = configFromJson(JsonValue::parse(h.json));
        } else {
            if (configPath.empty()) throw std::runtime_error("train needs --config for a new run");
            cfg = configFromJson(JsonValue::parse(readText(configPath)));
            if (cfg.vocabSize == 0) {  // "vocab_size": 0 = the data's vocabulary, padded to a multiple of 64
                if (!dataVocab) throw std::runtime_error("vocab_size 0 needs meta.json with vocab_size in --data");
                cfg.vocabSize = uint32_t(roundUp(dataVocab, 64));
                std::cout << "vocab_size from data: " << dataVocab << " -> " << cfg.vocabSize << "\n";
            }
        }
        if (fs::exists(meta)) {
            JsonValue j = JsonValue::parse(readText(meta.string()));
            if (j.has("vocab_size") && uint32_t(j["vocab_size"].asInt()) > cfg.vocabSize)
                throw std::runtime_error("data vocabulary (" + std::to_string(j["vocab_size"].asInt()) +
                                         ") is larger than the model's vocab_size");
        }
        ComputeContext ctx(shaders, device);
        std::cout << "device: " << ctx.deviceName() << " (" << ctx.deviceLocalBytes() / (1ull << 20) << " MiB)\n";
        GptOssTrainer trainer(ctx, cfg, o);
        trainer.train();
        return 0;
    }
    if (mode == "gradcheck") {
        if (configPath.empty() || tokensFile.empty() || out.empty())
            throw std::runtime_error("gradcheck needs --config, --tokens-file and --out");
        cfg = configFromJson(JsonValue::parse(readText(configPath)));
        o.qatStart = -2;
        if (o.schedule != "cosine" && o.schedule != "linear" && o.schedule != "wsd")
            throw std::runtime_error("--schedule must be cosine, linear or wsd");
        if (o.precision == "auto") o.precision = "fp32";  // exact gradients unless fp16 is asked for
        ComputeContext ctx(shaders, device);
        GptOssTrainer trainer(ctx, cfg, o);
        std::vector<float> params, grads;
        const float loss = trainer.gradientCheck(parseIdList(readText(tokensFile)), params, grads);
        fs::create_directories(out);
        writeFloats((fs::path(out) / "params.bin").string(), params);
        writeFloats((fs::path(out) / "grads.bin").string(), grads);
        std::ofstream lj(fs::path(out) / "layout.json");
        lj << "{\"loss\": " << std::setprecision(9) << loss << ", \"tensors\": [";
        const auto& L = trainer.layout();
        for (size_t i = 0; i < L.tensors.size(); ++i) {
            const auto& t = L.tensors[i];
            lj << (i ? "," : "") << "\n  {\"name\": \"" << t.name << "\", \"offset\": " << t.offset << ", \"shape\": [";
            for (size_t d = 0; d < t.shape.size(); ++d) lj << (d ? "," : "") << t.shape[d];
            lj << "]}";
        }
        lj << "]}\n";
        std::cout << "gradcheck: loss " << std::setprecision(7) << loss << ", wrote " << out << "\n";
        return 0;
    }
    std::cerr << kUsage;
    return 2;
}

}  // namespace lve::ml
