#include "memorilla.hpp"

#include "ml_json.hpp"
#include "ml_safetensors.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace lve::ml {

namespace fs = std::filesystem;

namespace {

const char* kMemSrc = "slangml/memorilla/memorilla_f32.slang";
const char* kGemmSrc = "slangml/train/gemm_train_f32.slang";
const char* kGemmTcSrc = "slangml/train/gemm_train_coopmat.slang";
const char* kOpsSrc = "slangml/train/train_ops_f32.slang";

constexpr uint32_t DT_FP32 = 0, DT_I32 = 5;
constexpr uint32_t GEMM_BIAS = 1, GEMM_RESID = 2, GEMM_ACCUM = 4;
constexpr uint32_t TILE = 128;
constexpr uint32_t kMaxGroups = 65535;
constexpr uint32_t MEM_ACCUM = 1, MEM_MASK = 2, MEM_TRANSPOSED = 4, MEM_NO_DX = 8;
constexpr char kStoreMagic[8] = { 'N', 'E', 'O', 'N', 'M', 'E', 'M', '1' };
constexpr char kStoreEnd[8] = { 'N', 'E', 'O', 'N', 'M', 'E', 'N', 'D' };
constexpr char kSeqMagic[8] = { 'N', 'E', 'O', 'N', 'S', 'E', 'Q', '1' };

uint32_t ceilDiv(uint64_t a, uint64_t b) { return uint32_t((a + b - 1) / b); }
uint64_t roundUp(uint64_t a, uint64_t b) { return (a + b - 1) / b * b; }
uint32_t strideGroups(uint64_t n) { return std::max<uint32_t>(1, std::min<uint32_t>(ceilDiv(n, 256), kMaxGroups)); }

uint64_t splitmix(uint64_t& s) {
    uint64_t z = (s += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}
uint64_t hashName(const std::string& s) {
    uint64_t h = 1469598103934665603ull;
    for (char c : s) h = (h ^ uint8_t(c)) * 1099511628211ull;
    return h;
}
double uniform01(uint64_t& s) { return (double(splitmix(s) >> 11) + 0.5) * (1.0 / 9007199254740992.0); }

void writeAll(std::FILE* f, const void* p, size_t n, const std::string& path) {
    if (n && std::fwrite(p, 1, n, f) != n) throw std::runtime_error("write failed: " + path);
}
void readAll(std::FILE* f, void* p, size_t n, const std::string& path) {
    if (n && std::fread(p, 1, n, f) != n) throw std::runtime_error("truncated file: " + path);
}
std::string readText(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + path);
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}
std::string jsonEscape(const std::string& s) {
    std::string o;
    for (char c : s) {
        if (c == '"' || c == '\\') o += '\\';
        o += c;
    }
    return o;
}
std::string strOr(const JsonValue& j, const std::string& key, const std::string& fallback) {
    return j.has(key) ? j[key].asString() : fallback;
}

}  // namespace

// ---------------------------------------------------------------------------
// Config and layout
// ---------------------------------------------------------------------------
MemorillaConfig MemorillaConfig::fromJson(const std::string& text) {
    const JsonValue j = JsonValue::parse(text);
    MemorillaConfig c;
    c.slots = uint32_t(j.numberOr("slots", 16));
    c.encoderDim = uint32_t(j.numberOr("encoder_dim", 0));
    c.llmDim = uint32_t(j.numberOr("llm_dim", 0));
    c.heads = uint32_t(j.numberOr("heads", 8));
    c.refineLayers = uint32_t(j.numberOr("refine_layers", 2));
    c.ffnMult = uint32_t(j.numberOr("ffn_mult", 4));
    c.eps = float(j.numberOr("eps", 1.0e-5));
    c.encoder = strOr(j, "encoder", "self");
    c.encoderLayer = int32_t(j.numberOr("encoder_layer", -1));
    c.decoder = strOr(j, "decoder", "");
    return c;
}

std::string MemorillaConfig::toJson() const {
    std::ostringstream o;
    o << "{\n  \"slots\": " << slots << ",\n  \"encoder_dim\": " << encoderDim << ",\n  \"llm_dim\": " << llmDim
      << ",\n  \"heads\": " << heads << ",\n  \"refine_layers\": " << refineLayers << ",\n  \"ffn_mult\": " << ffnMult
      << ",\n  \"eps\": " << std::setprecision(9) << eps << ",\n  \"encoder\": \"" << jsonEscape(encoder)
      << "\",\n  \"encoder_layer\": " << encoderLayer << ",\n  \"decoder\": \"" << jsonEscape(decoder) << "\"\n}\n";
    return o.str();
}

ParamLayout buildMemorillaLayout(const MemorillaConfig& c) {
    if (!c.encoderDim || !c.llmDim) throw std::runtime_error("memorilla: encoder_dim and llm_dim must be set");
    if (c.encoderDim % c.heads) throw std::runtime_error("memorilla: encoder_dim must be a multiple of heads");
    if (c.encoderDim % 4 || c.llmDim % 4) throw std::runtime_error("memorilla: dims must be multiples of 4");
    const uint64_t h = c.encoderDim, F = c.ffnDim(), hL = c.llmDim, K = c.slots;
    ParamLayout L;
    auto add = [&](const std::string& name, std::vector<uint64_t> shape, bool decay) {
        ParamTensor t;
        t.name = name;
        t.shape = shape;
        t.count = 1;
        for (uint64_t s : shape) t.count *= s;
        t.offset = L.total;
        t.decay = decay;
        t.exportAs = ParamTensor::Export::F32;
        L.total = roundUp(L.total + t.count, 4);
        L.tensors.push_back(t);
    };
    // Decayed (linear weights) first, then everything else.
    add("cond.v.weight", { h, h }, true);
    add("cond.o.weight", { h, h }, true);
    for (uint32_t l = 0; l < c.refineLayers; ++l) {
        const std::string b = "block." + std::to_string(l) + ".";
        for (const char* p : { "attn.q", "attn.k", "attn.v", "attn.o" }) add(b + p + ".weight", { h, h }, true);
        add(b + "ffn.up.weight", { F, h }, true);
        add(b + "ffn.down.weight", { h, F }, true);
    }
    add("proj.weight", { hL, h }, true);
    L.decayEnd = L.total;
    add("slots", { K, h }, false);
    add("cond.v.bias", { h }, false);
    add("cond.o.bias", { h }, false);
    add("cond.norm.scale", { h }, false);
    add("cond.norm.bias", { h }, false);
    add("docs.norm.scale", { h }, false);
    add("docs.norm.bias", { h }, false);
    for (uint32_t l = 0; l < c.refineLayers; ++l) {
        const std::string b = "block." + std::to_string(l) + ".";
        add(b + "norm1.scale", { h }, false);
        add(b + "norm1.bias", { h }, false);
        for (const char* p : { "attn.q", "attn.k", "attn.v", "attn.o" }) add(b + p + ".bias", { h }, false);
        add(b + "norm2.scale", { h }, false);
        add(b + "norm2.bias", { h }, false);
        add(b + "ffn.up.bias", { F }, false);
        add(b + "ffn.down.bias", { h }, false);
    }
    add("out.norm.scale", { h }, false);
    add("out.norm.bias", { h }, false);
    add("proj.bias", { hL }, false);
    return L;
}

uint64_t memorillaPaperParamCount(const MemorillaConfig& c) {
    uint64_t n = 0;
    for (const auto& t : buildMemorillaLayout(c).tensors) n += t.count;
    const uint64_t h = c.encoderDim;
    return n + 2 * (h * h + h);  // the paper's unused W_q, W_k (+ biases) of the conditioning attention
}

std::string decoderFingerprint(const GptOssConfig& c) {
    std::ostringstream o;
    o << "gpt-oss L" << c.numLayers << " H" << c.hidden << " I" << c.intermediate << " E" << c.numExperts << "k"
      << c.expertsPerToken << " V" << c.vocabSize << " heads" << c.numHeads << "/" << c.numKVHeads << " win"
      << c.slidingWindow;
    return o.str();
}

// ---------------------------------------------------------------------------
// Store
// ---------------------------------------------------------------------------
size_t MemoryStore::liveCount() const {
    size_t n = 0;
    for (const Doc& d : docs) n += (d.flags & 1u) ? 0 : 1;
    return n;
}

void MemoryStore::add(const float* v, uint64_t id, uint32_t tokens) {
    if (!dim) throw std::runtime_error("memory store: dim not set");
    vectors.insert(vectors.end(), v, v + dim);
    docs.push_back({ id, tokens, 0u });
}

bool MemoryStore::remove(uint64_t id) {
    bool found = false;
    for (Doc& d : docs)
        if (d.id == id && !(d.flags & 1u)) {
            d.flags |= 1u;
            found = true;
        }
    return found;
}

void MemoryStore::compact() {
    std::vector<float> v;
    std::vector<Doc> d;
    for (size_t i = 0; i < docs.size(); ++i)
        if (!(docs[i].flags & 1u)) {
            v.insert(v.end(), vector(i), vector(i) + dim);
            d.push_back(docs[i]);
        }
    vectors.swap(v);
    docs.swap(d);
}

std::vector<float> MemoryStore::liveVectors() const {
    std::vector<float> v;
    v.reserve(liveCount() * dim);
    for (size_t i = 0; i < docs.size(); ++i)
        if (!(docs[i].flags & 1u)) v.insert(v.end(), vector(i), vector(i) + dim);
    return v;
}

MemoryStore MemoryStore::load(const std::string& path) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("cannot open " + path);
    MemoryStore s;
    try {
        char magic[8];
        uint32_t version = 0, len = 0;
        readAll(f, magic, 8, path);
        if (std::memcmp(magic, kStoreMagic, 8) != 0) throw std::runtime_error(path + " is not a .nmem store");
        readAll(f, &version, 4, path);
        readAll(f, &len, 4, path);
        if (version != 1 || len > (1u << 20)) throw std::runtime_error(path + ": unsupported store version");
        std::string json(len, '\0');
        readAll(f, json.data(), len, path);
        const JsonValue j = JsonValue::parse(json);
        s.encoder = strOr(j, "encoder", "self");
        s.encoderLayer = int32_t(j.numberOr("encoder_layer", -1));
        s.pooling = strOr(j, "pooling", "mean");
        s.decoder = strOr(j, "decoder", "");
        s.dim = uint32_t(j.numberOr("dim", 0));
        if (strOr(j, "dtype", "f32") != "f32") throw std::runtime_error(path + ": only f32 stores are supported");
        uint64_t count = 0;
        readAll(f, &count, 8, path);
        s.vectors.resize(count * s.dim);
        readAll(f, s.vectors.data(), s.vectors.size() * 4, path);
        s.docs.resize(count);
        for (Doc& d : s.docs) {
            readAll(f, &d.id, 8, path);
            readAll(f, &d.tokens, 4, path);
            readAll(f, &d.flags, 4, path);
        }
        readAll(f, magic, 8, path);
        if (std::memcmp(magic, kStoreEnd, 8) != 0) throw std::runtime_error(path + ": bad end marker");
    } catch (...) {
        std::fclose(f);
        throw;
    }
    std::fclose(f);
    return s;
}

void MemoryStore::save(const std::string& path) const {
    std::ostringstream o;
    o << "{\"encoder\": \"" << jsonEscape(encoder) << "\", \"encoder_layer\": " << encoderLayer << ", \"pooling\": \""
      << jsonEscape(pooling) << "\", \"dim\": " << dim << ", \"dtype\": \"f32\", \"decoder\": \"" << jsonEscape(decoder)
      << "\"}";
    const std::string json = o.str();
    const std::string tmp = path + ".tmp";
    std::FILE* f = std::fopen(tmp.c_str(), "wb");
    if (!f) throw std::runtime_error("cannot write " + tmp);
    const uint32_t version = 1, len = uint32_t(json.size());
    const uint64_t count = docs.size();
    writeAll(f, kStoreMagic, 8, tmp);
    writeAll(f, &version, 4, tmp);
    writeAll(f, &len, 4, tmp);
    writeAll(f, json.data(), json.size(), tmp);
    writeAll(f, &count, 8, tmp);
    writeAll(f, vectors.data(), vectors.size() * 4, tmp);
    for (const Doc& d : docs) {
        writeAll(f, &d.id, 8, tmp);
        writeAll(f, &d.tokens, 4, tmp);
        writeAll(f, &d.flags, 4, tmp);
    }
    writeAll(f, kStoreEnd, 8, tmp);
    if (std::fflush(f) != 0 || std::fclose(f) != 0) throw std::runtime_error("write failed: " + tmp);
    std::error_code ec;
    fs::remove(path, ec);
    fs::rename(tmp, path);
}

std::vector<std::vector<uint32_t>> readSequencesFile(const std::string& path) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("cannot open " + path);
    std::vector<std::vector<uint32_t>> out;
    try {
        char magic[8];
        uint64_t count = 0;
        readAll(f, magic, 8, path);
        if (std::memcmp(magic, kSeqMagic, 8) != 0) throw std::runtime_error(path + " is not a sequences file");
        readAll(f, &count, 8, path);
        out.resize(count);
        for (auto& s : out) {
            uint32_t n = 0;
            readAll(f, &n, 4, path);
            s.resize(n);
            readAll(f, s.data(), size_t(n) * 4, path);
        }
    } catch (...) {
        std::fclose(f);
        throw;
    }
    std::fclose(f);
    return out;
}

// ---------------------------------------------------------------------------
// Module
// ---------------------------------------------------------------------------
MemorillaModule::MemorillaModule(ComputeContext& c, const MemorillaConfig& config, bool tc, bool train)
    : ctx(c), cfg(config), lay(buildMemorillaLayout(config)), tensorCores(tc), trainable(train) {
    arena = std::make_unique<ParamArena>(ctx, 4u << 20);
    auto dev = [&](uint64_t floats) { return ctx.createBuffer(std::max<uint64_t>(floats, 4) * 4, MemKind::Device); };
    P = dev(lay.total);
    if (trainable) {
        G = dev(lay.total);
        Mo = dev(lay.total);
        Ve = dev(lay.total);
        ctx.fillZero(G);
        ctx.fillZero(Mo);
        ctx.fillZero(Ve);
        normPartial = dev(kMaxGroups + 1);
        normSq = ctx.createBuffer(16, MemKind::Host);
    }
}

MemorillaModule::~MemorillaModule() {
    std::vector<GpuBuffer*> all = { &P, &G, &Mo, &Ve, &docsIn, &countsIn, &queryIn, &u1, &u, &pre, &cm, &cr, &zn,
                                    &dmn, &drs, &rf, &fm, &fr, &M, &dM, &dR, &dN, &dQ, &dK, &dV, &dZn, &dP, &dO,
                                    &dH, &gx, &dPre, &du, &du1, &normPartial, &normSq, &splitBuf };
    for (GpuBuffer* b : all) ctx.destroyBuffer(*b);
    for (auto& r : res) ctx.destroyBuffer(r);
    for (auto& b : blocks)
        for (GpuBuffer* x : { &b.n1, &b.m1, &b.r1, &b.q, &b.k, &b.v, &b.P, &b.O, &b.n2, &b.m2, &b.r2, &b.hu, &b.ha })
            ctx.destroyBuffer(*x);
}

void MemorillaModule::allocate(uint32_t B, uint32_t N) {
    if (capB) throw std::runtime_error("memorilla: allocate called twice");
    capB = std::max(1u, B);
    capN = std::max(1u, N);
    const uint64_t h = cfg.encoderDim, F = cfg.ffnDim(), hL = cfg.llmDim, K = cfg.slots, nh = cfg.heads;
    const uint64_t rk = uint64_t(capB) * K, rn = uint64_t(capB) * capN;
    auto dev = [&](uint64_t floats) { return ctx.createBuffer(std::max<uint64_t>(floats, 4) * 4, MemKind::Device); };
    auto host = [&](uint64_t floats) { return ctx.createBuffer(std::max<uint64_t>(floats, 4) * 4, MemKind::Host); };
    docsIn = host(rn * h);
    countsIn = host(capB);
    queryIn = host(uint64_t(capB) * h);
    std::memset(docsIn.mapped, 0, docsIn.size);
    std::memset(countsIn.mapped, 0, countsIn.size);
    std::memset(queryIn.mapped, 0, queryIn.size);
    u1 = dev(capB * h);
    u = dev(capB * h);
    pre = dev(rk * h);
    cm = dev(rk);
    cr = dev(rk);
    zn = dev(rn * h);
    dmn = dev(rn);
    drs = dev(rn);
    res.resize(2 * cfg.refineLayers + 1);
    for (auto& r : res) r = dev(rk * h);
    blocks.resize(cfg.refineLayers);
    for (auto& b : blocks) {
        b.n1 = dev(rk * h);
        b.m1 = dev(rk);
        b.r1 = dev(rk);
        b.q = dev(rk * h);
        b.k = dev(rn * h);
        b.v = dev(rn * h);
        b.P = dev(capB * nh * K * capN);
        b.O = dev(rk * h);
        b.n2 = dev(rk * h);
        b.m2 = dev(rk);
        b.r2 = dev(rk);
        b.hu = dev(rk * F);
        b.ha = dev(rk * F);
    }
    rf = dev(rk * h);
    fm = dev(rk);
    fr = dev(rk);
    M = dev(rk * hL);
    if (trainable) {
        dM = dev(rk * hL);
        dR = dev(rk * h);
        dN = dev(rk * h);
        dQ = dev(rk * h);
        dK = dev(rn * h);
        dV = dev(rn * h);
        dZn = dev(rn * h);
        dP = dev(capB * nh * K * capN);
        dO = dev(rk * h);
        dH = dev(rk * F);
        gx = dev(std::max(rn, rk) * h);
        dPre = dev(rk * h);
        du = dev(capB * h);
        du1 = dev(capB * h);
    }
    splitCap = uint32_t(8 * std::max({ h * h, F * h, hL * h }));
    splitBuf = dev(splitCap);
}

uint64_t MemorillaModule::pAddr(const std::string& name) const { return P.address + lay.at(name).offset * 4; }
uint64_t MemorillaModule::gAddr(const std::string& name) const { return G.address + lay.at(name).offset * 4; }

void MemorillaModule::initParams(uint64_t seed, float projStd) {
    std::vector<float> host(lay.total, 0.0f);
    for (const auto& t : lay.tensors) {
        float* p = host.data() + t.offset;
        const std::string& n = t.name;
        auto endsWith = [&](const char* s) {
            const size_t k = std::strlen(s);
            return n.size() >= k && n.compare(n.size() - k, k, s) == 0;
        };
        uint64_t s = seed ^ hashName(n);
        if (endsWith(".scale")) {  // every LayerNorm gain (norm, norm1, norm2)
            std::fill(p, p + t.count, 1.0f);
        } else if (n == "slots") {  // Xavier uniform
            const double bound = std::sqrt(6.0 / double(t.shape[0] + t.shape[1]));
            for (uint64_t i = 0; i < t.count; ++i) p[i] = float((2.0 * uniform01(s) - 1.0) * bound);
        } else if (n == "proj.weight") {  // normal, scaled to the decoder's embedding rows
            for (uint64_t i = 0; i < t.count; i += 2) {
                const double r = std::sqrt(-2.0 * std::log(uniform01(s)));
                const double a = 6.283185307179586 * uniform01(s);
                p[i] = float(projStd * r * std::cos(a));
                if (i + 1 < t.count) p[i + 1] = float(projStd * r * std::sin(a));
            }
        } else if (t.decay) {  // linear weights: U(-1/sqrt(in), 1/sqrt(in)) (PyTorch's default)
            const double bound = 1.0 / std::sqrt(double(t.shape[1]));
            for (uint64_t i = 0; i < t.count; ++i) p[i] = float((2.0 * uniform01(s) - 1.0) * bound);
        }
        // biases and norm shifts stay 0
    }
    setParams(host);
}

void MemorillaModule::setParams(const std::vector<float>& host) {
    if (host.size() != lay.total) throw std::runtime_error("memorilla: parameter count mismatch");
    ctx.upload(P, 0, host.data(), host.size() * 4);
}

std::vector<float> MemorillaModule::params() const {
    std::vector<float> h(lay.total);
    ctx.download(P, 0, h.data(), h.size() * 4);
    return h;
}

std::vector<float> MemorillaModule::grads() const {
    std::vector<float> h(lay.total);
    if (G) ctx.download(G, 0, h.data(), h.size() * 4);
    return h;
}

void MemorillaModule::zeroGrads() {
    if (G) ctx.fillZero(G);
}

// ---- recording helpers ----
struct MemGemmParams {  // TrainGemmParams (gemm_train_*.slang)
    uint32_t M, N, K, flags, sAm, sAk, sBk, sBn, ldC, mode, kChunk, strideBG, strideBiasG, strideCG;
    float alpha;
    uint32_t reserved;
};

// Same operand rules as GptOssTrainer::gemm (tensor cores when the layout allows,
// split-K for weight gradients with few output tiles), plain (mode 0) GEMMs only.
void MemorillaModule::gemm(ComputeContext::Recorder& rec, const GemmArgs& a) {
    bool tc = false;
    std::string entry = "sml_train_gemm_fwd_f32";
    if (tensorCores) {
        const bool aK = a.sAk == 1, aM = !aK && a.sAm == 1, bK = a.sBk == 1, bN = !bK && a.sBn == 1;
        const uint32_t ldA = aK ? a.sAm : a.sAk, ldB = bK ? a.sBn : a.sBk;
        tc = (aK || aM) && (bK || bN) && ldA % 4 == 0 && ldB % 4 == 0 && a.ldC % 4 == 0 && a.A % 16 == 0 &&
             a.B % 16 == 0 && a.C % 16 == 0 && a.bias % 16 == 0 && a.R % 16 == 0 && (a.M * a.N) % 4 == 0;
        if (tc) entry = std::string("sml_train_gemm_tc_") + (aK ? "k" : "m") + (bK ? "k" : "n");
    }
    const Pipeline& pipe = tc ? ctx.pipeline(kGemmTcSrc, entry) : ctx.pipeline(kGemmSrc, entry);
    MemGemmParams p{};
    p.M = a.M; p.N = a.N; p.K = a.K; p.flags = a.flags;
    p.sAm = a.sAm; p.sAk = a.sAk; p.sBk = a.sBk; p.sBn = a.sBn; p.ldC = a.ldC;
    p.alpha = 1.0f;
    const uint32_t gx = ceilDiv(a.N, TILE), gy = ceilDiv(a.M, TILE);
    uint64_t C = a.C;
    uint32_t splits = 1;
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
    std::vector<ParamArena::TensorRef> t = { { a.A, DT_FP32 }, { a.B, DT_FP32 }, { C, DT_FP32 },
                                             { a.bias, DT_FP32 }, { a.R, DT_FP32 }, { 0, DT_I32 } };
    rec.dispatch(pipe, arena->params(t, p), gx, gy, splits);
    if (splits > 1) {
        struct { uint32_t n, splits, stride, accum; } r{ a.M * a.N, splits, a.M * a.N, (a.flags & GEMM_ACCUM) ? 1u : 0u };
        std::vector<ParamArena::TensorRef> rt = { { splitBuf.address, DT_FP32 }, { a.C, DT_FP32 } };
        rec.dispatch(ctx.pipeline(kGemmSrc, "sml_train_splitk_reduce_f32"), arena->params(rt, r), ceilDiv(r.n, 256));
    }
}

void MemorillaModule::linear(ComputeContext::Recorder& rec, uint64_t X, const std::string& w, uint64_t Y,
                             uint32_t rows, uint32_t in, uint32_t out, uint64_t resid) {
    GemmArgs a;
    a.A = X; a.B = pAddr(w + ".weight"); a.C = Y; a.bias = pAddr(w + ".bias"); a.R = resid;
    a.M = rows; a.N = out; a.K = in;
    a.sAm = in; a.sAk = 1; a.sBk = 1; a.sBn = in; a.ldC = out;
    a.flags = GEMM_BIAS | (resid ? GEMM_RESID : 0);
    gemm(rec, a);
}

void MemorillaModule::linearDX(ComputeContext::Recorder& rec, uint64_t dY, const std::string& w, uint64_t dX,
                               uint32_t rows, uint32_t in, uint32_t out, bool accum) {
    GemmArgs a;
    a.A = dY; a.B = pAddr(w + ".weight"); a.C = dX;
    a.M = rows; a.N = in; a.K = out;
    a.sAm = out; a.sAk = 1; a.sBk = in; a.sBn = 1; a.ldC = in;
    a.flags = accum ? GEMM_ACCUM : 0;
    gemm(rec, a);
}

void MemorillaModule::linearDW(ComputeContext::Recorder& rec, uint64_t dY, uint64_t X, const std::string& w,
                               uint32_t rows, uint32_t in, uint32_t out) {
    GemmArgs a;
    a.A = dY; a.B = X; a.C = gAddr(w + ".weight");
    a.M = out; a.N = in; a.K = rows;
    a.sAm = 1; a.sAk = out; a.sBk = in; a.sBn = 1; a.ldC = in;
    a.flags = GEMM_ACCUM;
    gemm(rec, a);
    colsum(rec, dY, gAddr(w + ".bias"), rows, out);
}

void MemorillaModule::colsum(ComputeContext::Recorder& rec, uint64_t in, uint64_t out, uint32_t rows, uint32_t cols) {
    struct { uint32_t rows, cols, ld, flags, strideOut, r0; } p{ rows, cols, cols, 1u, 0, 0 };  // accumulate
    std::vector<ParamArena::TensorRef> t = { { in, DT_FP32 }, { out, DT_FP32 }, { 0, DT_I32 } };
    rec.dispatch(ctx.pipeline(kOpsSrc, "sml_train_colsum_f32"), arena->params(t, p), ceilDiv(cols, 64), 1, 1);
}

void MemorillaModule::layerNorm(ComputeContext::Recorder& rec, uint64_t x, const std::string& name, uint64_t y,
                                uint64_t mean, uint64_t rstd, uint32_t rows) {
    struct { uint32_t rows, H; float eps; uint32_t flags; } p{ rows, cfg.encoderDim, cfg.eps, 0 };
    std::vector<ParamArena::TensorRef> t = { { x, DT_FP32 },    { pAddr(name + ".scale"), DT_FP32 },
                                             { pAddr(name + ".bias"), DT_FP32 }, { y, DT_FP32 },
                                             { mean, DT_FP32 }, { rstd, DT_FP32 } };
    rec.dispatch(ctx.pipeline(kMemSrc, "sml_mem_layernorm_fwd_f32"), arena->params(t, p), rows);
}

void MemorillaModule::layerNormBwd(ComputeContext::Recorder& rec, uint64_t x, const std::string& name, uint64_t mean,
                                   uint64_t rstd, uint64_t dy, uint64_t dx, uint32_t rows, bool accum, bool noDx) {
    const uint32_t h = cfg.encoderDim;
    struct { uint32_t rows, H; float eps; uint32_t flags; } p{ rows, h, cfg.eps,
                                                               (accum ? MEM_ACCUM : 0u) | (noDx ? MEM_NO_DX : 0u) };
    std::vector<ParamArena::TensorRef> t = { { x, DT_FP32 },  { pAddr(name + ".scale"), DT_FP32 },
                                             { mean, DT_FP32 }, { rstd, DT_FP32 },
                                             { dy, DT_FP32 }, { dx, DT_FP32 },
                                             { gx.address, DT_FP32 } };
    rec.dispatch(ctx.pipeline(kMemSrc, "sml_mem_layernorm_bwd_f32"), arena->params(t, p), rows);
    colsum(rec, gx.address, gAddr(name + ".scale"), rows, h);
    colsum(rec, dy, gAddr(name + ".bias"), rows, h);
}

struct MemAttnPayload {
    uint32_t B, I, J, nh, dh, ldA, ldB;
    float scale;
    uint32_t flags, stride, r0, r1;
};

void MemorillaModule::attnDispatch(ComputeContext::Recorder& rec, const char* entry, uint64_t t0, uint64_t t1,
                                   uint64_t t2, uint64_t t3, uint32_t I, uint32_t J, uint32_t ldA, uint32_t ldB,
                                   float scale, uint32_t flags, uint64_t work) {
    const uint32_t g = strideGroups(work);
    MemAttnPayload p{ curB, I, J, cfg.heads, cfg.headDim(), ldA, ldB, scale, flags, g * 256u, 0, 0 };
    std::vector<ParamArena::TensorRef> t = { { t0, DT_FP32 }, { t1, DT_FP32 }, { t2, DT_FP32 }, { t3, DT_I32 } };
    rec.dispatch(ctx.pipeline(kMemSrc, entry), arena->params(t, p), g);
}

void MemorillaModule::recordForward(ComputeContext::Recorder& rec, uint32_t B, uint32_t Nd) {
    if (!capB) throw std::runtime_error("memorilla: allocate() first");
    if (B > capB || Nd > capN || B == 0 || Nd == 0) throw std::runtime_error("memorilla: batch exceeds allocation");
    curB = B;
    curN = Nd;
    const uint32_t h = cfg.encoderDim, F = cfg.ffnDim(), hL = cfg.llmDim, K = cfg.slots, nh = cfg.heads;
    const uint32_t rk = B * K, rn = B * Nd;
    const float scale = 1.0f / std::sqrt(float(cfg.headDim()));

    // Stage 2: u = W_o (W_v z_q + b_v) + b_o; R0 = LN_c(slots + u).
    linear(rec, queryIn.address, "cond.v", u1.address, B, h, h);
    linear(rec, u1.address, "cond.o", u.address, B, h, h);
    {
        const uint32_t g = strideGroups(uint64_t(rk) * h);
        struct { uint32_t B, K, H, stride; } p{ B, K, h, g * 256u };
        std::vector<ParamArena::TensorRef> t = { { pAddr("slots"), DT_FP32 }, { u.address, DT_FP32 }, { pre.address, DT_FP32 } };
        rec.dispatch(ctx.pipeline(kMemSrc, "sml_mem_bcast_add_f32"), arena->params(t, p), g);
    }
    layerNorm(rec, pre.address, "cond.norm", res[0].address, cm.address, cr.address, rk);
    layerNorm(rec, docsIn.address, "docs.norm", zn.address, dmn.address, drs.address, rn);

    for (uint32_t l = 0; l < cfg.refineLayers; ++l) {
        Block& b = blocks[l];
        const std::string n = "block." + std::to_string(l) + ".";
        const uint64_t rin = res[2 * l].address, rmid = res[2 * l + 1].address, rout = res[2 * l + 2].address;
        layerNorm(rec, rin, n + "norm1", b.n1.address, b.m1.address, b.r1.address, rk);
        linear(rec, b.n1.address, n + "attn.q", b.q.address, rk, h, h);
        linear(rec, zn.address, n + "attn.k", b.k.address, rn, h, h);
        linear(rec, zn.address, n + "attn.v", b.v.address, rn, h, h);
        attnDispatch(rec, "sml_mem_rowdot_f32", b.q.address, b.k.address, b.P.address, countsIn.address, K, Nd, h, h,
                     scale, MEM_MASK, uint64_t(B) * nh * K * Nd);
        {
            struct { uint32_t rows, J; float scale; uint32_t r0; } p{ B * nh * K, Nd, 1.0f, 0 };
            std::vector<ParamArena::TensorRef> t = { { b.P.address, DT_FP32 } };
            rec.dispatch(ctx.pipeline(kMemSrc, "sml_mem_softmax_fwd_f32"), arena->params(t, p), B * nh * K);
        }
        attnDispatch(rec, "sml_mem_wsum_f32", b.P.address, b.v.address, b.O.address, 0, K, Nd, h, h, 1.0f, 0,
                     uint64_t(rk) * h);
        linear(rec, b.O.address, n + "attn.o", rmid, rk, h, h, rin);
        layerNorm(rec, rmid, n + "norm2", b.n2.address, b.m2.address, b.r2.address, rk);
        linear(rec, b.n2.address, n + "ffn.up", b.hu.address, rk, h, F);
        {
            const uint64_t cnt = uint64_t(rk) * F;
            const uint32_t g = strideGroups(cnt);
            struct { uint32_t n, stride, r0, r1; } p{ uint32_t(cnt), g * 256u, 0, 0 };
            std::vector<ParamArena::TensorRef> t = { { b.hu.address, DT_FP32 }, { b.ha.address, DT_FP32 } };
            rec.dispatch(ctx.pipeline(kMemSrc, "sml_mem_gelu_fwd_f32"), arena->params(t, p), g);
        }
        linear(rec, b.ha.address, n + "ffn.down", rout, rk, F, h, rmid);
    }
    layerNorm(rec, res[2 * cfg.refineLayers].address, "out.norm", rf.address, fm.address, fr.address, rk);
    linear(rec, rf.address, "proj", M.address, rk, h, hL);
}

void MemorillaModule::recordBackward(ComputeContext::Recorder& rec) {
    if (!trainable) throw std::runtime_error("memorilla: module was created without training buffers");
    const uint32_t B = curB, Nd = curN;
    const uint32_t h = cfg.encoderDim, F = cfg.ffnDim(), hL = cfg.llmDim, K = cfg.slots, nh = cfg.heads;
    const uint32_t rk = B * K, rn = B * Nd;
    const float scale = 1.0f / std::sqrt(float(cfg.headDim()));

    linearDW(rec, dM.address, rf.address, "proj", rk, h, hL);
    linearDX(rec, dM.address, "proj", dN.address, rk, h, hL, false);
    layerNormBwd(rec, res[2 * cfg.refineLayers].address, "out.norm", fm.address, fr.address, dN.address, dR.address,
                 rk, false, false);
    rec.fillZero(dZn);
    for (int li = int(cfg.refineLayers) - 1; li >= 0; --li) {
        const uint32_t l = uint32_t(li);
        Block& b = blocks[l];
        const std::string n = "block." + std::to_string(l) + ".";
        const uint64_t rin = res[2 * l].address, rmid = res[2 * l + 1].address;
        // FFN: R_out = R_mid + down(gelu(up(LN_2(R_mid))))
        linearDW(rec, dR.address, b.ha.address, n + "ffn.down", rk, F, h);
        linearDX(rec, dR.address, n + "ffn.down", dH.address, rk, F, h, false);
        {
            const uint64_t cnt = uint64_t(rk) * F;
            const uint32_t g = strideGroups(cnt);
            struct { uint32_t n, stride, r0, r1; } p{ uint32_t(cnt), g * 256u, 0, 0 };
            std::vector<ParamArena::TensorRef> t = { { b.hu.address, DT_FP32 }, { dH.address, DT_FP32 } };
            rec.dispatch(ctx.pipeline(kMemSrc, "sml_mem_gelu_bwd_f32"), arena->params(t, p), g);
        }
        linearDW(rec, dH.address, b.n2.address, n + "ffn.up", rk, h, F);
        linearDX(rec, dH.address, n + "ffn.up", dN.address, rk, h, F, false);
        layerNormBwd(rec, rmid, n + "norm2", b.m2.address, b.r2.address, dN.address, dR.address, rk, true, false);
        // Attention: R_mid = R_in + o(MHA(q(LN_1(R_in)), k(Z^), v(Z^)))
        linearDW(rec, dR.address, b.O.address, n + "attn.o", rk, h, h);
        linearDX(rec, dR.address, n + "attn.o", dO.address, rk, h, h, false);
        attnDispatch(rec, "sml_mem_rowdot_f32", dO.address, b.v.address, dP.address, countsIn.address, K, Nd, h, h,
                     1.0f, 0, uint64_t(B) * nh * K * Nd);
        {
            struct { uint32_t rows, J; float scale; uint32_t r0; } p{ B * nh * K, Nd, scale, 0 };
            std::vector<ParamArena::TensorRef> t = { { b.P.address, DT_FP32 }, { dP.address, DT_FP32 } };
            rec.dispatch(ctx.pipeline(kMemSrc, "sml_mem_softmax_bwd_f32"), arena->params(t, p), B * nh * K);
        }
        attnDispatch(rec, "sml_mem_wsum_f32", dP.address, b.k.address, dQ.address, 0, K, Nd, h, h, 1.0f, 0,
                     uint64_t(rk) * h);
        attnDispatch(rec, "sml_mem_wsum_f32", dP.address, b.q.address, dK.address, 0, Nd, K, h, h, 1.0f,
                     MEM_TRANSPOSED, uint64_t(rn) * h);
        attnDispatch(rec, "sml_mem_wsum_f32", b.P.address, dO.address, dV.address, 0, Nd, K, h, h, 1.0f,
                     MEM_TRANSPOSED, uint64_t(rn) * h);
        linearDW(rec, dQ.address, b.n1.address, n + "attn.q", rk, h, h);
        linearDX(rec, dQ.address, n + "attn.q", dN.address, rk, h, h, false);
        linearDW(rec, dK.address, zn.address, n + "attn.k", rn, h, h);
        linearDX(rec, dK.address, n + "attn.k", dZn.address, rn, h, h, true);
        linearDW(rec, dV.address, zn.address, n + "attn.v", rn, h, h);
        linearDX(rec, dV.address, n + "attn.v", dZn.address, rn, h, h, true);
        layerNormBwd(rec, rin, n + "norm1", b.m1.address, b.r1.address, dN.address, dR.address, rk, true, false);
    }
    // Document norm: only its scale / bias need gradients (the store is frozen).
    layerNormBwd(rec, docsIn.address, "docs.norm", dmn.address, drs.address, dZn.address, 0, rn, false, true);
    // Stage 2.
    layerNormBwd(rec, pre.address, "cond.norm", cm.address, cr.address, dR.address, dPre.address, rk, false, false);
    {
        struct { uint32_t G, R, C, groupStride, rowStride, flags, r0, r1; } p{ K, B, h, h, K * h, MEM_ACCUM, 0, 0 };
        std::vector<ParamArena::TensorRef> t = { { dPre.address, DT_FP32 }, { gAddr("slots"), DT_FP32 } };
        rec.dispatch(ctx.pipeline(kMemSrc, "sml_mem_group_sum_f32"), arena->params(t, p), ceilDiv(h, 256), K);
        struct { uint32_t G, R, C, groupStride, rowStride, flags, r0, r1; } q{ B, K, h, K * h, h, 0, 0, 0 };
        std::vector<ParamArena::TensorRef> t2 = { { dPre.address, DT_FP32 }, { du.address, DT_FP32 } };
        rec.dispatch(ctx.pipeline(kMemSrc, "sml_mem_group_sum_f32"), arena->params(t2, q), ceilDiv(h, 256), B);
    }
    linearDW(rec, du.address, u1.address, "cond.o", B, h, h);
    linearDX(rec, du.address, "cond.o", du1.address, B, h, h, false);
    linearDW(rec, du1.address, queryIn.address, "cond.v", B, h, h);
}

void MemorillaModule::recordRowsCopy(ComputeContext::Recorder& rec, uint64_t in, uint64_t out, uint32_t B, uint32_t K,
                                     uint32_t H, uint32_t inT, uint32_t inOff, uint32_t outT, uint32_t outOff, bool accum) {
    const uint32_t g = strideGroups(uint64_t(B) * K * H);
    struct { uint32_t B, K, H, inT, inOff, outT, outOff, flags, stride, r0, r1, r2; } p{
        B, K, H, inT, inOff, outT, outOff, accum ? MEM_ACCUM : 0u, g * 256u, 0, 0, 0 };
    std::vector<ParamArena::TensorRef> t = { { in, DT_FP32 }, { out, DT_FP32 } };
    rec.dispatch(ctx.pipeline(kMemSrc, "sml_mem_rows_copy_f32"), arena->params(t, p), g);
}

void MemorillaModule::recordOptimizer(ComputeContext::Recorder& rec, float lr, uint64_t step, float gradScale,
                                      float beta1, float beta2, float eps, float wd, float clip) {
    if (!trainable) throw std::runtime_error("memorilla: module was created without training buffers");
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
    const double t = double(step + 1);
    const uint32_t g = std::min<uint32_t>(ceilDiv(n, 256), kMaxGroups);
    struct { uint32_t n, decayEnd; float lr, b1, b2, eps, wd, bc1, bc2, clip, gradScale; uint32_t stride; } p{
        uint32_t(n), uint32_t(lay.decayEnd), lr, beta1, beta2, eps, wd,
        float(1.0 - std::pow(double(beta1), t)), float(1.0 - std::pow(double(beta2), t)), clip, gradScale, g * 256u };
    std::vector<ParamArena::TensorRef> tt = { { P.address, DT_FP32 }, { G.address, DT_FP32 }, { Mo.address, DT_FP32 },
                                              { Ve.address, DT_FP32 }, { normSq.address, DT_FP32 } };
    rec.dispatch(ctx.pipeline(kOpsSrc, "sml_train_adamw_f32"), arena->params(tt, p), g);
}

// ---- files ----
MemorillaConfig MemorillaModule::loadConfig(const std::string& dir) {
    return MemorillaConfig::fromJson(readText((fs::path(dir) / "memorilla.json").string()));
}

void MemorillaModule::save(const std::string& dir) const {
    fs::create_directories(dir);
    std::ofstream(fs::path(dir) / "memorilla.json", std::ios::binary) << cfg.toJson();
    const std::vector<float> host = params();
    std::ostringstream h;
    h << "{";
    uint64_t off = 0;
    for (size_t i = 0; i < lay.tensors.size(); ++i) {
        const ParamTensor& t = lay.tensors[i];
        h << (i ? "," : "") << "\"" << t.name << "\":{\"dtype\":\"F32\",\"shape\":[";
        for (size_t d = 0; d < t.shape.size(); ++d) h << (d ? "," : "") << t.shape[d];
        h << "],\"data_offsets\":[" << off << "," << off + t.count * 4 << "]}";
        off += t.count * 4;
    }
    h << "}";
    std::string header = h.str();
    while (header.size() % 8) header += ' ';
    const std::string path = (fs::path(dir) / "memorilla.safetensors").string(), tmp = path + ".tmp";
    std::FILE* f = std::fopen(tmp.c_str(), "wb");
    if (!f) throw std::runtime_error("cannot write " + tmp);
    const uint64_t hl = header.size();
    writeAll(f, &hl, 8, tmp);
    writeAll(f, header.data(), header.size(), tmp);
    for (const ParamTensor& t : lay.tensors) writeAll(f, host.data() + t.offset, t.count * 4, tmp);
    if (std::fflush(f) != 0 || std::fclose(f) != 0) throw std::runtime_error("write failed: " + tmp);
    std::error_code ec;
    fs::remove(path, ec);
    fs::rename(tmp, path);
}

void MemorillaModule::load(const std::string& dir) {
    SafeTensors st(dir);
    std::vector<float> host(lay.total, 0.0f);
    for (const ParamTensor& t : lay.tensors) {
        const SafeTensorInfo& info = st.info(t.name);
        if (info.dtype != "F32" || info.numel() != t.count)
            throw std::runtime_error("memorilla: tensor " + t.name + " has the wrong type or shape in " + dir);
        st.read(t.name, host.data() + t.offset, t.count * 4);
    }
    setParams(host);
}

void MemorillaModule::saveState(std::FILE* f, const std::string& path) const {
    std::vector<float> host(lay.total);
    for (const GpuBuffer* b : { &P, &Mo, &Ve }) {
        ctx.download(*b, 0, host.data(), host.size() * 4);
        writeAll(f, host.data(), host.size() * 4, path);
    }
}

void MemorillaModule::loadState(std::FILE* f, const std::string& path) {
    std::vector<float> host(lay.total);
    for (const GpuBuffer* b : { &P, &Mo, &Ve }) {
        readAll(f, host.data(), host.size() * 4, path);
        ctx.upload(*b, 0, host.data(), host.size() * 4);
    }
}

}  // namespace lve::ml
