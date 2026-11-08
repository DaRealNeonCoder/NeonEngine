// Memorilla command line (docs: shaders/ML/docs/memorilla_plan.md):
//
//   NeonML --memorilla encode    --decoder <ckpt> --sequences docs.seq --out docs.nmem
//   NeonML --memorilla train     --decoder <ckpt> --data <dir> --out <dir> [--mem-config m.json]
//   NeonML --memorilla eval      --decoder <ckpt> --data <dir> --module <dir> [--ablation closed|noise]
//   NeonML --memorilla gradcheck --config dec.json --data <dir> --out <dir> (tiny random decoder, fp32)
//   NeonML --memorilla params    [--mem-config m.json] (parameter count; paper config by default)
//
// The decoder is always frozen: these commands build the trainer with
// TrainOptions::frozen, so it has no weight gradients and no optimizer state.
//
// Data directory (train/prepare_memorilla_data.py + the encode command):
//   {split}_docs.nmem, {split}_queries.nmem       document / query embeddings
//   {split}_examples.bin                          examples (format below)
// Example record: doc_begin, doc_end (into the docs store), query (index into the
// query store), nq, na, answer_bytes, then nq query tokens and na answer tokens (uint32).
// Sequence layout per example (seq rows): [K memory rows][query][answer][padding];
// only rows whose next token is an answer token carry a loss.

#include "memorilla.hpp"

#include "gpt_oss_train.hpp"
#include "ml_json.hpp"

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
#include <random>
#include <sstream>
#include <stdexcept>
#include <tuple>

namespace lve::ml {

namespace fs = std::filesystem;

namespace {

constexpr uint32_t kInvalid = 0xFFFFFFFFu;
constexpr char kExMagic[8] = { 'N', 'E', 'O', 'N', 'M', 'E', 'X', '1' };
constexpr char kCkMagic[8] = { 'N', 'E', 'O', 'N', 'M', 'C', 'K', '1' };
constexpr char kCkEnd[8] = { 'N', 'E', 'O', 'N', 'M', 'C', 'E', '1' };

std::string readText(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + path);
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}
void writeAll(std::FILE* f, const void* p, size_t n, const std::string& path) {
    if (n && std::fwrite(p, 1, n, f) != n) throw std::runtime_error("write failed: " + path);
}
void readAll(std::FILE* f, void* p, size_t n, const std::string& path) {
    if (n && std::fread(p, 1, n, f) != n) throw std::runtime_error("truncated file: " + path);
}
void writeFloats(const std::string& path, const float* p, size_t n) {
    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) throw std::runtime_error("cannot write " + path);
    writeAll(f, p, n * 4, path);
    std::fclose(f);
}
uint64_t mix(uint64_t a, uint64_t b) {
    uint64_t z = a * 0x9E3779B97F4A7C15ull ^ (b + 0x632BE59BD9B4E019ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}
uint64_t fnv(const float* p, size_t n) {
    uint64_t h = 1469598103934665603ull;
    const uint8_t* b = reinterpret_cast<const uint8_t*>(p);
    for (size_t i = 0; i < n * 4; ++i) h = (h ^ b[i]) * 1099511628211ull;
    return h;
}

std::string findDecoderCheckpoint(const std::string& p) {
    if (fs::is_directory(p)) {
        CheckpointInfo best;
        for (const char* n : { "latest.nckpt", "prev.nckpt" }) {
            CheckpointInfo i = probeCheckpoint((fs::path(p) / n).string());
            if (i.valid && (!best.valid || i.step > best.step)) best = i;
        }
        if (!best.valid) throw std::runtime_error("no valid decoder checkpoint in " + p);
        return best.path;
    }
    return p;
}

GptOssConfig configFromFile(const std::string& path) {
    const JsonValue j = JsonValue::parse(readText(path));
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

struct Example {
    uint32_t docBegin = 0, docEnd = 0, query = 0, answerBytes = 0;
    std::vector<uint32_t> q, a;
};

std::vector<Example> readExamples(const std::string& path) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("cannot open " + path);
    std::vector<Example> out;
    try {
        char magic[8];
        uint64_t count = 0;
        readAll(f, magic, 8, path);
        if (std::memcmp(magic, kExMagic, 8) != 0) throw std::runtime_error(path + " is not a Memorilla examples file");
        readAll(f, &count, 8, path);
        out.resize(count);
        for (Example& e : out) {
            uint32_t h[6];
            readAll(f, h, sizeof(h), path);
            e.docBegin = h[0];
            e.docEnd = h[1];
            e.query = h[2];
            e.q.resize(h[3]);
            e.a.resize(h[4]);
            e.answerBytes = h[5];
            readAll(f, e.q.data(), e.q.size() * 4, path);
            readAll(f, e.a.data(), e.a.size() * 4, path);
        }
    } catch (...) {
        std::fclose(f);
        throw;
    }
    std::fclose(f);
    return out;
}

struct Split {
    MemoryStore docs, queries;
    std::vector<Example> examples;
    std::vector<float> noiseMean, noiseStd;  // per-dimension stats of the doc store (noise ablation)
};

Split loadSplit(const std::string& dir, const std::string& name) {
    Split s;
    s.docs = MemoryStore::load((fs::path(dir) / (name + "_docs.nmem")).string());
    s.queries = MemoryStore::load((fs::path(dir) / (name + "_queries.nmem")).string());
    s.examples = readExamples((fs::path(dir) / (name + "_examples.bin")).string());
    if (s.docs.dim != s.queries.dim) throw std::runtime_error("memorilla: doc and query stores differ in width");
    for (const Example& e : s.examples)
        if (e.docEnd > s.docs.count() || e.docBegin > e.docEnd || e.query >= s.queries.count())
            throw std::runtime_error("memorilla: example points outside the stores (" + dir + ", " + name + ")");
    const uint32_t d = s.docs.dim;
    s.noiseMean.assign(d, 0.0f);
    s.noiseStd.assign(d, 0.0f);
    const size_t n = std::max<size_t>(1, s.docs.count());
    for (size_t i = 0; i < s.docs.count(); ++i)
        for (uint32_t c = 0; c < d; ++c) s.noiseMean[c] += s.docs.vector(i)[c] / float(n);
    for (size_t i = 0; i < s.docs.count(); ++i)
        for (uint32_t c = 0; c < d; ++c) {
            const float x = s.docs.vector(i)[c] - s.noiseMean[c];
            s.noiseStd[c] += x * x / float(n);
        }
    for (float& v : s.noiseStd) v = std::sqrt(v);
    return s;
}

std::atomic<bool> gStop{ false };
void onMemSignal(int) { gStop = true; }

struct RunOptions {
    std::string decoder, dataDir, outDir = "memrun", mirrorDir, memConfig, moduleDir, shaders = "shaders/ML", device;
    std::string ablation = "none";  // eval: none | closed | noise
    uint32_t batch = 16, seq = 128, maxDocs = 64, accum = 1;
    uint64_t steps = 2000, seed = 1234;
    float warmupFrac = 0.03f, lr = 3.0e-4f, minLr = 0.0f, wd = 1.0e-2f, clip = 1.0f;
    float beta1 = 0.9f, beta2 = 0.999f, eps = 1.0e-8f;
    uint32_t evalEvery = 200, evalBatches = 20, logEvery = 10;
    double ckptMinutes = 20.0;
    std::string precision = "auto";
    bool fresh = false, mxfp4 = true;
};

// Frozen decoder + memory module, one micro-batch at a time.
class MemorillaRunner {
public:
    MemorillaRunner(ComputeContext& c, const GptOssConfig& dcfg, const MemorillaConfig& mcfg, const RunOptions& o,
                    bool trainable, uint32_t memRows)
        : ctx(c), ro(o), K(memRows) {
        TrainOptions t;
        t.batch = o.batch;
        t.seq = o.seq;
        t.precision = o.precision;
        t.frozen = true;
        t.memRows = memRows;
        t.auxCoef = 0.0f;
        t.qatStart = -2;
        t.lmChunk = std::min<uint32_t>(2048, o.batch * o.seq);
        t.seed = o.seed;
        trainer = std::make_unique<GptOssTrainer>(ctx, dcfg, t);
        module = std::make_unique<MemorillaModule>(ctx, mcfg, trainer->usesTensorCores(), trainable);
        module->allocate(o.batch, o.maxDocs);
        trainer->setMemorySource(module->outputAddress());
        if (memRows != 0 && memRows != mcfg.slots) throw std::runtime_error("memorilla: memRows must be 0 or K");
    }

    GptOssTrainer& dec() { return *trainer; }
    MemorillaModule& mem() { return *module; }

    // Fills the decoder batch and the module inputs. Returns (answer tokens, answer bytes).
    std::pair<uint64_t, double> setBatch(const Split& s, const std::vector<uint32_t>& idx, bool noise, uint64_t noiseSeed) {
        const uint32_t B = ro.batch, T = ro.seq, h = module->config().encoderDim;
        std::vector<uint32_t> in(size_t(B) * T, 0), tg(size_t(B) * T, kInvalid);
        Nd = 1;
        for (uint32_t b = 0; b < B; ++b) {
            const Example& e = s.examples[idx[b]];
            Nd = std::max(Nd, std::min(ro.maxDocs, e.docEnd - e.docBegin));
        }
        float* docs = module->docsHost();
        uint32_t* counts = module->countsHost();
        float* query = module->queryHost();
        std::memset(docs, 0, size_t(B) * Nd * h * 4);
        std::mt19937_64 rng(noiseSeed);
        std::normal_distribution<float> nd(0.0f, 1.0f);
        uint64_t answerTokens = 0;
        double answerBytes = 0.0;
        for (uint32_t b = 0; b < B; ++b) {
            const Example& e = s.examples[idx[b]];
            const uint32_t n = std::min(ro.maxDocs, e.docEnd - e.docBegin);
            counts[b] = n;
            for (uint32_t i = 0; i < n; ++i) {
                float* dst = docs + (size_t(b) * Nd + i) * h;
                if (noise)
                    for (uint32_t c = 0; c < h; ++c) dst[c] = s.noiseMean[c] + s.noiseStd[c] * nd(rng);
                else
                    std::memcpy(dst, s.docs.vector(e.docBegin + i), size_t(h) * 4);
            }
            std::memcpy(query + size_t(b) * h, s.queries.vector(e.query), size_t(h) * 4);
            // [K memory rows][query][answer][pad]
            const uint32_t room = T > K + uint32_t(e.q.size()) ? T - K - uint32_t(e.q.size()) : 0;
            const uint32_t na = std::min<uint32_t>(uint32_t(e.a.size()), room);
            if (na == 0) throw std::runtime_error("memorilla: --seq too short for an example's query");
            uint32_t r = K;
            for (uint32_t t : e.q) in[size_t(b) * T + r++] = t;
            const uint32_t a0 = r;
            for (uint32_t i = 0; i < na; ++i) in[size_t(b) * T + r++] = e.a[i];
            for (uint32_t i = 0; i < na; ++i)
                if (a0 + i >= 1) tg[size_t(b) * T + a0 + i - 1] = e.a[i];
            answerTokens += na;
            answerBytes += double(e.answerBytes) * double(na) / double(std::max<size_t>(1, e.a.size()));
        }
        trainer->setBatch(in, tg);
        trainer->setLossDenominator(float(answerTokens));
        return { answerTokens, answerBytes };
    }

    // One micro-batch; returns the summed loss (nats) over answer tokens.
    double step(bool backward) {
        module->beginRecording();
        trainer->beginRecording();
        ComputeContext::Recorder rec(ctx);
        if (K) module->recordForward(rec, ro.batch, Nd);
        trainer->recordStep(rec, backward);
        if (backward && K) {
            module->recordRowsCopy(rec, trainer->inputGradAddress(), module->outputGradAddress(), ro.batch, K,
                                   trainer->config().hidden, ro.seq, 0, K, 0, false);
            module->recordBackward(rec);
        }
        rec.submitAndWait();
        return trainer->lossSum();
    }

    uint32_t docsInBatch() const { return Nd; }

private:
    ComputeContext& ctx;
    RunOptions ro;
    uint32_t K = 0;
    uint32_t Nd = 1;
    std::unique_ptr<GptOssTrainer> trainer;
    std::unique_ptr<MemorillaModule> module;
};

float lrAt(const RunOptions& o, uint64_t step) {
    const uint64_t warm = std::max<uint64_t>(1, uint64_t(double(o.steps) * o.warmupFrac));
    if (step < warm) return o.lr * float(step + 1) / float(warm);
    const double t = double(step - warm) / double(std::max<uint64_t>(1, o.steps - warm));
    return float(o.minLr + 0.5 * (o.lr - o.minLr) * (1.0 + std::cos(3.14159265358979323846 * std::min(1.0, t))));
}

// Mean answer loss and bits per byte over the first `batches` batches of a split.
std::pair<double, double> evaluate(MemorillaRunner& run, const Split& s, const RunOptions& o, bool noise) {
    double loss = 0.0, bytes = 0.0;
    uint64_t tokens = 0;
    const uint64_t n = s.examples.size();
    for (uint32_t i = 0; i < o.evalBatches; ++i) {
        std::vector<uint32_t> idx(o.batch);
        for (uint32_t b = 0; b < o.batch; ++b) idx[b] = uint32_t((uint64_t(i) * o.batch + b) % n);
        auto [t, by] = run.setBatch(s, idx, noise, 0xE7A1ull + i);
        loss += run.step(false);
        tokens += t;
        bytes += by;
    }
    return { loss / double(std::max<uint64_t>(1, tokens)), bytes > 0 ? loss / (std::log(2.0) * bytes) : -1.0 };
}

std::vector<float> loadDecoderParams(const std::string& ckpt, GptOssConfig& cfg) {
    std::vector<float> params;
    cfg = readCheckpointParams(findDecoderCheckpoint(ckpt), params);
    return params;
}

MemorillaConfig moduleConfig(const RunOptions& o, const GptOssConfig& dcfg, uint32_t encDim) {
    MemorillaConfig m = o.memConfig.empty() ? MemorillaConfig{} : MemorillaConfig::fromJson(readText(o.memConfig));
    if (m.encoderDim && m.encoderDim != encDim)
        throw std::runtime_error("memorilla: encoder_dim in --mem-config differs from the store width");
    m.encoderDim = encDim;
    m.llmDim = dcfg.hidden;
    m.decoder = decoderFingerprint(dcfg);
    return m;
}

float embeddingRms(const std::vector<float>& params, const ParamLayout& lay) {
    const ParamTensor& t = lay.at("embedding.weight");
    double s = 0.0;
    for (uint64_t i = 0; i < t.count; ++i) s += double(params[t.offset + i]) * params[t.offset + i];
    return float(std::sqrt(s / double(std::max<uint64_t>(1, t.count))));
}

void saveRunCheckpoint(MemorillaRunner& run, const std::string& dir, const std::string& mirror, uint64_t step,
                       double bestVal) {
    fs::create_directories(dir);
    const fs::path path = fs::path(dir) / "mem_latest.nckpt";
    const std::string tmp = path.string() + ".tmp";
    std::ostringstream j;
    j << std::setprecision(17) << "{\"step\": " << step << ", \"best_val\": " << bestVal
      << ", \"loss_scale\": " << run.dec().lossScaleValue() << ", \"params\": " << run.mem().layout().total << "}";
    const std::string json = j.str();
    std::FILE* f = std::fopen(tmp.c_str(), "wb");
    if (!f) throw std::runtime_error("cannot write " + tmp);
    const uint32_t len = uint32_t(json.size());
    writeAll(f, kCkMagic, 8, tmp);
    writeAll(f, &len, 4, tmp);
    writeAll(f, json.data(), json.size(), tmp);
    run.mem().saveState(f, tmp);
    writeAll(f, kCkEnd, 8, tmp);
    if (std::fflush(f) != 0 || std::fclose(f) != 0) throw std::runtime_error("write failed: " + tmp);
    std::error_code ec;
    fs::remove(path, ec);
    fs::rename(tmp, path);
    std::ofstream(fs::path(dir) / "memorilla.json", std::ios::binary) << run.mem().config().toJson();
    if (!mirror.empty()) {
        fs::create_directories(mirror, ec);
        const fs::path dst = fs::path(mirror) / "mem_latest.nckpt", mtmp = dst.string() + ".tmp";
        fs::copy_file(path, mtmp, fs::copy_options::overwrite_existing, ec);
        if (!ec) {
            fs::remove(dst, ec);
            fs::rename(mtmp, dst, ec);
            fs::copy_file(fs::path(dir) / "memorilla.json", fs::path(mirror) / "memorilla.json",
                          fs::copy_options::overwrite_existing, ec);
        }
        if (ec) std::cerr << "mirror: copy failed (" << ec.message() << "); training continues\n";
    }
    std::cout << "checkpoint: step " << step << " -> " << path.string() << "\n";
}

bool loadRunCheckpoint(MemorillaRunner& run, const RunOptions& o, uint64_t& step, double& bestVal) {
    for (const std::string& d : { o.outDir, o.mirrorDir }) {
        if (d.empty()) continue;
        const std::string path = (fs::path(d) / "mem_latest.nckpt").string();
        std::FILE* f = std::fopen(path.c_str(), "rb");
        if (!f) continue;
        try {
            char magic[8];
            uint32_t len = 0;
            readAll(f, magic, 8, path);
            if (std::memcmp(magic, kCkMagic, 8) != 0) throw std::runtime_error("bad magic");
            readAll(f, &len, 4, path);
            std::string json(len, '\0');
            readAll(f, json.data(), len, path);
            const JsonValue j = JsonValue::parse(json);
            if (uint64_t(j["params"].asInt()) != run.mem().layout().total)
                throw std::runtime_error("module config differs from the checkpoint");
            run.mem().loadState(f, path);
            readAll(f, magic, 8, path);
            if (std::memcmp(magic, kCkEnd, 8) != 0) throw std::runtime_error("bad end marker");
            step = uint64_t(j["step"].asInt());
            bestVal = j.numberOr("best_val", 1e30);
            if (run.dec().usesTensorCores()) run.dec().setLossScale(float(j.numberOr("loss_scale", 65536.0)));
            std::fclose(f);
            std::cout << "resumed from " << path << " at step " << step << "\n";
            return true;
        } catch (const std::exception& e) {
            std::fclose(f);
            std::cerr << "ignoring " << path << " (" << e.what() << ")\n";
        }
    }
    return false;
}

// ---- modes ----
int runEncode(const RunOptions& o, const std::string& seqPath, const std::string& out, int32_t layer) {
    GptOssConfig cfg;
    const std::vector<float> params = loadDecoderParams(o.decoder, cfg);
    const auto seqs = readSequencesFile(seqPath);
    ComputeContext ctx(o.shaders, o.device);
    TrainOptions t;
    t.batch = o.batch;
    t.seq = o.seq;
    t.precision = o.precision;
    t.frozen = true;
    t.auxCoef = 0.0f;
    t.qatStart = -2;
    t.lmChunk = std::min<uint32_t>(2048, o.batch * o.seq);
    GptOssTrainer dec(ctx, cfg, t);
    dec.setParams(params);
    if (o.mxfp4) dec.useMxfp4Experts();
    if (layer >= int32_t(cfg.numLayers)) throw std::runtime_error("--layer out of range");
    const GpuBuffer& hidden = layer < 0 ? dec.finalHidden() : dec.residual(2 * uint32_t(layer) + 2);
    const uint32_t B = o.batch, T = o.seq, H = cfg.hidden;
    MemoryStore store;
    store.encoder = "self";
    store.encoderLayer = layer;
    store.decoder = decoderFingerprint(cfg);
    store.dim = H;
    std::vector<float> hs(size_t(B) * T * H), v(H);
    std::vector<uint32_t> in(size_t(B) * T), tg(size_t(B) * T, kInvalid);
    size_t truncated = 0;
    const auto t0 = std::chrono::steady_clock::now();
    for (size_t s0 = 0; s0 < seqs.size(); s0 += B) {
        std::fill(in.begin(), in.end(), 0u);
        std::vector<uint32_t> len(B, 1);
        for (uint32_t b = 0; b < B && s0 + b < seqs.size(); ++b) {
            const auto& q = seqs[s0 + b];
            if (q.empty()) throw std::runtime_error("encode: empty sequence " + std::to_string(s0 + b));
            if (q.size() > T) ++truncated;
            len[b] = uint32_t(std::min<size_t>(q.size(), T));
            for (uint32_t i = 0; i < len[b]; ++i) in[size_t(b) * T + i] = q[i];
        }
        dec.setBatch(in, tg);
        dec.beginRecording();
        ComputeContext::Recorder rec(ctx);
        dec.recordForwardOnly(rec);
        rec.submitAndWait();
        ctx.download(hidden, 0, hs.data(), hs.size() * 4);
        for (uint32_t b = 0; b < B && s0 + b < seqs.size(); ++b) {
            std::fill(v.begin(), v.end(), 0.0f);
            for (uint32_t i = 0; i < len[b]; ++i)
                for (uint32_t c = 0; c < H; ++c) v[c] += hs[(size_t(b) * T + i) * H + c];
            for (float& x : v) x /= float(len[b]);
            store.add(v.data(), s0 + b, len[b]);
        }
    }
    store.save(out);
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::cout << "encoded " << seqs.size() << " sequences (" << truncated << " truncated to " << T << " tokens) in "
              << std::fixed << std::setprecision(1) << secs << " s -> " << out << " (dim " << H << ")\n";
    return 0;
}

int runParams(const RunOptions& o) {
    MemorillaConfig m;
    if (!o.memConfig.empty()) m = MemorillaConfig::fromJson(readText(o.memConfig));
    if (!m.encoderDim) m.encoderDim = 2560;  // the paper: Qwen3-Embedding-4B -> Qwen3-8B
    if (!m.llmDim) m.llmDim = 4096;
    uint64_t n = 0;
    for (const auto& t : buildMemorillaLayout(m).tensors) n += t.count;
    std::cout << "memorilla K=" << m.slots << " h_e=" << m.encoderDim << " h_L=" << m.llmDim << " heads=" << m.heads
              << " blocks=" << m.refineLayers << " ffn x" << m.ffnMult << ": " << n
              << " parameters; with the paper's unused W_q/W_k: " << memorillaPaperParamCount(m) << "\n";
    return 0;
}

int runGradcheck(const RunOptions& o, const std::string& configPath) {
    const GptOssConfig cfg = configFromFile(configPath);
    Split s = loadSplit(o.dataDir, "train");
    const MemorillaConfig mcfg = moduleConfig(o, cfg, s.docs.dim);
    RunOptions ro = o;
    if (ro.precision == "auto") ro.precision = "fp32";
    ComputeContext ctx(o.shaders, o.device);
    MemorillaRunner run(ctx, cfg, mcfg, ro, true, mcfg.slots);
    run.dec().initRandom();
    run.mem().initParams(o.seed, 0.02f);
    std::vector<uint32_t> idx(o.batch);
    for (uint32_t b = 0; b < o.batch; ++b) idx[b] = b % uint32_t(s.examples.size());
    auto [tokens, bytes] = run.setBatch(s, idx, false, 0);
    (void)bytes;
    const double lossSum = run.step(true);
    const float loss = float(lossSum / double(tokens));
    fs::create_directories(o.outDir);
    const fs::path out(o.outDir);
    const std::vector<float> dp = run.dec().paramsHost();
    writeFloats((out / "params.bin").string(), dp.data(), dp.size());
    std::vector<float> mp = run.mem().params(), mg = run.mem().grads();
    for (float& g : mg) g /= run.dec().lossScaleValue();
    writeFloats((out / "mem_params.bin").string(), mp.data(), mp.size());
    writeFloats((out / "mem_grads.bin").string(), mg.data(), mg.size());
    // The exact batch the GPU saw.
    const uint32_t B = o.batch, Nd = run.docsInBatch(), h = mcfg.encoderDim;
    writeFloats((out / "docs.bin").string(), run.mem().docsHost(), size_t(B) * Nd * h);
    writeFloats((out / "query.bin").string(), run.mem().queryHost(), size_t(B) * h);
    {
        std::ofstream j(out / "batch.json");
        j << "{\"loss\": " << std::setprecision(9) << loss << ", \"batch\": " << B << ", \"seq\": " << o.seq
          << ", \"docs\": " << Nd << ", \"slots\": " << mcfg.slots << ", \"counts\": [";
        for (uint32_t b = 0; b < B; ++b) j << (b ? "," : "") << run.mem().countsHost()[b];
        j << "], \"examples\": [";
        for (uint32_t b = 0; b < B; ++b) j << (b ? "," : "") << idx[b];
        j << "]}\n";
    }
    auto dumpLayout = [&](const ParamLayout& L, const std::string& name) {
        std::ofstream lj(out / name);
        lj << "{\"tensors\": [";
        for (size_t i = 0; i < L.tensors.size(); ++i) {
            const auto& t = L.tensors[i];
            lj << (i ? "," : "") << "\n  {\"name\": \"" << t.name << "\", \"offset\": " << t.offset << ", \"shape\": [";
            for (size_t d = 0; d < t.shape.size(); ++d) lj << (d ? "," : "") << t.shape[d];
            lj << "]}";
        }
        lj << "]}\n";
    };
    dumpLayout(run.dec().layout(), "layout.json");
    dumpLayout(run.mem().layout(), "mem_layout.json");
    std::ofstream(out / "memorilla.json") << mcfg.toJson();
    std::cout << "memorilla gradcheck: loss " << std::setprecision(7) << loss << " over " << tokens
              << " answer tokens, wrote " << o.outDir << "\n";
    return 0;
}

int runTrainOrEval(const RunOptions& o, bool train) {
    using clock = std::chrono::steady_clock;
    GptOssConfig cfg;
    const std::vector<float> params = loadDecoderParams(o.decoder, cfg);
    Split trainSplit;
    if (train) trainSplit = loadSplit(o.dataDir, "train");
    const bool haveVal = fs::exists(fs::path(o.dataDir) / "val_examples.bin");
    Split val;
    if (haveVal) val = loadSplit(o.dataDir, "val");
    const uint32_t encDim = train ? trainSplit.docs.dim : val.docs.dim;
    const MemoryStore& probe = train ? trainSplit.docs : val.docs;
    if (probe.encoder == "self" && !probe.decoder.empty() && probe.decoder != decoderFingerprint(cfg))
        std::cerr << "warning: the store was encoded with a different decoder (" << probe.decoder << ")\n";

    MemorillaConfig mcfg;
    if (train) {
        mcfg = moduleConfig(o, cfg, encDim);
    } else {
        if (o.moduleDir.empty()) throw std::runtime_error("eval needs --module");
        mcfg = MemorillaModule::loadConfig(o.moduleDir);
        if (mcfg.encoderDim != encDim || mcfg.llmDim != cfg.hidden)
            throw std::runtime_error("memorilla: module dims do not match the store / decoder");
        if (!mcfg.decoder.empty() && mcfg.decoder != decoderFingerprint(cfg))
            std::cerr << "warning: the module was trained for a different decoder (" << mcfg.decoder << ")\n";
    }
    const bool closed = o.ablation == "closed";
    ComputeContext ctx(o.shaders, o.device);
    std::cout << "device: " << ctx.deviceName() << " (" << ctx.deviceLocalBytes() / (1ull << 20) << " MiB)\n";
    MemorillaRunner run(ctx, cfg, mcfg, o, train, closed ? 0u : mcfg.slots);
    run.dec().setParams(params);
    if (o.mxfp4) run.dec().useMxfp4Experts();
    uint64_t n = 0;
    for (const auto& t : run.mem().layout().tensors) n += t.count;
    std::cout << "memorilla: K=" << mcfg.slots << " h_e=" << mcfg.encoderDim << " h_L=" << mcfg.llmDim << ", "
              << std::fixed << std::setprecision(2) << double(n) / 1e6 << " M trainable parameters; decoder frozen ("
              << decoderFingerprint(cfg) << ")\n";

    if (!train) {
        if (o.ablation == "untrained") run.mem().initParams(o.seed, embeddingRms(params, run.dec().layout()) / std::sqrt(float(encDim)));
        else if (!closed) run.mem().load(o.moduleDir);
        if (!haveVal) throw std::runtime_error("eval needs val_examples.bin in --data");
        auto [loss, bpb] = evaluate(run, val, o, o.ablation == "noise");
        std::cout << "eval (" << o.ablation << "): answer loss " << std::setprecision(4) << loss << " nats/token";
        if (bpb >= 0) std::cout << ", " << bpb << " bits/byte";
        std::cout << "\n";
        return 0;
    }

    const uint64_t decHash = fnv(params.data(), params.size());
    const float projStd = embeddingRms(params, run.dec().layout()) / std::sqrt(float(encDim));
    run.mem().initParams(o.seed, projStd);
    uint64_t step = 0;
    double bestVal = 1e30;
    if (!o.fresh) loadRunCheckpoint(run, o, step, bestVal);
    fs::create_directories(o.outDir);
    const std::string logPath = (fs::path(o.outDir) / "mem_log.csv").string();
    if (!fs::exists(logPath)) std::ofstream(logPath) << "step,loss,val_loss,val_bpb,lr,gnorm,examples_per_s,loss_scale\n";
    std::signal(SIGINT, onMemSignal);
    std::signal(SIGTERM, onMemSignal);
    std::cout << "training: " << o.batch << " examples x " << o.accum << " micro-batches per step, seq " << o.seq
              << ", up to " << o.maxDocs << " docs per example, steps " << step << " -> " << o.steps << ", "
              << trainSplit.examples.size() << " training examples\n";
    if (haveVal && step == 0) {
        auto [l, b] = evaluate(run, val, o, false);
        std::cout << "step 0  val " << std::setprecision(4) << l << (b >= 0 ? " (" + std::to_string(b) + " bpb)" : "")
                  << "  [untrained module]\n";
    }
    auto lastCkpt = clock::now(), tLog = clock::now();
    double lossAcc = 0.0;
    uint64_t lossTok = 0, goodSteps = 0;
    uint32_t badInARow = 0, logged = 0;
    const uint64_t count = trainSplit.examples.size();
    while (step < o.steps && !gStop) {
        for (uint32_t micro = 0; micro < o.accum; ++micro) {
            std::vector<uint32_t> idx(o.batch);
            for (uint32_t b = 0; b < o.batch; ++b)
                idx[b] = uint32_t(mix(o.seed, (step * o.accum + micro) * 4096 + b) % count);
            auto [tok, bytes] = run.setBatch(trainSplit, idx, false, 0);
            (void)bytes;
            lossAcc += run.step(true);
            lossTok += tok;
        }
        const float lr = lrAt(o, step), scale = run.dec().lossScaleValue();
        {
            run.mem().beginRecording();
            ComputeContext::Recorder rec(ctx);
            run.mem().recordOptimizer(rec, lr, step, 1.0f / (float(o.accum) * scale), o.beta1, o.beta2, o.eps, o.wd,
                                      o.clip);
            rec.submitAndWait();
        }
        const float ns = run.mem().lastNormSq();
        if (!std::isfinite(ns)) {
            goodSteps = 0;
            if (++badInARow >= 40) throw std::runtime_error("gradients not finite for 40 steps in a row");
            if (run.dec().usesTensorCores() && scale > 1.0f) {
                run.dec().setLossScale(std::max(1.0f, scale * 0.5f));
                std::cout << "fp16 gradient overflow: step skipped, loss scale -> " << run.dec().lossScaleValue() << "\n";
            } else {
                std::cout << "warning: non-finite gradient, step skipped\n";
            }
        } else {
            badInARow = 0;
            if (run.dec().usesTensorCores() && ++goodSteps >= 1000 && scale < 16777216.0f) {
                goodSteps = 0;
                run.dec().setLossScale(scale * 2.0f);
            }
        }
        ++step;
        ++logged;
        double valLoss = -1.0, valBpb = -1.0;
        if (haveVal && o.evalEvery && (step % o.evalEvery == 0 || step == o.steps)) {
            std::tie(valLoss, valBpb) = evaluate(run, val, o, false);
            if (valLoss < bestVal) {
                bestVal = valLoss;
                run.mem().save((fs::path(o.outDir) / "best").string());
            }
        }
        if (step % o.logEvery == 0 || valLoss >= 0 || step == o.steps) {
            const double secs = std::chrono::duration<double>(clock::now() - tLog).count();
            tLog = clock::now();
            const double trainLoss = lossAcc / double(std::max<uint64_t>(1, lossTok));
            const float gnorm = std::isfinite(ns) ? std::sqrt(ns) / (float(o.accum) * scale) : -1.0f;
            const double eps = double(logged) * o.batch * o.accum / std::max(secs, 1e-9);
            std::cout << "step " << step << "  loss " << std::fixed << std::setprecision(4) << trainLoss;
            if (valLoss >= 0) std::cout << "  val " << valLoss;
            if (valBpb >= 0) std::cout << " (" << valBpb << " bpb)";
            std::cout << "  lr " << std::scientific << std::setprecision(2) << lr << std::fixed << "  gnorm "
                      << std::setprecision(3) << gnorm << "  " << std::setprecision(1) << eps << " ex/s\n"
                      << std::flush;
            std::ofstream(logPath, std::ios::app) << step << "," << trainLoss << "," << valLoss << "," << valBpb << ","
                                                  << lr << "," << gnorm << "," << eps << "," << scale << "\n";
            lossAcc = 0.0;
            lossTok = 0;
            logged = 0;
        }
        if (std::chrono::duration<double>(clock::now() - lastCkpt).count() >= o.ckptMinutes * 60.0) {
            saveRunCheckpoint(run, o.outDir, o.mirrorDir, step, bestVal);
            lastCkpt = clock::now();
        }
    }
    if (gStop) std::cout << "stop requested: saving checkpoint\n";
    saveRunCheckpoint(run, o.outDir, o.mirrorDir, step, bestVal);
    run.mem().save((fs::path(o.outDir) / "module").string());
    const std::vector<float> after = run.dec().paramsHost();
    std::cout << "module -> " << (fs::path(o.outDir) / "module").string() << "; decoder parameters "
              << (fnv(after.data(), after.size()) == decHash ? "unchanged (frozen)" : "CHANGED (bug)") << "\n";
    if (fnv(after.data(), after.size()) != decHash) return 1;
    return 0;
}

const char* kUsage =
    "usage:\n"
    "  NeonML --memorilla encode    --decoder <ckpt> --sequences f.seq --out f.nmem [--layer -1] [--batch 32] [--seq 128]\n"
    "  NeonML --memorilla train     --decoder <ckpt> --data <dir> --out <dir> [--mem-config m.json] [--steps N]\n"
    "                               [--batch 16] [--accum 1] [--seq 128] [--max-docs 64] [--lr 3e-4] [--min-lr 0]\n"
    "                               [--wd 0.01] [--warmup-frac 0.03] [--clip 1] [--eval-every 200] [--eval-batches 20]\n"
    "                               [--mirror <dir>] [--fresh] [--precision auto|fp16|fp32] [--no-mxfp4]\n"
    "  NeonML --memorilla eval      --decoder <ckpt> --data <dir> --module <dir> [--ablation none|closed|noise|untrained]\n"
    "  NeonML --memorilla gradcheck --config dec.json --data <dir> --out <dir> [--mem-config m.json]\n"
    "  NeonML --memorilla params    [--mem-config m.json]\n"
    "  NeonML --memorilla init      --mem-config m.json --out <dir> [--seed N]   (random module, for tests)\n";

}  // namespace

int runMemorillaCli(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << kUsage;
        return 2;
    }
    const std::string mode = argv[2];
    RunOptions o;
    std::string configPath, seqPath, out;
    int32_t layer = -1;
    bool batchSet = false;
    for (int i = 3; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::runtime_error("missing value after " + a);
            return argv[++i];
        };
        if (a == "--decoder") o.decoder = next();
        else if (a == "--data") o.dataDir = next();
        else if (a == "--out") out = next();
        else if (a == "--mirror") o.mirrorDir = next();
        else if (a == "--mem-config") o.memConfig = next();
        else if (a == "--module") o.moduleDir = next();
        else if (a == "--config") configPath = next();
        else if (a == "--sequences") seqPath = next();
        else if (a == "--layer") layer = int32_t(std::stol(next()));
        else if (a == "--ablation") o.ablation = next();
        else if (a == "--batch") { o.batch = uint32_t(std::stoul(next())); batchSet = true; }
        else if (a == "--accum") o.accum = std::max(1u, uint32_t(std::stoul(next())));
        else if (a == "--seq") o.seq = uint32_t(std::stoul(next()));
        else if (a == "--max-docs") o.maxDocs = uint32_t(std::stoul(next()));
        else if (a == "--steps") o.steps = std::stoull(next());
        else if (a == "--warmup-frac") o.warmupFrac = std::stof(next());
        else if (a == "--lr") o.lr = std::stof(next());
        else if (a == "--min-lr") o.minLr = std::stof(next());
        else if (a == "--wd") o.wd = std::stof(next());
        else if (a == "--clip") o.clip = std::stof(next());
        else if (a == "--eval-every") o.evalEvery = uint32_t(std::stoul(next()));
        else if (a == "--eval-batches") o.evalBatches = uint32_t(std::stoul(next()));
        else if (a == "--log-every") o.logEvery = std::max(1u, uint32_t(std::stoul(next())));
        else if (a == "--ckpt-minutes") o.ckptMinutes = std::stod(next());
        else if (a == "--seed") o.seed = std::stoull(next());
        else if (a == "--precision") o.precision = next();
        else if (a == "--fresh") o.fresh = true;
        else if (a == "--no-mxfp4") o.mxfp4 = false;
        else if (a == "--shaders") o.shaders = next();
        else if (a == "--device") o.device = next();
        else throw std::runtime_error("unknown option " + a + "\n" + kUsage);
    }
    if (!out.empty()) o.outDir = out;
    if (o.seq % 64) throw std::runtime_error("--seq must be a multiple of 64");
    if (o.ablation != "none" && o.ablation != "closed" && o.ablation != "noise" && o.ablation != "untrained")
        throw std::runtime_error("--ablation must be none, closed, noise or untrained");

    if (mode == "params") return runParams(o);
    if (mode == "init") {  // a randomly initialized module (tests); dims from --mem-config
        if (o.memConfig.empty() || out.empty()) throw std::runtime_error("init needs --mem-config and --out");
        const MemorillaConfig m = MemorillaConfig::fromJson(readText(o.memConfig));
        ComputeContext ctx(o.shaders, o.device);
        MemorillaModule mod(ctx, m, false, false);
        mod.initParams(o.seed, 0.02f);
        mod.save(out);
        std::cout << "memorilla: random module -> " << out << "\n";
        return 0;
    }
    if (mode == "encode") {
        if (o.decoder.empty() || seqPath.empty() || out.empty()) throw std::runtime_error("encode needs --decoder, --sequences, --out");
        if (!batchSet) o.batch = 32;
        return runEncode(o, seqPath, out, layer);
    }
    if (mode == "gradcheck") {
        if (configPath.empty() || o.dataDir.empty() || out.empty())
            throw std::runtime_error("gradcheck needs --config, --data and --out");
        if (!batchSet) o.batch = 2;
        return runGradcheck(o, configPath);
    }
    if (mode == "train" || mode == "eval") {
        if (o.decoder.empty() || o.dataDir.empty()) throw std::runtime_error(mode + " needs --decoder and --data");
        return runTrainOrEval(o, mode == "train");
    }
    std::cerr << kUsage;
    return 2;
}

}  // namespace lve::ml
