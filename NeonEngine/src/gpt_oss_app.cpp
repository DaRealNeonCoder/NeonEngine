#include "gpt_oss_app.hpp"

#include "gpt_oss_model.hpp"
#include "memorilla.hpp"
#include "ml_compute.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace lve {

namespace {

std::vector<uint32_t> parseIds(const std::string& text) {
    std::vector<uint32_t> ids;
    std::string cur;
    auto flush = [&] {
        if (!cur.empty()) {
            ids.push_back(uint32_t(std::stoul(cur)));
            cur.clear();
        }
    };
    for (char c : text) {
        if (c >= '0' && c <= '9') cur += c;
        else flush();
    }
    flush();
    return ids;
}

std::string readFile(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + path);
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

// Samples a token from logits with temperature and optional top-k.
uint32_t sampleToken(const std::vector<float>& logits, float temperature, uint32_t topK, std::mt19937_64& rng) {
    std::vector<uint32_t> idx(logits.size());
    for (uint32_t i = 0; i < idx.size(); ++i) idx[i] = i;
    size_t keep = idx.size();
    if (topK > 0 && topK < keep) {
        std::partial_sort(idx.begin(), idx.begin() + topK, idx.end(),
                          [&](uint32_t a, uint32_t b) { return logits[a] > logits[b]; });
        keep = topK;
    }
    float mx = -3.0e38f;
    for (size_t i = 0; i < keep; ++i) mx = std::max(mx, logits[idx[i]]);
    std::vector<double> w(keep);
    for (size_t i = 0; i < keep; ++i) w[i] = std::exp(double(logits[idx[i]] - mx) / double(temperature));
    std::discrete_distribution<size_t> dist(w.begin(), w.end());
    return idx[dist(rng)];
}

}  // namespace

int runGptOssCli(int argc, char** argv) {
    using namespace lve::ml;
    if (argc < 3) {
        std::cerr << "usage: LveEngine --gpt-oss <checkpoint original/ dir> [--tokens ids] [--max-new N] ...\n";
        return 2;
    }
    const std::string modelDir = argv[2];
    std::string shaders = "shaders/ML";
    std::string deviceName;
    std::string dumpPath;
    std::vector<uint32_t> prompt;
    std::vector<uint32_t> stops = { 200002, 200012 };
    uint32_t maxNew = 32;
    bool score = false;
    float temperature = 0.0f;  // 0 = greedy
    uint32_t topK = 0;
    uint64_t seed = 0;
    GptOssOptions opts;
    MemorillaOptions mem;          // off unless --memorilla is given
    std::string softPrefixPath;    // --soft-prefix: raw fp32 input vectors fed before the prompt (testing)
    std::string memDumpPath;       // --mem-dump: write z_q and the memory tokens (testing)
    std::string memQueryVecPath;   // --mem-query-vec: z_q for stores made by an external encoder

    for (int i = 3; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::runtime_error("missing value for " + a);
            return argv[++i];
        };
        if (a == "--tokens") prompt = parseIds(next());
        else if (a == "--tokens-file") prompt = parseIds(readFile(next()));
        else if (a == "--max-new") maxNew = uint32_t(std::stoul(next()));
        else if (a == "--stop") stops = parseIds(next());
        else if (a == "--ctx") opts.maxContext = uint32_t(std::stoul(next()));
        else if (a == "--dense") {
            std::string v = next();
            if (v == "bf16") opts.dense = DenseFormat::BF16;
            else if (v == "q8") opts.dense = DenseFormat::Q8;
            else throw std::runtime_error("--dense must be bf16 or q8");
        } else if (a == "--experts") {
            std::string v = next();
            if (v == "auto") opts.experts = ExpertPlacement::Auto;
            else if (v == "device") opts.experts = ExpertPlacement::Device;
            else if (v == "host") opts.experts = ExpertPlacement::Host;
            else throw std::runtime_error("--experts must be auto, device or host");
        } else if (a == "--score") score = true;
        else if (a == "--dump-logits") dumpPath = next();
        else if (a == "--shaders") shaders = next();
        else if (a == "--device") deviceName = next();
        else if (a == "--quiet") opts.verbose = false;
        else if (a == "--temperature") temperature = std::stof(next());
        else if (a == "--top-k") topK = uint32_t(std::stoul(next()));
        else if (a == "--seed") seed = std::stoull(next());
        else if (a == "--memorilla") { mem.enabled = true; mem.moduleDir = next(); }
        else if (a == "--mem-store") mem.storePath = next();
        else if (a == "--mem-k") mem.slotsToUse = uint32_t(std::stoul(next()));
        else if (a == "--mem-query") mem.queryTokens = parseIds(next());
        else if (a == "--mem-query-file") mem.queryTokens = parseIds(readFile(next()));
        else if (a == "--mem-query-vec") memQueryVecPath = next();
        else if (a == "--mem-dump") memDumpPath = next();
        else if (a == "--soft-prefix") softPrefixPath = next();
        else throw std::runtime_error("unknown option " + a);
    }
    if (prompt.empty()) throw std::runtime_error("no prompt tokens (use --tokens or --tokens-file)");

    ComputeContext ctx(shaders, deviceName);
    std::cout << "device: " << ctx.deviceName() << " (subgroup " << ctx.subgroupSize()
              << (ctx.hasCoopMat() ? ", coopmat" : "") << ")\n";
    GptOssModel model(ctx, modelDir, opts);
    if (!mem.enabled && (!mem.storePath.empty() || mem.slotsToUse || !mem.queryTokens.empty() ||
                         !memDumpPath.empty() || !memQueryVecPath.empty()))
        throw std::runtime_error("--mem-* options need --memorilla <module dir>");
    if (mem.enabled && !softPrefixPath.empty()) throw std::runtime_error("use either --memorilla or --soft-prefix");

    // Input vectors fed before the prompt: Memorilla memory tokens or a raw --soft-prefix file.
    std::vector<float> prefix;
    const uint32_t H = model.config().hidden;
    if (!softPrefixPath.empty()) {
        const std::string raw = readFile(softPrefixPath);
        if (raw.size() % (size_t(H) * 4)) throw std::runtime_error("--soft-prefix size is not a multiple of hidden * 4 bytes");
        prefix.resize(raw.size() / 4);
        std::memcpy(prefix.data(), raw.data(), raw.size());
    }
    if (mem.enabled) {
        using clk = std::chrono::steady_clock;
        if (mem.storePath.empty()) throw std::runtime_error("--memorilla needs --mem-store <file.nmem>");
        const MemorillaConfig mcfg = MemorillaModule::loadConfig(mem.moduleDir);
        const MemoryStore store = MemoryStore::load(mem.storePath);
        if (mcfg.llmDim != H) throw std::runtime_error("memorilla: the module was built for hidden size " + std::to_string(mcfg.llmDim));
        if (store.dim != mcfg.encoderDim) throw std::runtime_error("memorilla: store width differs from the module's encoder_dim");
        if (store.liveCount() == 0) throw std::runtime_error("memorilla: the store has no documents");
        const std::string fp = decoderFingerprint(model.config());
        if (!mcfg.decoder.empty() && mcfg.decoder != fp)
            std::cerr << "warning: the module was trained for a different decoder (" << mcfg.decoder << ")\n";
        // z_q: the self encoder (mean of rmsnorm(x_L) over the query tokens), or a given vector.
        const auto t0 = clk::now();
        std::vector<float> zq(mcfg.encoderDim, 0.0f);
        if (!memQueryVecPath.empty()) {
            const std::string raw = readFile(memQueryVecPath);
            if (raw.size() != zq.size() * 4) throw std::runtime_error("--mem-query-vec must hold encoder_dim floats");
            std::memcpy(zq.data(), raw.data(), raw.size());
        } else {
            if (store.encoder != "self" || mcfg.encoderDim != H || store.encoderLayer != -1)
                throw std::runtime_error("memorilla: this store needs --mem-query-vec (not encoded by this decoder's final layer)");
            if (!store.decoder.empty() && store.decoder != fp)
                std::cerr << "warning: the store was encoded with a different decoder (" << store.decoder << ")\n";
            const std::vector<uint32_t>& q = mem.queryTokens.empty() ? prompt : mem.queryTokens;
            if (q.size() > opts.maxContext) throw std::runtime_error("memorilla: query longer than --ctx");
            std::vector<float> hid;
            for (uint32_t t : q) {
                model.forward(t, false, nullptr, nullptr, &hid);
                for (uint32_t c = 0; c < H; ++c) zq[c] += hid[c] / float(q.size());
            }
            model.reset();
        }
        const double encS = std::chrono::duration<double>(clk::now() - t0).count();
        // Memory module (fp32 GEMMs; its cost is negligible next to decoding).
        const uint32_t n = uint32_t(store.liveCount());
        MemorillaModule module(ctx, mcfg, false, false);
        module.allocate(1, n);
        module.load(mem.moduleDir);
        const std::vector<float> docs = store.liveVectors();
        std::memcpy(module.docsHost(), docs.data(), docs.size() * 4);
        module.countsHost()[0] = n;
        std::memcpy(module.queryHost(), zq.data(), zq.size() * 4);
        module.beginRecording();
        double modMs = 0.0;
        {
            ComputeContext::Recorder rec(ctx);
            rec.timestamp(0);
            module.recordForward(rec, 1, n);
            rec.timestamp(1);
            modMs = rec.submitAndWait();
        }
        const uint32_t k = mem.slotsToUse ? std::min(mem.slotsToUse, mcfg.slots) : mcfg.slots;
        std::vector<float> M(size_t(mcfg.slots) * H);
        ctx.download(module.outputBuffer(), 0, M.data(), M.size() * 4);
        prefix.assign(M.begin(), M.begin() + size_t(k) * H);
        if (!memDumpPath.empty()) {
            std::FILE* f = std::fopen(memDumpPath.c_str(), "wb");
            if (!f) throw std::runtime_error("cannot write " + memDumpPath);
            std::fwrite(zq.data(), 4, zq.size(), f);
            std::fwrite(M.data(), 4, M.size(), f);
            std::fclose(f);
        }
        std::cout << "memorilla: " << k << " memory tokens from " << n << " documents (query encode " << std::fixed
                  << std::setprecision(3) << encS << " s, module " << modMs << " ms)\n";
    }
    const uint32_t prefixRows = uint32_t(prefix.size() / H);
    if (prefixRows + prompt.size() + (score ? 0 : maxNew) > opts.maxContext)
        throw std::runtime_error("prompt + generation exceeds --ctx");
    for (uint32_t r = 0; r < prefixRows; ++r) model.forwardEmbedding(prefix.data() + size_t(r) * H, false);

    std::FILE* dump = nullptr;
    if (!dumpPath.empty()) {
        dump = std::fopen(dumpPath.c_str(), "wb");
        if (!dump) throw std::runtime_error("cannot write " + dumpPath);
    }
    using clock = std::chrono::steady_clock;
    std::vector<float> logits;
    std::vector<float>* logitsOut = (dump || temperature > 0.0f) ? &logits : nullptr;
    std::mt19937_64 rng(seed);

    // Prompt.
    const auto t0 = clock::now();
    uint32_t next = UINT32_MAX;
    double gpuMsTotal = 0.0;
    for (size_t i = 0; i < prompt.size(); ++i) {
        const bool last = i + 1 == prompt.size();
        double ms = 0.0;
        next = model.forward(prompt[i], score || last, logitsOut, &ms);
        if (last && !score && temperature > 0.0f) next = sampleToken(logits, temperature, topK, rng);
        gpuMsTotal += ms;
        if (dump && (score || last)) std::fwrite(logits.data(), 4, logits.size(), dump);
    }
    const double promptS = std::chrono::duration<double>(clock::now() - t0).count();
    std::cout << "prompt: " << prompt.size() << " tokens in " << std::fixed << std::setprecision(3) << promptS << " s ("
              << std::setprecision(1) << double(prompt.size()) / promptS << " tok/s, GPU " << gpuMsTotal / prompt.size()
              << " ms/token)\n";

    if (score) {
        if (dump) std::fclose(dump);
        std::cout << "next token after prompt: " << next << "\n";
        return 0;
    }

    // Generation (greedy, or sampled with --temperature / --top-k).
    std::vector<uint32_t> generated;
    const auto t1 = clock::now();
    gpuMsTotal = 0.0;
    for (uint32_t n = 0; n < maxNew; ++n) {
        generated.push_back(next);
        std::cout << next << (n + 1 < maxNew ? "," : "") << std::flush;
        if (std::find(stops.begin(), stops.end(), next) != stops.end()) break;
        if (n + 1 == maxNew) break;
        double ms = 0.0;
        next = model.forward(next, true, logitsOut, &ms);
        if (temperature > 0.0f) next = sampleToken(logits, temperature, topK, rng);
        gpuMsTotal += ms;
        if (dump) std::fwrite(logits.data(), 4, logits.size(), dump);
    }
    const double genS = std::chrono::duration<double>(clock::now() - t1).count();
    std::cout << "\ngenerated " << generated.size() << " tokens";
    if (generated.size() > 1)
        std::cout << " (" << std::setprecision(1) << double(generated.size() - 1) / genS << " tok/s, GPU "
                  << gpuMsTotal / double(generated.size() - 1) << " ms/token)";
    std::cout << "\n";
    if (dump) std::fclose(dump);
    return 0;
}

}  // namespace lve
