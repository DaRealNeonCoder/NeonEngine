#include "mcts_app.hpp"

#include "mcts_gpu.hpp"

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
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// =============================================================================
// MCTS experiment harness (all four variants).
//   1. sanity   : tactical positions, legal move, anomaly counter == 0
//   2. bench    : throughput / per-kernel time at a fixed iteration budget
//   3. bias     : mirror-symmetry tests (detects column bias)
//   4. matches  : round robin, paired colour-swapped random openings, equal time per move,
//                 JSONL logs (-> analyze_matches.py)
// =============================================================================

namespace lve {

    namespace {

        constexpr int COLS = MctsGpu::COLS;
        constexpr int ROWS = MctsGpu::ROWS;
        constexpr int CELLS = MctsGpu::CELLS;
        using Board = std::array<int, CELLS>;

        // ---- experiment settings ------------------------------------------------
        constexpr bool RUN_SANITY = true;
        constexpr bool RUN_BENCH = true;
        constexpr bool RUN_BIAS = true;
        constexpr bool RUN_MATCHES = true;

        constexpr int NUM_TREES = 64;
        constexpr int MAX_ITERS = 4000;        // node-pool cap; matches log how often it was hit (must be rare)
        constexpr int ITERS_PER_SUBMIT = 50;
        constexpr float UCB_C = 1.4f;          // MUST equal the CPU reference's c when comparing against it
        constexpr uint32_t EXP_SEED = 20241;

        constexpr int BENCH_ITERS = 200, BENCH_REPS = 10;
        constexpr int BIAS_REPS = 64, BIAS_ITERS = 20;
        constexpr int SYM_REPS = 200, SYM_ITERS = 4;

        constexpr double MOVE_MS = 200.0;      // equal wall-clock budget per move for every engine
        constexpr int N_OPENINGS = 10;         // each opening is played twice (colours swapped); use >= 200 for real runs
        constexpr int OPENING_PLIES = 4;       // random plies before the engines take over
        constexpr bool INCLUDE_SELFPLAY = false;

        const char* SHADER_PATH = "C:\\Users\\ZyBros\\Downloads\\NeonEngine\\NeonEngine\\shaders\\mcts_connect4.slang.spv";  // adjust; must be the NEW build
        const char* OUT_DIR = "C:\\Users\\ZyBros\\Downloads\\NeonEngine\\mcts_results";  // created if missing; "" = current working dir
        const char* OUT_PREFIX = "mcts_";      // output files: <prefix><expid>_meta.json / _games.jsonl / _moves.jsonl

        const std::array<Algo, 4> ALGOS = { Algo::AcpProdigal, Algo::AcpThrifty, Algo::OcpProdigal, Algo::OcpThrifty };
        using Engines = std::vector<std::unique_ptr<MctsGpu>>;

        // ---- small utilities ------------------------------------------------------
        uint32_t h32(uint32_t x) {
            x ^= x >> 16; x *= 0x7feb352dU;
            x ^= x >> 15; x *= 0x846ca68bU;
            x ^= x >> 16;
            return x;
        }
        uint32_t mix3(uint32_t a, uint32_t b, uint32_t c) { return h32(a ^ h32(b + 0x9E3779B9u * (c + 1u))); }

        std::string fnvHex(const std::string& s) {
            uint32_t h = 2166136261u;
            for (unsigned char ch : s) { h ^= ch; h *= 16777619u; }
            char buf[16];
            std::snprintf(buf, sizeof(buf), "%08x", h);
            return buf;
        }

        std::string jsonEscape(const std::string& s) {
            std::string o;
            for (char ch : s) {
                if (ch == '\\' || ch == '"') o += '\\';
                o += ch;
            }
            return o;
        }

        struct Acc {
            int n = 0; double s = 0, s2 = 0;
            void add(double x) { n++; s += x; s2 += x * x; }
            double mean() const { return n ? s / n : 0.0; }
            double sd() const {
                if (n < 2) return 0.0;
                const double m = mean();
                return std::sqrt(std::max(0.0, (s2 - n * m * m) / (n - 1)));
            }
        };

        const char* nameOf(const MctsGpu& e) { return algoName(e.config().algo); }

        // rows[0] is the TOP row. 'X' = +1 (moves first), 'O' = -1, '.' = empty.
        Board parseBoard(const std::array<std::string, ROWS>& rows) {
            Board b{};
            for (int r = 0; r < ROWS; r++) {
                const std::string& line = rows[ROWS - 1 - r];
                for (int c = 0; c < COLS; c++) b[r * COLS + c] = (line[c] == 'X') ? 1 : (line[c] == 'O') ? -1 : 0;
            }
            return b;
        }

        Board mirrorBoard(const Board& b) {
            Board m{};
            for (int r = 0; r < ROWS; r++)
                for (int c = 0; c < COLS; c++) m[r * COLS + (COLS - 1 - c)] = b[r * COLS + c];
            return m;
        }

        SearchLimits iterLimits(int n) { SearchLimits l; l.maxIterations = n; return l; }

        // ---- host-side Connect Four (used only to run matches) --------------------
        struct C4 {
            Board cells{};
            std::array<int, COLS> h{};
            int turn = 1;   // +1 (black / X) always moves first
            int moves = 0;

            bool legal(int c) const { return c >= 0 && c < COLS && h[c] < ROWS; }

            bool wins(int col, int row, int p) const {
                static const int DC[4] = { 1, 0, 1, 1 }, DR[4] = { 0, 1, 1, -1 };
                for (int d = 0; d < 4; d++) {
                    int count = 1;
                    for (int sgn = -1; sgn <= 1; sgn += 2) {
                        int c = col + sgn * DC[d], r = row + sgn * DR[d];
                        while (c >= 0 && c < COLS && r >= 0 && r < ROWS && cells[r * COLS + c] == p) {
                            count++; c += sgn * DC[d]; r += sgn * DR[d];
                        }
                    }
                    if (count >= 4) return true;
                }
                return false;
            }

            // returns true if the move wins
            bool play(int c) {
                const int r = h[c], p = turn;
                cells[r * COLS + c] = p;
                h[c] = r + 1;
                moves++;
                turn = -p;
                return wins(c, r, p);
            }
        };

        // ============================================================================
        // 1. SANITY
        // ============================================================================
        struct TestCase { const char* name; Board board; int turn; int expected; };

        void runSanity(Engines& engines) {
            std::cout << "\n######## 1. SANITY (tactics, legality, anomalies) ########\n";
            const std::vector<TestCase> tests = {
                { "empty board", Board{}, +1, 3 },
                { "X wins in col 3", parseBoard({ ".......", ".......", ".......", ".......", ".......", "XXX.OOO" }), +1, 3 },
                { "X must block col 3", parseBoard({ ".......", ".......", ".......", ".......", "....X..", "OOO.XX." }), +1, 3 },
            };
            for (auto& e : engines) {
                for (const auto& t : tests) {
                    auto r = e->search(t.board, t.turn, iterLimits(200));
                    const bool ok = (r.bestAction == t.expected);
                    std::printf("%-13s %-20s best=%d %-12s anomalies=%lld  %7.1f ms  visits:",
                        nameOf(*e), t.name, r.bestAction, ok ? "[OK]" : "[UNEXPECTED]",
                        static_cast<long long>(r.stats.anomalies), r.ms);
                    for (int c = 0; c < COLS; c++) std::printf(" %d", r.visits[c]);
                    std::printf("\n");
                    if (r.stats.anomalies != 0) std::printf("   !!! anomaly counter non-zero\n");
                }
            }
        }

        // ============================================================================
        // 2. BENCHMARK (fixed iterations; isolates speed from strength)
        // ============================================================================
        void runBench(Engines& engines) {
            std::cout << "\n######## 2. BENCHMARK (" << BENCH_ITERS << " iterations, " << BENCH_REPS
                << " reps after 2 warm-ups) ########\n";
            const Board mid = parseBoard({ ".......", ".......", "...O...", "..XX...", "..OXO..", ".XOXOX." });
            const std::vector<std::pair<const char*, std::pair<Board, int>>> positions = {
                { "empty", { Board{}, +1 } }, { "midgame", { mid, -1 } } };

            std::printf("%-13s %-8s %9s %9s %9s %7s %12s %12s %7s %9s %7s\n", "algo", "pos", "wall ms", "(sd)", "gpu ms",
                "ovhd%", "real pl/s", "credit pl/s", "depth", "nodes", "blkUtl");
            for (auto& e : engines) {
                for (const auto& p : positions) {
                    for (int i = 0; i < 2; i++) e->search(p.second.first, p.second.second, iterLimits(BENCH_ITERS));
                    Acc wall, gpu, ovh, real, cred, depth, nodes, util;
                    for (int i = 0; i < BENCH_REPS; i++) {
                        auto r = e->search(p.second.first, p.second.second, iterLimits(BENCH_ITERS));
                        const auto& s = r.stats;
                        wall.add(s.wallMs); gpu.add(s.gpuMs); ovh.add(100.0 * s.overheadFrac);
                        real.add(s.playoutsRealPerSec); cred.add(s.playoutsCreditedPerSec);
                        depth.add(s.leafDepthMean); nodes.add(double(s.nodesTotal)); util.add(s.blockUtilReal);
                    }
                    std::printf("%-13s %-8s %9.2f %9.2f %9.2f %7.1f %12.0f %12.0f %7.2f %9.0f %7.2f\n", nameOf(*e), p.first,
                        wall.mean(), wall.sd(), gpu.mean(), ovh.mean(), real.mean(), cred.mean(), depth.mean(), nodes.mean(),
                        util.mean());
                }
            }
            std::cout << "(real pl/s counts simulated playouts only; credit pl/s also counts virtual playouts of terminal children)\n";
        }

        // ============================================================================
        // 3. BIAS: mirror symmetry. Connect Four is left-right symmetric, so searching a position and its
        //    mirror must give statistically identical (mirrored) visit shares, and the best action on the
        //    empty board must be symmetric in distribution.
        // ============================================================================
        void runBias(Engines& engines) {
            std::cout << "\n######## 3. BIAS (mirror symmetry) ########\n";
            const Board P = parseBoard({ ".......", ".......", "...O...", "..XX...", "..OXO..", ".XOXOX." });
            const Board M = mirrorBoard(P);
            const int turn = -1;

            for (auto& e : engines) {
                // (a) visit-share mirror test
                std::array<Acc, COLS> accP, accM;
                int64_t anomalies = 0;
                for (int i = 0; i < BIAS_REPS; i++) {
                    auto rp = e->search(P, turn, iterLimits(BIAS_ITERS));
                    auto rm = e->search(M, turn, iterLimits(BIAS_ITERS));
                    anomalies += rp.stats.anomalies + rm.stats.anomalies;
                    double tp = 0, tm = 0;
                    for (int c = 0; c < COLS; c++) { tp += rp.visits[c]; tm += rm.visits[c]; }
                    for (int c = 0; c < COLS; c++) {
                        accP[c].add(tp > 0 ? rp.visits[c] / tp : 0.0);
                        accM[c].add(tm > 0 ? rm.visits[c] / tm : 0.0);
                    }
                }
                double maxZ = 0; int maxC = 0;
                for (int c = 0; c < COLS; c++) {
                    const Acc& a = accP[c]; const Acc& b = accM[COLS - 1 - c];
                    const double se = std::sqrt(a.sd() * a.sd() / a.n + b.sd() * b.sd() / b.n);
                    const double z = se > 1e-12 ? (a.mean() - b.mean()) / se : 0.0;
                    if (std::fabs(z) > std::fabs(maxZ)) { maxZ = z; maxC = c; }
                }

                // (b) empty-board best-action symmetry (small budget => many ties => exposes tie-break bias)
                std::array<int, COLS> cnt{};
                for (int i = 0; i < SYM_REPS; i++) {
                    auto r = e->search(Board{}, +1, iterLimits(SYM_ITERS));
                    if (r.bestAction >= 0) cnt[r.bestAction]++;
                }
                double chi2 = 0;
                for (int c = 0; c < COLS / 2; c++) {
                    const double n1 = cnt[c], n2 = cnt[COLS - 1 - c];
                    if (n1 + n2 > 0) chi2 += (n1 - n2) * (n1 - n2) / (n1 + n2);
                }

                std::printf("%-13s visit-share mirror: max|z|=%.2f (col %d)  %s | empty-board best-action counts:", nameOf(*e),
                    std::fabs(maxZ), maxC, std::fabs(maxZ) > 3.5 ? "[SUSPECT]" : "[ok]");
                for (int c = 0; c < COLS; c++) std::printf(" %d", cnt[c]);
                std::printf("  chi2(df=3)=%.2f %s  anomalies=%lld\n", chi2, chi2 > 11.34 ? "[SUSPECT]" : "[ok]",
                    static_cast<long long>(anomalies));
            }
            std::cout << "(max|z| over 7 columns > 3.5 or chi2 > 11.34 means a left/right preference worth investigating)\n";
        }

        // ============================================================================
        // 4. MATCHES
        // ============================================================================
        struct EngineAcc {
            Acc wallMs, iters, realPlPerSec;
            int moves = 0, capped = 0;
            int64_t anomalies = 0;
        };

        std::vector<int> makeOpening(int k) {
            std::mt19937 rng(mix3(EXP_SEED, 0xBEEF, static_cast<uint32_t>(k)));
            while (true) {
                C4 g;
                std::vector<int> moves;
                bool terminal = false;
                for (int p = 0; p < OPENING_PLIES && !terminal; p++) {
                    int c;
                    do { c = static_cast<int>(rng() % COLS); } while (!g.legal(c));
                    moves.push_back(c);
                    terminal = g.play(c);
                }
                if (!terminal) return moves;
            }
        }

        // returns +1 black won, 0 draw, -1 white won. Black = the player who moved first (turn +1).
        int playGame(int gameId, int openingId, const std::string& matchup, int blackIdx, int whiteIdx,
            const std::vector<int>& opening, Engines& engines, std::vector<EngineAcc>& accs, std::ofstream& movesOut,
            int& pliesOut) {
            C4 g;
            for (int c : opening) g.play(c);
            int ply = static_cast<int>(opening.size());
            int outcome = 0;
            while (true) {
                const int idx = (g.turn == 1) ? blackIdx : whiteIdx;
                MctsGpu& e = *engines[idx];
                SearchLimits lim;
                lim.maxIterations = MAX_ITERS;
                lim.maxMs = MOVE_MS;
                auto r = e.search(g.cells, g.turn, lim, mix3(EXP_SEED, static_cast<uint32_t>(gameId), static_cast<uint32_t>(ply)));
                if (!g.legal(r.bestAction)) {
                    throw std::runtime_error(std::string("engine ") + nameOf(e) + " returned illegal action " +
                        std::to_string(r.bestAction) + " in game " + std::to_string(gameId));
                }
                EngineAcc& a = accs[idx];
                a.moves++;
                a.wallMs.add(r.stats.wallMs);
                a.iters.add(r.stats.iterations);
                a.realPlPerSec.add(r.stats.playoutsRealPerSec);
                a.anomalies += r.stats.anomalies;
                if (r.stats.iterations >= MAX_ITERS) a.capped++;

                movesOut << "{\"game\":" << gameId << ",\"matchup\":\"" << matchup << "\",\"opening_id\":" << openingId
                    << ",\"ply\":" << ply << ",\"player\":\"" << nameOf(e) << "\",\"turn\":" << g.turn
                    << ",\"action\":" << r.bestAction << ",\"stats\":" << r.stats.toJson() << "}\n";

                const bool win = g.play(r.bestAction);
                ply++;
                if (win) { outcome = (g.turn == -1) ? +1 : -1; break; }  // turn already flipped: mover = -turn
                if (g.moves == CELLS) { outcome = 0; break; }
            }
            pliesOut = ply;
            return outcome;
        }

        void runMatches(LveDevice& dev, Engines& engines) {
            std::cout << "\n######## 4. MATCHES (round robin, " << N_OPENINGS << " paired openings, " << MOVE_MS
                << " ms/move) ########\n";

            // ---- experiment metadata + id ----
            std::ostringstream meta;
            meta << "{\"exp_seed\":" << EXP_SEED << ",\"num_trees\":" << NUM_TREES << ",\"max_iters\":" << MAX_ITERS
                << ",\"iters_per_submit\":" << ITERS_PER_SUBMIT << ",\"ucb_c\":" << UCB_C << ",\"move_ms\":" << MOVE_MS
                << ",\"opening_plies\":" << OPENING_PLIES << ",\"n_openings\":" << N_OPENINGS
                << ",\"playouts_per_child\":" << kPlayoutsPerChild << ",\"device\":\""
                << jsonEscape(dev.properties.deviceName) << "\",\"driver_version\":" << dev.properties.driverVersion
                << ",\"timestamp_period_ns\":" << dev.properties.limits.timestampPeriod << ",\"shader\":\""
                << jsonEscape(SHADER_PATH) << "\",\"algos\":[";
            for (size_t i = 0; i < engines.size(); i++) meta << (i ? "," : "") << "\"" << nameOf(*engines[i]) << "\"";
            meta << "]}";
            const std::string expId = fnvHex(meta.str());
            namespace fs = std::filesystem;
            fs::path dir = std::string(OUT_DIR).empty() ? fs::current_path() : fs::path(OUT_DIR);
            std::error_code ec;
            fs::create_directories(dir, ec);
            const std::string base = (dir / (std::string(OUT_PREFIX) + expId)).string();
            { std::ofstream m(base + "_meta.json"); m << meta.str() << "\n"; }
            std::ofstream gamesOut(base + "_games.jsonl");
            std::ofstream movesOut(base + "_moves.jsonl");
            if (!gamesOut.is_open() || !movesOut.is_open())
                throw std::runtime_error("cannot open output files under " + dir.string());
            std::cout << "experiment id " << expId << "\nwriting to: " << fs::absolute(dir).string() << "\n";

            // ---- warm-up so first-use costs don't land in the first game ----
            for (auto& e : engines) for (int i = 0; i < 3; i++) e->search(Board{}, +1, iterLimits(100));

            std::vector<EngineAcc> accs(engines.size());
            int gameId = 0;

            for (size_t ia = 0; ia < engines.size(); ia++) {
                for (size_t ib = ia + (INCLUDE_SELFPLAY ? 0 : 1); ib < engines.size(); ib++) {
                    const std::string an = nameOf(*engines[ia]), bn = nameOf(*engines[ib]);
                    const std::string matchup = an + "__vs__" + bn;
                    int w = 0, d = 0, l = 0;  // from A's perspective
                    for (int k = 0; k < N_OPENINGS; k++) {
                        const std::vector<int> opening = makeOpening(k);
                        for (int swap = 0; swap < 2; swap++) {
                            const bool aBlack = (swap == 0);
                            const int blackIdx = aBlack ? static_cast<int>(ia) : static_cast<int>(ib);
                            const int whiteIdx = aBlack ? static_cast<int>(ib) : static_cast<int>(ia);
                            int plies = 0;
                            const int outcome = playGame(++gameId, k, matchup, blackIdx, whiteIdx, opening, engines, accs, movesOut, plies);
                            gamesOut << "{\"game\":" << gameId << ",\"matchup\":\"" << matchup << "\",\"a\":\"" << an
                                << "\",\"b\":\"" << bn << "\",\"opening_id\":" << k << ",\"black\":\""
                                << (aBlack ? "a" : "b") << "\",\"outcome\":" << outcome << ",\"plies\":" << plies
                                << ",\"budget\":{\"type\":\"time_ms\",\"value\":" << MOVE_MS << "}}\n";
                            gamesOut.flush();
                            const double sa = aBlack ? 0.5 * (outcome + 1.0) : 1.0 - 0.5 * (outcome + 1.0);
                            if (sa == 1.0) w++; else if (sa == 0.0) l++; else d++;
                        }
                    }
                    std::printf("%-34s  A: W %2d  D %2d  L %2d   score(A)=%.3f\n", matchup.c_str(), w, d, l,
                        (w + 0.5 * d) / std::max(1, w + d + l));
                }
            }
            movesOut.flush();

            // ---- fairness / validity report ----
            std::cout << "\nper-engine move statistics (all matches):\n";
            std::printf("%-13s %7s %10s %10s %12s %9s %9s\n", "algo", "moves", "wall ms", "iters", "real pl/s", "capped", "anomal.");
            double lo = 1e300, hi = 0;
            for (size_t i = 0; i < engines.size(); i++) {
                const EngineAcc& a = accs[i];
                if (a.moves == 0) continue;
                lo = std::min(lo, a.wallMs.mean()); hi = std::max(hi, a.wallMs.mean());
                std::printf("%-13s %7d %10.2f %10.1f %12.0f %8.1f%% %9lld\n", nameOf(*engines[i]), a.moves, a.wallMs.mean(),
                    a.iters.mean(), a.realPlPerSec.mean(), 100.0 * a.capped / a.moves, static_cast<long long>(a.anomalies));
                if (a.capped > 0.01 * a.moves)
                    std::printf("   !!! %s hit the iteration cap on >1%% of moves: raise MAX_ITERS, otherwise the time budget is not equal\n", nameOf(*engines[i]));
                if (a.anomalies) std::printf("   !!! %s had anomalies\n", nameOf(*engines[i]));
            }
            if (hi > 0 && (hi - lo) / hi > 0.10)
                std::printf("!!! mean time per move differs by >10%% between engines (overshoot of the time limit): compare with care\n");
            std::cout << "\nNext: python analyze_matches.py " << base << "_games.jsonl\n";
        }

    }  // namespace

    void MctsApp::run() {
        Engines engines;
        for (Algo a : ALGOS) {
            MctsGpu::Config cfg;
            cfg.algo = a;
            cfg.numTrees = NUM_TREES;
            cfg.maxIterations = MAX_ITERS;
            cfg.itersPerSubmit = ITERS_PER_SUBMIT;
            cfg.ucbC = UCB_C;
            cfg.seed = EXP_SEED;
            cfg.shaderPath = SHADER_PATH;
            engines.push_back(std::make_unique<MctsGpu>(lveDevice, cfg));
        }

        if (RUN_SANITY) runSanity(engines);
        if (RUN_BENCH) runBench(engines);
        if (RUN_BIAS) runBias(engines);
        if (RUN_MATCHES) runMatches(lveDevice, engines);

        std::cout << "\nDone.\n";
    }

}  // namespace lve