// MctsBench: command-line entry for the MCTS-VK experiment harness (see mcts_app.cpp).
//   MctsBench                  -> every experiment (long: includes the round-robin matches)
//   MctsBench ncbench          -> timing matched to MCTS-NC's bench_mctsnc.py
//   MctsBench sanity verify determinism ncbench bench bias replay counters matches   (any subset)
//     verify      : results bit-identical for every slots/group setting
//     determinism : repeated identical searches give identical visits
// Options: --profile        per-kernel GPU timestamps in ncbench
//          --counters       instrumentation counters on in ncbench (lane efficiency, playout length)
//          --slots N        children per playout group (0 = auto, 1..8)
//          --trees N        number of trees (default 64)
//          --ips N          iterations per submit in the determinism test
//          --shader PATH    mcts_connect4.slang.spv to load
#include "mcts_app.hpp"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    std::vector<std::string> args(argv + 1, argv + argc);
    try {
        lve::MctsApp app{};
        app.run(args);
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
