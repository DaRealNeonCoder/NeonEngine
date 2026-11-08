// NeonML: the headless ML entry points of NeonEngine without the windowed apps
// (no GLFW), so it builds and runs on cloud GPUs (Linux, Google Colab).
//
//   NeonML --gpt-oss <checkpoint dir> ...           inference (gpt_oss_app.hpp)
//   NeonML --gpt-oss-train train|export|gradcheck   training (gpt_oss_train.hpp)
//   NeonML --memorilla encode|train|eval|...         Memorilla memory module (memorilla.hpp)

#include "gpt_oss_app.hpp"
#include "gpt_oss_train.hpp"
#include "memorilla.hpp"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

int main(int argc, char** argv) {
    const std::string mode = argc > 1 ? argv[1] : "";
    try {
        if (mode == "--gpt-oss") return lve::runGptOssCli(argc, argv);
        if (mode == "--gpt-oss-train") return lve::ml::runGptOssTrainCli(argc, argv);
        if (mode == "--memorilla") return lve::ml::runMemorillaCli(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return EXIT_FAILURE;
    }
    std::cerr << "usage: NeonML --gpt-oss <checkpoint dir> [options]\n"
                 "       NeonML --gpt-oss-train train|export|gradcheck [options]\n"
                 "       NeonML --memorilla encode|train|eval|gradcheck|params [options]\n";
    return 2;
}
