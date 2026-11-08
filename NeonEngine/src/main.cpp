
#include "first_app.hpp"
#include "water_app.hpp"
#include "raytracing_app.hpp"
#include "mcts_app.hpp"
#include "digit_recognizer_app.hpp"
#include "gpt_oss_app.hpp"
#include "gpt_oss_train.hpp"
#include "memorilla.hpp"

// std
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

int main(int argc, char** argv) {
    // TODO: first app is broken. I think its a sampler/descriptor set mismatch or smth,
    // cuz we moved from three frames in flight to one.


    // lve::FirstApp app{};
    //lve::RayTracingApp app{};
   //lve::WaterApp app{};
   //lve::MctsApp app{};
  // "--gpt-oss <dir> ..." runs the gpt-oss LLM headless (see gpt_oss_app.hpp).
  if (argc > 1 && std::string(argv[1]) == "--gpt-oss") {
    try {
      return lve::runGptOssCli(argc, argv);
    } catch (const std::exception &e) {
      std::cerr << e.what() << '\n';
      return EXIT_FAILURE;
    }
  }
  // "--gpt-oss-train train|export|gradcheck ..." trains gpt-oss from scratch (see gpt_oss_train.hpp).
  if (argc > 1 && std::string(argv[1]) == "--gpt-oss-train") {
    try {
      return lve::ml::runGptOssTrainCli(argc, argv);
    } catch (const std::exception &e) {
      std::cerr << e.what() << '\n';
      return EXIT_FAILURE;
    }
  }
  // "--memorilla encode|train|eval|..." trains / evaluates a Memorilla memory module (see memorilla.hpp).
  if (argc > 1 && std::string(argv[1]) == "--memorilla") {
    try {
      return lve::ml::runMemorillaCli(argc, argv);
    } catch (const std::exception &e) {
      std::cerr << e.what() << '\n';
      return EXIT_FAILURE;
    }
  }

  // "--serve" runs the Vulkan digit model as a process for the drawing app.
  const bool serveMode = argc > 1 && std::string(argv[1]) == "--serve";
  if (serveMode) lve::beginServeProtocol();  // must precede app construction (see digit_recognizer_app.hpp)

   lve::DigitRecognizerApp app{};


  try {
    if (serveMode) {
      app.serve();
    } else {
      app.run();
    }
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}