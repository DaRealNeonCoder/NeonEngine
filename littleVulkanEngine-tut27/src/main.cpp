
#include "first_app.hpp"
#include "water_app.hpp"
#include "raytracing_app.hpp"

// std
#include <cstdlib>
#include <iostream>
#include <stdexcept>

int main() {
   // lve::FirstApp app{};
   // lve::RayTracingApp app{};
   
   lve::WaterApp app{};


  try {
    app.run();
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}