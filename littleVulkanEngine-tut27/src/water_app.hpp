#pragma once

#include "lve_descriptors.hpp"
#include "lve_device.hpp"
#include "lve_game_object.hpp"
#include "lve_renderer.hpp"
#include "lve_window.hpp"

// std
#include <memory>
#include <vector>

namespace lve {
class WaterApp {
 public:
  std::vector<LveGameObject> waterParticles;
  static constexpr int WIDTH = 800;
  static constexpr int HEIGHT = 600;

  WaterApp();
  ~WaterApp();

  WaterApp(const WaterApp &) = delete;
  WaterApp &operator=(const WaterApp &) = delete;

  void run();

 private:
  void loadGameObjects();

  LveWindow lveWindow{WIDTH, HEIGHT, "Vulkan Tutorial"};
  LveDevice lveDevice{lveWindow};
  LveRenderer lveRenderer{lveWindow, lveDevice};

  // note: order of declarations matters
  std::unique_ptr<LveDescriptorPool> globalPool{};
  LveGameObject::Map gameObjects;
  std::vector<glm::vec3> colors;

  glm::vec4 boxDim{1.f,1.f,1.f,1.f};
};
}  // namespace lve
