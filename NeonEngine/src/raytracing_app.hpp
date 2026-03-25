#pragma once
#include "raytracing_rast.hpp" // contains GBufferRenderTargets
#include "lve_descriptors.hpp"
#include "lve_device.hpp"
#include "lve_game_object.hpp"
#include "lve_renderer.hpp"
#include "lve_window.hpp"

// std
#include <memory>
#include <vector>

namespace lve {
class RayTracingApp {
 public:
  static constexpr int WIDTH = 800;
  static constexpr int HEIGHT = 600;

  RayTracingApp();
  ~RayTracingApp();

  RayTracingApp(const RayTracingApp &) = delete;
  RayTracingApp &operator=(const RayTracingApp &) = delete;

  void run();
  void recreateGBuffer(VkRenderPass gBufferRenderPass, GBufferRenderTargets& gBuffers, VkFramebuffer& frameBuffer, RayTracingRast& rayTracingRast, VkExtent2D extents);
  void transitionHistoryToGeneral(VkCommandBuffer cmd, VkImage image);
 private:
  void loadGameObjects();

  LveWindow lveWindow{WIDTH, HEIGHT, "Vulkan Tutorial"};
  LveDevice lveDevice{lveWindow};
  LveRenderer lveRenderer{lveWindow, lveDevice};

  // note: order of declarations matters
  std::unique_ptr<LveDescriptorPool> globalPool{};
  LveGameObject::Map gameObjects;

};
}  // namespace lve
