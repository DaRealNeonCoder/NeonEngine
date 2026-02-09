#pragma once

#include "lve_camera.hpp"
#include "lve_device.hpp"
#include "lve_frame_info.hpp"
#include "lve_game_object.hpp"
#include "lve_pipeline.hpp"
#include "water_physics.hpp"

// std
#include <memory>
#include <vector>

namespace lve {
class WaterRenderSystem {
  struct InstanceData {
    glm::vec4 posRadius;  // xyz + particle radius
    glm::vec4 color;      // rgb + density/pressure/etc
  };

 public:
  WaterRenderSystem(
      LveDevice &device,
      VkRenderPass renderPass,
      VkDescriptorSetLayout globalSetLayout,
      int _particleCount,
      std::vector<LveModel::Vertex> particleVerts);
  ~WaterRenderSystem();

  WaterRenderSystem(const WaterRenderSystem &) = delete;
  WaterRenderSystem &operator=(const WaterRenderSystem &) = delete;

    void renderGameObjects(WaterFrameInfo &frameInfo, WaterPhysics &waterPhys);
  void updateBuffers(std::vector<glm::vec4> &positions, std::vector<glm::vec3> &colors);
  int particleVert;
 private:
  void createPipelineLayout(VkDescriptorSetLayout globalSetLayout);
  void createPipeline(VkRenderPass renderPass);
  void createBuffers(int particleCount, std::vector<LveModel::Vertex> particleVerts);

  LveDevice &lveDevice;
  std::unique_ptr<LveBuffer> meshBuffer;
  std::unique_ptr<LveBuffer> instanceBuffer;

  std::unique_ptr<LvePipeline> lvePipeline;
  std::vector<InstanceData> particleInstances;

  VkPipelineLayout pipelineLayout;

  int particleCount;
};
}  // namespace lve
