#pragma once

#include <vulkan.h>

#include <glm.hpp>
#include <memory>
#include <vector>

#include "lve_device.hpp"
#include "lve_frame_info.hpp"
#include "lve_pipeline.hpp"
#include "lve_swap_chain.hpp"  // for LveSwapChain::MAX_FRAMES_IN_FLIGHT

namespace lve {

class ShadowSystem {
 public:
  ShadowSystem(LveDevice& device, VkDescriptorSetLayout globalSetLayout);
  ShadowSystem(const ShadowSystem&) = delete;
  ShadowSystem& operator=(const ShadowSystem&) = delete;

  void destroy();

  void update(FrameInfo& frameInfo, GlobalUbo& ubo);
  void render(FrameInfo& frameInfo);

  // Accessors for the light-space matrix and light dir (for writing into GlobalUbo)
  glm::mat4 getLightSpaceMatrix() const { return lightSpaceMatrix_; }
  glm::vec4 getLightDirection() const { return lightDir_; }

  // returns the descriptor info for the shadow map corresponding to a particular frame-in-flight
  VkDescriptorImageInfo getShadowMapDescriptor(uint32_t frameIndex) const;

 private:
  void createShadowResources();
  void createRenderPass();
  void createFramebuffer();
  void createSampler();
  void createPipelineLayout(VkDescriptorSetLayout globalSetLayout);
  void createPipeline();

 private:
  // --- in ShadowSystem class private members ---
  LveDevice& lveDevice;

  // single shadow resources (instead of per-frame vectors)
  VkImage shadowImage = VK_NULL_HANDLE;
  VkDeviceMemory shadowImageMemory = VK_NULL_HANDLE;
  VkImageView shadowImageView = VK_NULL_HANDLE;
  VkFramebuffer shadowFramebuffer = VK_NULL_HANDLE;

  VkSampler shadowSampler = VK_NULL_HANDLE;
  VkRenderPass shadowRenderPass = VK_NULL_HANDLE;
  VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
  std::unique_ptr<LvePipeline> shadowPipeline;

  glm::mat4 lightSpaceMatrix_{};

  glm::vec4 lightDir_;

  static constexpr uint32_t SHADOW_MAP_SIZE = 4096;
};

}  // namespace lve
