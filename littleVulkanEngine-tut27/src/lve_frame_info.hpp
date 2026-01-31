#pragma once

#include "lve_camera.hpp"
#include "lve_game_object.hpp"

// lib
#include <vulkan.h>

namespace lve {

#define MAX_LIGHTS 1

struct PointLight {
  glm::vec4 position{};  // ignore w
  glm::vec4 color{};     // w is intensity
};

struct GlobalUbo {
  glm::mat4 projection{1.f};
  glm::mat4 inverseProjection{1.f};
  glm::mat4 view{1.f};
  glm::mat4 inverseView{1.f};
  glm::vec4 ambientLightColor{1.f, 1.f, 1.f, .02f};  // w is intensity
  PointLight pointLights[MAX_LIGHTS];
  glm::vec4 numLights;

  glm::mat4 lightSpaceMatrix;  
  glm::vec4 lightDir; 
};
struct WaterUbo {
  glm::mat4 projection{1.f};
  glm::mat4 inverseProjection{1.f};
  glm::mat4 view{1.f};
  glm::mat4 inverseView{1.f};
};

struct FrameInfo {
  int frameIndex;
  float frameTime;
  VkCommandBuffer commandBuffer;
  LveCamera &camera;
  VkDescriptorSet globalDescriptorSet;
  LveGameObject::Map &gameObjects;
};

struct WaterFrameInfo {
  float frameTime;
  VkCommandBuffer commandBuffer;
  LveCamera &camera;
  VkDescriptorSet globalDescriptorSet;
  VkDescriptorSet computeDescriptorSet;
  LveGameObject::Map &gameObjects;
};
}  // namespace lve
