#include "water_render_system.hpp"
#include "water_physics.hpp"

// libs
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm.hpp>
#include <gtc/constants.hpp>

// std
#include <array>
#include <cassert>
#include <stdexcept>
#include <iostream>
#include "lve_model.hpp"

namespace lve {

struct WaterPushConstantData {
  glm::mat4 modelMatrix{1.f};
  glm::mat4 normalMatrix{1.f};
  glm::vec4 color{0.f};
};

WaterRenderSystem::WaterRenderSystem(
    LveDevice& device,
    VkRenderPass renderPass,
    VkDescriptorSetLayout globalSetLayout,
    int _particleCount,
    std::vector<LveModel::Vertex> particleVerts)
    : lveDevice{device} {
  createPipelineLayout(globalSetLayout);
  createPipeline(renderPass);

  particleCount = _particleCount;
  particleInstances.resize(particleCount);
  std::cout << particleInstances.size()<< " that was the instances size \n";
  createBuffers(particleCount, particleVerts);
}

WaterRenderSystem::~WaterRenderSystem() {
  vkDestroyPipelineLayout(lveDevice.device(), pipelineLayout, nullptr);
}
void WaterRenderSystem::updateBuffers(std::vector<glm::vec4>& positions, std::vector<glm::vec3> &colors) {
    for (size_t i = 0; i < particleInstances.size(); i++) {
    particleInstances[i].posRadius = glm::vec4(positions[i].x, positions[i].y, positions[i].z, 0.17f);
    particleInstances[i].color = glm::vec4(0.f, 0.4f, 0.8f, 0.f);

  }
    instanceBuffer->writeToBuffer(particleInstances.data());
    instanceBuffer->flush();
}

void WaterRenderSystem::createPipelineLayout(VkDescriptorSetLayout globalSetLayout) {
  VkPushConstantRange pushConstantRange{};
  pushConstantRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
  pushConstantRange.offset = 0;
  pushConstantRange.size = sizeof(WaterPushConstantData);

  std::vector<VkDescriptorSetLayout> descriptorSetLayouts{globalSetLayout};

  VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
  pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  pipelineLayoutInfo.setLayoutCount = static_cast<uint32_t>(descriptorSetLayouts.size());
  pipelineLayoutInfo.pSetLayouts = descriptorSetLayouts.data();
  pipelineLayoutInfo.pushConstantRangeCount = 1;
  pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;
  if (vkCreatePipelineLayout(lveDevice.device(), &pipelineLayoutInfo, nullptr, &pipelineLayout) !=
      VK_SUCCESS) {
    throw std::runtime_error("failed to create pipeline layout!");
  }
}
void WaterRenderSystem::createPipeline(VkRenderPass renderPass) {
  assert(pipelineLayout != nullptr && "Cannot create pipeline before pipeline layout");

  PipelineConfigInfo pipelineConfig{};
  LvePipeline::defaultPipelineConfigInfo(pipelineConfig);

  // =========================
  // TWO INSTANCE BINDINGS
  // =========================
  std::vector<VkVertexInputBindingDescription> bindings(2);

  bindings[0].binding = 0;
  bindings[0].stride = sizeof(glm::vec4);
  bindings[0].inputRate = VK_VERTEX_INPUT_RATE_INSTANCE;

  bindings[1].binding = 1;
  bindings[1].stride = sizeof(glm::vec4);
  bindings[1].inputRate = VK_VERTEX_INPUT_RATE_INSTANCE;

  pipelineConfig.bindingDescriptions = bindings;

  // =========================
  // ATTRIBUTES
  // =========================
  std::vector<VkVertexInputAttributeDescription> attributes(2);

  attributes[0].location = 0;
  attributes[0].binding = 0;
  attributes[0].format = VK_FORMAT_R32G32B32A32_SFLOAT;
  attributes[0].offset = 0;

  attributes[1].location = 1;
  attributes[1].binding = 1;
  attributes[1].format = VK_FORMAT_R32G32B32A32_SFLOAT;
  attributes[1].offset = 0;

  pipelineConfig.attributeDescriptions = attributes;

  pipelineConfig.renderPass = renderPass;
  pipelineConfig.pipelineLayout = pipelineLayout;

  lvePipeline = std::make_unique<LvePipeline>(
      lveDevice,
      "C:\\Users\\ZyBros\\Downloads\\NeonEngine\\NeonEngine\\shaders\\water_shader.vert.spv",
      "C:\\Users\\ZyBros\\Downloads\\NeonEngine\\NeonEngine\\shaders\\water_shader.frag.spv",
      pipelineConfig);
}
void WaterRenderSystem::createBuffers(
    int particleCount, std::vector<LveModel::Vertex> particleVerts) {
  instanceBuffer = std::make_unique<LveBuffer>(
      lveDevice,
      sizeof(glm::vec4),  // ONLY positions now
      particleCount,
      VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

  instanceBuffer->map();
}
void WaterRenderSystem::renderGameObjects(WaterFrameInfo& frameInfo, WaterPhysics& waterPhys) {
  lvePipeline->bind(frameInfo.commandBuffer);

  vkCmdBindDescriptorSets(
      frameInfo.commandBuffer,
      VK_PIPELINE_BIND_POINT_GRAPHICS,
      pipelineLayout,
      0,
      1,
      &frameInfo.globalDescriptorSet,
      0,
      nullptr);

  // Use compute-written position buffer (outputBuffer) + compute-written color buffer
  VkBuffer buffers[] = {waterPhys.partPosBuff->getBuffer(), waterPhys.colorsBuff->getBuffer()};
  VkDeviceSize offsets[] = {0, 0};

  // Bind instance buffers at binding 0 & 1
  vkCmdBindVertexBuffers(frameInfo.commandBuffer, 0, 2, buffers, offsets);

  // Draw (6 vertices per instance quad)
  vkCmdDraw(frameInfo.commandBuffer, 6, particleCount, 0, 0);
}

}  // namespace lve
