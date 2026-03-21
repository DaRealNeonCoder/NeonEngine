

#include "shadow_system.hpp"

// glm
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm.hpp>
#include <gtc/matrix_transform.hpp>

// std
#include <array>
#include <iostream>
#include <stdexcept>

namespace lve {

ShadowSystem::ShadowSystem(LveDevice& device, VkDescriptorSetLayout globalSetLayout)
    : lveDevice{device} {
  std::cout << "Pass 2.1 \n";

  createShadowResources();

  std::cout << "Pass 2.2 \n";

  createRenderPass();
  std::cout << "Pass 2.3 \n";

  createFramebuffer();
  std::cout << "Pass 2.4 \n";

  createSampler();
  std::cout << "Pass 2.5 \n";

  createPipelineLayout(globalSetLayout);
  std::cout << "Pass 2.6 \n";

  createPipeline();
  std::cout << "Pass 2.7 \n";
}

void ShadowSystem::destroy() {
  vkDeviceWaitIdle(lveDevice.device());

  // destroy single framebuffer / view / image / memory
  if (shadowFramebuffer != VK_NULL_HANDLE) {
    vkDestroyFramebuffer(lveDevice.device(), shadowFramebuffer, nullptr);
    shadowFramebuffer = VK_NULL_HANDLE;
  }

  if (shadowImageView != VK_NULL_HANDLE) {
    vkDestroyImageView(lveDevice.device(), shadowImageView, nullptr);
    shadowImageView = VK_NULL_HANDLE;
  }

  if (shadowImage != VK_NULL_HANDLE) {
    vkDestroyImage(lveDevice.device(), shadowImage, nullptr);
    shadowImage = VK_NULL_HANDLE;
  }

  if (shadowImageMemory != VK_NULL_HANDLE) {
    vkFreeMemory(lveDevice.device(), shadowImageMemory, nullptr);
    shadowImageMemory = VK_NULL_HANDLE;
  }

  if (shadowSampler != VK_NULL_HANDLE) {
    vkDestroySampler(lveDevice.device(), shadowSampler, nullptr);
    shadowSampler = VK_NULL_HANDLE;
  }

  if (shadowRenderPass != VK_NULL_HANDLE) {
    vkDestroyRenderPass(lveDevice.device(), shadowRenderPass, nullptr);
    shadowRenderPass = VK_NULL_HANDLE;
  }

  if (pipelineLayout != VK_NULL_HANDLE) {
    vkDestroyPipelineLayout(lveDevice.device(), pipelineLayout, nullptr);
    pipelineLayout = VK_NULL_HANDLE;
  }
}

//COMPLETE
void ShadowSystem::createShadowResources() {
  VkFormat depthFormat = VK_FORMAT_D16_UNORM;

  VkImageCreateInfo imageInfo{};
  imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  imageInfo.imageType = VK_IMAGE_TYPE_2D;
  imageInfo.extent = {SHADOW_MAP_SIZE, SHADOW_MAP_SIZE, 1};
  imageInfo.mipLevels = 1;
  imageInfo.arrayLayers = 1;
  imageInfo.format = depthFormat;
  imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
  imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  imageInfo.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
  imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
  imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  lveDevice.createImageWithInfo(
      imageInfo,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
      shadowImage,
      shadowImageMemory);

  VkImageViewCreateInfo viewInfo{};
  viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
  viewInfo.image = shadowImage;
  viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
  viewInfo.format = depthFormat;
  viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
  viewInfo.subresourceRange.baseMipLevel = 0;
  viewInfo.subresourceRange.levelCount = 1;
  viewInfo.subresourceRange.baseArrayLayer = 0;
  viewInfo.subresourceRange.layerCount = 1;

  if (vkCreateImageView(lveDevice.device(), &viewInfo, nullptr, &shadowImageView) != VK_SUCCESS) {
    throw std::runtime_error("failed to create shadow image view");
  }
}


//COMPLETE
void ShadowSystem::createRenderPass() {
  VkAttachmentDescription depthAttachment{};
  depthAttachment.format = VK_FORMAT_D16_UNORM;
  depthAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
  depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
  depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
  depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
  depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
  depthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  depthAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;

  VkAttachmentReference depthRef{};
  depthRef.attachment = 0;
  depthRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

  VkSubpassDescription subpass{};
  subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
  subpass.colorAttachmentCount = 0;  // No color attachments
  subpass.pDepthStencilAttachment = &depthRef;

  std::array<VkSubpassDependency, 2> dependencies{};

  dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
  dependencies[0].dstSubpass = 0;
  dependencies[0].srcStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
  dependencies[0].dstStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
  dependencies[0].srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
  dependencies[0].dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
  dependencies[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

  dependencies[1].srcSubpass = 0;
  dependencies[1].dstSubpass = VK_SUBPASS_EXTERNAL;
  dependencies[1].srcStageMask = VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
  dependencies[1].dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
  dependencies[1].srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
  dependencies[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  dependencies[1].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

  VkRenderPassCreateInfo renderPassInfo{};
  renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
  renderPassInfo.attachmentCount = 1;
  renderPassInfo.pAttachments = &depthAttachment;
  renderPassInfo.subpassCount = 1;
  renderPassInfo.pSubpasses = &subpass;
  renderPassInfo.dependencyCount = static_cast<uint32_t>(dependencies.size());
  renderPassInfo.pDependencies = dependencies.data();

  if (vkCreateRenderPass(lveDevice.device(), &renderPassInfo, nullptr, &shadowRenderPass) !=
      VK_SUCCESS) {
    throw std::runtime_error("failed to create shadow render pass");
  }
}

//COMPLETE
void ShadowSystem::createFramebuffer() {
  VkFramebufferCreateInfo fbInfo{};
  fbInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
  fbInfo.renderPass = shadowRenderPass;
  fbInfo.attachmentCount = 1;
  fbInfo.pAttachments = &shadowImageView;
  fbInfo.width = SHADOW_MAP_SIZE;
  fbInfo.height = SHADOW_MAP_SIZE;
  fbInfo.layers = 1;

  if (vkCreateFramebuffer(lveDevice.device(), &fbInfo, nullptr, &shadowFramebuffer) != VK_SUCCESS) {
    throw std::runtime_error("failed to create shadow framebuffer");
  }
}

//COMPLETE
void ShadowSystem::createSampler() {
  VkSamplerCreateInfo samplerInfo{};
  samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
  samplerInfo.magFilter = VK_FILTER_LINEAR;
  samplerInfo.minFilter = VK_FILTER_LINEAR;
  samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  samplerInfo.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
  samplerInfo.compareEnable = VK_TRUE;
  samplerInfo.compareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
  samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
  samplerInfo.minLod = 0.0f;
  samplerInfo.maxLod = 1.0f;
  samplerInfo.mipLodBias = 0.0f;
  samplerInfo.maxAnisotropy = 1.0f;

  if (vkCreateSampler(lveDevice.device(), &samplerInfo, nullptr, &shadowSampler) != VK_SUCCESS) {
    throw std::runtime_error("failed to create shadow sampler");
  }
}

void ShadowSystem::createPipelineLayout(VkDescriptorSetLayout globalSetLayout) {
  VkPushConstantRange pushConstantRange{};
  pushConstantRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
  pushConstantRange.offset = 0;
  pushConstantRange.size = sizeof(glm::mat4);  // model matrix

  VkPipelineLayoutCreateInfo layoutInfo{};
  layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  layoutInfo.setLayoutCount = 1;
  layoutInfo.pSetLayouts = &globalSetLayout;
  layoutInfo.pushConstantRangeCount = 1;
  layoutInfo.pPushConstantRanges = &pushConstantRange;
  if (vkCreatePipelineLayout(lveDevice.device(), &layoutInfo, nullptr, &pipelineLayout) !=
      VK_SUCCESS) {
    throw std::runtime_error("failed to create shadow pipeline layout");
  }
}
void ShadowSystem::createPipeline() {
  PipelineConfigInfo config{};
  LvePipeline::defaultPipelineConfigInfo(config);

  config.renderPass = shadowRenderPass;
  config.pipelineLayout = pipelineLayout;


 config.inputAssemblyInfo.flags = 0;

  // depth-only
  config.colorBlendInfo.attachmentCount = 0;
  config.depthStencilInfo.depthWriteEnable = VK_TRUE;
  config.depthStencilInfo.depthTestEnable = VK_TRUE;
  config.depthStencilInfo.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;

  config.rasterizationInfo.frontFace =VK_FRONT_FACE_COUNTER_CLOCKWISE;                                
  config.rasterizationInfo.cullMode = VK_CULL_MODE_NONE;  
  config.colorBlendAttachment.blendEnable = VK_FALSE;


  VkPipelineRasterizationStateCreateInfo pipelineRasterizationStateCreateInfo{};
  pipelineRasterizationStateCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
  pipelineRasterizationStateCreateInfo.polygonMode = VK_POLYGON_MODE_FILL;
  pipelineRasterizationStateCreateInfo.cullMode = VK_CULL_MODE_BACK_BIT;
  pipelineRasterizationStateCreateInfo.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
  pipelineRasterizationStateCreateInfo.flags = 0;
  pipelineRasterizationStateCreateInfo.depthClampEnable = VK_FALSE;
  pipelineRasterizationStateCreateInfo.lineWidth = 1.0f;
  config.rasterizationInfo = pipelineRasterizationStateCreateInfo;
  
  config.colorBlendAttachment.colorWriteMask = 0xf;
  config.colorBlendAttachment.blendEnable = VK_TRUE;
  config.colorBlendInfo.flags = 1;
  config.colorBlendInfo.pAttachments = &config.colorBlendAttachment;

  config.depthStencilInfo.back.compareOp = VK_COMPARE_OP_ALWAYS;
  config.depthStencilInfo.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
  config.depthStencilInfo.depthTestEnable= VK_TRUE;
  config.depthStencilInfo.depthWriteEnable = VK_TRUE;


  // Configure depth bias to reduce shadow acne
  config.rasterizationInfo.depthBiasEnable = VK_TRUE;


  			// Set depth bias (aka "Polygon offset")
  // Required to avoid shadow mapping artifacts

  shadowPipeline = std::make_unique<LvePipeline>(
      lveDevice,
      "C:\\Users\\ZyBros\\Downloads\\NeonEngine\\NeonEngine\\shaders\\shadow_depth.vert.spv",
      "",
      config);
}

void ShadowSystem::update(FrameInfo&, GlobalUbo& ubo) {
    // let it be known that I debugged for 10 days STRAIGHT, only to realize that my shadows werent rendering because of some weird y 
    // coordinate thing vulkan does.
    // nice.
  glm::vec3 lightPos = glm::vec3(-3.0f, -3.0f, -3.0f);
  glm::vec3 lightTarget = glm::vec3(0.0f, 0.0f, 0.0f);

  float orthoSize = 10.0f;
  float near = 0.1f;
  float far = 100.0f;


  glm::mat4 depthProjectionMatrix = glm::perspective(glm::radians(45.0f), 1.0f, 1.0f, 8.0f);
  glm::mat4 depthViewMatrix = glm::lookAt(lightPos, glm::vec3(0.0f), glm::vec3(0, 1, 0));
  glm::mat4 depthModelMatrix = glm::mat4(1.0f);

  //depthProjectionMatrix[1][1] *= -1.0f;

  ubo.lightSpaceMatrix = depthProjectionMatrix * depthViewMatrix;

  glm::vec3 lightDir = glm::normalize(lightTarget - lightPos);
  ubo.lightDir = glm::vec4(lightDir, 0.0f);
}

void ShadowSystem::render(FrameInfo& frameInfo) {
  VkClearValue clearValues[2]{};
  clearValues[0].depthStencil = {1.0f, 0};

  VkRenderPassBeginInfo rpInfo{};
  rpInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
  rpInfo.renderPass = shadowRenderPass;
  rpInfo.framebuffer = shadowFramebuffer;  // single framebuffer
  rpInfo.renderArea.offset = {0, 0};
  rpInfo.renderArea.extent = {SHADOW_MAP_SIZE, SHADOW_MAP_SIZE};
  rpInfo.clearValueCount = 2;
  rpInfo.pClearValues = clearValues;

  vkCmdBeginRenderPass(frameInfo.commandBuffer, &rpInfo, VK_SUBPASS_CONTENTS_INLINE);

  VkViewport viewport{};
  viewport.x = 0.0f;
  viewport.y = 0.0f;
  viewport.width = static_cast<float>(SHADOW_MAP_SIZE);
  viewport.height = static_cast<float>(SHADOW_MAP_SIZE);
  viewport.minDepth = 0.0f;
  viewport.maxDepth = 1.0f;

  VkRect2D scissor{};
  scissor.offset = {0, 0};
  scissor.extent = {SHADOW_MAP_SIZE, SHADOW_MAP_SIZE};

  vkCmdSetViewport(frameInfo.commandBuffer, 0, 1, &viewport);
  vkCmdSetScissor(frameInfo.commandBuffer, 0, 1, &scissor);
  vkCmdSetDepthBias(frameInfo.commandBuffer, 1.25f, 0.0f, 1.75f);
  shadowPipeline->bind(frameInfo.commandBuffer);


  vkCmdBindDescriptorSets(
      frameInfo.commandBuffer,
      VK_PIPELINE_BIND_POINT_GRAPHICS,
      pipelineLayout,
      0,
      1,
      &frameInfo.globalDescriptorSet,
      0,
      nullptr);

  for (auto& kv : frameInfo.gameObjects) {
    auto& obj = kv.second;
    if (!obj.model) continue;

    struct ShadowPushConstants {
      glm::mat4 model{};
    } push;

    push.model = obj.transform.mat4();
    vkCmdPushConstants(
        frameInfo.commandBuffer,
        pipelineLayout,
        VK_SHADER_STAGE_VERTEX_BIT,
        0,
        sizeof(push),
        &push);

    obj.model->bind(frameInfo.commandBuffer);
    obj.model->draw(frameInfo.commandBuffer);
  }

  vkCmdEndRenderPass(frameInfo.commandBuffer);
}

VkDescriptorImageInfo ShadowSystem::getShadowMapDescriptor(uint32_t x) const {
  VkDescriptorImageInfo info{};
  info.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
  info.imageView = shadowImageView;
  info.sampler = shadowSampler;
  return info;
}

}  
