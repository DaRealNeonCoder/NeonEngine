#pragma once

// ============================
// Vulkan
// ============================
#include <vulkan.h>

#include "lve_frame_info.hpp"
// ============================
// GLM
// ============================
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm.hpp>
#include <gtc/matrix_transform.hpp>

// ============================
// Standard library
// ============================
#include <array>
#include <cstdint>
#include <memory>
#include <vector>

#include "lve_buffer.hpp"

namespace lve {

// Forward declarations
struct RayTracingScratchBuffer {
  uint64_t deviceAddress = 0;
  VkBuffer handle = VK_NULL_HANDLE;
  VkDeviceMemory memory = VK_NULL_HANDLE;
};

struct AccelerationStructure {
  VkAccelerationStructureKHR handle = VK_NULL_HANDLE;
  uint64_t deviceAddress = 0;
  VkDeviceMemory memory = VK_NULL_HANDLE;
  VkBuffer buffer = VK_NULL_HANDLE;
};
struct RayTracingVertex {
  glm::vec4 pos;
  glm::vec4 normal;
  alignas(16) uint32_t materialIndex;
};

struct Material {
  glm::vec4 emission;
  glm::vec4 albedo;
  glm::vec4 misc;//x = roughness, y = IOR(only for glass), z = material type (0 = normal, 67 = reflective), w = empty
};
class RayTracingSystem {
 public:
  explicit RayTracingSystem(
      LveDevice& device3,
      VkFormat format,
      VkDescriptorSetLayout globalSetLayout,
      std::vector<RayTracingVertex> allVertex,
      std::vector<uint32_t> allIndicies,
      uint32_t width,
      uint32_t height);
  ~RayTracingSystem();
  void VK_CHECK_RESULT(VkResult f);

  void prepare();
  void render(FrameInfo& frameInfo);
  void handleResize(
      uint32_t width, uint32_t height, const std::vector<VkDescriptorSet>& descriptorSets);
  void updateUniforms(uint32_t frameIndex, const glm::mat4& view, const glm::mat4& proj);
  void createMaterialBuffer();
  void copyStorageImageToSwapChain(
      VkCommandBuffer commandBuffer,
      VkImage swapChainImage,
      uint32_t width,
      uint32_t height,
      uint32_t frameIndex);

  VkDescriptorImageInfo getStorageImageDescriptor(uint32_t frameIndex) const;
  VkAccelerationStructureKHR getTLAS() const;
  VkDescriptorBufferInfo getMaterialBufferDescriptor() const {
    return materialBuffer->descriptorInfo();
  }
  VkDescriptorBufferInfo getVertexBufferDescriptor() const {
    return vertexBuffer->descriptorInfo();
  }
  VkDescriptorBufferInfo getIndexBufferDescriptor() const {
    return indexBuffer->descriptorInfo();
  }
  void resetFrameId() { frameID = 0; }
 private:
  LveDevice& vulkanDevice;
  VkDevice device;

  PFN_vkGetBufferDeviceAddressKHR vkGetBufferDeviceAddressKHR{nullptr};
  PFN_vkCreateAccelerationStructureKHR vkCreateAccelerationStructureKHR{nullptr};
  PFN_vkDestroyAccelerationStructureKHR vkDestroyAccelerationStructureKHR{nullptr};
  PFN_vkGetAccelerationStructureBuildSizesKHR vkGetAccelerationStructureBuildSizesKHR{nullptr};
  PFN_vkGetAccelerationStructureDeviceAddressKHR vkGetAccelerationStructureDeviceAddressKHR{
      nullptr};
  PFN_vkCmdBuildAccelerationStructuresKHR vkCmdBuildAccelerationStructuresKHR{nullptr};
  PFN_vkBuildAccelerationStructuresKHR vkBuildAccelerationStructuresKHR{nullptr};
  PFN_vkCmdTraceRaysKHR vkCmdTraceRaysKHR{nullptr};
  PFN_vkGetRayTracingShaderGroupHandlesKHR vkGetRayTracingShaderGroupHandlesKHR{nullptr};
  PFN_vkCreateRayTracingPipelinesKHR vkCreateRayTracingPipelinesKHR{nullptr};

  VkPhysicalDeviceRayTracingPipelinePropertiesKHR rayTracingPipelineProperties{};
  VkPhysicalDeviceAccelerationStructureFeaturesKHR accelerationStructureFeatures{};

  VkPhysicalDeviceBufferDeviceAddressFeatures enabledBufferDeviceAddressFeatures{};
  VkPhysicalDeviceRayTracingPipelineFeaturesKHR enabledRayTracingPipelineFeatures{};
  VkPhysicalDeviceAccelerationStructureFeaturesKHR enabledAccelerationStructureFeatures{};

  AccelerationStructure bottomLevelAS{};
  AccelerationStructure topLevelAS{};

  VkFormat swapChainImageFormat;

  std::unique_ptr<LveBuffer> vertexBuffer;
  std::unique_ptr<LveBuffer> indexBuffer;
  std::unique_ptr<LveBuffer> instanceBuffer;

  uint32_t indexCount{0};
  std::unique_ptr<LveBuffer> transformBuffer;
  VkPipeline rayTracingPipeline = VK_NULL_HANDLE;

  // ==========================================================
  // Shader Binding Tables
  // ==========================================================
  std::vector<VkRayTracingShaderGroupCreateInfoKHR> shaderGroups{};
  std::unique_ptr<LveBuffer> raygenShaderBindingTable;
  std::unique_ptr<LveBuffer> missShaderBindingTable;
  std::unique_ptr<LveBuffer> hitShaderBindingTable;
  std::unique_ptr<LveBuffer> materialBuffer;

  std::vector<RayTracingVertex> vertices;
  std::vector<uint32_t> indices;
  // ==========================================================
  // Storage Image - NOW PER FRAME
  // ==========================================================
  struct StorageImage {
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkImage image = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkFormat format{};
    uint32_t width;
    uint32_t height;
  };
  StorageImage storageImage;
  StorageImage bloomImage;  
  StorageImage bloomTempImage;  
  
  struct UniformData {
    glm::mat4 viewInverse;
    glm::mat4 projInverse;
  };
  std::vector<std::unique_ptr<LveBuffer>> uniformBuffers;

  // ==========================================================
  // Pipeline & descriptors
  // ==========================================================
  VkPipeline pipeline{VK_NULL_HANDLE};
  VkPipelineLayout pipelineLayout{VK_NULL_HANDLE};
  VkDescriptorPool descriptorPool{VK_NULL_HANDLE};
  std::vector<VkDescriptorSet> descriptorSets;
  bool resized;
  // ==========================================================
  // Internal helpers
  // ==========================================================
  RayTracingScratchBuffer createScratchBuffer(VkDeviceSize size);
  void deleteScratchBuffer(RayTracingScratchBuffer& scratchBuffer);

  void createAccelerationStructureBuffer(
      AccelerationStructure& accelerationStructure,
      const VkAccelerationStructureBuildSizesInfoKHR& buildSizeInfo);

  uint64_t getBufferDeviceAddress(VkBuffer buffer);

  void createStorageImage(StorageImage& storageImage, uint32_t width, uint32_t height);
  void destroyStorageImage(StorageImage& storageImage);
  void createBottomLevelAccelerationStructure();
  void createTopLevelAccelerationStructure();
  void createRayTracingPipeline(VkDescriptorSetLayout globalSetLayout);
  void createShaderBindingTable();

  void loadFunctionPointers();
  uint32_t frameID;
};

}  // namespace lve