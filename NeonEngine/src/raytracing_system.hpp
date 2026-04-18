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
  glm::vec4 uv;
  uint32_t materialIndex;
  uint32_t pad1;
  uint32_t pad2;
  uint32_t pad3;
};

struct RayTracingMaterial {
  glm::vec4 emission;
  glm::vec4 albedo;
  glm::vec4 position;
  glm::vec4 misc;//x = roughness, y = IOR(only for glass), z = material type (0 = normal, 67 = reflective), w = precomputed area for lights (NEE)
};

class RayTracingSystem {
 public:
     explicit RayTracingSystem(
         LveDevice& device3,
         VkFormat format,
         VkDescriptorSetLayout globalSetLayout,
         VkDescriptorSetLayout restirSetLayout,
         std::vector<RayTracingVertex> allVertex,
         std::vector<uint32_t> allIndicies,
         std::vector<LveMaterial> allMaterials,
         std::vector<glm::vec4> allLightPos,
         uint32_t width,
         uint32_t height);

  ~RayTracingSystem();
  void VK_CHECK_RESULT(VkResult f);

  void prepare();
  void runShaders(ReSTIRFrameInfo& frameInfo);
  void render(FrameInfo& frameInfo);
  void handleResize(
      uint32_t width, uint32_t height, const std::vector<VkDescriptorSet>& descriptorSets);
  void updateUniforms(uint32_t frameIndex, const glm::mat4& view, const glm::mat4& proj);
  void createMaterialBuffer(std::vector<LveMaterial> materials, std::vector<glm::vec4> allLightPos);
  void copyStorageImageToSwapChain(
      VkCommandBuffer commandBuffer,
      VkImage swapChainImage,
      uint32_t width,
      uint32_t height,
      uint32_t frameIndex);
  void resizeReSTIRBuffers(uint32_t width, uint32_t height);
  VkImage& getStorageImage() { return storageImage.image; };
  VkDescriptorImageInfo getStorageImageDescriptor(uint32_t frameIndex) const;
  VkDescriptorImageInfo getDirectLightingDescriptor() const;
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
  VkDescriptorBufferInfo getLightBufferDescriptor() const {
      return lightBuffer->descriptorInfo();
  }

  VkDescriptorBufferInfo getVBufferDescriptor() const {
    return vBuffer->descriptorInfo();
}
VkDescriptorBufferInfo getTemporalVBufferDescriptor() const {
    return temporalVBuffer->descriptorInfo();
}
VkDescriptorBufferInfo getNRooksPatternDescriptor() const {
    return nRooksPatternBuffer->descriptorInfo();
}
VkDescriptorBufferInfo getNeighborOffsetDescriptor() const {
    return neighborOffsetBuffer->descriptorInfo();
}
VkDescriptorBufferInfo getOutputReservoirDescriptor() const {
    return outputReservoirBuffer->descriptorInfo();
}
VkDescriptorBufferInfo getTemporalReservoirDescriptor() const {
    return temporalReservoirBuffer->descriptorInfo();
}
VkDescriptorBufferInfo getReconnectionBufferDescriptor() const {
    return reconnectionBuffer->descriptorInfo();
}
VkDescriptorBufferInfo getMISWeightBufferDescriptor() const {
    return misWeightBuffer->descriptorInfo();
}



/*
VkDescriptorBufferInfo getLightBufferDescriptor() const {
    return lightBuffer->descriptorInfo();
}


what



*/ 



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
  std::unique_ptr<LveBuffer> lightBuffer;
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
  //ReSTIR_PT
  std::unique_ptr<LveBuffer> vBuffer;
  std::unique_ptr<LveBuffer> temporalVBuffer;
  std::unique_ptr<LveBuffer> nRooksPatternBuffer;
  std::unique_ptr<LveBuffer> neighborOffsetBuffer;
  std::unique_ptr<LveBuffer> outputReservoirBuffer;
  std::unique_ptr<LveBuffer> temporalReservoirBuffer;
  std::unique_ptr<LveBuffer> reconnectionBuffer;
  std::unique_ptr<LveBuffer> misWeightBuffer;
  std::unique_ptr<LveBuffer> rayDataBuffer;

  std::unique_ptr<LveBuffer> stagingNRooks;
  std::unique_ptr<LveBuffer> stagingNeighborOffsets;

  std::vector<RayTracingVertex> vertices;
  std::vector<uint32_t> indices;

  // Pipeline layouts
  VkPipelineLayout restirPipelineLayout{ VK_NULL_HANDLE };

  // Pipelines
  VkPipeline temporalRetracePipeline{ VK_NULL_HANDLE };
  VkPipeline temporalReusePipeline{ VK_NULL_HANDLE };
  VkPipeline spatialRetracePipeline{ VK_NULL_HANDLE };
  VkPipeline spatialReusePipeline{ VK_NULL_HANDLE };
  VkPipeline misWeightsPipeline{ VK_NULL_HANDLE };

  // Shader modules
  VkShaderModule temporalRetraceModule{ VK_NULL_HANDLE };
  VkShaderModule temporalReuseModule{ VK_NULL_HANDLE };
  VkShaderModule spatialRetraceModule{ VK_NULL_HANDLE };
  VkShaderModule spatialReuseModule{ VK_NULL_HANDLE };
  VkShaderModule misWeightsModule{ VK_NULL_HANDLE };

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
  StorageImage directLightingImage; // this is so messy. image buffers should be handled in one of the root classes, or in a new root class (similar to LveBuffer or smthing)
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
  void createReSTIRBuffers(uint32_t width, uint32_t height);
  void createTopLevelAccelerationStructure();
  void createRayTracingPipeline(VkDescriptorSetLayout globalSetLayout);
  void createShaderBindingTable();
  void createComputePipelineLayout(VkDescriptorSetLayout setLayout);
  void createComputePipelines();
  void loadFunctionPointers();
  
  const int NEIGHBOR_OFFSET_COUNT = 16;
  static constexpr int RCDATA_PATH_NUM = 16;

  //restir strucs
  struct HitInfo
  {
      glm::vec4 misc1;

      /*

          packed vector of the following:
          vec2 barycentrics;
          uint primitveID; // The triangle we hit on the model.
                           // original implementation had another uint for the mesh instance ID, which imma ignore.

      */
  };

  //cls
  struct PathReservoir
  {
      glm::vec4 F; // cached integrand (always updated after a new path is chosen in RIS)
      glm::vec4 cachedJacobian; // saved previous vertex scatter PDF, scatter PDF, and geometry term at rcVertex (used when rcVertex is not v2)

      glm::vec4 rcVertexWi[2]; // incident direction on reconnection vertex
      glm::vec4 rcVertexIrradiance[2]; // sampled irradiance on reconnection vertex

      HitInfo hitInfo;

      glm::vec4 rcVertexBSDFLightSamplingIrradiance;

      float M; // this is a float, because temporal history length is allowed to be a fraction. 
      float weight; // during RIS and when used as a "RisState", this is w_sum; during RIS when used as an incoming reseroivr or after RIS, this is 1/p(y) * 1/M * w_sum
      uint32_t pathFlags; // this is a path type indicator, see the struct definition for details
      uint32_t rcRandomSeed; // saved random seed after rcVertex (due to the need of blending half-vector reuse and random number replay)

      float lightPdf; // NEE light pdf (might change after shift if transmission is included since light sampling considers "upperHemisphere" of the previous bounce)
      uint32_t initRandomSeed; // saved random seed at the first bounce (for recovering the random distance threshold for hybrid shift)
      float rcLightPdf;

  };
  //cls
  struct ReconnectionData
  {
      HitInfo rcPrevHit;
      glm::vec4 rcPrevWo;
      glm::vec4 pathThroughput;
  };

  //cls
  struct PixelReconnectionData
  {
      ReconnectionData data[RCDATA_PATH_NUM];
  };

  struct PathReuseMISWeight
  {
      float rcBSDFMISWeight;
      float rcNEEMISWeight;

      float pad1;
      float pad2;
  };
  uint32_t frameID;
};

}  // namespace lve