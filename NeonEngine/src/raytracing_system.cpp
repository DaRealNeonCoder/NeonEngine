#include "raytracing_system.hpp"
#include "lve_pipeline.hpp"
#include "lve_swap_chain.hpp"
#include "lve_descriptors.hpp"

#include <iostream>


// std
#include <cstring>
#include <memory>
#include <stdexcept>

namespace lve {

// ============================================================
// Constructor / Destructor
// ============================================================
RayTracingSystem::RayTracingSystem(
    LveDevice& device3,
    VkFormat format,
    VkDescriptorSetLayout globalSetLayout, 
    VkDescriptorSetLayout restirSetLayout,
    std::vector<RayTracingVertex> allVertex,
    std::vector<uint32_t> allIndicies,
    std::vector<LveMaterial> allMaterials,
    std::vector<glm::vec4> allLightPos,
    uint32_t width,
    uint32_t height)
    : vulkanDevice{device3}, device{device3.device()} {
  loadFunctionPointers();

  rayTracingPipelineProperties.sType =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR;
  
  VkPhysicalDeviceProperties2 deviceProperties2{};
  deviceProperties2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
  deviceProperties2.pNext = &rayTracingPipelineProperties;
  vkGetPhysicalDeviceProperties2(vulkanDevice.getPhysicalDevice(), &deviceProperties2);

  swapChainImageFormat = format;
  std::cout << "Ray tracing properties:" << std::endl;
  std::cout << "  shaderGroupHandleSize: " << rayTracingPipelineProperties.shaderGroupHandleSize
            << std::endl;
  std::cout << "  shaderGroupHandleAlignment: "
            << rayTracingPipelineProperties.shaderGroupHandleAlignment << std::endl;

  if (rayTracingPipelineProperties.shaderGroupHandleSize == 0) {
    throw std::runtime_error("Failed to query ray tracing properties! shaderGroupHandleSize is 0");
  }


  vertices = allVertex;
  indices = allIndicies;


  createStorageImage(storageImage, width, height);

  createMaterialBuffer(allMaterials, allLightPos);
  createBottomLevelAccelerationStructure();
  createTopLevelAccelerationStructure();
  createReSTIRBuffers(width, height);
  createRayTracingPipeline(globalSetLayout);
  createShaderBindingTable();

  createComputePipelineLayout(restirSetLayout);
  createComputePipelines();
}

// Updated destructor
RayTracingSystem::~RayTracingSystem() {
  vkDestroyPipeline(device, pipeline, nullptr);
  vkDestroyPipelineLayout(device, pipelineLayout, nullptr);

  // Destroy all storage images
    destroyStorageImage(storageImage);

  vkDestroyAccelerationStructureKHR(device, bottomLevelAS.handle, nullptr);
  vkFreeMemory(device, bottomLevelAS.memory, nullptr);
  vkDestroyBuffer(device, bottomLevelAS.buffer, nullptr);

  vkDestroyAccelerationStructureKHR(device, topLevelAS.handle, nullptr);
  vkFreeMemory(device, topLevelAS.memory, nullptr);
  vkDestroyBuffer(device, topLevelAS.buffer, nullptr);
}

// ============================================================
// Public API
// ============================================================

struct PushData {
  glm::vec4 accum{0};
};
// Updated render function
int temp;
void RayTracingSystem::render(FrameInfo& frameInfo) {
  
   
    
    vkCmdBindPipeline(frameInfo.commandBuffer, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, pipeline);

  vkCmdBindDescriptorSets(
      frameInfo.commandBuffer,
      VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR,
      pipelineLayout,
      0,
      1,
      &frameInfo.globalDescriptorSet,
      0,
      nullptr);

  uint32_t handleSizeAligned = (rayTracingPipelineProperties.shaderGroupHandleSize +
                                rayTracingPipelineProperties.shaderGroupHandleAlignment - 1) &
                               ~(rayTracingPipelineProperties.shaderGroupHandleAlignment - 1);

  VkStridedDeviceAddressRegionKHR raygen{};
  raygen.deviceAddress = getBufferDeviceAddress(raygenShaderBindingTable->getBuffer());
  raygen.stride = handleSizeAligned;
  raygen.size = handleSizeAligned;

  VkStridedDeviceAddressRegionKHR miss{};
  miss.deviceAddress = getBufferDeviceAddress(missShaderBindingTable->getBuffer());
  miss.stride = handleSizeAligned;
  miss.size = handleSizeAligned * 2;

  VkStridedDeviceAddressRegionKHR hit{};
  hit.deviceAddress = getBufferDeviceAddress(hitShaderBindingTable->getBuffer());
  hit.stride = handleSizeAligned;
  hit.size = handleSizeAligned;

  VkStridedDeviceAddressRegionKHR callable{};
  PushData push{};

  if (temp == UINT32_MAX) {
    temp = 0;
  }
  push.accum.x = rand()/ 1000.0;
  temp++;
  vkCmdPushConstants(
      frameInfo.commandBuffer,
      pipelineLayout,
      VK_SHADER_STAGE_RAYGEN_BIT_KHR,
      0,
      sizeof(PushData),
      &push);

  // Use the storage image for the current frame
  const auto& currentStorageImage = storageImage;
  vkCmdTraceRaysKHR(
      frameInfo.commandBuffer,
      &raygen,
      &miss,
      &hit,
      &callable,
      currentStorageImage.width,
      currentStorageImage.height,
      1);

}

// Updated handleResize function
void RayTracingSystem::handleResize(
    uint32_t width, uint32_t height, const std::vector<VkDescriptorSet>& descriptorSets) {
  // Recreate all storage images with new dimensions
  // Recreate all storage images
    destroyStorageImage(storageImage);
    createStorageImage(storageImage, width, height);

  // Update descriptor sets with new image views
  for (size_t i = 0; i < descriptorSets.size(); i++) {
    VkDescriptorImageInfo imageInfo = getStorageImageDescriptor(0);

    VkWriteDescriptorSet writeDescriptorSet{};
    writeDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeDescriptorSet.dstSet = descriptorSets[i];
    writeDescriptorSet.dstBinding = 1;  // Storage image binding
    writeDescriptorSet.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    writeDescriptorSet.descriptorCount = 1;
    writeDescriptorSet.pImageInfo = &imageInfo;

    vkUpdateDescriptorSets(device, 1, &writeDescriptorSet, 0, nullptr);
  }


  resizeReSTIRBuffers(width, height);
}

void RayTracingSystem::copyStorageImageToSwapChain(
    VkCommandBuffer commandBuffer,
    VkImage swapChainImage,
    uint32_t width,
    uint32_t height,
    uint32_t frameIndex) {

    // Wait for COMPUTE (not ray tracing) to finish writing
    VkImageMemoryBarrier storageBarrier{};
    storageBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    storageBarrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    storageBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    storageBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    storageBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    storageBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    storageBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    storageBarrier.image = storageImage.image;
    storageBarrier.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

    vkCmdPipelineBarrier(
        commandBuffer,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,  
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        0, 0, nullptr, 0, nullptr,
        1, &storageBarrier);

    // Transition swapchain image to transfer dst
    VkImageMemoryBarrier swapChainBarrierBefore{};
    swapChainBarrierBefore.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    swapChainBarrierBefore.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    swapChainBarrierBefore.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    swapChainBarrierBefore.srcAccessMask = 0;
    swapChainBarrierBefore.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    swapChainBarrierBefore.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    swapChainBarrierBefore.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    swapChainBarrierBefore.image = swapChainImage;
    swapChainBarrierBefore.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

    vkCmdPipelineBarrier(
        commandBuffer,
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        0, 0, nullptr, 0, nullptr,
        1, &swapChainBarrierBefore);

    // Copy
    VkImageCopy copyRegion{};
    copyRegion.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    copyRegion.srcOffset = { 0, 0, 0 };
    copyRegion.dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    copyRegion.dstOffset = { 0, 0, 0 };
    copyRegion.extent = { width, height, 1 };

    vkCmdCopyImage(
        commandBuffer,
        storageImage.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        swapChainImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        1, &copyRegion);

    // Restore storage image to GENERAL for next frame's ray tracing write
    storageBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    storageBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    storageBarrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    storageBarrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;

    vkCmdPipelineBarrier(
        commandBuffer,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR, // next frame writes via RT
        0, 0, nullptr, 0, nullptr,
        1, &storageBarrier);

    // Transition swapchain to present
    VkImageMemoryBarrier swapChainBarrierAfter{};
    swapChainBarrierAfter.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    swapChainBarrierAfter.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    swapChainBarrierAfter.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    swapChainBarrierAfter.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    swapChainBarrierAfter.dstAccessMask = 0;
    swapChainBarrierAfter.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    swapChainBarrierAfter.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    swapChainBarrierAfter.image = swapChainImage;
    swapChainBarrierAfter.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

    vkCmdPipelineBarrier(
        commandBuffer,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
        0, 0, nullptr, 0, nullptr,
        1, &swapChainBarrierAfter);
}

void RayTracingSystem::updateUniforms(
    uint32_t frameIndex, const glm::mat4& view, const glm::mat4& proj) {
  UniformData ubo{};
  ubo.viewInverse = glm::inverse(view);
  ubo.projInverse = glm::inverse(proj);

  uniformBuffers[frameIndex]->writeToBuffer(&ubo);
}

void RayTracingSystem::loadFunctionPointers() {
  vkGetBufferDeviceAddressKHR = reinterpret_cast<PFN_vkGetBufferDeviceAddressKHR>(
      vkGetDeviceProcAddr(device, "vkGetBufferDeviceAddressKHR"));
  vkCreateAccelerationStructureKHR = reinterpret_cast<PFN_vkCreateAccelerationStructureKHR>(
      vkGetDeviceProcAddr(device, "vkCreateAccelerationStructureKHR"));
  vkDestroyAccelerationStructureKHR = reinterpret_cast<PFN_vkDestroyAccelerationStructureKHR>(
      vkGetDeviceProcAddr(device, "vkDestroyAccelerationStructureKHR"));
  vkGetAccelerationStructureBuildSizesKHR =
      reinterpret_cast<PFN_vkGetAccelerationStructureBuildSizesKHR>(
          vkGetDeviceProcAddr(device, "vkGetAccelerationStructureBuildSizesKHR"));
  vkGetAccelerationStructureDeviceAddressKHR =
      reinterpret_cast<PFN_vkGetAccelerationStructureDeviceAddressKHR>(
          vkGetDeviceProcAddr(device, "vkGetAccelerationStructureDeviceAddressKHR"));
  vkCmdBuildAccelerationStructuresKHR = reinterpret_cast<PFN_vkCmdBuildAccelerationStructuresKHR>(
      vkGetDeviceProcAddr(device, "vkCmdBuildAccelerationStructuresKHR"));
  vkCmdTraceRaysKHR =
      reinterpret_cast<PFN_vkCmdTraceRaysKHR>(vkGetDeviceProcAddr(device, "vkCmdTraceRaysKHR"));
  vkGetRayTracingShaderGroupHandlesKHR = reinterpret_cast<PFN_vkGetRayTracingShaderGroupHandlesKHR>(
      vkGetDeviceProcAddr(device, "vkGetRayTracingShaderGroupHandlesKHR"));
  vkCreateRayTracingPipelinesKHR = reinterpret_cast<PFN_vkCreateRayTracingPipelinesKHR>(
      vkGetDeviceProcAddr(device, "vkCreateRayTracingPipelinesKHR"));
}

RayTracingScratchBuffer RayTracingSystem::createScratchBuffer(VkDeviceSize size) {
  RayTracingScratchBuffer scratch{};

  VkBufferCreateInfo bufferCI{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  bufferCI.size = size;
  bufferCI.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;

  VK_CHECK_RESULT(vkCreateBuffer(device, &bufferCI, nullptr, &scratch.handle));

  VkMemoryRequirements memReq;
  vkGetBufferMemoryRequirements(device, scratch.handle, &memReq);

  VkMemoryAllocateFlagsInfo flags{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
  flags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;

  VkMemoryAllocateInfo alloc{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  alloc.pNext = &flags;
  alloc.allocationSize = memReq.size;
  alloc.memoryTypeIndex =
      vulkanDevice.findMemoryType(memReq.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

  VK_CHECK_RESULT(vkAllocateMemory(device, &alloc, nullptr, &scratch.memory));
  VK_CHECK_RESULT(vkBindBufferMemory(device, scratch.handle, scratch.memory, 0));

  VkBufferDeviceAddressInfo addrInfo{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
  addrInfo.buffer = scratch.handle;
  scratch.deviceAddress = vkGetBufferDeviceAddressKHR(device, &addrInfo);

  return scratch;
}

void RayTracingSystem::deleteScratchBuffer(RayTracingScratchBuffer& scratch) {
  vkFreeMemory(device, scratch.memory, nullptr);
  vkDestroyBuffer(device, scratch.handle, nullptr);
}

// ============================================================
// Acceleration Structures (BLAS / TLAS)
// ============================================================
void RayTracingSystem::createBottomLevelAccelerationStructure() {
  // Vertex struct + data


  indexCount = static_cast<uint32_t>(indices.size());
  const uint32_t numTriangles = indexCount / 3;  // one triangle

  auto stagingBuffer = std::make_unique<LveBuffer>(
      vulkanDevice,
      sizeof(RayTracingVertex),                // or sizeof(uint32_t) / sizeof(VkTransformMatrixKHR)
      static_cast<uint32_t>(vertices.size()),  // or indices.size() / 1
      VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  stagingBuffer->map();
  stagingBuffer->writeToBuffer(vertices.data());  // your data here

  // Create device local buffer
  vertexBuffer = std::make_unique<LveBuffer>(
      vulkanDevice,
      sizeof(RayTracingVertex),
      static_cast<uint32_t>(vertices.size()),
      VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
          VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

  // Copy via command buffer
  VkCommandBuffer cmd2 = vulkanDevice.beginSingleTimeCommands();

  VkBufferCopy copyRegion{};
  copyRegion.size = sizeof(RayTracingVertex) * vertices.size();
  vkCmdCopyBuffer(cmd2, stagingBuffer->getBuffer(), vertexBuffer->getBuffer(), 1, &copyRegion);

  vulkanDevice.endSingleTimeCommands(cmd2);

  indexBuffer = std::make_unique<LveBuffer>(
      vulkanDevice,
      sizeof(uint32_t),
      static_cast<uint32_t>(indices.size()),
      VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
          VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  indexBuffer->map();
  indexBuffer->writeToBuffer(indices.data());

  // Identity transform matrix for the triangle (VkTransformMatrixKHR is 3x4)
  VkTransformMatrixKHR transformMatrix = {
      {{1.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f, 0.0f}}};
  transformBuffer = std::make_unique<LveBuffer>(
      vulkanDevice,
      sizeof(VkTransformMatrixKHR),
      1u,
      VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
          VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  transformBuffer->map();
  transformBuffer->writeToBuffer(&transformMatrix);

  // Device addresses for geometry inputs
  VkDeviceOrHostAddressConstKHR vertexAddr{};
  vertexAddr.deviceAddress = getBufferDeviceAddress(vertexBuffer->getBuffer());

  VkDeviceOrHostAddressConstKHR indexAddr{};
  indexAddr.deviceAddress = getBufferDeviceAddress(indexBuffer->getBuffer());

  VkDeviceOrHostAddressConstKHR transformAddr{};
  transformAddr.deviceAddress = getBufferDeviceAddress(transformBuffer->getBuffer());

  // Build geometry description
  VkAccelerationStructureGeometryKHR geometry{
      VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
  geometry.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
  geometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
  geometry.geometry.triangles.sType =
      VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
  geometry.geometry.triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
  geometry.geometry.triangles.vertexData = vertexAddr;
  geometry.geometry.triangles.vertexStride = sizeof(RayTracingVertex);
  geometry.geometry.triangles.maxVertex = static_cast<uint32_t>(vertices.size() - 1);
  geometry.geometry.triangles.indexType = VK_INDEX_TYPE_UINT32;
  geometry.geometry.triangles.indexData = indexAddr;
  geometry.geometry.triangles.transformData = transformAddr;

  // Query sizes needed for BLAS and scratch
  VkAccelerationStructureBuildGeometryInfoKHR buildSizeInfo{
      VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
  buildSizeInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
  buildSizeInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
  buildSizeInfo.geometryCount = 1;
  buildSizeInfo.pGeometries = &geometry;

  VkAccelerationStructureBuildSizesInfoKHR sizeInfo{
      VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
  vkGetAccelerationStructureBuildSizesKHR(
      device,
      VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
      &buildSizeInfo,
      &numTriangles,
      &sizeInfo);

  // Create buffer to hold the acceleration structure
  // createAccelerationStructureBuffer should allocate a VkBuffer large enough and store it in
  // bottomLevelAS.buffer (match your helpers)
  createAccelerationStructureBuffer(bottomLevelAS, sizeInfo);

  // Create the acceleration structure handle
  VkAccelerationStructureCreateInfoKHR accCreateInfo{
      VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
  accCreateInfo.buffer =
      bottomLevelAS.buffer;  // assumes bottomLevelAS.buffer is VkBuffer (matches Sascha style)
  accCreateInfo.size = sizeInfo.accelerationStructureSize;
  accCreateInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;

  vkCreateAccelerationStructureKHR(device, &accCreateInfo, nullptr, &bottomLevelAS.handle);

  // Scratch buffer for the build
  RayTracingScratchBuffer scratchBuffer = createScratchBuffer(sizeInfo.buildScratchSize);

  // Build info for actual build command
  VkAccelerationStructureBuildGeometryInfoKHR buildInfo{
      VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
  buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
  buildInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
  buildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
  buildInfo.dstAccelerationStructure = bottomLevelAS.handle;
  buildInfo.geometryCount = 1;
  buildInfo.pGeometries = &geometry;
  buildInfo.scratchData.deviceAddress = scratchBuffer.deviceAddress;

  // Range info (primitive counts)
  VkAccelerationStructureBuildRangeInfoKHR buildRangeInfo{};
  buildRangeInfo.primitiveCount = numTriangles;
  buildRangeInfo.primitiveOffset = 0;
  buildRangeInfo.firstVertex = 0;
  buildRangeInfo.transformOffset = 0;
  std::vector<VkAccelerationStructureBuildRangeInfoKHR*> pBuildRangeInfos = {&buildRangeInfo};

  // Issue build on a one-time command buffer
  VkCommandBuffer cmd = vulkanDevice.beginSingleTimeCommands();

  // Single AS build info and range info
  vkCmdBuildAccelerationStructuresKHR(cmd, 1, &buildInfo, pBuildRangeInfos.data());

  vulkanDevice.endSingleTimeCommands(cmd);  // optionally pass the queue

  // Get device address for the BLAS
  VkAccelerationStructureDeviceAddressInfoKHR addressInfo{
      VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR};
  addressInfo.accelerationStructure = bottomLevelAS.handle;
  bottomLevelAS.deviceAddress = vkGetAccelerationStructureDeviceAddressKHR(device, &addressInfo);

  // Cleanup scratch
  deleteScratchBuffer(scratchBuffer);
}



void RayTracingSystem::createStorageImage(
    StorageImage& storageImage, uint32_t width, uint32_t height) {
  storageImage.width = width;
  storageImage.height = height;

  VkImageCreateInfo imageInfo{};
  imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  imageInfo.imageType = VK_IMAGE_TYPE_2D;
  imageInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
  imageInfo.extent = {width, height, 1};
  imageInfo.mipLevels = 1;
  imageInfo.arrayLayers = 1;
  imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
  imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
  imageInfo.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
  imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

  vulkanDevice.createImageWithInfo(
      imageInfo,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
      storageImage.image,
      storageImage.memory);

  VkImageViewCreateInfo viewInfo{};
  viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
  viewInfo.image = storageImage.image;
  viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
  viewInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
  viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  viewInfo.subresourceRange.baseMipLevel = 0;
  viewInfo.subresourceRange.levelCount = 1;
  viewInfo.subresourceRange.baseArrayLayer = 0;
  viewInfo.subresourceRange.layerCount = 1;

  if (vkCreateImageView(vulkanDevice.device(), &viewInfo, nullptr, &storageImage.view) !=
      VK_SUCCESS) {
    throw std::runtime_error("failed to create storage image view!");
  }

  VkCommandBuffer cmd = vulkanDevice.beginSingleTimeCommands();

  VkImageMemoryBarrier barrier{};
  barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
  barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.image = storageImage.image;
  barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  barrier.subresourceRange.baseMipLevel = 0;
  barrier.subresourceRange.levelCount = 1;
  barrier.subresourceRange.baseArrayLayer = 0;
  barrier.subresourceRange.layerCount = 1;
  barrier.srcAccessMask = 0;
  barrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;

  vkCmdPipelineBarrier(
      cmd,
      VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
      VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
      0,
      0,
      nullptr,
      0,
      nullptr,
      1,
      &barrier);

  vulkanDevice.endSingleTimeCommands(cmd);
}



/// New helper function to destroy a single storage image
void RayTracingSystem::destroyStorageImage(StorageImage& storageImage) {
  vkDestroyImageView(device, storageImage.view, nullptr);
  vkDestroyImage(device, storageImage.image, nullptr);
  vkFreeMemory(device, storageImage.memory, nullptr);

  storageImage.view = VK_NULL_HANDLE;
  storageImage.image = VK_NULL_HANDLE;
  storageImage.memory = VK_NULL_HANDLE;
}

// Updated getStorageImageDescriptor function
VkDescriptorImageInfo RayTracingSystem::getStorageImageDescriptor(uint32_t frameIndex) const {
  VkDescriptorImageInfo imageInfo{};
  imageInfo.imageView = storageImage.view;
  imageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
  imageInfo.sampler = VK_NULL_HANDLE;

  return imageInfo;
}

VkDescriptorImageInfo RayTracingSystem::getDirectLightingDescriptor() const {
    VkDescriptorImageInfo info{};
    info.imageView = directLightingImage.view;
    info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    info.sampler = VK_NULL_HANDLE;  
    return info;
}
uint64_t RayTracingSystem::getBufferDeviceAddress(VkBuffer buffer) {
  VkBufferDeviceAddressInfoKHR bufferDeviceAI{};
  bufferDeviceAI.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
  bufferDeviceAI.buffer = buffer;
  return vkGetBufferDeviceAddressKHR(device, &bufferDeviceAI);
}

// Currently lights are simple enough to be passed in like this.
// Ideally we have a more flexible system for this typa stuff.
// Assumes lights are first.
void RayTracingSystem::createMaterialBuffer(std::vector<LveMaterial> materials, std::vector<glm::vec4> allLightPos)  {

    std::vector<RayTracingMaterial> materialsFinal;

    std::cout << "materials thang" << materials.size() << "\n";
    std::cout <<"light thang" << allLightPos.size() << "\n";

    for (size_t i = 0; i < allLightPos.size(); i++)
    {
        RayTracingMaterial lightMat{};
        lightMat.emission = materials[i].emission;
        lightMat.position = allLightPos[i];
        lightMat.albedo = glm::vec4{ 1 };
        lightMat.misc.x = 0.5f;
        materialsFinal.push_back(lightMat);
    }
    
    for (size_t i = allLightPos.size(); i < materials.size(); i++)
    {
        RayTracingMaterial mat{};
        mat.misc.x = 1.f;
        mat.misc.z = materials[i].type;
        mat.albedo = materials[i].albedo;
        //mat.albedo = glm::vec4(1,0,0,1);

        materialsFinal.push_back(mat);
    }
   
    for (size_t i = 0; i < materialsFinal.size(); i++)
    {
        const RayTracingMaterial& m = materialsFinal[i];

        std::cout << "Material [" << i << "]\n";

        std::cout << "  emission : ("
            << m.emission.x << ", "
            << m.emission.y << ", "
            << m.emission.z << ", "
            << m.emission.w << ")\n";

        std::cout << "  albedo   : ("
            << m.albedo.x << ", "
            << m.albedo.y << ", "
            << m.albedo.z << ", "
            << m.albedo.w << ")\n";

        std::cout << "  position : ("
            << m.position.x << ", "
            << m.position.y << ", "
            << m.position.z << ", "
            << m.position.w << ")\n";

        std::cout << "  misc     : ("
            << m.misc.x << ", "
            << m.misc.y << ", "
            << m.misc.z << ", "
            << m.misc.w << ")\n";

        std::cout << "-------------------------------------\n";
    }

    materialBuffer = std::make_unique<LveBuffer>(
        vulkanDevice,
        sizeof(RayTracingMaterial),
        static_cast<uint32_t>(materialsFinal.size()),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

  materialBuffer->map();
  materialBuffer->writeToBuffer(materialsFinal.data());
}

void RayTracingSystem::createTopLevelAccelerationStructure() {
  // ----- Create instance -----
  VkTransformMatrixKHR
      transformMatrix{1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f};

  VkAccelerationStructureInstanceKHR instance{};
  instance.transform = transformMatrix;
  instance.instanceCustomIndex = 0;
  instance.mask = 0xFF;
  instance.instanceShaderBindingTableRecordOffset = 0;
  instance.flags = 0;
  instance.accelerationStructureReference = bottomLevelAS.deviceAddress;

  // ----- Instance buffer using LveBuffer -----
  instanceBuffer = std::make_unique<LveBuffer>(
      vulkanDevice,
      sizeof(VkAccelerationStructureInstanceKHR),
      1,
      VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
          VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  instanceBuffer->map();
  instanceBuffer->writeToBuffer(&instance);

  VkDeviceOrHostAddressConstKHR instanceData{};
  instanceData.deviceAddress = getBufferDeviceAddress(instanceBuffer->getBuffer());

  // ----- Geometry info -----
  VkAccelerationStructureGeometryKHR asGeometry{
      VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
  asGeometry.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
  asGeometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
  asGeometry.geometry.instances.sType =
      VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
  asGeometry.geometry.instances.arrayOfPointers = VK_FALSE;
  asGeometry.geometry.instances.data = instanceData;

  uint32_t primitiveCount = 1;

  // ----- Get build sizes -----
  VkAccelerationStructureBuildGeometryInfoKHR buildInfo{
      VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
  buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
  buildInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
  buildInfo.geometryCount = 1;
  buildInfo.pGeometries = &asGeometry;

  VkAccelerationStructureBuildSizesInfoKHR sizeInfo{
      VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
  vkGetAccelerationStructureBuildSizesKHR(
      device,
      VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
      &buildInfo,
      &primitiveCount,
      &sizeInfo);

  // ----- Create TLAS buffer -----
  createAccelerationStructureBuffer(topLevelAS, sizeInfo);

  VkAccelerationStructureCreateInfoKHR asCreateInfo{
      VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
  asCreateInfo.buffer = topLevelAS.buffer;
  asCreateInfo.size = sizeInfo.accelerationStructureSize;
  asCreateInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
  vkCreateAccelerationStructureKHR(device, &asCreateInfo, nullptr, &topLevelAS.handle);

  // ----- Scratch buffer -----
  auto scratchBuffer = createScratchBuffer(sizeInfo.buildScratchSize);

  // ----- Build TLAS -----
  VkAccelerationStructureBuildGeometryInfoKHR buildGeomInfo{
      VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
  buildGeomInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
  buildGeomInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
  buildGeomInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
  buildGeomInfo.dstAccelerationStructure = topLevelAS.handle;
  buildGeomInfo.geometryCount = 1;
  buildGeomInfo.pGeometries = &asGeometry;
  buildGeomInfo.scratchData.deviceAddress = scratchBuffer.deviceAddress;

  VkAccelerationStructureBuildRangeInfoKHR buildRange{};
  buildRange.primitiveCount = 1;
  buildRange.primitiveOffset = 0;
  buildRange.firstVertex = 0;
  buildRange.transformOffset = 0;
  std::vector<VkAccelerationStructureBuildRangeInfoKHR*> buildRanges = {&buildRange};

  VkCommandBuffer commandBuffer = vulkanDevice.beginSingleTimeCommands();

  vkCmdBuildAccelerationStructuresKHR(commandBuffer, 1, &buildGeomInfo, buildRanges.data());

  vulkanDevice.endSingleTimeCommands(commandBuffer);

  // ----- TLAS device address -----
  VkAccelerationStructureDeviceAddressInfoKHR deviceAddressInfo{
      VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR};
  deviceAddressInfo.accelerationStructure = topLevelAS.handle;
  topLevelAS.deviceAddress = vkGetAccelerationStructureDeviceAddressKHR(device, &deviceAddressInfo);

  // ----- Cleanup -----
  deleteScratchBuffer(scratchBuffer);
}

void RayTracingSystem::createReSTIRBuffers(uint32_t width, uint32_t height) {
    uint32_t pixelCount = width * height;

    auto makeDeviceLocal = [&](VkDeviceSize elemSize, uint32_t count) {
        return std::make_unique<LveBuffer>(
            vulkanDevice,
            elemSize,
            count,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    };

    auto makeDeviceLocal2 = [&](VkDeviceSize elemSize, uint32_t count) {
        return std::make_unique<LveBuffer>(
            vulkanDevice,
            elemSize,
            count,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    };

    auto makeHostVisible = [&](VkDeviceSize elemSize, uint32_t count) {
        return std::make_unique<LveBuffer>(
            vulkanDevice,
            elemSize,
            count,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    };

    // =====================================================
    // GBUFFER / VISIBILITY (binding 0, 1)
    // =====================================================
    vBuffer = makeDeviceLocal(sizeof(HitInfo), pixelCount);         // binding 0
    temporalVBuffer = makeDeviceLocal(sizeof(HitInfo), pixelCount);         // binding 1

    // =====================================================
    // SAMPLING / PATTERNS (binding 2, 3)
    // =====================================================
    nRooksPatternBuffer = makeDeviceLocal2(sizeof(uint32_t), 65536);    // binding 2
    neighborOffsetBuffer = makeDeviceLocal2(sizeof(uint32_t), NEIGHBOR_OFFSET_COUNT); // binding 3

    // =====================================================
    // RESERVOIRS (binding 4, 5)
    // =====================================================
    outputReservoirBuffer = makeDeviceLocal(sizeof(PathReservoir), pixelCount); // binding 4
    temporalReservoirBuffer = makeDeviceLocal(sizeof(PathReservoir), pixelCount); // binding 5

    // =====================================================
    // RECONNECTION / REUSE (binding 6, 7)
    // =====================================================
    reconnectionBuffer = makeDeviceLocal(sizeof(PixelReconnectionData), pixelCount); // binding 6
    misWeightBuffer = makeDeviceLocal(sizeof(PathReuseMISWeight), pixelCount);    // binding 7

    // =====================================================
    // UPLOAD STAGING (for bindings 2, 3 that need CPU->GPU transfer)
    // =====================================================
    stagingNRooks = makeHostVisible(sizeof(uint32_t), 65536);
    stagingNeighborOffsets = makeHostVisible(sizeof(uint32_t), NEIGHBOR_OFFSET_COUNT);

    debugBuffer = std::make_unique<LveBuffer>(
        vulkanDevice,
        sizeof(float),
        10000,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    debugBuffer->map();

    //upload rooks buffer
    {
    // Load from file, same as Falcor reference
    FILE* f = nullptr;
    fopen_s(&f, "C:\\users\\zybros\\Downloads\\NeonEngine\\NeonEngine\\misc\\16RooksPattern256.txt", "r");
    if (!f) throw std::runtime_error("Failed to open 16RooksPattern256.txt");

    std::vector<uint8_t> nRookArray(65536);
    for (int i = 0; i < 8192; i++) {
        for (int j = 0; j < 8; j++) {
            int temp1, temp2;
            fscanf_s(f, "%d %d", &temp1, &temp2);
            nRookArray[8 * i + j] = (uint8_t)((temp2 << 4) | temp1);
        }
    }
    fclose(f);

    stagingNRooks->map();
    stagingNRooks->writeToBuffer(nRookArray.data());

    VkCommandBuffer cmd = vulkanDevice.beginSingleTimeCommands();

    VkBufferCopy copyRegion{};
    copyRegion.size = 65536;

    vkCmdCopyBuffer(cmd, stagingNRooks->getBuffer(), nRooksPatternBuffer->getBuffer(), 1, &copyRegion);

    vulkanDevice.endSingleTimeCommands(cmd);

}
    /*
    
    Since your direct lighting pass writes every pixel, just drop VK_IMAGE_USAGE_TRANSFER_DST_BIT from the usage flags and forget about it. The shader handles it.

    */
    //directLightingImage
    {
        directLightingImage.width = width;
        directLightingImage.height = height;

        VkImageCreateInfo imageInfo{};
        imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        imageInfo.imageType = VK_IMAGE_TYPE_2D;
        imageInfo.format = VK_FORMAT_R32G32B32A32_SFLOAT;  // rgba32f to match shader
        imageInfo.extent = { width, height, 1 };
        imageInfo.mipLevels = 1;
        imageInfo.arrayLayers = 1;
        imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        imageInfo.usage = VK_IMAGE_USAGE_STORAGE_BIT
            | VK_IMAGE_USAGE_TRANSFER_DST_BIT  // for clearing
            | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

        vulkanDevice.createImageWithInfo(
            imageInfo,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            directLightingImage.image,
            directLightingImage.memory);

        VkImageViewCreateInfo viewInfo{};
        viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewInfo.image = directLightingImage.image;
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format = VK_FORMAT_R32G32B32A32_SFLOAT;
        viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        viewInfo.subresourceRange.baseMipLevel = 0;
        viewInfo.subresourceRange.levelCount = 1;
        viewInfo.subresourceRange.baseArrayLayer = 0;
        viewInfo.subresourceRange.layerCount = 1;

        if (vkCreateImageView(
            vulkanDevice.device(), &viewInfo, nullptr, &directLightingImage.view) != VK_SUCCESS) {
            throw std::runtime_error("failed to create direct lighting image view!");
        }

        // Transition to GENERAL for shader read/write
        VkCommandBuffer cmd = vulkanDevice.beginSingleTimeCommands();

        VkImageMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = directLightingImage.image;
        barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        barrier.subresourceRange.baseMipLevel = 0;
        barrier.subresourceRange.levelCount = 1;
        barrier.subresourceRange.baseArrayLayer = 0;
        barrier.subresourceRange.layerCount = 1;
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT
            | VK_ACCESS_SHADER_READ_BIT;

        vkCmdPipelineBarrier(
            cmd,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
            0, 0, nullptr, 0, nullptr,
            1, &barrier);

        vulkanDevice.endSingleTimeCommands(cmd);
    
    }
}

void RayTracingSystem::resizeReSTIRBuffers(uint32_t width, uint32_t height) {
    // Wait for GPU to finish before destroying anything
    vkDeviceWaitIdle(vulkanDevice.device());

    // Destroy the direct lighting image manually (not a unique_ptr)
    if (directLightingImage.view != VK_NULL_HANDLE) {
        vkDestroyImageView(vulkanDevice.device(), directLightingImage.view, nullptr);
        directLightingImage.view = VK_NULL_HANDLE;
    }
    if (directLightingImage.image != VK_NULL_HANDLE) {
        vkDestroyImage(vulkanDevice.device(), directLightingImage.image, nullptr);
        directLightingImage.image = VK_NULL_HANDLE;
    }
    if (directLightingImage.memory != VK_NULL_HANDLE) {
        vkFreeMemory(vulkanDevice.device(), directLightingImage.memory, nullptr);
        directLightingImage.memory = VK_NULL_HANDLE;
    }

    // Buffers are unique_ptrs — reassignment destroys the old ones automatically
    createReSTIRBuffers(width, height);
}

void RayTracingSystem::createAccelerationStructureBuffer(
    AccelerationStructure& accelerationStructure,
    const VkAccelerationStructureBuildSizesInfoKHR& buildSizeInfo) {
  VkBufferCreateInfo bufferCreateInfo{};
  bufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  bufferCreateInfo.size = buildSizeInfo.accelerationStructureSize;
  bufferCreateInfo.usage = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR |
                           VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;

  vkCreateBuffer(device, &bufferCreateInfo, nullptr, &accelerationStructure.buffer);

  VkMemoryRequirements memoryRequirements{};
  vkGetBufferMemoryRequirements(device, accelerationStructure.buffer, &memoryRequirements);
  VkMemoryAllocateFlagsInfo memoryAllocateFlagsInfo{};
  memoryAllocateFlagsInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
  memoryAllocateFlagsInfo.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT_KHR;
  VkMemoryAllocateInfo memoryAllocateInfo{};
  memoryAllocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  memoryAllocateInfo.pNext = &memoryAllocateFlagsInfo;
  memoryAllocateInfo.allocationSize = memoryRequirements.size;
  memoryAllocateInfo.memoryTypeIndex = vulkanDevice.findMemoryType(
      memoryRequirements.memoryTypeBits,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

  vkAllocateMemory(device, &memoryAllocateInfo, nullptr, &accelerationStructure.memory);
  vkBindBufferMemory(device, accelerationStructure.buffer, accelerationStructure.memory, 0);
}
void RayTracingSystem::createRayTracingPipeline(VkDescriptorSetLayout globalSetLayout) {
  // Destroy old layout/pipeline if recreating
  if (pipeline != VK_NULL_HANDLE) {
    vkDestroyPipeline(device, pipeline, nullptr);
    pipeline = VK_NULL_HANDLE;
  }
  if (pipelineLayout != VK_NULL_HANDLE) {
    vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
    pipelineLayout = VK_NULL_HANDLE;
  }

  // Create pipeline layout FIRST
  VkPushConstantRange pushConstantRange{};
  pushConstantRange.stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
  pushConstantRange.offset = 0;
  pushConstantRange.size = sizeof(PushData);

  VkPipelineLayoutCreateInfo pipelineLayoutCI{};
  pipelineLayoutCI.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  pipelineLayoutCI.setLayoutCount = 1;
  pipelineLayoutCI.pSetLayouts = &globalSetLayout;
  pipelineLayoutCI.pushConstantRangeCount = 1;
  pipelineLayoutCI.pPushConstantRanges = &pushConstantRange;

  if (vkCreatePipelineLayout(device, &pipelineLayoutCI, nullptr, &pipelineLayout) != VK_SUCCESS) {
    throw std::runtime_error("Failed to create ray tracing pipeline layout!");
  }

  // Clear shader groups in case of recreation
  shaderGroups.clear();

  std::vector<VkPipelineShaderStageCreateInfo> shaderStages;

  // Raygen
  shaderStages.push_back(LvePipeline::loadShaderCreateInfo(
      "C:\\Users\\ZyBros\\Downloads\\NeonEngine\\NeonEngine\\shaders\\raygen.rgen.spv",
      VK_SHADER_STAGE_RAYGEN_BIT_KHR,
      device));
  {
    VkRayTracingShaderGroupCreateInfoKHR group{};
    group.sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
    group.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
    group.generalShader = static_cast<uint32_t>(shaderStages.size() - 1);
    group.closestHitShader = VK_SHADER_UNUSED_KHR;
    group.anyHitShader = VK_SHADER_UNUSED_KHR;
    group.intersectionShader = VK_SHADER_UNUSED_KHR;
    shaderGroups.push_back(group);
  }

  // Miss
  shaderStages.push_back(LvePipeline::loadShaderCreateInfo(
      "C:\\Users\\ZyBros\\Downloads\\NeonEngine\\NeonEngine\\shaders\\miss.rmiss.spv",
      VK_SHADER_STAGE_MISS_BIT_KHR,
      device));
  {
    VkRayTracingShaderGroupCreateInfoKHR group{};
    group.sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
    group.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
    group.generalShader = static_cast<uint32_t>(shaderStages.size() - 1);
    group.closestHitShader = VK_SHADER_UNUSED_KHR;
    group.anyHitShader = VK_SHADER_UNUSED_KHR;
    group.intersectionShader = VK_SHADER_UNUSED_KHR;
    shaderGroups.push_back(group);
  }

  // Shadow miss - index 2
  shaderStages.push_back(LvePipeline::loadShaderCreateInfo(
      "C:\\Users\\ZyBros\\Downloads\\NeonEngine\\NeonEngine\\shaders\\shadow.rmiss.spv",
      VK_SHADER_STAGE_MISS_BIT_KHR,
      device));
  {
      VkRayTracingShaderGroupCreateInfoKHR group{};
      group.sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
      group.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
      group.generalShader = static_cast<uint32_t>(shaderStages.size() - 1);
      group.closestHitShader = VK_SHADER_UNUSED_KHR;
      group.anyHitShader = VK_SHADER_UNUSED_KHR;
      group.intersectionShader = VK_SHADER_UNUSED_KHR;
      shaderGroups.push_back(group);
  }

  // Closest hit
  shaderStages.push_back(LvePipeline::loadShaderCreateInfo(
      "C:\\Users\\ZyBros\\Downloads\\NeonEngine\\NeonEngine\\shaders\\closesthit.rchit.spv",
      VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR,
      device));
  {
    VkRayTracingShaderGroupCreateInfoKHR group{};
    group.sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
    group.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
    group.generalShader = VK_SHADER_UNUSED_KHR;
    group.closestHitShader = static_cast<uint32_t>(shaderStages.size() - 1);
    group.anyHitShader = VK_SHADER_UNUSED_KHR;
    group.intersectionShader = VK_SHADER_UNUSED_KHR;
    shaderGroups.push_back(group);
  }

  VkRayTracingPipelineCreateInfoKHR rayTracingPipelineCI{};
  rayTracingPipelineCI.sType = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR;
  rayTracingPipelineCI.stageCount = static_cast<uint32_t>(shaderStages.size());
  rayTracingPipelineCI.pStages = shaderStages.data();
  rayTracingPipelineCI.groupCount = static_cast<uint32_t>(shaderGroups.size());
  rayTracingPipelineCI.pGroups = shaderGroups.data();
  rayTracingPipelineCI.maxPipelineRayRecursionDepth = 16;
  rayTracingPipelineCI.layout = pipelineLayout;

  VkResult result = vkCreateRayTracingPipelinesKHR(
      device,
      VK_NULL_HANDLE,
      VK_NULL_HANDLE,
      1,
      &rayTracingPipelineCI,
      nullptr,
      &pipeline);

  if (result != VK_SUCCESS) {
    throw std::runtime_error("Failed to create ray tracing pipeline!");
  }
}


void RayTracingSystem::createShaderBindingTable() {
  if (shaderGroups.empty()) {
    throw std::runtime_error("Shader groups are empty! Pipeline may have failed to create.");
  }

  const uint32_t handleSize = rayTracingPipelineProperties.shaderGroupHandleSize;
  const uint32_t handleAlignment = rayTracingPipelineProperties.shaderGroupHandleAlignment;
  const uint32_t groupCount = static_cast<uint32_t>(shaderGroups.size());

  // Align the handle size
  const uint32_t handleSizeAligned =
      static_cast<uint32_t>(LveBuffer::getAlignment(handleSize, handleAlignment));
  const uint32_t sbtSize = groupCount * handleSizeAligned;

  // Get shader group handles (all groups packed with padding/alignment)
  std::vector<uint8_t> shaderHandleStorage(sbtSize);
  VkResult result = vkGetRayTracingShaderGroupHandlesKHR(
      device,
      pipeline,
      0,
      groupCount,
      sbtSize,
      shaderHandleStorage.data());
  if (result != VK_SUCCESS) {
    throw std::runtime_error("vkGetRayTracingShaderGroupHandlesKHR failed");
  }

  // We expect 3 groups in your setup: raygen(0), miss(1), hit(2)
  // Create buffers sized to handleSizeAligned so writes with offsets are safe.
  raygenShaderBindingTable = std::make_unique<LveBuffer>(
      vulkanDevice,
      handleSizeAligned,  // allocate aligned size
      1,
      VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  raygenShaderBindingTable->map();

  missShaderBindingTable = std::make_unique<LveBuffer>(
      vulkanDevice,
      handleSizeAligned,
      2,                  // <-- was 1
      VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  missShaderBindingTable->map();

  
  hitShaderBindingTable = std::make_unique<LveBuffer>(
      vulkanDevice,
      handleSizeAligned,
      1,
      VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  hitShaderBindingTable->map();

 // Offsets shift because hit group is now index 3
  const size_t raygenOffset = 0;
  const size_t missOffset = handleSizeAligned * 1;
  const size_t shadowOffset = handleSizeAligned * 2;
  const size_t hitOffset = handleSizeAligned * 3;

  // If your LveBuffer::writeToBuffer supports (data, size, offset) then prefer that.
  // Otherwise use distinct buffers like here.
  raygenShaderBindingTable->writeToBuffer(shaderHandleStorage.data() + raygenOffset, handleSize);
  missShaderBindingTable->writeToBuffer(shaderHandleStorage.data() + missOffset, handleSize * 2);
  hitShaderBindingTable->writeToBuffer(shaderHandleStorage.data() + hitOffset, handleSize);

  // Optional sanity checks (print addresses)
  auto raygenAddr = getBufferDeviceAddress(raygenShaderBindingTable->getBuffer());
  auto missAddr = getBufferDeviceAddress(missShaderBindingTable->getBuffer());
  auto hitAddr = getBufferDeviceAddress(hitShaderBindingTable->getBuffer());
  std::cout << "SBT device addresses: raygen=" << raygenAddr << " miss=" << missAddr
            << " hit=" << hitAddr << " handleSizeAligned=" << handleSizeAligned << std::endl;
}

VkAccelerationStructureKHR RayTracingSystem::getTLAS() const { return topLevelAS.handle; }


void RayTracingSystem::runShaders(ReSTIRFrameInfo& frameInfo) {
    const uint32_t localSizeX = 8;
    const uint32_t localSizeY = 8;
    uint32_t groupsX = (frameInfo.width + localSizeX - 1) / localSizeX;
    uint32_t groupsY = (frameInfo.height + localSizeY - 1) / localSizeY;

    const uint32_t kSpatialRounds = 3; // tune as needed


    auto makeBufferBarrier = [&](VkBuffer buf, VkAccessFlags src, VkAccessFlags dst)
    {
        VkBufferMemoryBarrier b{ VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER };
        b.srcAccessMask = src;
        b.dstAccessMask = dst;
        b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.buffer = buf;
        b.offset = 0;
        b.size = VK_WHOLE_SIZE;
        return b;
    };

    auto makeImageBarrier = [&](VkImage image,
        VkAccessFlags src, VkAccessFlags dst,
        VkImageLayout oldLayout, VkImageLayout newLayout)
    {
        VkImageMemoryBarrier b{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
        b.srcAccessMask = src;
        b.dstAccessMask = dst;
        b.oldLayout = oldLayout;
        b.newLayout = newLayout;
        b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image = image;
        b.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        return b;
    };

    auto computeBarrier = [&](std::vector<VkBufferMemoryBarrier> bufBarriers,
        std::vector<VkImageMemoryBarrier>  imgBarriers)
    {
        vkCmdPipelineBarrier(
            frameInfo.commandBuffer,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0,
            0, nullptr,
            (uint32_t)bufBarriers.size(), bufBarriers.data(),
            (uint32_t)imgBarriers.size(), imgBarriers.data());
    };

    auto pushConstants = [&](uint32_t roundId, bool isLastRound)
    {
        RestirPushConstants pc{};
        pc.gSpatialRoundId = roundId;
        pc.gIsLastRound = isLastRound ? 1 : 0;
        vkCmdPushConstants(
            frameInfo.commandBuffer,
            restirPipelineLayout,           
            VK_SHADER_STAGE_COMPUTE_BIT,
            0, sizeof(pc), &pc);
    };


    vkCmdBindDescriptorSets(
        frameInfo.commandBuffer,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        restirPipelineLayout,
        0, 1, &frameInfo.globalDescriptorSet,
        0, nullptr);

    // =========================================================================
    // PASS 1 — Temporal Reuse

    pushConstants(0, false);
    vkCmdBindPipeline(frameInfo.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, temporalReusePipeline);
    vkCmdDispatch(frameInfo.commandBuffer, groupsX, groupsY, 1);

    computeBarrier(
        { makeBufferBarrier(outputReservoirBuffer->getBuffer(),
            VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT) },
        {});

    // =========================================================================
    // PASS 2 — Spatial Reuse  (multiple rounds)

    vkCmdBindPipeline(frameInfo.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, spatialReusePipeline);

    for (uint32_t round = 0; round < kSpatialRounds; ++round)
    {
        const bool isLastSpatialRound = (round == kSpatialRounds - 1);
        pushConstants(round, isLastSpatialRound);

        vkCmdDispatch(frameInfo.commandBuffer, groupsX, groupsY, 1);

        computeBarrier(
            { makeBufferBarrier(outputReservoirBuffer->getBuffer(),
                VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT) },
            {});
    }

    // =========================================================================
    // PASS 3 — Temporal Path Retrace

    pushConstants(0, false);
    vkCmdBindPipeline(frameInfo.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, temporalRetracePipeline);
    vkCmdDispatch(frameInfo.commandBuffer, groupsX, groupsY, 1);

    computeBarrier(
        { makeBufferBarrier(outputReservoirBuffer->getBuffer(),
              VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT),
          makeBufferBarrier(temporalReservoirBuffer->getBuffer(),
              VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT) },
        {});

    // =========================================================================
    // PASS 4 — Spatial Path Retrace  (multiple rounds, mirrors spatial reuse)
    // =========================================================================

    vkCmdBindPipeline(frameInfo.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, spatialRetracePipeline);

    for (uint32_t round = 0; round < kSpatialRounds; ++round)
    {
        const bool isLastSpatialRound = (round == kSpatialRounds - 1);
        pushConstants(round, isLastSpatialRound);

        vkCmdDispatch(frameInfo.commandBuffer, groupsX, groupsY, 1);

        computeBarrier(
            { makeBufferBarrier(outputReservoirBuffer->getBuffer(),
                VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT) },
            {});
    }

    // =========================================================================
    // PASS 5 — MIS Weights
    // =========================================================================

    pushConstants(0, true); // gIsLastRound=1 signals final normalization
    vkCmdBindPipeline(frameInfo.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, misWeightsPipeline);
    vkCmdDispatch(frameInfo.commandBuffer, groupsX, groupsY, 1);

    // Final barrier — downstream shading reads the resolved reservoir
    computeBarrier(
        { makeBufferBarrier(outputReservoirBuffer->getBuffer(),
            VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT) },
        {});
}

void RayTracingSystem::createComputePipelineLayout(VkDescriptorSetLayout setLayout) {

    VkPushConstantRange pushConstantRange{};
    pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushConstantRange.offset = 0;
    pushConstantRange.size = sizeof(RestirPushConstants); // your merged struct

    VkPipelineLayoutCreateInfo layoutInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    layoutInfo.setLayoutCount = 1;
    layoutInfo.pSetLayouts = &setLayout;
    layoutInfo.pushConstantRangeCount = 1;
    layoutInfo.pPushConstantRanges = &pushConstantRange;

    if (vkCreatePipelineLayout(vulkanDevice.device(), &layoutInfo, nullptr, &restirPipelineLayout) != VK_SUCCESS) {
        throw std::runtime_error("ReSTIR PT: failed to create pipeline layout");
    }
}

static VkPipeline createSingleComputePipeline(
    LveDevice& device,
    VkPipelineLayout layout,
    VkShaderModule   shaderModule)
{
    VkPipelineShaderStageCreateInfo stageInfo{ VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO };
    stageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stageInfo.module = shaderModule;
    stageInfo.pName = "main";

    VkComputePipelineCreateInfo pipelineInfo{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
    pipelineInfo.stage = stageInfo;
    pipelineInfo.layout = layout;

    VkPipeline pipeline;
    if (vkCreateComputePipelines(device.device(), VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline) != VK_SUCCESS) {
        throw std::runtime_error("ReSTIR PT: failed to create compute pipeline");
    }
    return pipeline;
}

void RayTracingSystem::createComputePipelines() {
    const std::string shaderBase = "C:\\Users\\ZyBros\\Downloads\\NeonEngine\\NeonEngine\\shaders\\";

    //  shader modules
    temporalRetraceModule = LvePipeline::loadShaderModule((shaderBase + "restir_temporal_retrace.comp.spv").c_str(), vulkanDevice.device());
    temporalReuseModule = LvePipeline::loadShaderModule((shaderBase + "restir_temporal_reuse.comp.spv").c_str(), vulkanDevice.device());
    spatialRetraceModule = LvePipeline::loadShaderModule((shaderBase + "restir_spatial_retrace.comp.spv").c_str(), vulkanDevice.device());
    spatialReuseModule = LvePipeline::loadShaderModule((shaderBase + "restir_spatial_reuse.comp.spv").c_str(), vulkanDevice.device());
    misWeightsModule = LvePipeline::loadShaderModule((shaderBase + "restir_mis_reuse.comp.spv").c_str(), vulkanDevice.device());

    //pipelines
    temporalRetracePipeline = createSingleComputePipeline(vulkanDevice, restirPipelineLayout, temporalRetraceModule);
    temporalReusePipeline = createSingleComputePipeline(vulkanDevice, restirPipelineLayout, temporalReuseModule);
    spatialRetracePipeline = createSingleComputePipeline(vulkanDevice, restirPipelineLayout, spatialRetraceModule);
    spatialReusePipeline = createSingleComputePipeline(vulkanDevice, restirPipelineLayout, spatialReuseModule);
    misWeightsPipeline = createSingleComputePipeline(vulkanDevice, restirPipelineLayout, misWeightsModule);
}

/*
void ReSTIRpt::DestroyPipelines() {
    // Pipelines
    vkDestroyPipeline(device.device(), temporalRetracePipeline, nullptr);
    vkDestroyPipeline(device.device(), temporalReusePipeline, nullptr);
    vkDestroyPipeline(device.device(), spatialRetracePipeline, nullptr);
    vkDestroyPipeline(device.device(), spatialReusePipeline, nullptr);
    vkDestroyPipeline(device.device(), misWeightsPipeline, nullptr);

    // Shader modules — safe to destroy after pipeline creation
    vkDestroyShaderModule(device.device(), temporalRetraceModule, nullptr);
    vkDestroyShaderModule(device.device(), temporalReuseModule, nullptr);
    vkDestroyShaderModule(device.device(), spatialRetraceModule, nullptr);
    vkDestroyShaderModule(device.device(), spatialReuseModule, nullptr);
    vkDestroyShaderModule(device.device(), misWeightsModule, nullptr);

    // Layout
    vkDestroyPipelineLayout(device.device(), restirPipelineLayout, nullptr);
}
*/

void RayTracingSystem::VK_CHECK_RESULT(VkResult f) {
  VkResult res = (f);
  if (res != VK_SUCCESS) {
    std::cout << std::endl << std::endl << "Fatal: VkResult is " << f << std::endl;
  }
}  // namespace lve





}  // namespace lve