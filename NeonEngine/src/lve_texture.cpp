#include "lve_texture.hpp"

#include <cstring>
#include <iostream>

#include "lve_buffer.hpp"

// Implement STB only once, in this cpp
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

namespace lve {

// ---------------------------------------------------------
// Public methods
// ---------------------------------------------------------
LveTexture::AllocatedImage LveTexture::memoryStuff(const char* file, LveDevice& lveDevice) {
  int texWidth, texHeight, texChannels;
  stbi_uc* pixels = stbi_load(file, &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);

  if (!pixels) {
    std::cerr << "Failed to load texture file " << file << std::endl;
    return AllocatedImage{};
  }

  VkDeviceSize imageSize = static_cast<VkDeviceSize>(texWidth) * texHeight * 4;

  // Create staging buffer
  LveBuffer stagingBuffer{
      lveDevice,
      imageSize,
      1,
      VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT};

  stagingBuffer.map();
  stagingBuffer.writeToBuffer(pixels);

  stbi_image_free(pixels);

  // Create GPU image
  VkExtent3D extent{static_cast<uint32_t>(texWidth), static_cast<uint32_t>(texHeight), 1};
  VkFormat format = VK_FORMAT_R8G8B8A8_SRGB;
  AllocatedImage image;
  createImage(
      lveDevice,
      extent,
      format,
      VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
      image);

  uploadImageData(lveDevice, stagingBuffer.getBuffer(), image);

  createImageView(lveDevice, image);
  createSampler(lveDevice, image);

  return image;
}

void LveTexture::destroyImage(LveDevice& device, AllocatedImage& image) {
  vkDestroySampler(device.device(), image.sampler, nullptr);
  vkDestroyImageView(device.device(), image.imageView, nullptr);
  vkDestroyImage(device.device(), image.image, nullptr);
  vkFreeMemory(device.device(), image.memory, nullptr);
}

// ---------------------------------------------------------
// Private helpers
// ---------------------------------------------------------
void LveTexture::createImage(
    LveDevice& device,
    VkExtent3D extent,
    VkFormat format,
    VkImageUsageFlags usage,
    AllocatedImage& outImage) {
  outImage.imageExtent = extent;
  outImage.imageFormat = format;

  VkImageCreateInfo imageInfo{};
  imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  imageInfo.imageType = VK_IMAGE_TYPE_2D;
  imageInfo.extent = extent;
  imageInfo.mipLevels = 1;
  imageInfo.arrayLayers = 1;
  imageInfo.format = format;
  imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
  imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  imageInfo.usage = usage;
  imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
  imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  device.createImageWithInfo(
      imageInfo,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
      outImage.image,
      outImage.memory);
}

void LveTexture::createImageView(LveDevice& device, AllocatedImage& image) {
  VkImageViewCreateInfo viewInfo{};
  viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
  viewInfo.image = image.image;
  viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
  viewInfo.format = image.imageFormat;

  viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  viewInfo.subresourceRange.baseMipLevel = 0;
  viewInfo.subresourceRange.levelCount = 1;
  viewInfo.subresourceRange.baseArrayLayer = 0;
  viewInfo.subresourceRange.layerCount = 1;

  if (vkCreateImageView(device.device(), &viewInfo, nullptr, &image.imageView) != VK_SUCCESS) {
    throw std::runtime_error("failed to create image view");
  }
}

void LveTexture::createSampler(LveDevice& device, AllocatedImage& image) {
  VkSamplerCreateInfo samplerInfo{};
  samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
  samplerInfo.magFilter = VK_FILTER_LINEAR;
  samplerInfo.minFilter = VK_FILTER_LINEAR;

  samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
  samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
  samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;

  samplerInfo.anisotropyEnable = VK_TRUE;
  samplerInfo.maxAnisotropy = 16.0f;

  samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
  samplerInfo.unnormalizedCoordinates = VK_FALSE;
  samplerInfo.compareEnable = VK_FALSE;
  samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;

  if (vkCreateSampler(device.device(), &samplerInfo, nullptr, &image.sampler) != VK_SUCCESS) {
    throw std::runtime_error("failed to create sampler");
  }
}

void LveTexture::uploadImageData(LveDevice& device, VkBuffer stagingBuffer, AllocatedImage& image) {
  VkCommandBuffer cmd = device.beginSingleTimeCommands();

  VkImageSubresourceRange range{};
  range.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  range.baseMipLevel = 0;
  range.levelCount = 1;
  range.baseArrayLayer = 0;
  range.layerCount = 1;

  VkImageMemoryBarrier toTransfer{};
  toTransfer.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  toTransfer.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  toTransfer.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
  toTransfer.image = image.image;
  toTransfer.subresourceRange = range;
  toTransfer.srcAccessMask = 0;
  toTransfer.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

  vkCmdPipelineBarrier(
      cmd,
      VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
      VK_PIPELINE_STAGE_TRANSFER_BIT,
      0,
      0,
      nullptr,
      0,
      nullptr,
      1,
      &toTransfer);

  VkBufferImageCopy copy{};
  copy.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  copy.imageSubresource.mipLevel = 0;
  copy.imageSubresource.baseArrayLayer = 0;
  copy.imageSubresource.layerCount = 1;
  copy.imageExtent = image.imageExtent;

  vkCmdCopyBufferToImage(
      cmd,
      stagingBuffer,
      image.image,
      VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
      1,
      &copy);

  VkImageMemoryBarrier toReadable = toTransfer;
  toReadable.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
  toReadable.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  toReadable.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  toReadable.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

  vkCmdPipelineBarrier(
      cmd,
      VK_PIPELINE_STAGE_TRANSFER_BIT,
      VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
      0,
      0,
      nullptr,
      0,
      nullptr,
      1,
      &toReadable);

  device.endSingleTimeCommands(cmd);
}
VkDescriptorImageInfo LveTexture::getDescriptor(AllocatedImage& img) {
  VkDescriptorImageInfo info{};
  info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  info.imageView = img.imageView;
  info.sampler = img.sampler;
  return info;
}


LveTexture::AllocatedImage LveTexture::loadHDR(const char* file, LveDevice& lveDevice) {
    int texWidth, texHeight, texChannels;
    float* pixels = stbi_loadf(file, &texWidth, &texHeight, &texChannels, 4);

    if (!pixels) {
        std::cerr << "Failed to load HDR file " << file << std::endl;
        return AllocatedImage{};
    }

    VkDeviceSize imageSize = static_cast<VkDeviceSize>(texWidth) * texHeight * 4 * sizeof(float);

    LveBuffer stagingBuffer{
        lveDevice,
        imageSize,
        1,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT };

    stagingBuffer.map();
    stagingBuffer.writeToBuffer(pixels);

    stbi_image_free(pixels);

    VkExtent3D extent{ static_cast<uint32_t>(texWidth), static_cast<uint32_t>(texHeight), 1 };
    VkFormat format = VK_FORMAT_R32G32B32A32_SFLOAT;
    AllocatedImage image;
    createImage(
        lveDevice,
        extent,
        format,
        VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        image);

    uploadImageData(lveDevice, stagingBuffer.getBuffer(), image);

    createImageView(lveDevice, image);
    createSampler(lveDevice, image);

    return image;
}

}  // namespace lve

 
