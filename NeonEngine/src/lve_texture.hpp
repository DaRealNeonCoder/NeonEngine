#pragma once

#include <vulkan.h>

#include "lve_buffer.hpp"
#include "lve_device.hpp"

namespace lve {

class LveTexture {
 public:

  struct AllocatedImage {
    VkImage image;
    VkDeviceMemory memory;
    VkImageView imageView;
    VkSampler sampler;
    VkExtent3D imageExtent;
    VkFormat imageFormat;
  };
  // Public interface
  AllocatedImage memoryStuff(const char* file, LveDevice& lveDevice);
  void destroyImage(LveDevice& device, AllocatedImage& image);
  VkDescriptorImageInfo getDescriptor(AllocatedImage& img);
  AllocatedImage loadHDR(const char* file, LveDevice& lveDevice);
 private:
  // Helpers (all defined in the .cpp)
  void createImage(
      LveDevice& device,
      VkExtent3D extent,
      VkFormat format,
      VkImageUsageFlags usage,
      AllocatedImage& outImage);

  void createImageView(LveDevice& device, AllocatedImage& image);
  void createSampler(LveDevice& device, AllocatedImage& image);
  void uploadImageData(LveDevice& device, VkBuffer stagingBuffer, AllocatedImage& image);
};

}  // namespace lve
