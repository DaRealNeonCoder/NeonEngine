#include "lve_descriptors.hpp"
#include <vulkan.h>

// std
#include <cassert>
#include <stdexcept>

namespace lve {

// *************** Descriptor Set Layout Builder *********************

LveDescriptorSetLayout::Builder &LveDescriptorSetLayout::Builder::addBinding(
    uint32_t binding,
    VkDescriptorType descriptorType,
    VkShaderStageFlags stageFlags,
    uint32_t count) {
  assert(bindings.count(binding) == 0 && "Binding already in use");
  VkDescriptorSetLayoutBinding layoutBinding{};
  layoutBinding.binding = binding;
  layoutBinding.descriptorType = descriptorType;
  layoutBinding.descriptorCount = count;
  layoutBinding.stageFlags = stageFlags;
  bindings[binding] = layoutBinding;
  return *this;
}

LveDescriptorSetLayout::Builder& LveDescriptorSetLayout::Builder::addBindlessBinding(
    uint32_t binding, VkDescriptorType descriptorType,
    VkShaderStageFlags stageFlags, uint32_t maxCount) {
    assert(bindings.count(binding) == 0 && "Binding already in use");

    VkDescriptorSetLayoutBinding layoutBinding{};
    layoutBinding.binding = binding;
    layoutBinding.descriptorType = descriptorType;
    layoutBinding.descriptorCount = maxCount;
    layoutBinding.stageFlags = stageFlags;
    bindings[binding] = layoutBinding;

    bindingFlags[binding] = VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT |
        VK_DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT_BIT;
    return *this;
}

std::unique_ptr<LveDescriptorSetLayout> LveDescriptorSetLayout::Builder::build() const {
  return std::make_unique<LveDescriptorSetLayout>(lveDevice, bindings, bindingFlags);
}
// *************** Descriptor Set Layout *********************
// .cpp
LveDescriptorSetLayout::LveDescriptorSetLayout(
    LveDevice& lveDevice,
    std::unordered_map<uint32_t, VkDescriptorSetLayoutBinding> bindings,
    std::unordered_map<uint32_t, VkDescriptorBindingFlags> bindingFlags)
    : lveDevice{ lveDevice }, bindings{ bindings } {

    std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings{};
    for (const auto kv : bindings) setLayoutBindings.push_back(kv.second);

    VkDescriptorSetLayoutCreateInfo descriptorSetLayoutInfo{};
    descriptorSetLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    descriptorSetLayoutInfo.bindingCount = static_cast<uint32_t>(setLayoutBindings.size());
    descriptorSetLayoutInfo.pBindings = setLayoutBindings.data();

    // Only attach flags info if any bindless bindings were registered
    std::vector<VkDescriptorBindingFlags> flags;
    VkDescriptorSetLayoutBindingFlagsCreateInfo flagsInfo{};

    if (!bindingFlags.empty()) {
        // Must be one entry per binding, in the same order as setLayoutBindings
        flags.resize(setLayoutBindings.size(), 0);
        for (size_t i = 0; i < setLayoutBindings.size(); i++) {
            uint32_t b = setLayoutBindings[i].binding;
            if (bindingFlags.count(b)) flags[i] = bindingFlags.at(b);
        }

        flagsInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO;
        flagsInfo.bindingCount = static_cast<uint32_t>(flags.size());
        flagsInfo.pBindingFlags = flags.data();
        descriptorSetLayoutInfo.pNext = &flagsInfo;
    }

    if (vkCreateDescriptorSetLayout(lveDevice.device(), &descriptorSetLayoutInfo,
        nullptr, &descriptorSetLayout) != VK_SUCCESS) {
        throw std::runtime_error("failed to create descriptor set layout!");
    }
}


LveDescriptorSetLayout::~LveDescriptorSetLayout() {
  vkDestroyDescriptorSetLayout(lveDevice.device(), descriptorSetLayout, nullptr);
}

// *************** Descriptor Pool Builder *********************

LveDescriptorPool::Builder &LveDescriptorPool::Builder::addPoolSize(
    VkDescriptorType descriptorType, uint32_t count) {
  poolSizes.push_back({descriptorType, count});
  return *this;
}

LveDescriptorPool::Builder &LveDescriptorPool::Builder::setPoolFlags(
    VkDescriptorPoolCreateFlags flags) {
  poolFlags = flags;
  return *this;
}
LveDescriptorPool::Builder &LveDescriptorPool::Builder::setMaxSets(uint32_t count) {
  maxSets = count;
  return *this;
}

std::unique_ptr<LveDescriptorPool> LveDescriptorPool::Builder::build() const {
  return std::make_unique<LveDescriptorPool>(lveDevice, maxSets, poolFlags, poolSizes);
}

// *************** Descriptor Pool *********************

LveDescriptorPool::LveDescriptorPool(
    LveDevice &lveDevice,
    uint32_t maxSets,
    VkDescriptorPoolCreateFlags poolFlags,
    const std::vector<VkDescriptorPoolSize> &poolSizes)
    : lveDevice{lveDevice} {
  VkDescriptorPoolCreateInfo descriptorPoolInfo{};
  descriptorPoolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  descriptorPoolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
  descriptorPoolInfo.pPoolSizes = poolSizes.data();
  descriptorPoolInfo.maxSets = maxSets;
  descriptorPoolInfo.flags = poolFlags;

  if (vkCreateDescriptorPool(lveDevice.device(), &descriptorPoolInfo, nullptr, &descriptorPool) !=
      VK_SUCCESS) {
    throw std::runtime_error("failed to create descriptor pool!");
  }
}

LveDescriptorPool::~LveDescriptorPool() {
  vkDestroyDescriptorPool(lveDevice.device(), descriptorPool, nullptr);
}

bool LveDescriptorPool::allocateDescriptor(
    const VkDescriptorSetLayout descriptorSetLayout, VkDescriptorSet &descriptor) const {
  VkDescriptorSetAllocateInfo allocInfo{};
  allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  allocInfo.descriptorPool = descriptorPool;
  allocInfo.pSetLayouts = &descriptorSetLayout;
  allocInfo.descriptorSetCount = 1;

  // Might want to create a "DescriptorPoolManager" class that handles this case, and builds
  // a new pool whenever an old pool fills up. But this is beyond our current scope
  if (vkAllocateDescriptorSets(lveDevice.device(), &allocInfo, &descriptor) != VK_SUCCESS) {
    return false;
  }
  return true;
}

bool LveDescriptorPool::allocateDescriptor(
    const VkDescriptorSetLayout descriptorSetLayout,
    VkDescriptorSet& descriptor,
    uint32_t variableCount
) const {

    VkDescriptorSetVariableDescriptorCountAllocateInfo countInfo{};
    countInfo.sType =
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_VARIABLE_DESCRIPTOR_COUNT_ALLOCATE_INFO;
    countInfo.descriptorSetCount = 1;
    countInfo.pDescriptorCounts = &variableCount;

    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.pNext = &countInfo; 
    allocInfo.descriptorPool = descriptorPool;
    allocInfo.pSetLayouts = &descriptorSetLayout;
    allocInfo.descriptorSetCount = 1;

    if (vkAllocateDescriptorSets(lveDevice.device(), &allocInfo, &descriptor) != VK_SUCCESS) {
        return false;
    }
    return true;
}
void LveDescriptorPool::freeDescriptors(std::vector<VkDescriptorSet> &descriptors) const {
  vkFreeDescriptorSets(
      lveDevice.device(),
      descriptorPool,
      static_cast<uint32_t>(descriptors.size()),
      descriptors.data());
}

void LveDescriptorPool::resetPool() {
  vkResetDescriptorPool(lveDevice.device(), descriptorPool, 0);
}

// *************** Descriptor Writer *********************

LveDescriptorWriter::LveDescriptorWriter(LveDescriptorSetLayout &setLayout, LveDescriptorPool &pool)
    : setLayout{setLayout}, pool{pool} {}

LveDescriptorWriter &LveDescriptorWriter::writeBuffer(
    uint32_t binding, VkDescriptorBufferInfo *bufferInfo) {
  assert(setLayout.bindings.count(binding) == 1 && "Layout does not contain specified binding");

  auto &bindingDescription = setLayout.bindings[binding];

  assert(
      bindingDescription.descriptorCount == 1 &&
      "Binding single descriptor info, but binding expects multiple");

  VkWriteDescriptorSet write{};
  write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  write.descriptorType = bindingDescription.descriptorType;
  write.dstBinding = binding;
  write.pBufferInfo = bufferInfo;
  write.descriptorCount = 1;

  writes.push_back(write);
  return *this;
}

LveDescriptorWriter &LveDescriptorWriter::writeImage(
    uint32_t binding, VkDescriptorImageInfo *imageInfo) {
  assert(setLayout.bindings.count(binding) == 1 && "Layout does not contain specified binding");

  auto &bindingDescription = setLayout.bindings[binding];

  assert(
      bindingDescription.descriptorCount == 1 &&
      "Binding single descriptor info, but binding expects multiple");

  VkWriteDescriptorSet write{};
  write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  write.descriptorType = bindingDescription.descriptorType;
  write.dstBinding = binding;
  write.pImageInfo = imageInfo;
  write.descriptorCount = 1;
  writes.push_back(write);
  return *this;
}
LveDescriptorWriter& LveDescriptorWriter::writeImageArray(
    uint32_t binding, std::vector<VkDescriptorImageInfo>& imageInfos) {
    assert(setLayout.bindings.count(binding) == 1 && "Layout does not contain specified binding");

    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    write.dstBinding = binding;
    write.dstArrayElement = 0;
    write.descriptorCount = static_cast<uint32_t>(imageInfos.size());
    write.pImageInfo = imageInfos.data();

    writes.push_back(write);
    return *this;
}
LveDescriptorWriter &LveDescriptorWriter::writeAccelerationStructure(
    uint32_t binding, VkWriteDescriptorSetAccelerationStructureKHR *asInfo, VkDescriptorSet &set) {
  assert(setLayout.bindings.count(binding) == 1 && "Layout does not contain specified binding");

  const auto &bindingDescription = setLayout.bindings[binding];

  assert(
      bindingDescription.descriptorType == VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR &&
      "Binding type mismatch: expected ACCELERATION_STRUCTURE_KHR");

  assert(
      bindingDescription.descriptorCount == 1 &&
      "Binding single descriptor info, but binding expects multiple");

  VkWriteDescriptorSet write{};
  write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  write.descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
  write.dstBinding = binding;
  write.descriptorCount = 1;
  write.pNext = asInfo;
  write.dstSet = set;

  writes.push_back(write);
  return *this;
}

bool LveDescriptorWriter::build(VkDescriptorSet &set) {
  bool success = pool.allocateDescriptor(setLayout.getDescriptorSetLayout(), set, 10);// 10 = number of descriptors for bindless bindings (arrays of textures for example)

  if (!success) {
    return false;
  }
  overwrite(set);
  return true;
}

void LveDescriptorWriter::overwrite(VkDescriptorSet &set) {
  for (auto &write : writes) {
    write.dstSet = set;
  }
  vkUpdateDescriptorSets(pool.lveDevice.device(), writes.size(), writes.data(), 0, nullptr);
}

}  // namespace lve
