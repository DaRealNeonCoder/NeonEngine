#include "ml_compute.hpp"

#include "slang_runtime_compiler.hpp"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace lve::ml {

namespace fs = std::filesystem;

namespace {

void vkCheck(VkResult r, const char* what) {
    if (r != VK_SUCCESS) throw std::runtime_error(std::string("Vulkan: ") + what + " failed (" + std::to_string(r) + ")");
}

std::string lower(std::string s) {
    for (char& c : s) c = char(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

uint64_t fnv1a(const void* data, size_t n, uint64_t h = 1469598103934665603ull) {
    const uint8_t* p = static_cast<const uint8_t*>(data);
    for (size_t i = 0; i < n; ++i) {
        h ^= p[i];
        h *= 1099511628211ull;
    }
    return h;
}

uint64_t fnv1a(const std::string& s, uint64_t h) { return fnv1a(s.data(), s.size(), h); }

std::string hex16(uint64_t v) {
    static const char* digits = "0123456789abcdef";
    std::string s(16, '0');
    for (int i = 15; i >= 0; --i) {
        s[size_t(i)] = digits[v & 15];
        v >>= 4;
    }
    return s;
}

constexpr VkDeviceSize kStagingChunk = 64ull << 20;

}  // namespace

// Holds the Slang global session for the context's lifetime (creating it is slow).
static std::unique_ptr<SlangRuntimeCompiler>& compilerInstance() {
    static std::unique_ptr<SlangRuntimeCompiler> c;
    return c;
}

ComputeContext::ComputeContext(const std::string& shaderRoot, const std::string& deviceFilter)
    : shaderRoot_(shaderRoot) {
    VkApplicationInfo app{ VK_STRUCTURE_TYPE_APPLICATION_INFO };
    app.pApplicationName = "neonengine_llm";
    app.pEngineName = "neonengine";
    app.apiVersion = VK_API_VERSION_1_3;

    VkInstanceCreateInfo ici{ VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
    ici.pApplicationInfo = &app;
    const char* validation = "VK_LAYER_KHRONOS_validation";
    const char* env = std::getenv("NEON_VK_VALIDATION");
    if (env && std::string(env) == "1") {
        ici.enabledLayerCount = 1;
        ici.ppEnabledLayerNames = &validation;
    }
    vkCheck(vkCreateInstance(&ici, nullptr, &instance_), "vkCreateInstance");

    pickDevice(deviceFilter);
    createDevice();

    VkCommandPoolCreateInfo pci{ VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
    pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pci.queueFamilyIndex = queueFamily_;
    vkCheck(vkCreateCommandPool(device_, &pci, nullptr, &pool_), "vkCreateCommandPool");

    VkPushConstantRange pcr{ VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants) };
    VkPipelineLayoutCreateInfo lci{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    lci.pushConstantRangeCount = 1;
    lci.pPushConstantRanges = &pcr;
    vkCheck(vkCreatePipelineLayout(device_, &lci, nullptr, &layout_), "vkCreatePipelineLayout");

    VkQueryPoolCreateInfo qci{ VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO };
    qci.queryType = VK_QUERY_TYPE_TIMESTAMP;
    qci.queryCount = 2;
    vkCheck(vkCreateQueryPool(device_, &qci, nullptr, &queryPool_), "vkCreateQueryPool");

    VkFenceCreateInfo fci{ VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    vkCheck(vkCreateFence(device_, &fci, nullptr, &fence_), "vkCreateFence");
}

ComputeContext::~ComputeContext() {
    if (device_) vkDeviceWaitIdle(device_);
    destroyBuffer(staging_);
    for (auto& [key, p] : pipelines_) vkDestroyPipeline(device_, p.pipeline, nullptr);
    for (VkShaderModule m : modules_) vkDestroyShaderModule(device_, m, nullptr);
    if (fence_) vkDestroyFence(device_, fence_, nullptr);
    if (queryPool_) vkDestroyQueryPool(device_, queryPool_, nullptr);
    if (profPool_) vkDestroyQueryPool(device_, profPool_, nullptr);
    if (layout_) vkDestroyPipelineLayout(device_, layout_, nullptr);
    if (pool_) vkDestroyCommandPool(device_, pool_, nullptr);
    if (device_) vkDestroyDevice(device_, nullptr);
    if (instance_) vkDestroyInstance(instance_, nullptr);
}

void ComputeContext::pickDevice(const std::string& filter) {
    uint32_t count = 0;
    vkEnumeratePhysicalDevices(instance_, &count, nullptr);
    std::vector<VkPhysicalDevice> devices(count);
    vkEnumeratePhysicalDevices(instance_, &count, devices.data());

    const std::string want = lower(filter);
    int bestScore = -1;
    for (VkPhysicalDevice d : devices) {
        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(d, &props);
        if (!want.empty() && lower(props.deviceName).find(want) == std::string::npos) continue;
        if (props.apiVersion < VK_API_VERSION_1_3) continue;
        uint32_t qn = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(d, &qn, nullptr);
        std::vector<VkQueueFamilyProperties> qf(qn);
        vkGetPhysicalDeviceQueueFamilyProperties(d, &qn, qf.data());
        int family = -1;
        for (uint32_t i = 0; i < qn; ++i) {
            if (qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                family = int(i);
                break;
            }
        }
        if (family < 0) continue;
        int score = props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU ? 2 : 1;
        if (score > bestScore) {
            bestScore = score;
            physical_ = d;
            queueFamily_ = uint32_t(family);
            deviceName_ = props.deviceName;
            timestampPeriodNs_ = props.limits.timestampPeriod;
        }
    }
    if (!physical_) throw std::runtime_error("No Vulkan 1.3 compute device found" + (filter.empty() ? "" : " matching '" + filter + "'"));

    VkPhysicalDeviceSubgroupProperties sub{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES };
    VkPhysicalDeviceMaintenance3Properties m3{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_3_PROPERTIES };
    sub.pNext = &m3;
    VkPhysicalDeviceProperties2 p2{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2 };
    p2.pNext = &sub;
    vkGetPhysicalDeviceProperties2(physical_, &p2);
    subgroupSize_ = sub.subgroupSize;
    maxAllocation_ = m3.maxMemoryAllocationSize;

    vkGetPhysicalDeviceMemoryProperties(physical_, &memProps_);
    for (uint32_t i = 0; i < memProps_.memoryHeapCount; ++i) {
        if (memProps_.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
            deviceLocalBytes_ = std::max<uint64_t>(deviceLocalBytes_, memProps_.memoryHeaps[i].size);
    }
}

void ComputeContext::createDevice() {
    uint32_t extCount = 0;
    vkEnumerateDeviceExtensionProperties(physical_, nullptr, &extCount, nullptr);
    std::vector<VkExtensionProperties> exts(extCount);
    vkEnumerateDeviceExtensionProperties(physical_, nullptr, &extCount, exts.data());
    bool extCoopMat = false;
    for (const auto& e : exts)
        if (std::strcmp(e.extensionName, VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME) == 0) extCoopMat = true;

    // Query.
    VkPhysicalDeviceVulkan11Features q11{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES };
    VkPhysicalDeviceVulkan12Features q12{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES };
    VkPhysicalDeviceVulkan13Features q13{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES };
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR qcm{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR };
    q12.pNext = &q11;
    q13.pNext = &q12;
    void* chain = &q13;
    if (extCoopMat) {
        qcm.pNext = &q13;
        chain = &qcm;
    }
    VkPhysicalDeviceFeatures2 q2{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2 };
    q2.pNext = chain;
    vkGetPhysicalDeviceFeatures2(physical_, &q2);

    if (!q12.bufferDeviceAddress || !q12.scalarBlockLayout || !q2.features.shaderInt64)
        throw std::runtime_error("Device lacks bufferDeviceAddress / scalarBlockLayout / shaderInt64");
    if (!q12.shaderInt8 || !q12.storageBuffer8BitAccess || !q11.storageBuffer16BitAccess || !q13.shaderIntegerDotProduct)
        throw std::runtime_error("Device lacks int8 / 8-16-bit storage / integer dot product (required by the LLM kernels)");
    hasCoopMat_ = extCoopMat && qcm.cooperativeMatrix;
    hasFullSubgroups_ = q13.computeFullSubgroups;

    // Enable.
    VkPhysicalDeviceVulkan11Features f11{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES };
    f11.storageBuffer16BitAccess = q11.storageBuffer16BitAccess;
    f11.uniformAndStorageBuffer16BitAccess = q11.uniformAndStorageBuffer16BitAccess;
    VkPhysicalDeviceVulkan12Features f12{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES };
    f12.pNext = &f11;
    f12.bufferDeviceAddress = VK_TRUE;
    f12.scalarBlockLayout = VK_TRUE;
    f12.shaderFloat16 = q12.shaderFloat16;
    f12.shaderInt8 = q12.shaderInt8;
    f12.storageBuffer8BitAccess = q12.storageBuffer8BitAccess;
    f12.uniformAndStorageBuffer8BitAccess = q12.uniformAndStorageBuffer8BitAccess;
    f12.vulkanMemoryModel = q12.vulkanMemoryModel;
    f12.vulkanMemoryModelDeviceScope = q12.vulkanMemoryModelDeviceScope;
    f12.shaderSubgroupExtendedTypes = q12.shaderSubgroupExtendedTypes;
    f12.shaderBufferInt64Atomics = q12.shaderBufferInt64Atomics;
    VkPhysicalDeviceVulkan13Features f13{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES };
    f13.pNext = &f12;
    f13.subgroupSizeControl = q13.subgroupSizeControl;
    f13.computeFullSubgroups = q13.computeFullSubgroups;
    f13.shaderIntegerDotProduct = q13.shaderIntegerDotProduct;
    f13.maintenance4 = q13.maintenance4;
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR fcm{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR };
    void* enableChain = &f13;
    std::vector<const char*> enabledExts;
    if (hasCoopMat_) {
        fcm.pNext = &f13;
        fcm.cooperativeMatrix = VK_TRUE;
        enableChain = &fcm;
        enabledExts.push_back(VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME);
    }
    VkPhysicalDeviceFeatures2 features{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2 };
    features.pNext = enableChain;
    features.features.shaderInt64 = VK_TRUE;
    features.features.shaderInt16 = q2.features.shaderInt16;

    float priority = 1.0f;
    VkDeviceQueueCreateInfo qci{ VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO };
    qci.queueFamilyIndex = queueFamily_;
    qci.queueCount = 1;
    qci.pQueuePriorities = &priority;

    VkDeviceCreateInfo dci{ VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
    dci.pNext = &features;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = uint32_t(enabledExts.size());
    dci.ppEnabledExtensionNames = enabledExts.empty() ? nullptr : enabledExts.data();
    vkCheck(vkCreateDevice(physical_, &dci, nullptr, &device_), "vkCreateDevice");
    vkGetDeviceQueue(device_, queueFamily_, 0, &queue_);
}

uint32_t ComputeContext::memoryType(uint32_t typeBits, VkMemoryPropertyFlags want, VkMemoryPropertyFlags avoid) const {
    for (int pass = 0; pass < 2; ++pass) {
        for (uint32_t i = 0; i < memProps_.memoryTypeCount; ++i) {
            VkMemoryPropertyFlags f = memProps_.memoryTypes[i].propertyFlags;
            if (!(typeBits & (1u << i)) || (f & want) != want) continue;
            if (pass == 0 && (f & avoid)) continue;
            return i;
        }
    }
    return UINT32_MAX;
}

GpuBuffer ComputeContext::createBuffer(size_t bytes, MemKind kind) {
    bytes = std::max<size_t>((bytes + 15) & ~size_t(15), 16);
    if (maxAllocation_ && bytes > maxAllocation_)
        throw std::runtime_error("createBuffer: " + std::to_string(bytes) + " bytes exceeds maxMemoryAllocationSize " +
                                 std::to_string(maxAllocation_));
    GpuBuffer b;
    b.size = bytes;
    VkBufferCreateInfo bci{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    bci.size = bytes;
    bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    vkCheck(vkCreateBuffer(device_, &bci, nullptr, &b.buffer), "vkCreateBuffer");
    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(device_, b.buffer, &req);

    const VkMemoryPropertyFlags hostFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    uint32_t type = UINT32_MAX;
    switch (kind) {
        case MemKind::Device:
            type = memoryType(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
            break;
        case MemKind::Host:
            type = memoryType(req.memoryTypeBits, hostFlags, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            break;
        case MemKind::HostDevice:
            type = memoryType(req.memoryTypeBits, hostFlags | VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0);
            if (type == UINT32_MAX) type = memoryType(req.memoryTypeBits, hostFlags, 0);
            break;
    }
    if (type == UINT32_MAX) throw std::runtime_error("createBuffer: no suitable memory type");

    VkMemoryAllocateFlagsInfo flags{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO };
    flags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
    VkMemoryAllocateInfo mai{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    mai.pNext = &flags;
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = type;
    VkResult r = vkAllocateMemory(device_, &mai, nullptr, &b.memory);
    if (r != VK_SUCCESS) {
        vkDestroyBuffer(device_, b.buffer, nullptr);
        throw std::runtime_error("createBuffer: vkAllocateMemory failed for " + std::to_string(bytes) +
                                 " bytes (" + std::to_string(r) + "); out of " +
                                 (kind == MemKind::Device ? "VRAM" : "host-visible memory"));
    }
    vkCheck(vkBindBufferMemory(device_, b.buffer, b.memory, 0), "vkBindBufferMemory");

    VkMemoryPropertyFlags props = memProps_.memoryTypes[type].propertyFlags;
    b.deviceLocal = (props & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0;
    if (props & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
        vkCheck(vkMapMemory(device_, b.memory, 0, VK_WHOLE_SIZE, 0, &b.mapped), "vkMapMemory");

    VkBufferDeviceAddressInfo ai{ VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO };
    ai.buffer = b.buffer;
    b.address = vkGetBufferDeviceAddress(device_, &ai);
    return b;
}

void ComputeContext::destroyBuffer(GpuBuffer& b) {
    if (!b.buffer) return;
    if (b.mapped) vkUnmapMemory(device_, b.memory);
    vkDestroyBuffer(device_, b.buffer, nullptr);
    vkFreeMemory(device_, b.memory, nullptr);
    b = GpuBuffer{};
}

GpuBuffer& ComputeContext::staging(size_t bytes) {
    if (!staging_ || staging_.size < bytes) {
        destroyBuffer(staging_);
        staging_ = createBuffer(std::max<size_t>(bytes, 1u << 20), MemKind::Host);
    }
    return staging_;
}

VkCommandBuffer ComputeContext::beginOneShot() {
    VkCommandBufferAllocateInfo ai{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
    ai.commandPool = pool_;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    VkCommandBuffer cmd;
    vkCheck(vkAllocateCommandBuffers(device_, &ai, &cmd), "vkAllocateCommandBuffers");
    VkCommandBufferBeginInfo bi{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &bi);
    return cmd;
}

void ComputeContext::endOneShot(VkCommandBuffer cmd) {
    vkEndCommandBuffer(cmd);
    VkSubmitInfo si{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    vkCheck(vkQueueSubmit(queue_, 1, &si, fence_), "vkQueueSubmit");
    vkCheck(vkWaitForFences(device_, 1, &fence_, VK_TRUE, UINT64_MAX), "vkWaitForFences");
    vkResetFences(device_, 1, &fence_);
    vkFreeCommandBuffers(device_, pool_, 1, &cmd);
}

void ComputeContext::upload(const GpuBuffer& dst, size_t dstOffset, const void* src, size_t bytes) {
    if (bytes == 0) return;
    if (dstOffset + bytes > dst.size) throw std::runtime_error("upload: out of range");
    if (dst.mapped) {
        std::memcpy(static_cast<uint8_t*>(dst.mapped) + dstOffset, src, bytes);
        return;
    }
    GpuBuffer& st = staging(size_t(std::min<VkDeviceSize>(bytes, kStagingChunk)));
    const uint8_t* p = static_cast<const uint8_t*>(src);
    size_t done = 0;
    while (done < bytes) {
        size_t n = std::min(bytes - done, st.size);
        std::memcpy(st.mapped, p + done, n);
        VkCommandBuffer cmd = beginOneShot();
        VkBufferCopy copy{ 0, VkDeviceSize(dstOffset + done), VkDeviceSize(n) };
        vkCmdCopyBuffer(cmd, st.buffer, dst.buffer, 1, &copy);
        endOneShot(cmd);
        done += n;
    }
}

void ComputeContext::download(const GpuBuffer& src, size_t srcOffset, void* dst, size_t bytes) {
    if (bytes == 0) return;
    if (srcOffset + bytes > src.size) throw std::runtime_error("download: out of range");
    if (src.mapped) {
        std::memcpy(dst, static_cast<const uint8_t*>(src.mapped) + srcOffset, bytes);
        return;
    }
    GpuBuffer& st = staging(size_t(std::min<VkDeviceSize>(bytes, kStagingChunk)));
    uint8_t* p = static_cast<uint8_t*>(dst);
    size_t done = 0;
    while (done < bytes) {
        size_t n = std::min(bytes - done, st.size);
        VkCommandBuffer cmd = beginOneShot();
        VkBufferCopy copy{ VkDeviceSize(srcOffset + done), 0, VkDeviceSize(n) };
        vkCmdCopyBuffer(cmd, src.buffer, st.buffer, 1, &copy);
        endOneShot(cmd);
        std::memcpy(p + done, st.mapped, n);
        done += n;
    }
}

void ComputeContext::fillZero(const GpuBuffer& b) {
    VkCommandBuffer cmd = beginOneShot();
    vkCmdFillBuffer(cmd, b.buffer, 0, VK_WHOLE_SIZE, 0);
    endOneShot(cmd);
}

std::vector<uint32_t> ComputeContext::loadOrCompile(const std::string& sourceRel, const std::string& entry,
                                                    const Defines& defines) {
    const fs::path includeDir = fs::path(shaderRoot_) / "slangml";
    if (sourceHash_ == 0) {
        // Any module can be imported by any kernel, so every .slang file is part of the key.
        std::vector<fs::path> files;
        for (const auto& e : fs::recursive_directory_iterator(includeDir))
            if (e.is_regular_file() && e.path().extension() == ".slang") files.push_back(e.path());
        std::sort(files.begin(), files.end());
        uint64_t h = 1469598103934665603ull;
        for (const auto& f : files) {
            std::ifstream in(f, std::ios::binary);
            std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
            h = fnv1a(f.filename().string(), h);
            h = fnv1a(text, h);
        }
        sourceHash_ = h;
    }
    uint64_t key = fnv1a(sourceRel, sourceHash_);
    key = fnv1a(entry, key);
    for (const auto& d : defines) key = fnv1a(d.first + "=" + d.second + ";", key);

    const fs::path cacheDir = fs::temp_directory_path() / "neonengine_spv";
    const fs::path cached = cacheDir / (entry + "_" + hex16(key) + ".spv");
    if (fs::exists(cached)) {
        std::ifstream in(cached, std::ios::binary);
        std::vector<char> bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        if (!bytes.empty() && bytes.size() % 4 == 0) {
            std::vector<uint32_t> words(bytes.size() / 4);
            std::memcpy(words.data(), bytes.data(), bytes.size());
            return words;
        }
    }

    auto& compiler = compilerInstance();
    if (!compiler) compiler = std::make_unique<SlangRuntimeCompiler>();
    const fs::path source = fs::path(shaderRoot_) / sourceRel;
    std::vector<uint32_t> words = compiler->compileEntryPoint(source.string(), includeDir.string(), entry, defines);

    std::error_code ec;
    fs::create_directories(cacheDir, ec);
    std::ofstream out(cached, std::ios::binary);
    out.write(reinterpret_cast<const char*>(words.data()), std::streamsize(words.size() * 4));
    return words;
}

const Pipeline& ComputeContext::pipeline(const std::string& sourceRel, const std::string& entry, const Defines& defines) {
    std::string key = sourceRel + "|" + entry;
    for (const auto& d : defines) key += "|" + d.first + "=" + d.second;
    auto it = pipelines_.find(key);
    if (it != pipelines_.end()) return it->second;

    std::vector<uint32_t> spirv = loadOrCompile(sourceRel, entry, defines);
    VkShaderModuleCreateInfo smi{ VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
    smi.codeSize = spirv.size() * 4;
    smi.pCode = spirv.data();
    VkShaderModule module;
    vkCheck(vkCreateShaderModule(device_, &smi, nullptr, &module), "vkCreateShaderModule");
    modules_.push_back(module);

    VkComputePipelineCreateInfo ci{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
    ci.stage = VkPipelineShaderStageCreateInfo{ VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO };
    ci.stage.flags = hasFullSubgroups_ ? VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT : 0;
    ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    ci.stage.module = module;
    ci.stage.pName = entry.c_str();
    ci.layout = layout_;
    Pipeline p;
    p.entry = entry;
    vkCheck(vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &ci, nullptr, &p.pipeline),
            ("vkCreateComputePipelines " + entry).c_str());
    return pipelines_.emplace(key, p).first->second;
}

// ---------------------------------------------------------------------------
// Recorder
// ---------------------------------------------------------------------------
namespace {
void fullBarrier(VkCommandBuffer cmd, VkPipelineStageFlags dstStage, VkAccessFlags dstAccess) {
    VkMemoryBarrier mb{ VK_STRUCTURE_TYPE_MEMORY_BARRIER };
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    mb.dstAccessMask = dstAccess;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, dstStage, 0, 1,
                         &mb, 0, nullptr, 0, nullptr);
}
constexpr VkAccessFlags kComputeAccess = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT |
                                         VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
constexpr VkPipelineStageFlags kComputeStages = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT;
}  // namespace

ComputeContext::Recorder::Recorder(ComputeContext& c) : ctx(c) { cmd = ctx.beginOneShot(); }

ComputeContext::Recorder::~Recorder() {
    if (!submitted && cmd) {
        vkEndCommandBuffer(cmd);
        vkFreeCommandBuffers(ctx.device_, ctx.pool_, 1, &cmd);
    }
}

void ComputeContext::Recorder::barrier() {
    fullBarrier(cmd, kComputeStages, kComputeAccess);
    needBarrier = false;
}

void ComputeContext::Recorder::dispatch(const Pipeline& p, uint64_t paramAddress, uint32_t gx, uint32_t gy, uint32_t gz) {
    if (needBarrier) barrier();
    const bool prof = ctx.profiling_ && profLabels.size() + 2 <= kProfQueries;
    if (prof && profLabels.empty()) {
        vkCmdResetQueryPool(cmd, ctx.profPool_, 0, kProfQueries);
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, ctx.profPool_, 0);
    }
    PushConstants pc{ paramAddress, 0, 0, 0 };
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p.pipeline);
    vkCmdPushConstants(cmd, ctx.layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cmd, gx, gy, gz);
    if (prof) {
        // Dispatches are serialized by barriers, so consecutive timestamps bracket one dispatch.
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, ctx.profPool_, uint32_t(profLabels.size() + 1));
        profLabels.push_back(nextLabel.empty() ? p.entry : nextLabel);
    }
    nextLabel.clear();
    needBarrier = true;
}

void ComputeContext::Recorder::fillZero(const GpuBuffer& b) {
    if (needBarrier) barrier();
    vkCmdFillBuffer(cmd, b.buffer, 0, VK_WHOLE_SIZE, 0);
    needBarrier = true;
}

void ComputeContext::Recorder::timestamp(uint32_t query) {
    if (query == 0) {
        vkCmdResetQueryPool(cmd, ctx.queryPool_, 0, 2);
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, ctx.queryPool_, 0);
        timed = true;
    } else {
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, ctx.queryPool_, 1);
    }
}

double ComputeContext::Recorder::submitAndWait() {
    fullBarrier(cmd, VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_HOST_READ_BIT);
    ctx.endOneShot(cmd);
    submitted = true;
    cmd = VK_NULL_HANDLE;
    if (!profLabels.empty()) {
        std::vector<uint64_t> ts(profLabels.size() + 1);
        if (vkGetQueryPoolResults(ctx.device_, ctx.profPool_, 0, uint32_t(ts.size()), ts.size() * 8, ts.data(), 8,
                                  VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT) == VK_SUCCESS)
            for (size_t i = 0; i < profLabels.size(); ++i) {
                auto& e = ctx.profile_[profLabels[i]];
                e.ms += double(ts[i + 1] - ts[i]) * double(ctx.timestampPeriodNs_) * 1e-6;
                ++e.count;
            }
        profLabels.clear();
    }
    if (!timed) return 0.0;
    uint64_t ts[2] = { 0, 0 };
    if (vkGetQueryPoolResults(ctx.device_, ctx.queryPool_, 0, 2, sizeof(ts), ts, 8,
                              VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT) != VK_SUCCESS)
        return 0.0;
    return double(ts[1] - ts[0]) * double(ctx.timestampPeriodNs_) * 1e-6;
}

void ComputeContext::setProfiling(bool on) {
    if (on && !profPool_) {
        VkQueryPoolCreateInfo qci{ VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO };
        qci.queryType = VK_QUERY_TYPE_TIMESTAMP;
        qci.queryCount = kProfQueries;
        vkCheck(vkCreateQueryPool(device_, &qci, nullptr, &profPool_), "vkCreateQueryPool (profile)");
    }
    profiling_ = on;
}

void ComputeContext::printProfile(std::ostream& os, double per, const char* perName) const {
    std::vector<std::pair<std::string, ProfEntry>> rows(profile_.begin(), profile_.end());
    std::sort(rows.begin(), rows.end(), [](const auto& a, const auto& b) { return a.second.ms > b.second.ms; });
    double total = 0.0;
    for (const auto& r : rows) total += r.second.ms;
    const auto flags = os.flags();
    const auto prec = os.precision();
    os << "GPU profile (ms per " << perName << ", " << std::fixed;
    os.precision(2);
    os << total / per << " ms total):\n";
    for (const auto& [name, e] : rows) {
        os.width(9);
        os << e.ms / per << " ms ";
        os.width(5);
        os.precision(1);
        os << 100.0 * e.ms / std::max(total, 1e-12) << "%  x";
        os.width(4);
        os << std::left << double(e.count) / per << std::right << "  " << name << "\n";
        os.precision(2);
    }
    os.flags(flags);
    os.precision(prec);
}

// ---------------------------------------------------------------------------
// ParamArena
// ---------------------------------------------------------------------------
ParamArena::ParamArena(ComputeContext& c, size_t bytes) : ctx(c) {
    buffer = ctx.createBuffer(bytes, MemKind::HostDevice);
}

ParamArena::~ParamArena() { ctx.destroyBuffer(buffer); }

uint8_t* ParamArena::alloc(size_t bytes, uint64_t& gpuAddress) {
    size_t start = (used + 15) & ~size_t(15);
    if (start + bytes > buffer.size)
        throw std::runtime_error("ParamArena: out of space (" + std::to_string(buffer.size) + " bytes); reset it per batch");
    used = start + bytes;
    gpuAddress = buffer.address + start;
    return static_cast<uint8_t*>(buffer.mapped) + start;
}

uint64_t ParamArena::params(const std::vector<TensorRef>& tensors, const void* payload, size_t payloadBytes,
                            uint32_t numel) {
    uint64_t tableAddr = 0;
    uint8_t* table = alloc(std::max<size_t>(tensors.size(), 1) * sizeof(TensorDesc), tableAddr);
    for (size_t i = 0; i < tensors.size(); ++i) {
        TensorDesc d{};
        d.dataAddress = tensors[i].address;
        d.dtype = tensors[i].dtype;
        d.rank = 1;
        d.flags = (1u << 0) | (1u << 4) | (1u << 6) | (1u << 7);  // contiguous, aligned16, inner stride 1, canonical
        std::memcpy(table + i * sizeof(TensorDesc), &d, sizeof(d));
    }
    uint64_t paramAddr = 0;
    uint8_t* block = alloc(sizeof(KernelHeader) + payloadBytes, paramAddr);
    KernelHeader h{};
    h.numel = numel;
    h.tensorCount = uint32_t(tensors.size());
    h.tensorTableAddress = tableAddr;
    std::memcpy(block, &h, sizeof(h));
    if (payloadBytes) std::memcpy(block + sizeof(h), payload, payloadBytes);
    return paramAddr;
}

}  // namespace lve::ml
