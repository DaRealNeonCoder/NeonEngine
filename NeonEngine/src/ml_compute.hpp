#pragma once

// Headless Vulkan compute context for the SlangML LLM kernels.
//
// Unlike LveDevice it needs no window or surface (so it runs on cloud GPUs) and
// it enables what the LLM kernels require: Vulkan 1.3, buffer device address,
// scalar block layout, 8/16-bit storage, fp16/int8 arithmetic, integer dot
// product, subgroup size control and (when present) cooperative matrix. This
// mirrors shaders/ML/tests/llm_harness.py, which is the reference host.
//
// Kernel ABI (shaders/ML/docs/CONTRACT.md): every dispatch gets a parameter
// block (KernelHeader + hot scalars) and a TensorDesc table, both addressed
// through BDA from a 24-byte push-constant block. ParamArena holds those
// blocks; Recorder records dispatches with compute->compute barriers.

#include <vulkan/vulkan.h>

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace lve::ml {

// ---- kernel ABI structs (scalar layout; must match common/tensor.slang) ----
struct KernelHeader {
    uint32_t dispatchBase[3];
    uint32_t numel;
    uint32_t flags;
    uint32_t tensorTableOffset;
    uint32_t tensorCount;
    uint32_t reserved0;
    uint64_t tensorTableAddress;
    uint32_t reserved1;
    uint32_t reserved2;
};
static_assert(sizeof(KernelHeader) == 48, "KernelHeader layout");

struct TensorDesc {
    uint32_t shape[8];
    int32_t stride[8];
    int64_t baseOffset;
    uint64_t dataAddress;
    uint32_t dtype;
    uint32_t rank;
    uint32_t flags;
    uint32_t bufferIndex;
};
static_assert(sizeof(TensorDesc) == 96, "TensorDesc layout");

struct PushConstants {
    uint64_t headerAddress;
    uint64_t workspaceAddress;
    uint32_t flags;
    uint32_t reserved;
};
static_assert(sizeof(PushConstants) == 24, "PushConstants layout");

// Where a buffer lives.
enum class MemKind {
    Device,      // VRAM, not mappable (weights, activations)
    Host,        // system RAM, mapped, GPU-readable over PCIe (offloaded experts, embedding table)
    HostDevice,  // VRAM that is host-visible if the device has it (resizable BAR), else Host
};

struct GpuBuffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    uint64_t address = 0;     // buffer device address
    size_t size = 0;
    void* mapped = nullptr;   // non-null for host-visible kinds
    bool deviceLocal = false;
    explicit operator bool() const { return buffer != VK_NULL_HANDLE; }
};

struct Pipeline {
    VkPipeline pipeline = VK_NULL_HANDLE;
    std::string entry;
};

using Defines = std::vector<std::pair<std::string, std::string>>;

class ComputeContext {
public:
    // deviceFilter: case-insensitive substring of the device name ("" = prefer a discrete GPU).
    // shaderRoot: the shaders/ML directory (contains slangml/).
    ComputeContext(const std::string& shaderRoot, const std::string& deviceFilter = "");
    ~ComputeContext();

    ComputeContext(const ComputeContext&) = delete;
    ComputeContext& operator=(const ComputeContext&) = delete;

    // ---- device info
    const std::string& deviceName() const { return deviceName_; }
    uint32_t subgroupSize() const { return subgroupSize_; }
    bool hasCoopMat() const { return hasCoopMat_; }
    uint64_t deviceLocalBytes() const { return deviceLocalBytes_; }
    uint64_t maxAllocationBytes() const { return maxAllocation_; }

    // ---- buffers (sizes are rounded up to 16 bytes)
    GpuBuffer createBuffer(size_t bytes, MemKind kind);
    void destroyBuffer(GpuBuffer& b);
    // Copies host data into a buffer (direct memcpy for mapped buffers, staged otherwise).
    void upload(const GpuBuffer& dst, size_t dstOffset, const void* src, size_t bytes);
    void download(const GpuBuffer& src, size_t srcOffset, void* dst, size_t bytes);
    void fillZero(const GpuBuffer& b);

    // ---- pipelines: compiled from Slang at runtime, SPIR-V cached on disk
    // sourceRel is relative to shaderRoot, e.g. "slangml/quant/gemv_quant.slang".
    const Pipeline& pipeline(const std::string& sourceRel, const std::string& entry, const Defines& defines = {});

    // ---- profiling: when on, every Recorder times each dispatch (GPU timestamps
    // around it) and sums the time per label (the entry name, or Recorder::label).
    void setProfiling(bool on);
    bool profiling() const { return profiling_; }
    void resetProfile() { profile_.clear(); }
    // Prints the table sorted by time; `per` divides the totals (e.g. steps).
    void printProfile(std::ostream& os, double per = 1.0, const char* perName = "step") const;

    // ---- command recording
    class Recorder {
    public:
        explicit Recorder(ComputeContext& ctx);
        ~Recorder();
        // Label for the next dispatch in the profile (default: the pipeline's entry name).
        void label(std::string text) { nextLabel = std::move(text); }
        void dispatch(const Pipeline& p, uint64_t paramAddress, uint32_t gx, uint32_t gy = 1, uint32_t gz = 1);
        void barrier();  // compute writes -> compute reads/writes
        void fillZero(const GpuBuffer& b);
        void timestamp(uint32_t query);  // 0 = start, 1 = end
        // Submits, waits, returns GPU milliseconds between timestamps 0 and 1 (0 if not written).
        double submitAndWait();

    private:
        ComputeContext& ctx;
        VkCommandBuffer cmd = VK_NULL_HANDLE;
        bool needBarrier = false;
        bool timed = false;
        bool submitted = false;
        std::string nextLabel;
        std::vector<std::string> profLabels;  // label of the dispatch ending at query i + 1
    };

    VkDevice device() const { return device_; }

private:
    void pickDevice(const std::string& filter);
    void createDevice();
    uint32_t memoryType(uint32_t typeBits, VkMemoryPropertyFlags want, VkMemoryPropertyFlags avoid) const;
    GpuBuffer& staging(size_t bytes);
    VkCommandBuffer beginOneShot();
    void endOneShot(VkCommandBuffer cmd);
    std::vector<uint32_t> loadOrCompile(const std::string& sourceRel, const std::string& entry, const Defines& defines);

    std::string shaderRoot_;
    VkInstance instance_ = VK_NULL_HANDLE;
    VkPhysicalDevice physical_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue queue_ = VK_NULL_HANDLE;
    uint32_t queueFamily_ = 0;
    VkCommandPool pool_ = VK_NULL_HANDLE;
    VkPipelineLayout layout_ = VK_NULL_HANDLE;
    VkQueryPool queryPool_ = VK_NULL_HANDLE;
    VkQueryPool profPool_ = VK_NULL_HANDLE;
    static constexpr uint32_t kProfQueries = 8192;
    bool profiling_ = false;
    struct ProfEntry { double ms = 0.0; uint64_t count = 0; };
    std::map<std::string, ProfEntry> profile_;
    VkFence fence_ = VK_NULL_HANDLE;
    VkPhysicalDeviceMemoryProperties memProps_{};

    std::string deviceName_;
    uint32_t subgroupSize_ = 32;
    float timestampPeriodNs_ = 1.0f;
    bool hasCoopMat_ = false;
    bool hasFullSubgroups_ = false;
    uint64_t deviceLocalBytes_ = 0;
    uint64_t maxAllocation_ = 0;

    GpuBuffer staging_;
    std::map<std::string, Pipeline> pipelines_;
    std::vector<VkShaderModule> modules_;
    uint64_t sourceHash_ = 0;  // hash of every .slang file (SPIR-V cache key)
};

// Bump allocator over one host-visible buffer for per-dispatch parameter
// blocks and tensor tables. Reset it once the GPU is done with the previous batch.
class ParamArena {
public:
    ParamArena(ComputeContext& ctx, size_t bytes);
    ~ParamArena();

    struct TensorRef {
        uint64_t address = 0;  // byte address (0 = null descriptor)
        uint32_t dtype = 0;
    };

    // Writes a TensorDesc table and a parameter block (header + payload) and
    // returns the parameter block's GPU address (for PushConstants.headerAddress).
    uint64_t params(const std::vector<TensorRef>& tensors, const void* payload, size_t payloadBytes,
                    uint32_t numel = 0);
    template <class T>
    uint64_t params(const std::vector<TensorRef>& tensors, const T& payload, uint32_t numel = 0) {
        return params(tensors, &payload, sizeof(T), numel);
    }

    void reset() { used = 0; }
    size_t bytesUsed() const { return used; }

private:
    uint8_t* alloc(size_t bytes, uint64_t& gpuAddress);

    ComputeContext& ctx;
    GpuBuffer buffer;
    size_t used = 0;
};

}  // namespace lve::ml
