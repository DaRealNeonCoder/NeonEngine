"""GPU harness for the SlangML LLM kernels (tests and benchmarks).

Differences from af3_harness.py:
  * buffers are device-local (uploads/downloads go through a staging buffer),
    so timings measure VRAM bandwidth rather than PCIe;
  * fp16/int8/8-bit storage, integer dot product, subgroup size control,
    Vulkan memory model and cooperative matrix are enabled when supported;
  * several dispatches can be recorded in one command buffer with
    compute->compute barriers, and timed with GPU timestamps;
  * pipelines and SPIR-V are cached per (source, entry, defines).

Select a device with SLANGML_DEVICE=<substring of the device name>; by default
a discrete GPU is preferred.

ABI (see slangml/common/tensor.slang):
  KernelHeader   48 bytes, TensorDesc 96 bytes, push constants 24 bytes.
"""

from __future__ import annotations

import hashlib
import os
import struct
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import vulkan as vk

TESTS_DIR = Path(__file__).resolve().parent
ML_ROOT = TESTS_DIR.parent
SLANGML_DIR = ML_ROOT / "slangml"
SLANGC = Path(os.environ.get("SLANGC", r"C:\VulkanSDK\1.4.328.1\Bin\slangc.exe"))
SPV_CACHE = Path(tempfile.gettempdir()) / "slangml_llm_spv"

BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT = 0x20000
MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT = 0x2
BUFFER_DEVICE_ADDRESS_INFO = 1000244001
PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT = 0x2

DTYPE_FP32, DTYPE_FP16, DTYPE_BF16, DTYPE_INT8, DTYPE_UINT8, DTYPE_INT32 = 0, 1, 2, 3, 4, 5
FLAGS_FAST = (1 << 0) | (1 << 4) | (1 << 6) | (1 << 7)  # contiguous, aligned16, inner stride 1, canonical

HEADER_FMT = "<3I5IQII"
TENSOR_FMT = "<8I8iqQ4I"
PUSH_FMT = "<QQII"
assert struct.calcsize(HEADER_FMT) == 48
assert struct.calcsize(TENSOR_FMT) == 96
assert struct.calcsize(PUSH_FMT) == 24


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def compile_entry(source_rel: str, entry: str, defines: dict | None = None) -> bytes:
    """Compile one entry point (cached on disk by source text + flags)."""
    defines = defines or {}
    src = ML_ROOT / source_rel
    key = hashlib.sha1()
    key.update(entry.encode())
    key.update(repr(sorted(defines.items())).encode())
    for path in sorted(SLANGML_DIR.rglob("*.slang")):  # common modules affect every entry
        key.update(path.read_bytes())
    SPV_CACHE.mkdir(parents=True, exist_ok=True)
    out = SPV_CACHE / f"{entry}_{key.hexdigest()[:16]}.spv"
    if out.exists():
        return out.read_bytes()
    cmd = [
        str(SLANGC), str(src), "-I", str(SLANGML_DIR),
        "-target", "spirv", "-profile", "spirv_1_5", "-stage", "compute",
        "-entry", entry, "-fvk-use-entrypoint-name", "-fvk-use-scalar-layout",
        "-O3", "-warnings-disable", "41012", "-o", str(out),
    ]
    for name, value in defines.items():
        cmd += ["-D", f"{name}={value}"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"slangc failed for {entry}:\n{proc.stdout}\n{proc.stderr}")
    spirv_val = SLANGC.parent / "spirv-val.exe"
    if spirv_val.exists():
        val = subprocess.run([str(spirv_val), "--target-env", "vulkan1.3", str(out)], capture_output=True, text=True)
        if val.returncode != 0:
            out.unlink()
            raise RuntimeError(f"spirv-val failed for {entry}:\n{val.stdout}\n{val.stderr}")
    return out.read_bytes()


@dataclass
class Buffer:
    gpu: "Gpu"
    buf: object
    mem: object
    address: int
    nbytes: int
    host_visible: bool = False


@dataclass
class Pipeline:
    pipeline: object
    layout: object
    module: object
    entry: str


@dataclass
class Dispatch:
    pipeline: Pipeline
    push: bytes
    groups: tuple[int, int, int]
    keep: list = field(default_factory=list)  # buffers that must outlive the submit


class Gpu:
    def __init__(self) -> None:
        app = vk.VkApplicationInfo(
            pApplicationName="slangml_llm", applicationVersion=1,
            pEngineName="slangml_llm", engineVersion=1,
            apiVersion=(1 << 22) | (3 << 12),
        )
        self.instance = vk.vkCreateInstance(vk.VkInstanceCreateInfo(pApplicationInfo=app), None)
        self.physical, self.queue_family = self._pick_device()
        self.device = self._create_device()
        self.queue = vk.vkGetDeviceQueue(self.device, self.queue_family, 0)
        self.pool = vk.vkCreateCommandPool(
            self.device,
            vk.VkCommandPoolCreateInfo(
                flags=vk.VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
                queueFamilyIndex=self.queue_family),
            None,
        )
        self.mem_props = vk.vkGetPhysicalDeviceMemoryProperties(self.physical)
        self._pipelines: dict = {}
        self._staging: Buffer | None = None
        self.query_pool = vk.vkCreateQueryPool(
            self.device,
            vk.VkQueryPoolCreateInfo(queryType=vk.VK_QUERY_TYPE_TIMESTAMP, queryCount=2),
            None,
        )

    # ------------------------------------------------------------------ setup
    def _pick_device(self):
        wanted = os.environ.get("SLANGML_DEVICE", "").lower()
        candidates = []
        for dev in vk.vkEnumeratePhysicalDevices(self.instance):
            props = vk.vkGetPhysicalDeviceProperties(dev)
            name = props.deviceName
            if wanted and wanted not in name.lower():
                continue
            family = None
            for index, fam in enumerate(vk.vkGetPhysicalDeviceQueueFamilyProperties(dev)):
                if fam.queueFlags & vk.VK_QUEUE_COMPUTE_BIT:
                    family = index
                    break
            if family is None:
                continue
            discrete = props.deviceType == vk.VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU
            candidates.append((0 if discrete else 1, dev, family, props))
        if not candidates:
            raise RuntimeError("No Vulkan compute device found")
        candidates.sort(key=lambda c: c[0])
        _, dev, family, props = candidates[0]
        self.device_name = props.deviceName
        sub = vk.VkPhysicalDeviceSubgroupProperties(sType=vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES)
        props2 = vk.VkPhysicalDeviceProperties2(sType=vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, pNext=sub)
        vk.vkGetPhysicalDeviceProperties2(dev, props2)
        self.subgroup_size = int(sub.subgroupSize)
        self.timestamp_period_ns = props.limits.timestampPeriod
        self.max_shared = props.limits.maxComputeSharedMemorySize
        return dev, family

    def _create_device(self):
        exts = {e.extensionName for e in vk.vkEnumerateDeviceExtensionProperties(self.physical, None)}
        q11 = vk.VkPhysicalDeviceVulkan11Features(sType=vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES)
        q12 = vk.VkPhysicalDeviceVulkan12Features(sType=vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, pNext=q11)
        q13 = vk.VkPhysicalDeviceVulkan13Features(sType=vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES, pNext=q12)
        has_cm = "VK_KHR_cooperative_matrix" in exts
        chain = q13
        qcm = None
        if has_cm:
            qcm = vk.VkPhysicalDeviceCooperativeMatrixFeaturesKHR(
                sType=vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR, pNext=q13)
            chain = qcm
        q2 = vk.VkPhysicalDeviceFeatures2(sType=vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, pNext=chain)
        vk.vkGetPhysicalDeviceFeatures2(self.physical, q2)
        if not (q12.bufferDeviceAddress and q12.scalarBlockLayout and q2.features.shaderInt64):
            raise RuntimeError("Device lacks bufferDeviceAddress / scalarBlockLayout / shaderInt64")

        self.has_fp16 = bool(q12.shaderFloat16 and q11.storageBuffer16BitAccess)
        self.has_int8 = bool(q12.shaderInt8 and q12.storageBuffer8BitAccess)
        self.has_coopmat = bool(has_cm and qcm.cooperativeMatrix)
        self.has_full_subgroups = bool(q13.computeFullSubgroups)

        f11 = vk.VkPhysicalDeviceVulkan11Features(
            sType=vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES,
            storageBuffer16BitAccess=q11.storageBuffer16BitAccess,
            uniformAndStorageBuffer16BitAccess=q11.uniformAndStorageBuffer16BitAccess,
        )
        f12 = vk.VkPhysicalDeviceVulkan12Features(
            sType=vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, pNext=f11,
            bufferDeviceAddress=True, scalarBlockLayout=True,
            shaderFloat16=q12.shaderFloat16, shaderInt8=q12.shaderInt8,
            storageBuffer8BitAccess=q12.storageBuffer8BitAccess,
            uniformAndStorageBuffer8BitAccess=q12.uniformAndStorageBuffer8BitAccess,
            vulkanMemoryModel=q12.vulkanMemoryModel,
            vulkanMemoryModelDeviceScope=q12.vulkanMemoryModelDeviceScope,
            shaderSubgroupExtendedTypes=q12.shaderSubgroupExtendedTypes,
            shaderBufferInt64Atomics=q12.shaderBufferInt64Atomics,
        )
        f13 = vk.VkPhysicalDeviceVulkan13Features(
            sType=vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES, pNext=f12,
            subgroupSizeControl=q13.subgroupSizeControl,
            computeFullSubgroups=q13.computeFullSubgroups,
            shaderIntegerDotProduct=q13.shaderIntegerDotProduct,
            maintenance4=q13.maintenance4,
        )
        chain = f13
        enabled_exts = []
        if self.has_coopmat:
            chain = vk.VkPhysicalDeviceCooperativeMatrixFeaturesKHR(
                sType=vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR,
                pNext=f13, cooperativeMatrix=True)
            enabled_exts.append("VK_KHR_cooperative_matrix")
        core = vk.VkPhysicalDeviceFeatures(shaderInt64=True, shaderInt16=q2.features.shaderInt16)
        features = vk.VkPhysicalDeviceFeatures2(
            sType=vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, pNext=chain, features=core)
        queue = vk.VkDeviceQueueCreateInfo(queueFamilyIndex=self.queue_family, queueCount=1, pQueuePriorities=[1.0])
        info = vk.VkDeviceCreateInfo(
            pNext=features, queueCreateInfoCount=1, pQueueCreateInfos=[queue],
            enabledExtensionCount=len(enabled_exts), ppEnabledExtensionNames=enabled_exts,
        )
        return vk.vkCreateDevice(self.physical, info, None)

    def _memory_type(self, type_bits: int, wanted: int) -> int:
        for index in range(self.mem_props.memoryTypeCount):
            flags = self.mem_props.memoryTypes[index].propertyFlags
            if (type_bits & (1 << index)) and (flags & wanted) == wanted:
                return index
        raise RuntimeError("No suitable memory type")

    # ---------------------------------------------------------------- buffers
    def buffer(self, nbytes: int, host_visible: bool = False) -> Buffer:
        nbytes = max(int(nbytes), 16)
        nbytes = (nbytes + 15) & ~15
        usage = (BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                 | vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT | vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT)
        buf = vk.vkCreateBuffer(self.device, vk.VkBufferCreateInfo(
            size=nbytes, usage=usage, sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE), None)
        req = vk.vkGetBufferMemoryRequirements(self.device, buf)
        wanted = (vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
                  if host_visible else vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
        mem = vk.vkAllocateMemory(self.device, vk.VkMemoryAllocateInfo(
            pNext=vk.VkMemoryAllocateFlagsInfo(flags=MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT),
            allocationSize=req.size,
            memoryTypeIndex=self._memory_type(req.memoryTypeBits, wanted)), None)
        vk.vkBindBufferMemory(self.device, buf, mem, 0)
        info_ptr = vk.ffi.new("VkBufferDeviceAddressInfo *", {"sType": BUFFER_DEVICE_ADDRESS_INFO, "buffer": buf})
        address = vk.lib.vkGetBufferDeviceAddress(self.device, info_ptr)
        return Buffer(self, buf, mem, int(address), nbytes, host_visible)

    def free(self, b: Buffer) -> None:
        vk.vkDestroyBuffer(self.device, b.buf, None)
        vk.vkFreeMemory(self.device, b.mem, None)

    def _staging_buffer(self, nbytes: int) -> Buffer:
        if self._staging is None or self._staging.nbytes < nbytes:
            if self._staging is not None:
                self.free(self._staging)
            self._staging = self.buffer(max(nbytes, 1 << 20), host_visible=True)
        return self._staging

    def _one_shot(self, record) -> None:
        cmd = vk.vkAllocateCommandBuffers(self.device, vk.VkCommandBufferAllocateInfo(
            commandPool=self.pool, level=vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY, commandBufferCount=1))[0]
        vk.vkBeginCommandBuffer(cmd, vk.VkCommandBufferBeginInfo(flags=vk.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT))
        record(cmd)
        vk.vkEndCommandBuffer(cmd)
        fence = vk.vkCreateFence(self.device, vk.VkFenceCreateInfo(), None)
        vk.vkQueueSubmit(self.queue, 1, [vk.VkSubmitInfo(commandBufferCount=1, pCommandBuffers=[cmd])], fence)
        vk.vkWaitForFences(self.device, 1, [fence], True, 60_000_000_000)
        vk.vkDestroyFence(self.device, fence, None)
        vk.vkFreeCommandBuffers(self.device, self.pool, 1, [cmd])

    def upload(self, dst: Buffer, data) -> None:
        raw = data.tobytes() if isinstance(data, np.ndarray) else bytes(data)
        if not raw:
            return
        stage = self._staging_buffer(len(raw))
        ptr = vk.vkMapMemory(self.device, stage.mem, 0, len(raw), 0)
        ptr[0:len(raw)] = raw
        vk.vkUnmapMemory(self.device, stage.mem)
        self._one_shot(lambda cmd: vk.vkCmdCopyBuffer(cmd, stage.buf, dst.buf, 1, [vk.VkBufferCopy(0, 0, len(raw))]))

    def download(self, src: Buffer, nbytes: int) -> bytes:
        stage = self._staging_buffer(nbytes)
        self._one_shot(lambda cmd: vk.vkCmdCopyBuffer(cmd, src.buf, stage.buf, 1, [vk.VkBufferCopy(0, 0, nbytes)]))
        ptr = vk.vkMapMemory(self.device, stage.mem, 0, nbytes, 0)
        data = bytes(ptr[0:nbytes])
        vk.vkUnmapMemory(self.device, stage.mem)
        return data

    def array(self, data: np.ndarray) -> Buffer:
        b = self.buffer(data.nbytes)
        self.upload(b, np.ascontiguousarray(data))
        return b

    def zeros(self, nbytes: int) -> Buffer:
        b = self.buffer(nbytes)
        self._one_shot(lambda cmd: vk.vkCmdFillBuffer(cmd, b.buf, 0, b.nbytes, 0))
        return b

    def read(self, b: Buffer, dtype, shape) -> np.ndarray:
        count = int(np.prod(shape))
        nbytes = count * np.dtype(dtype).itemsize
        return np.frombuffer(self.download(b, nbytes), dtype=dtype).reshape(shape).copy()

    # -------------------------------------------------------------- pipelines
    def pipeline(self, source_rel: str, entry: str, defines: dict | None = None) -> Pipeline:
        key = (source_rel, entry, tuple(sorted((defines or {}).items())))
        if key in self._pipelines:
            return self._pipelines[key]
        spv = compile_entry(source_rel, entry, defines)
        module = vk.vkCreateShaderModule(self.device, vk.VkShaderModuleCreateInfo(codeSize=len(spv), pCode=spv), None)
        push_range = vk.VkPushConstantRange(stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT, offset=0, size=24)
        layout = vk.vkCreatePipelineLayout(self.device, vk.VkPipelineLayoutCreateInfo(
            pushConstantRangeCount=1, pPushConstantRanges=[push_range]), None)
        flags = PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT if self.has_full_subgroups else 0
        stage = vk.VkPipelineShaderStageCreateInfo(
            flags=flags, stage=vk.VK_SHADER_STAGE_COMPUTE_BIT, module=module, pName=entry)
        pipe = vk.vkCreateComputePipelines(
            self.device, None, 1, [vk.VkComputePipelineCreateInfo(stage=stage, layout=layout)], None)[0]
        result = Pipeline(pipe, layout, module, entry)
        self._pipelines[key] = result
        return result

    # --------------------------------------------------------------- launches
    def launch(
        self,
        pipeline: Pipeline,
        tensors: list,
        payload: bytes,
        groups,
        dispatch_base=(0, 0, 0),
        numel: int = 0,
        workspace: Buffer | None = None,
    ) -> Dispatch:
        """Build the tensor table + parameter block for one dispatch.

        tensors: list of Buffer or None (None -> null descriptor). Each entry may
        also be (Buffer, dtype_tag, element_offset).
        """
        table = b""
        for t in tensors:
            if t is None:
                table += struct.pack(TENSOR_FMT, *([0] * 8), *([0] * 8), 0, 0, 0, 0, 0, 0)
                continue
            dtype, offset = DTYPE_FP32, 0
            if isinstance(t, tuple):
                t, dtype, offset = (list(t) + [DTYPE_FP32, 0])[:3]
            shape = (t.nbytes,) + (0,) * 7
            stride = (1,) + (0,) * 7
            table += struct.pack(TENSOR_FMT, *shape, *stride, offset, t.address, dtype, 1, FLAGS_FAST, 0)
        table_buf = self.buffer(max(len(table), 96))
        self.upload(table_buf, table)
        if isinstance(groups, int):
            groups = (groups, 1, 1)
        header = struct.pack(HEADER_FMT, *dispatch_base, numel, 0, 0, len(tensors), 0, table_buf.address, 0, 0)
        params = header + payload
        params_buf = self.buffer(len(params))
        self.upload(params_buf, params)
        push = struct.pack(PUSH_FMT, params_buf.address, workspace.address if workspace else 0, 0, 0)
        return Dispatch(pipeline, push, tuple(groups), keep=[table_buf, params_buf])

    def _record(self, cmd, dispatches: list[Dispatch]) -> None:
        barrier = vk.VkMemoryBarrier(
            srcAccessMask=vk.VK_ACCESS_SHADER_WRITE_BIT,
            dstAccessMask=vk.VK_ACCESS_SHADER_READ_BIT | vk.VK_ACCESS_SHADER_WRITE_BIT)
        for i, d in enumerate(dispatches):
            if i > 0:
                vk.vkCmdPipelineBarrier(cmd, vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                        vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, [barrier], 0, None, 0, None)
            vk.vkCmdBindPipeline(cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE, d.pipeline.pipeline)
            vk.vkCmdPushConstants(cmd, d.pipeline.layout, vk.VK_SHADER_STAGE_COMPUTE_BIT, 0, 24,
                                  vk.ffi.new("char[]", d.push))
            vk.vkCmdDispatch(cmd, *d.groups)

    def run(self, dispatches, repeat: int = 1, timed: bool = False) -> float:
        """Record dispatches (repeated `repeat` times) in one submit.

        Returns milliseconds per repetition when timed, else 0.
        """
        if isinstance(dispatches, Dispatch):
            dispatches = [dispatches]
        barrier = vk.VkMemoryBarrier(
            srcAccessMask=vk.VK_ACCESS_SHADER_WRITE_BIT,
            dstAccessMask=vk.VK_ACCESS_SHADER_READ_BIT | vk.VK_ACCESS_SHADER_WRITE_BIT)

        def record(cmd):
            if timed:
                vk.vkCmdResetQueryPool(cmd, self.query_pool, 0, 2)
                vk.vkCmdWriteTimestamp(cmd, vk.VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, self.query_pool, 0)
            for r in range(repeat):
                if r > 0:
                    vk.vkCmdPipelineBarrier(cmd, vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                            vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, [barrier], 0, None, 0, None)
                self._record(cmd, dispatches)
            if timed:
                vk.vkCmdWriteTimestamp(cmd, vk.VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, self.query_pool, 1)

        self._one_shot(record)
        if not timed:
            return 0.0
        data = vk.ffi.new("uint64_t[2]")
        vk.lib.vkGetQueryPoolResults(self.device, self.query_pool, 0, 2, 16, data, 8,
                                     vk.VK_QUERY_RESULT_64_BIT | vk.VK_QUERY_RESULT_WAIT_BIT)
        return (int(data[1]) - int(data[0])) * self.timestamp_period_ns / 1e6 / repeat

    def bench(self, dispatches, warmup_ms: float = 150.0, repeat: int = 20, trials: int = 5) -> float:
        """Best-of-`trials` milliseconds per repetition.

        Laptop GPUs drop to idle clocks between submits, so each benchmark first
        keeps the GPU busy for ~warmup_ms before timing.
        """
        one = self.run(dispatches, repeat=repeat, timed=True)
        spins = max(1, int(warmup_ms / max(one * repeat, 1e-3)))
        for _ in range(min(spins, 50)):
            self.run(dispatches, repeat=repeat)
        return min(self.run(dispatches, repeat=repeat, timed=True) for _ in range(trials))

    def copy_bandwidth_gbs(self, nbytes: int = 256 << 20, repeat: int = 10) -> float:
        """Device-local buffer copy bandwidth (read + write bytes per second)."""
        src, dst = self.buffer(nbytes), self.buffer(nbytes)

        def record(cmd):
            vk.vkCmdResetQueryPool(cmd, self.query_pool, 0, 2)
            vk.vkCmdWriteTimestamp(cmd, vk.VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, self.query_pool, 0)
            for _ in range(repeat):
                vk.vkCmdCopyBuffer(cmd, src.buf, dst.buf, 1, [vk.VkBufferCopy(0, 0, nbytes)])
            vk.vkCmdWriteTimestamp(cmd, vk.VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, self.query_pool, 1)

        self._one_shot(record)
        data = vk.ffi.new("uint64_t[2]")
        vk.lib.vkGetQueryPoolResults(self.device, self.query_pool, 0, 2, 16, data, 8,
                                     vk.VK_QUERY_RESULT_64_BIT | vk.VK_QUERY_RESULT_WAIT_BIT)
        ms = (int(data[1]) - int(data[0])) * self.timestamp_period_ns / 1e6 / repeat
        self.free(src)
        self.free(dst)
        return 2 * nbytes / (ms * 1e-3) / 1e9
