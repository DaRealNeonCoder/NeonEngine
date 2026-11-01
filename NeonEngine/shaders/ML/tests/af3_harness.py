"""Per-kernel test harness for the AlphaFold 3 SlangML entry points.

Runs one compute entry point at a time on the GPU through BDA and push constants,
then compares the outputs against a CPU reference.  It does not build or run a
model.  Shaders are compiled with the same flags as compile_af3.ps1.

Requires: pip install vulkan numpy, slangc, and a Vulkan 1.2 device with
bufferDeviceAddress, scalarBlockLayout, and shaderInt64.

ABI (see slangml/common/tensor.slang, memory.slang, config.slang):
  KernelHeader   48 bytes  dispatchBase[3], numel, flags, tensorTableOffset,
                           tensorCount, reserved0, tensorTableAddress (u64),
                           reserved1, reserved2
  TensorDesc     96 bytes  shape[8] u32, stride[8] i32, baseOffset i64,
                           dataAddress u64, dtype, rank, flags, bufferIndex
  Push constants 24 bytes  headerAddress u64, workspaceAddress u64, flags, reserved
  Params block           = KernelHeader followed by the kernel's own fields.
"""

from __future__ import annotations

import os
import struct
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import vulkan as vk

TESTS_DIR = Path(__file__).resolve().parent
ML_ROOT = TESTS_DIR.parent
SLANGML_DIR = ML_ROOT / "slangml"
SLANGC = Path(r"C:\VulkanSDK\1.4.328.1\Bin\slangc.exe")

# Core Vulkan 1.2 values that the Python binding does not export.
BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT = 0x20000
MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT = 0x2
BUFFER_DEVICE_ADDRESS_INFO = 1000244001

DTYPE_FP32 = 0
DTYPE_INT32 = 5
TENSOR_FLAG_CONTIGUOUS_CANONICAL = (1 << 0) | (1 << 7)

HEADER_FMT = "<3I5IQII"
TENSOR_FMT = "<8I8iqQ4I"
PUSH_FMT = "<QQII"
assert struct.calcsize(HEADER_FMT) == 48
assert struct.calcsize(TENSOR_FMT) == 96
assert struct.calcsize(PUSH_FMT) == 24


@dataclass
class Tensor:
    """A flat tensor passed to a kernel.  Outputs start zeroed and are read back."""

    data: np.ndarray
    output: bool = False

    def nbytes(self) -> int:
        return int(self.data.nbytes)


def compile_entry(source_rel: str, entry: str, out_dir: Path) -> bytes:
    """Compile one entry point with the flags used by compile_af3.ps1."""
    out = out_dir / f"{entry}.spv"
    cmd = [
        str(SLANGC),
        str(ML_ROOT / source_rel),
        "-I", str(SLANGML_DIR),
        "-target", "spirv",
        "-profile", "spirv_1_5",
        "-stage", "compute",
        "-entry", entry,
        "-fvk-use-entrypoint-name",
        "-fvk-use-scalar-layout",
        "-fspv-reflect",
        "-O3",
        "-o", str(out),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return out.read_bytes()


class Gpu:
    """Owns one Vulkan instance, device, queue, and command pool for the harness."""

    def __init__(self) -> None:
        app = vk.VkApplicationInfo(
            pApplicationName="af3_harness",
            applicationVersion=1,
            pEngineName="af3_harness",
            engineVersion=1,
            apiVersion=(1 << 22) | (2 << 12),
        )
        self.instance = vk.vkCreateInstance(vk.VkInstanceCreateInfo(pApplicationInfo=app), None)
        self.physical, self.queue_family = self._pick_device()
        self.device = self._create_device()
        self.queue = vk.vkGetDeviceQueue(self.device, self.queue_family, 0)
        self.pool = vk.vkCreateCommandPool(
            self.device,
            vk.VkCommandPoolCreateInfo(queueFamilyIndex=self.queue_family),
            None,
        )
        self.mem_props = vk.vkGetPhysicalDeviceMemoryProperties(self.physical)

    def _pick_device(self):
        for dev in vk.vkEnumeratePhysicalDevices(self.instance):
            props = vk.vkGetPhysicalDeviceProperties(dev)
            f12 = vk.VkPhysicalDeviceVulkan12Features(
                sType=vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES
            )
            f2 = vk.VkPhysicalDeviceFeatures2(
                sType=vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, pNext=f12
            )
            vk.vkGetPhysicalDeviceFeatures2(dev, f2)
            if not (f12.bufferDeviceAddress and f12.scalarBlockLayout and f2.features.shaderInt64):
                continue
            for index, family in enumerate(vk.vkGetPhysicalDeviceQueueFamilyProperties(dev)):
                if family.queueFlags & vk.VK_QUEUE_COMPUTE_BIT:
                    self.device_name = props.deviceName
                    return dev, index
        raise RuntimeError("No Vulkan device with bufferDeviceAddress + scalarBlockLayout + shaderInt64")

    def _create_device(self):
        f12 = vk.VkPhysicalDeviceVulkan12Features(
            sType=vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
            bufferDeviceAddress=True,
            scalarBlockLayout=True,
        )
        features = vk.VkPhysicalDeviceFeatures(shaderInt64=True)
        queue = vk.VkDeviceQueueCreateInfo(queueFamilyIndex=self.queue_family, queueCount=1, pQueuePriorities=[1.0])
        info = vk.VkDeviceCreateInfo(
            pNext=vk.VkPhysicalDeviceFeatures2(sType=vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, pNext=f12, features=features),
            queueCreateInfoCount=1,
            pQueueCreateInfos=[queue],
        )
        return vk.vkCreateDevice(self.physical, info, None)

    def _memory_type(self, type_bits: int, wanted: int) -> int:
        for index in range(self.mem_props.memoryTypeCount):
            flags = self.mem_props.memoryTypes[index].propertyFlags
            if (type_bits & (1 << index)) and (flags & wanted) == wanted:
                return index
        raise RuntimeError("No suitable memory type")

    def buffer(self, nbytes: int):
        nbytes = max(int(nbytes), 16)
        info = vk.VkBufferCreateInfo(
            size=nbytes,
            usage=BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE,
        )
        buf = vk.vkCreateBuffer(self.device, info, None)
        req = vk.vkGetBufferMemoryRequirements(self.device, buf)
        alloc = vk.VkMemoryAllocateInfo(
            pNext=vk.VkMemoryAllocateFlagsInfo(flags=MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT),
            allocationSize=req.size,
            memoryTypeIndex=self._memory_type(
                req.memoryTypeBits,
                vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            ),
        )
        mem = vk.vkAllocateMemory(self.device, alloc, None)
        vk.vkBindBufferMemory(self.device, buf, mem, 0)
        # The binding's vkGetBufferDeviceAddress wrapper returns the info struct, not the
        # address, so the C function is called directly with an explicit struct pointer.
        info_ptr = vk.ffi.new('VkBufferDeviceAddressInfo *', {'sType': BUFFER_DEVICE_ADDRESS_INFO, 'buffer': buf})
        address = vk.lib.vkGetBufferDeviceAddress(self.device, info_ptr)
        return _Buffer(self, buf, mem, address, nbytes)


class _Buffer:
    def __init__(self, gpu: Gpu, buf, mem, address: int, nbytes: int) -> None:
        self.gpu = gpu
        self.buf = buf
        self.mem = mem
        self.address = int(address)
        self.nbytes = nbytes

    def write(self, data: bytes) -> None:
        ptr = vk.vkMapMemory(self.gpu.device, self.mem, 0, self.nbytes, 0)
        ptr[0:len(data)] = data  # vkMapMemory returns a writable buffer object here
        vk.vkUnmapMemory(self.gpu.device, self.mem)

    def read(self, nbytes: int) -> bytes:
        ptr = vk.vkMapMemory(self.gpu.device, self.mem, 0, self.nbytes, 0)
        data = bytes(ptr[0:nbytes])
        vk.vkUnmapMemory(self.gpu.device, self.mem)
        return data


def _tensor_desc(address: int, numel: int, dtype: int) -> bytes:
    shape = (numel,) + (0,) * 7
    stride = (1,) + (0,) * 7
    return struct.pack(TENSOR_FMT, *shape, *stride, 0, address, dtype, 1, TENSOR_FLAG_CONTIGUOUS_CANONICAL, 0)


def run_entry(
    gpu: Gpu,
    spv: bytes,
    entry: str,
    payload: bytes,
    tensors: list[Tensor],
    groups: int,
    numel: int = 0,
    dispatch_base: tuple[int, int, int] = (0, 0, 0),
) -> list[np.ndarray | None]:
    """Dispatch one entry point and return the output tensors (None for inputs).

    dispatch_base is written to KernelHeader.dispatchBase.  A split dispatch sets it
    to the first global index of its slice (contract section 13).
    """
    gpu_bufs = []
    table = b""
    for t in tensors:
        buf = gpu.buffer(t.nbytes())
        payload_data = t.data.tobytes() if not t.output else bytes(t.nbytes())
        buf.write(payload_data)
        gpu_bufs.append(buf)
        dtype = DTYPE_INT32 if t.data.dtype == np.uint32 or t.data.dtype == np.int32 else DTYPE_FP32
        table += _tensor_desc(buf.address, t.data.size, dtype)

    table_buf = gpu.buffer(len(table))
    table_buf.write(table)

    header = struct.pack(HEADER_FMT, *dispatch_base, numel, 0, 0, len(tensors), 0, table_buf.address, 0, 0)
    params = header + payload
    params_buf = gpu.buffer(len(params))
    params_buf.write(params)
    push = struct.pack(PUSH_FMT, params_buf.address, 0, 0, 0)

    module_info = vk.VkShaderModuleCreateInfo(codeSize=len(spv), pCode=spv)
    module = vk.vkCreateShaderModule(gpu.device, module_info, None)
    push_range = vk.VkPushConstantRange(stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT, offset=0, size=len(push))
    layout = vk.vkCreatePipelineLayout(
        gpu.device,
        vk.VkPipelineLayoutCreateInfo(pushConstantRangeCount=1, pPushConstantRanges=[push_range]),
        None,
    )
    stage = vk.VkPipelineShaderStageCreateInfo(
        stage=vk.VK_SHADER_STAGE_COMPUTE_BIT, module=module, pName=entry
    )
    pipeline = vk.vkCreateComputePipelines(
        gpu.device, None, 1, [vk.VkComputePipelineCreateInfo(stage=stage, layout=layout)], None
    )[0]

    cmd = vk.vkAllocateCommandBuffers(
        gpu.device,
        vk.VkCommandBufferAllocateInfo(commandPool=gpu.pool, level=vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY, commandBufferCount=1),
    )[0]
    vk.vkBeginCommandBuffer(cmd, vk.VkCommandBufferBeginInfo(flags=vk.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT))
    vk.vkCmdBindPipeline(cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE, pipeline)
    vk.vkCmdPushConstants(cmd, layout, vk.VK_SHADER_STAGE_COMPUTE_BIT, 0, len(push), vk.ffi.new('char[]', push))
    vk.vkCmdDispatch(cmd, groups, 1, 1)
    vk.vkEndCommandBuffer(cmd)

    fence = vk.vkCreateFence(gpu.device, vk.VkFenceCreateInfo(), None)
    vk.vkQueueSubmit(gpu.queue, 1, [vk.VkSubmitInfo(commandBufferCount=1, pCommandBuffers=[cmd])], fence)
    vk.vkWaitForFences(gpu.device, 1, [fence], True, 10_000_000_000)

    results: list[np.ndarray | None] = []
    for t, buf in zip(tensors, gpu_bufs):
        if t.output:
            raw = buf.read(t.nbytes())
            results.append(np.frombuffer(raw, dtype=t.data.dtype).reshape(t.data.shape).copy())
        else:
            results.append(None)

    vk.vkDestroyFence(gpu.device, fence, None)
    vk.vkDestroyPipeline(gpu.device, pipeline, None)
    vk.vkDestroyPipelineLayout(gpu.device, layout, None)
    vk.vkDestroyShaderModule(gpu.device, module, None)
    return results


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b
