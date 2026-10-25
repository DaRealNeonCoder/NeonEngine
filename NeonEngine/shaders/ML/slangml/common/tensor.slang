implementing slangml_common;

#ifndef SLANGML_USE_BDA
#define SLANGML_USE_BDA 0
#endif

#ifndef SLANGML_INDEX64
#define SLANGML_INDEX64 0
#endif

// DType tags are stable ABI values.  Storage, compute, and accumulation types
// are selected independently by each kernel's generic parameters.
public static const uint SLANGML_DTYPE_FP32 = 0;
public static const uint SLANGML_DTYPE_FP16 = 1;
public static const uint SLANGML_DTYPE_BF16 = 2;
public static const uint SLANGML_DTYPE_INT8 = 3;
public static const uint SLANGML_DTYPE_UINT8 = 4;
public static const uint SLANGML_DTYPE_INT32 = 5;
public static const uint SLANGML_DTYPE_INT64 = 6;
public static const uint SLANGML_DTYPE_BOOL = 7;
public static const uint SLANGML_DTYPE_FP8_E4M3 = 8;
public static const uint SLANGML_DTYPE_FP8_E5M2 = 9;
public static const uint SLANGML_DTYPE_PACKED_INT4 = 10;

// Scalar-layout ABI (96 bytes, hot-path friendly):
//   shape[8]        0..31   uint elements
//   stride[8]      32..63   int elements
//   baseOffset     64..71   int64 elements; only 64-bit base arithmetic
//   dataAddress    72..79   uint64 BDA byte address
//   dtype          80..83   uint tag
//   rank           84..87   uint, in [0, 8]
//   flags          88..91   uint fast-path flags
//   bufferIndex    92..95   uint, bound-buffer fallback only
//
// Descriptors are stored in a BDA tensor table.  The bound-buffer index exists
// only for the small-tensor fallback and must not be read in BDA hot loops.
// Compile Vulkan shaders with -fvk-use-scalar-layout and keep this exact field
// order in the host-side descriptor upload.
public struct TensorDesc
{
    public uint shape[SLANGML_MAX_RANK];
    public int stride[SLANGML_MAX_RANK];
    public int64_t baseOffset;
    public uint64_t dataAddress;
    public uint dtype;
    public uint rank;
    public uint flags;
    public uint bufferIndex;
};

// Fixed at offset 0 of every parameter block.  The tensor table is a BDA array
// of TensorDesc records; tensorTableOffset is an element offset into it.
public struct KernelHeader
{
    public uint3 dispatchBase;
    public uint numel;
    public uint flags;
    public uint tensorTableOffset;
    public uint tensorCount;
    public uint reserved0;
    public uint64_t tensorTableAddress;
    public uint reserved1;
    public uint reserved2;
};

public struct KernelPushConstants
{
    public uint64_t headerAddress;
    public uint64_t workspaceAddress;
    public uint flags;
    public uint reserved;
};

public uint tensor_rank(TensorDesc t)
{
    return min(t.rank, SLANGML_MAX_RANK);
}

public bool tensor_has_zero_extent(TensorDesc t)
{
    for (uint axis = 0; axis < tensor_rank(t); ++axis)
        if (t.shape[axis] == 0u)
            return true;
    return false;
}

public uint tensor_numel32(TensorDesc t)
{
    uint result = 1u;
    uint rank = tensor_rank(t);
    if (rank == 0u)
        return result;
    for (uint axis = 0; axis < rank; ++axis)
        result *= t.shape[axis];
    return result;
}

#if SLANGML_INDEX64
public uint64_t tensor_numel64(TensorDesc t)
{
    uint64_t result = uint64_t(1);
    uint rank = tensor_rank(t);
    for (uint axis = 0; axis < rank; ++axis)
        result *= uint64_t(t.shape[axis]);
    return result;
}
#endif

public bool tensor_is_contiguous(TensorDesc t)
{
    if (tensor_has_zero_extent(t))
        return true;

    int expected = 1;
    uint rank = tensor_rank(t);
    for (uint reverse = 0; reverse < rank; ++reverse)
    {
        uint axis = rank - reverse - 1u;
        if (t.shape[axis] > 1u && t.stride[axis] != expected)
            return false;
        expected *= int(t.shape[axis]);
    }
    return true;
}

public bool tensor_uses_bda()
{
#if SLANGML_USE_BDA
    return true;
#else
    return false;
#endif
}

public bool tensor_is_fast_contiguous(TensorDesc t)
{
    return (t.flags & (SLANGML_FLAG_CONTIGUOUS | SLANGML_FLAG_CANONICAL))
        == (SLANGML_FLAG_CONTIGUOUS | SLANGML_FLAG_CANONICAL);
}

public bool tensor_is_fast_rank3(TensorDesc t)
{
    return (t.flags & SLANGML_FLAG_RANK_LE3) != 0u;
}

public bool tensor_is_fast_vec4(TensorDesc t)
{
    return (t.flags & SLANGML_FLAG_ALIGNED16) != 0u;
}

public TensorDesc* slangml_tensor_desc_ptr(uint64_t tensorTableAddress, uint index)
{
    TensorDesc* table = reinterpret<TensorDesc*>(tensorTableAddress);
    return table + index;
}

public KernelHeader* slangml_kernel_header_ptr(uint64_t headerAddress)
{
    return reinterpret<KernelHeader*>(headerAddress);
}

public TensorDesc slangml_tensor_desc(KernelHeader header, uint localIndex)
{
    TensorDesc* table = slangml_tensor_desc_ptr(
        header.tensorTableAddress,
        header.tensorTableOffset + localIndex);
    return table[0];
}

public uint tensor_element_size_bytes(uint dtype)
{
    switch (dtype)
    {
    case SLANGML_DTYPE_FP32: return 4u;
    case SLANGML_DTYPE_FP16: return 2u;
    case SLANGML_DTYPE_BF16: return 2u;
    case SLANGML_DTYPE_INT8: return 1u;
    case SLANGML_DTYPE_UINT8: return 1u;
    case SLANGML_DTYPE_INT32: return 4u;
    case SLANGML_DTYPE_INT64: return 8u;
    case SLANGML_DTYPE_BOOL: return 1u;
    case SLANGML_DTYPE_FP8_E4M3: return 1u;
    case SLANGML_DTYPE_FP8_E5M2: return 1u;
    case SLANGML_DTYPE_PACKED_INT4: return 0u; // two logical elements per byte
    default: return 0u;
    }
}
