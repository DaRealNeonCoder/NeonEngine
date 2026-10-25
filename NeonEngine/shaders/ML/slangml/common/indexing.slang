implementing slangml_common;

#ifndef SLANGML_INDEX64
#define SLANGML_INDEX64 0
#endif

// All default hot paths keep coordinates, extents, strides, and relative
// offsets in 32 bits.  Only the descriptor base offset and final byte address
// use 64-bit arithmetic.  Define SLANGML_INDEX64=1 for rare huge tensors.
public int slangml_linear_to_relative_offset32(TensorDesc t, uint linear)
{
    int relative = 0;
    uint rank = tensor_rank(t);
    for (uint reverse = 0; reverse < rank; ++reverse)
    {
        uint axis = rank - reverse - 1u;
        uint extent = t.shape[axis];
        uint coordinate = extent == 0u ? 0u : linear % extent;
        linear = extent == 0u ? 0u : linear / extent;
        relative += int(coordinate) * t.stride[axis];
    }
    return relative;
}

#if SLANGML_INDEX64
public int64_t slangml_linear_to_relative_offset64(TensorDesc t, uint64_t linear)
{
    int64_t relative = 0;
    uint rank = tensor_rank(t);
    for (uint reverse = 0; reverse < rank; ++reverse)
    {
        uint axis = rank - reverse - 1u;
        uint64_t extent = uint64_t(t.shape[axis]);
        uint64_t coordinate = extent == 0u ? 0u : linear % extent;
        linear = extent == 0u ? 0u : linear / extent;
        relative += int64_t(coordinate) * int64_t(t.stride[axis]);
    }
    return relative;
}
#endif

public void slangml_linear_to_indices32(TensorDesc t, uint linear, out uint indices[SLANGML_MAX_RANK])
{
    for (uint axis = 0; axis < SLANGML_MAX_RANK; ++axis)
        indices[axis] = 0u;

    uint rank = tensor_rank(t);
    for (uint reverse = 0; reverse < rank; ++reverse)
    {
        uint axis = rank - reverse - 1u;
        uint extent = t.shape[axis];
        indices[axis] = extent == 0u ? 0u : linear % extent;
        linear = extent == 0u ? 0u : linear / extent;
    }
}

public int slangml_indices_to_relative_offset32(TensorDesc t, uint indices[SLANGML_MAX_RANK])
{
    int relative = 0;
    uint rank = tensor_rank(t);
    for (uint axis = 0; axis < rank; ++axis)
        relative += int(indices[axis]) * t.stride[axis];
    return relative;
}

public int slangml_broadcast_relative_offset32(
    TensorDesc input,
    uint outputRank,
    uint outputIndices[SLANGML_MAX_RANK])
{
    int relative = 0;
    uint inRank = tensor_rank(input);
    if (inRank > outputRank)
        return relative;

    uint leading = outputRank - inRank;
    for (uint axis = 0; axis < inRank; ++axis)
    {
        uint outputAxis = leading + axis;
        uint coordinate = input.shape[axis] == 1u ? 0u : outputIndices[outputAxis];
        relative += int(coordinate) * input.stride[axis];
    }
    return relative;
}

public bool slangml_indices_in_bounds32(TensorDesc t, uint indices[SLANGML_MAX_RANK])
{
    uint rank = tensor_rank(t);
    for (uint axis = 0; axis < rank; ++axis)
        if (indices[axis] >= t.shape[axis])
            return false;
    return true;
}

public bool slangml_shapes_broadcast_compatible(TensorDesc a, TensorDesc b, uint outRank)
{
    uint ar = tensor_rank(a);
    uint br = tensor_rank(b);
    if (ar > outRank || br > outRank)
        return false;
    for (uint i = 0; i < outRank; ++i)
    {
        uint ai = i < outRank - ar ? 1u : a.shape[i - (outRank - ar)];
        uint bi = i < outRank - br ? 1u : b.shape[i - (outRank - br)];
        if (ai != bi && ai != 1u && bi != 1u)
            return false;
    }
    return true;
}

public uint slangml_grid_stride_index32(uint localIndex, uint workgroupSize, uint base, uint count)
{
    uint index = base + localIndex;
    return index < count ? index : SLANGML_INVALID_U32;
}
