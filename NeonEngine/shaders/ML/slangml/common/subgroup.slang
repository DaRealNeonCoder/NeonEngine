implementing slangml_common;

#ifndef SLANGML_ENABLE_SUBGROUP_OPS
#define SLANGML_ENABLE_SUBGROUP_OPS 1
#endif

public struct SubgroupConfig
{
    public uint lane;
    public uint laneCount;
    public uint subgroupId;
};

public SubgroupConfig slangml_subgroup_config(uint lane, uint laneCount, uint subgroupId)
{
    SubgroupConfig result;
    result.lane = lane;
    result.laneCount = laneCount;
    result.subgroupId = subgroupId;
    return result;
}

// The Wave* path is the baseline.  The portable path is deliberately explicit
// about its shared-memory scratch contract and is only for devices or kernels
// that cannot use subgroup arithmetic.
public uint slangml_subgroup_size()
{
    return SLANGML_SUBGROUP_SIZE;
}
public float slangml_subgroup_add_f32(float value)
{
#if SLANGML_ENABLE_SUBGROUP_OPS
    return WaveActiveSum(value);
#else
    return value;
#endif
}

public uint slangml_subgroup_add_u32(uint value)
{
#if SLANGML_ENABLE_SUBGROUP_OPS
    return WaveActiveSum(value);
#else
    return value;
#endif
}

public float slangml_subgroup_min_f32(float value)
{
#if SLANGML_ENABLE_SUBGROUP_OPS
    return WaveActiveMin(value);
#else
    return value;
#endif
}

public float slangml_subgroup_max_f32(float value)
{
#if SLANGML_ENABLE_SUBGROUP_OPS
    return WaveActiveMax(value);
#else
    return value;
#endif
}

public uint slangml_subgroup_ballot_any(bool value)
{
#if SLANGML_ENABLE_SUBGROUP_OPS
    return WaveActiveAnyTrue(value) ? 1u : 0u;
#else
    return value ? 1u : 0u;
#endif
}

public float slangml_subgroup_prefix_add_f32(float value)
{
#if SLANGML_ENABLE_SUBGROUP_OPS
    return WavePrefixSum(value);
#else
    return value;
#endif
}

public uint slangml_subgroup_prefix_add_u32(uint value)
{
#if SLANGML_ENABLE_SUBGROUP_OPS
    return WavePrefixSum(value);
#else
    return value;
#endif
}

public float slangml_workgroup_reduce_add_f32(
    float value,
    inout float scratch[SLANGML_MAX_WORKGROUP_LANES],
    uint lane,
    uint workgroupSize)
{
    // Entry barrier: a previous call's lanes may still be reading scratch[0].
    GroupMemoryBarrierWithGroupSync();
    scratch[lane] = value;
    GroupMemoryBarrierWithGroupSync();
    for (uint active = workgroupSize; active > 1u; active = (active + 1u) / 2u)
    {
        uint half = active / 2u;
        if (lane < half)
            scratch[lane] += scratch[lane + half];
        if ((active & 1u) != 0u && lane == 0u)
            scratch[0] += scratch[active - 1u];
        GroupMemoryBarrierWithGroupSync();
    }
    return scratch[0];
}

public float slangml_workgroup_reduce_add_f32_256(
    float value,
    inout float scratch[256],
    uint lane)
{
    // Entry barrier: a previous call's lanes may still be reading scratch[0].
    GroupMemoryBarrierWithGroupSync();
    scratch[lane] = value;
    GroupMemoryBarrierWithGroupSync();
    for (uint active = 256u; active > 1u; active = (active + 1u) / 2u)
    {
        uint half = active / 2u;
        if (lane < half)
            scratch[lane] += scratch[lane + half];
        if ((active & 1u) != 0u && lane == 0u)
            scratch[0] += scratch[active - 1u];
        GroupMemoryBarrierWithGroupSync();
    }
    return scratch[0];
}

public uint slangml_workgroup_reduce_add_u32(
    uint value,
    inout uint scratch[SLANGML_MAX_WORKGROUP_LANES],
    uint lane,
    uint workgroupSize)
{
    // Entry barrier: a previous call's lanes may still be reading scratch[0].
    GroupMemoryBarrierWithGroupSync();
    scratch[lane] = value;
    GroupMemoryBarrierWithGroupSync();
    for (uint active = workgroupSize; active > 1u; active = (active + 1u) / 2u)
    {
        uint half = active / 2u;
        if (lane < half)
            scratch[lane] += scratch[lane + half];
        if ((active & 1u) != 0u && lane == 0u)
            scratch[0] += scratch[active - 1u];
        GroupMemoryBarrierWithGroupSync();
    }
    return scratch[0];
}

public uint slangml_subgroup_shuffle_u32(uint value, uint lane)
{
#if SLANGML_ENABLE_SUBGROUP_OPS
    return WaveReadLaneAt(value, lane);
#else
    return value;
#endif
}

public float slangml_subgroup_shuffle_f32(float value, uint lane)
{
#if SLANGML_ENABLE_SUBGROUP_OPS
    return WaveReadLaneAt(value, lane);
#else
    return value;
#endif
}
