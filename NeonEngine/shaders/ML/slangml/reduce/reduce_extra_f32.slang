import slangml_common;

#ifndef SLANGML_ENABLE_SUBGROUP_OPS
#define SLANGML_ENABLE_SUBGROUP_OPS 1
#endif

struct ReduceExtraParams
{
    KernelHeader header;
    uint rows;
    uint cols;
    uint mode;
    uint reserved;
};

[[vk::push_constant]]
ConstantBuffer<KernelPushConstants> smlReduceExtraPush;

groupshared float smlReduceExtraFloat[256];
groupshared uint smlReduceExtraUint[256];

float sml_reduce_extra_sum(float value, uint lane)
{
    GroupMemoryBarrierWithGroupSync();
#if SLANGML_ENABLE_SUBGROUP_OPS
    float subgroupValue = slangml_subgroup_add_f32(value);
    uint subgroup = lane / SLANGML_SUBGROUP_SIZE;
    if (WaveGetLaneIndex() == 0u)
        smlReduceExtraFloat[subgroup] = subgroupValue;
    GroupMemoryBarrierWithGroupSync();
    uint subgroupCount = (256u + SLANGML_SUBGROUP_SIZE - 1u) / SLANGML_SUBGROUP_SIZE;
    if (subgroup == 0u)
    {
        float v = lane < subgroupCount ? smlReduceExtraFloat[lane] : 0.0f;
        v = slangml_subgroup_add_f32(v);
        if (WaveGetLaneIndex() == 0u)
            smlReduceExtraFloat[0] = v;
    }
    GroupMemoryBarrierWithGroupSync();
    return smlReduceExtraFloat[0];
#else
    return slangml_workgroup_reduce_add_f32_256(value, smlReduceExtraFloat, lane);
#endif
}

float sml_reduce_extra_product(float value, uint lane)
{
    GroupMemoryBarrierWithGroupSync();
    smlReduceExtraFloat[lane] = value;
    GroupMemoryBarrierWithGroupSync();
    for (uint active = 256u; active > 1u; active = (active + 1u) / 2u)
    {
        uint half = active / 2u;
        if (lane < half)
            smlReduceExtraFloat[lane] *= smlReduceExtraFloat[lane + half];
        if ((active & 1u) != 0u && lane == 0u)
            smlReduceExtraFloat[0] *= smlReduceExtraFloat[active - 1u];
        GroupMemoryBarrierWithGroupSync();
    }
    return smlReduceExtraFloat[0];
}

uint sml_reduce_extra_any(uint value, uint lane)
{
    GroupMemoryBarrierWithGroupSync();
#if SLANGML_ENABLE_SUBGROUP_OPS
    uint subgroup = lane / SLANGML_SUBGROUP_SIZE;
    uint subgroupValue = WaveActiveAnyTrue(value != 0u) ? 1u : 0u;
    if (WaveGetLaneIndex() == 0u)
        smlReduceExtraUint[subgroup] = subgroupValue;
    GroupMemoryBarrierWithGroupSync();
    uint subgroupCount = (256u + SLANGML_SUBGROUP_SIZE - 1u) / SLANGML_SUBGROUP_SIZE;
    if (subgroup == 0u)
    {
        bool anyValue = lane < subgroupCount ? smlReduceExtraUint[lane] != 0u : false;
        anyValue = WaveActiveAnyTrue(anyValue);
        if (WaveGetLaneIndex() == 0u)
            smlReduceExtraUint[0] = anyValue ? 1u : 0u;
    }
    GroupMemoryBarrierWithGroupSync();
    return smlReduceExtraUint[0];
#else
    smlReduceExtraUint[lane] = value != 0u ? 1u : 0u;
    GroupMemoryBarrierWithGroupSync();
    for (uint active = 256u; active > 1u; active = (active + 1u) / 2u)
    {
        uint half = active / 2u;
        if (lane < half)
            smlReduceExtraUint[lane] = max(smlReduceExtraUint[lane], smlReduceExtraUint[lane + half]);
        if ((active & 1u) != 0u && lane == 0u)
            smlReduceExtraUint[0] = max(smlReduceExtraUint[0], smlReduceExtraUint[active - 1u]);
        GroupMemoryBarrierWithGroupSync();
    }
    return smlReduceExtraUint[0];
#endif
}

uint sml_reduce_extra_all(uint value, uint lane)
{
    GroupMemoryBarrierWithGroupSync();
#if SLANGML_ENABLE_SUBGROUP_OPS
    uint subgroup = lane / SLANGML_SUBGROUP_SIZE;
    uint subgroupValue = WaveActiveAllTrue(value != 0u) ? 1u : 0u;
    if (WaveGetLaneIndex() == 0u)
        smlReduceExtraUint[subgroup] = subgroupValue;
    GroupMemoryBarrierWithGroupSync();
    uint subgroupCount = (256u + SLANGML_SUBGROUP_SIZE - 1u) / SLANGML_SUBGROUP_SIZE;
    if (subgroup == 0u)
    {
        bool allValue = lane < subgroupCount ? smlReduceExtraUint[lane] != 0u : true;
        allValue = WaveActiveAllTrue(allValue);
        if (WaveGetLaneIndex() == 0u)
            smlReduceExtraUint[0] = allValue ? 1u : 0u;
    }
    GroupMemoryBarrierWithGroupSync();
    return smlReduceExtraUint[0];
#else
    smlReduceExtraUint[lane] = value != 0u ? 1u : 0u;
    GroupMemoryBarrierWithGroupSync();
    for (uint active = 256u; active > 1u; active = (active + 1u) / 2u)
    {
        uint half = active / 2u;
        if (lane < half)
            smlReduceExtraUint[lane] = min(smlReduceExtraUint[lane], smlReduceExtraUint[lane + half]);
        if ((active & 1u) != 0u && lane == 0u)
            smlReduceExtraUint[0] = min(smlReduceExtraUint[0], smlReduceExtraUint[active - 1u]);
        GroupMemoryBarrierWithGroupSync();
    }
    return smlReduceExtraUint[0];
#endif
}

float sml_reduce_extra_load(TensorDesc input, uint row, uint col, uint cols)
{
    return slangml_bda_load_f32(slangml_final_byte_address(input, int(row * cols + col), 4u));
}

void sml_reduce_extra_rows(uint mode, uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID)
{
    ReduceExtraParams* params = reinterpret<ReduceExtraParams*>(smlReduceExtraPush.headerAddress);
    TensorDesc input = slangml_tensor_desc(params.header, 0u);
    TensorDesc output = slangml_tensor_desc(params.header, 1u);
    uint row = params.header.dispatchBase.x + gid.x;
    uint lane = tid.x;
    if (row >= params.rows)
        return;

    float result = 0.0f;
    if (mode == 0u)
    {
        float local = 1.0f;
        for (uint col = lane; col < params.cols; col += 256u)
            local *= sml_reduce_extra_load(input, row, col, params.cols);
        result = sml_reduce_extra_product(local, lane);
    }
    else
    {
        uint local = 0u;
        if (mode == 1u)
        {
            for (uint col = lane; col < params.cols; col += 256u)
                local = max(local, sml_reduce_extra_load(input, row, col, params.cols) != 0.0f ? 1u : 0u);
            result = float(sml_reduce_extra_any(local, lane));
        }
        else
        {
            local = 1u;
            for (uint col = lane; col < params.cols; col += 256u)
                local = min(local, sml_reduce_extra_load(input, row, col, params.cols) != 0.0f ? 1u : 0u);
            result = float(sml_reduce_extra_all(local, lane));
        }
    }
    if (lane == 0u)
        slangml_bda_store_f32(slangml_final_byte_address(output, int(row), 4u), result);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_reduce_prod_fwd_f32_subgroup(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID)
{
    sml_reduce_extra_rows(0u, gid, tid);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_reduce_any_fwd_f32_subgroup(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID)
{
    sml_reduce_extra_rows(1u, gid, tid);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_reduce_all_fwd_f32_subgroup(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID)
{
    sml_reduce_extra_rows(2u, gid, tid);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_reduce_prod_bwd_f32_subgroup(uint3 dtid : SV_DispatchThreadID)
{
    ReduceExtraParams* params = reinterpret<ReduceExtraParams*>(smlReduceExtraPush.headerAddress);
    TensorDesc input = slangml_tensor_desc(params.header, 0u);
    TensorDesc upstream = slangml_tensor_desc(params.header, 1u);
    TensorDesc output = slangml_tensor_desc(params.header, 2u);
    uint linear = params.header.dispatchBase.x + dtid.x;
    uint count = params.rows * params.cols;
    if (linear >= count)
        return;
    uint row = linear / params.cols;
    uint col = linear - row * params.cols;
    float x = sml_reduce_extra_load(input, row, col, params.cols);
    float productExcept = 1.0f;
    for (uint other = 0u; other < params.cols; ++other)
        if (other != col)
            productExcept *= sml_reduce_extra_load(input, row, other, params.cols);
    float dy = slangml_bda_load_f32(slangml_final_byte_address(upstream, int(row), 4u));
    slangml_bda_store_f32(slangml_final_byte_address(output, int(linear), 4u), dy * productExcept);
}
