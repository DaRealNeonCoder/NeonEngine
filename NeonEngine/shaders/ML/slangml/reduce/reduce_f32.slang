import slangml_common;

#ifndef SLANGML_ENABLE_SUBGROUP_OPS
#define SLANGML_ENABLE_SUBGROUP_OPS 1
#endif

struct ReduceParams
{
    KernelHeader header;
    uint rows;
    uint cols;
    uint axis;
    uint mode;
};

[[vk::push_constant]]
ConstantBuffer<KernelPushConstants> smlReducePush;

groupshared float smlReduceScratch[256];
groupshared uint smlReduceSubgroupCount;

float sml_reduce_input(TensorDesc input, uint row, uint col, uint cols)
{
    uint linear = row * cols + col;
    return slangml_bda_load_f32(slangml_final_byte_address(input, int(linear), 4u));
}

float sml_reduce_sum_workgroup(float value, uint localIndex)
{
    GroupMemoryBarrierWithGroupSync();
#if SLANGML_ENABLE_SUBGROUP_OPS
    float subgroupSum = slangml_subgroup_add_f32(value);
    uint lane = WaveGetLaneIndex();
    uint subgroup = localIndex / SLANGML_SUBGROUP_SIZE;
    uint subgroupCount = (256u + SLANGML_SUBGROUP_SIZE - 1u) / SLANGML_SUBGROUP_SIZE;
    if (lane == 0u)
        smlReduceScratch[subgroup] = subgroupSum;
    GroupMemoryBarrierWithGroupSync();
    if (subgroup == 0u)
    {
        float v = localIndex < subgroupCount ? smlReduceScratch[localIndex] : 0.0f;
        v = slangml_subgroup_add_f32(v);
        if (lane == 0u)
            smlReduceScratch[0] = v;
    }
    GroupMemoryBarrierWithGroupSync();
    return smlReduceScratch[0];
#else
    return slangml_workgroup_reduce_add_f32_256(value, smlReduceScratch, localIndex);
#endif
}

float sml_reduce_min_workgroup(float value, uint localIndex)
{
    GroupMemoryBarrierWithGroupSync();
#if SLANGML_ENABLE_SUBGROUP_OPS
    float subgroupValue = slangml_subgroup_min_f32(value);
    uint lane = WaveGetLaneIndex();
    uint subgroup = localIndex / SLANGML_SUBGROUP_SIZE;
    uint subgroupCount = (256u + SLANGML_SUBGROUP_SIZE - 1u) / SLANGML_SUBGROUP_SIZE;
    if (lane == 0u)
        smlReduceScratch[subgroup] = subgroupValue;
    GroupMemoryBarrierWithGroupSync();
    if (subgroup == 0u)
    {
        float v = localIndex < subgroupCount ? smlReduceScratch[localIndex] : asfloat(0x7F800000u);
        v = slangml_subgroup_min_f32(v);
        if (lane == 0u)
            smlReduceScratch[0] = v;
    }
    GroupMemoryBarrierWithGroupSync();
    return smlReduceScratch[0];
#else
    smlReduceScratch[localIndex] = value;
    GroupMemoryBarrierWithGroupSync();
    for (uint active = 256u; active > 1u; active = (active + 1u) / 2u)
    {
        uint half = active / 2u;
        if (localIndex < half)
            smlReduceScratch[localIndex] = min(smlReduceScratch[localIndex], smlReduceScratch[localIndex + half]);
        if ((active & 1u) != 0u && localIndex == 0u)
            smlReduceScratch[0] = min(smlReduceScratch[0], smlReduceScratch[active - 1u]);
        GroupMemoryBarrierWithGroupSync();
    }
    return smlReduceScratch[0];
#endif
}

float sml_reduce_max_workgroup(float value, uint localIndex)
{
    GroupMemoryBarrierWithGroupSync();
#if SLANGML_ENABLE_SUBGROUP_OPS
    float subgroupValue = slangml_subgroup_max_f32(value);
    uint lane = WaveGetLaneIndex();
    uint subgroup = localIndex / SLANGML_SUBGROUP_SIZE;
    uint subgroupCount = (256u + SLANGML_SUBGROUP_SIZE - 1u) / SLANGML_SUBGROUP_SIZE;
    if (lane == 0u)
        smlReduceScratch[subgroup] = subgroupValue;
    GroupMemoryBarrierWithGroupSync();
    if (subgroup == 0u)
    {
        float v = localIndex < subgroupCount ? smlReduceScratch[localIndex] : -asfloat(0x7F800000u);
        v = slangml_subgroup_max_f32(v);
        if (lane == 0u)
            smlReduceScratch[0] = v;
    }
    GroupMemoryBarrierWithGroupSync();
    return smlReduceScratch[0];
#else
    smlReduceScratch[localIndex] = value;
    GroupMemoryBarrierWithGroupSync();
    for (uint active = 256u; active > 1u; active = (active + 1u) / 2u)
    {
        uint half = active / 2u;
        if (localIndex < half)
            smlReduceScratch[localIndex] = max(smlReduceScratch[localIndex], smlReduceScratch[localIndex + half]);
        if ((active & 1u) != 0u && localIndex == 0u)
            smlReduceScratch[0] = max(smlReduceScratch[0], smlReduceScratch[active - 1u]);
        GroupMemoryBarrierWithGroupSync();
    }
    return smlReduceScratch[0];
#endif
}

void sml_reduce_rows(uint mode, uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID)
{
    ReduceParams* params = reinterpret<ReduceParams*>(smlReducePush.headerAddress);
    TensorDesc input = slangml_tensor_desc(params.header, 0u);
    TensorDesc output = slangml_tensor_desc(params.header, 1u);
    uint row = params.header.dispatchBase.x + gid.x;
    uint lane = tid.x;
    if (row >= params.rows)
        return;

    float partial = mode == 2u ? asfloat(0x7F800000u) : (mode == 3u ? -asfloat(0x7F800000u) : 0.0f);
    for (uint col = lane; col < params.cols; col += 256u)
    {
        float value = sml_reduce_input(input, row, col, params.cols);
        if (mode == 0u || mode == 1u)
            partial += value;
        else if (mode == 2u)
            partial = min(partial, value);
        else
            partial = max(partial, value);
    }

    float reduced = mode == 0u || mode == 1u
        ? sml_reduce_sum_workgroup(partial, lane)
        : (mode == 2u ? sml_reduce_min_workgroup(partial, lane) : sml_reduce_max_workgroup(partial, lane));
    if (lane == 0u)
    {
        if (mode == 1u)
            reduced /= float(params.cols);
        slangml_bda_store_f32(slangml_final_byte_address(output, int(row), 4u), reduced);
    }
}

struct ReduceBwdParams
{
    KernelHeader header;
    uint rows;
    uint cols;
    uint mode;
    uint reserved;
};

void sml_reduce_bwd(uint mode, uint3 dtid : SV_DispatchThreadID)
{
    ReduceBwdParams* params = reinterpret<ReduceBwdParams*>(smlReducePush.headerAddress);
    TensorDesc upstream = slangml_tensor_desc(params.header, 0u);
    TensorDesc output = slangml_tensor_desc(params.header, 1u);
    uint linear = params.header.dispatchBase.x + dtid.x;
    uint count = params.rows * params.cols;
    if (linear >= count)
        return;
    uint row = linear / params.cols;
    float dy = slangml_bda_load_f32(slangml_final_byte_address(upstream, int(row), 4u));
    float scale = mode == 0u ? 1.0f : 1.0f / float(params.cols);
    slangml_bda_store_f32(slangml_final_byte_address(output, int(linear), 4u), dy * scale);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_reduce_sum_fwd_f32_subgroup(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID)
{
    sml_reduce_rows(0u, gid, tid);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_reduce_mean_fwd_f32_subgroup(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID)
{
    sml_reduce_rows(1u, gid, tid);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_reduce_min_fwd_f32_subgroup(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID)
{
    sml_reduce_rows(2u, gid, tid);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_reduce_max_fwd_f32_subgroup(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID)
{
    sml_reduce_rows(3u, gid, tid);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_reduce_sum_bwd_f32_subgroup(uint3 dtid : SV_DispatchThreadID)
{
    sml_reduce_bwd(0u, dtid);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_reduce_mean_bwd_f32_subgroup(uint3 dtid : SV_DispatchThreadID)
{
    sml_reduce_bwd(1u, dtid);
}
