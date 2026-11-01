import slangml_common;

#ifndef SLANGML_ENABLE_SUBGROUP_OPS
#define SLANGML_ENABLE_SUBGROUP_OPS 1
#endif

struct ReduceStatsParams
{
    KernelHeader header;
    uint rows;
    uint cols;
    float epsilon;
    uint correction;
};

[[vk::push_constant]]
ConstantBuffer<KernelPushConstants> smlReduceStatsPush;

groupshared float smlReduceStatsScratch[256];

float sml_reduce_stats_sum(float value, uint lane)
{
    GroupMemoryBarrierWithGroupSync();
#if SLANGML_ENABLE_SUBGROUP_OPS
    float subgroupValue = slangml_subgroup_add_f32(value);
    uint subgroup = lane / SLANGML_SUBGROUP_SIZE;
    if (WaveGetLaneIndex() == 0u)
        smlReduceStatsScratch[subgroup] = subgroupValue;
    GroupMemoryBarrierWithGroupSync();
    uint subgroupCount = (256u + SLANGML_SUBGROUP_SIZE - 1u) / SLANGML_SUBGROUP_SIZE;
    if (subgroup == 0u)
    {
        float v = lane < subgroupCount ? smlReduceStatsScratch[lane] : 0.0f;
        v = slangml_subgroup_add_f32(v);
        if (WaveGetLaneIndex() == 0u)
            smlReduceStatsScratch[0] = v;
    }
    GroupMemoryBarrierWithGroupSync();
    return smlReduceStatsScratch[0];
#else
    return slangml_workgroup_reduce_add_f32_256(value, smlReduceStatsScratch, lane);
#endif
}

float sml_reduce_stats_max(float value, uint lane)
{
    GroupMemoryBarrierWithGroupSync();
#if SLANGML_ENABLE_SUBGROUP_OPS
    float subgroupValue = slangml_subgroup_max_f32(value);
    uint subgroup = lane / SLANGML_SUBGROUP_SIZE;
    if (WaveGetLaneIndex() == 0u)
        smlReduceStatsScratch[subgroup] = subgroupValue;
    GroupMemoryBarrierWithGroupSync();
    uint subgroupCount = (256u + SLANGML_SUBGROUP_SIZE - 1u) / SLANGML_SUBGROUP_SIZE;
    if (subgroup == 0u)
    {
        float v = lane < subgroupCount ? smlReduceStatsScratch[lane] : -asfloat(0x7F800000u);
        v = slangml_subgroup_max_f32(v);
        if (WaveGetLaneIndex() == 0u)
            smlReduceStatsScratch[0] = v;
    }
    GroupMemoryBarrierWithGroupSync();
    return smlReduceStatsScratch[0];
#else
    smlReduceStatsScratch[lane] = value;
    GroupMemoryBarrierWithGroupSync();
    for (uint active = 256u; active > 1u; active = (active + 1u) / 2u)
    {
        uint half = active / 2u;
        if (lane < half)
            smlReduceStatsScratch[lane] = max(smlReduceStatsScratch[lane], smlReduceStatsScratch[lane + half]);
        if ((active & 1u) != 0u && lane == 0u)
            smlReduceStatsScratch[0] = max(smlReduceStatsScratch[0], smlReduceStatsScratch[active - 1u]);
        GroupMemoryBarrierWithGroupSync();
    }
    return smlReduceStatsScratch[0];
#endif
}

float sml_reduce_stats_load(TensorDesc input, uint row, uint col, uint cols)
{
    return slangml_bda_load_f32(slangml_final_byte_address(input, int(row * cols + col), 4u));
}

void sml_reduce_stats(uint mode, uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID)
{
    ReduceStatsParams* params = reinterpret<ReduceStatsParams*>(smlReduceStatsPush.headerAddress);
    TensorDesc input = slangml_tensor_desc(params.header, 0u);
    TensorDesc output = slangml_tensor_desc(params.header, 1u);
    uint row = params.header.dispatchBase.x + gid.x;
    uint lane = tid.x;
    if (row >= params.rows)
        return;

    float result = 0.0f;
    if (mode == 0u)
    {
        float localMax = -asfloat(0x7F800000u);
        for (uint col = lane; col < params.cols; col += 256u)
            localMax = max(localMax, sml_reduce_stats_load(input, row, col, params.cols));
        float rowMax = sml_reduce_stats_max(localMax, lane);
        float localSum = 0.0f;
        for (uint col = lane; col < params.cols; col += 256u)
            localSum += exp(sml_reduce_stats_load(input, row, col, params.cols) - rowMax);
        result = rowMax + log(sml_reduce_stats_sum(localSum, lane));
    }
    else if (mode == 1u || mode == 2u)
    {
        float localSum = 0.0f;
        for (uint col = lane; col < params.cols; col += 256u)
            localSum += sml_reduce_stats_load(input, row, col, params.cols);
        float mean = sml_reduce_stats_sum(localSum, lane) / float(params.cols);
        float localSquares = 0.0f;
        for (uint col = lane; col < params.cols; col += 256u)
        {
            float centered = sml_reduce_stats_load(input, row, col, params.cols) - mean;
            localSquares += centered * centered;
        }
        float divisor = max(1.0f, float(params.cols - params.correction));
        result = sml_reduce_stats_sum(localSquares, lane) / divisor;
        if (mode == 2u)
            result = sqrt(result + params.epsilon);
    }
    else
    {
        float local = 0.0f;
        for (uint col = lane; col < params.cols; col += 256u)
        {
            float value = sml_reduce_stats_load(input, row, col, params.cols);
            local += mode == 3u ? abs(value) : value * value;
        }
        result = sml_reduce_stats_sum(local, lane);
        if (mode == 4u)
            result = sqrt(result);
    }

    if (lane == 0u)
        slangml_bda_store_f32(slangml_final_byte_address(output, int(row), 4u), result);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_reduce_logsumexp_fwd_f32_subgroup(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID)
{
    sml_reduce_stats(0u, gid, tid);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_reduce_var_fwd_f32_subgroup(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID)
{
    sml_reduce_stats(1u, gid, tid);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_reduce_std_fwd_f32_subgroup(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID)
{
    sml_reduce_stats(2u, gid, tid);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_reduce_l1_fwd_f32_subgroup(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID)
{
    sml_reduce_stats(3u, gid, tid);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_reduce_l2_fwd_f32_subgroup(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID)
{
    sml_reduce_stats(4u, gid, tid);
}
