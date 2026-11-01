import slangml_common;

#ifndef SLANGML_ENABLE_SUBGROUP_OPS
#define SLANGML_ENABLE_SUBGROUP_OPS 1
#endif

struct L2NormParams
{
    KernelHeader header;
    uint rows;
    uint cols;
    float epsilon;
    float reserved;
};

[[vk::push_constant]]
ConstantBuffer<KernelPushConstants> smlL2NormPush;

groupshared float smlL2NormScratch[256];

float sml_l2_reduce(float value, uint lane)
{
    GroupMemoryBarrierWithGroupSync();
#if SLANGML_ENABLE_SUBGROUP_OPS
    float subgroupValue = slangml_subgroup_add_f32(value);
    uint subgroup = lane / SLANGML_SUBGROUP_SIZE;
    if (WaveGetLaneIndex() == 0u)
        smlL2NormScratch[subgroup] = subgroupValue;
    GroupMemoryBarrierWithGroupSync();
    uint subgroupCount = (256u + SLANGML_SUBGROUP_SIZE - 1u) / SLANGML_SUBGROUP_SIZE;
    if (subgroup == 0u)
    {
        float v = lane < subgroupCount ? smlL2NormScratch[lane] : 0.0f;
        v = slangml_subgroup_add_f32(v);
        if (WaveGetLaneIndex() == 0u)
            smlL2NormScratch[0] = v;
    }
    GroupMemoryBarrierWithGroupSync();
    return smlL2NormScratch[0];
#else
    return slangml_workgroup_reduce_add_f32_256(value, smlL2NormScratch, lane);
#endif
}

float sml_l2_load(TensorDesc tensor, uint row, uint col, uint cols)
{
    return slangml_bda_load_f32(slangml_final_byte_address(tensor, int(row * cols + col), 4u));
}

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_norm_l2_normalize_fwd_f32_subgroup(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID)
{
    L2NormParams* params = reinterpret<L2NormParams*>(smlL2NormPush.headerAddress);
    TensorDesc input = slangml_tensor_desc(params.header, 0u);
    TensorDesc output = slangml_tensor_desc(params.header, 1u);
    uint row = params.header.dispatchBase.x + gid.x;
    uint lane = tid.x;
    if (row >= params.rows)
        return;
    float localSquares = 0.0f;
    for (uint col = lane; col < params.cols; col += 256u)
    {
        float x = sml_l2_load(input, row, col, params.cols);
        localSquares += x * x;
    }
    float invNorm = rsqrt(sml_l2_reduce(localSquares, lane) + params.epsilon);
    for (uint col = lane; col < params.cols; col += 256u)
    {
        uint linear = row * params.cols + col;
        float x = sml_l2_load(input, row, col, params.cols);
        slangml_bda_store_f32(slangml_final_byte_address(output, int(linear), 4u), x * invNorm);
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_norm_l2_normalize_bwd_f32_subgroup(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID)
{
    L2NormParams* params = reinterpret<L2NormParams*>(smlL2NormPush.headerAddress);
    TensorDesc input = slangml_tensor_desc(params.header, 0u);
    TensorDesc upstream = slangml_tensor_desc(params.header, 1u);
    TensorDesc output = slangml_tensor_desc(params.header, 2u);
    uint row = params.header.dispatchBase.x + gid.x;
    uint lane = tid.x;
    if (row >= params.rows)
        return;
    float localSquares = 0.0f;
    for (uint col = lane; col < params.cols; col += 256u)
    {
        float x = sml_l2_load(input, row, col, params.cols);
        localSquares += x * x;
    }
    float invNorm = rsqrt(sml_l2_reduce(localSquares, lane) + params.epsilon);
    float localDot = 0.0f;
    for (uint col = lane; col < params.cols; col += 256u)
    {
        float x = sml_l2_load(input, row, col, params.cols);
        float dy = sml_l2_load(upstream, row, col, params.cols);
        localDot += dy * x;
    }
    float dot = sml_l2_reduce(localDot, lane);
    for (uint col = lane; col < params.cols; col += 256u)
    {
        float x = sml_l2_load(input, row, col, params.cols);
        float dy = sml_l2_load(upstream, row, col, params.cols);
        float dx = invNorm * dy - x * invNorm * invNorm * invNorm * dot;
        slangml_bda_store_f32(slangml_final_byte_address(output, int(row * params.cols + col), 4u), dx);
    }
}
