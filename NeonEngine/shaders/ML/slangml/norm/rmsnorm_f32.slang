import slangml_common;

#ifndef SLANGML_ENABLE_SUBGROUP_OPS
#define SLANGML_ENABLE_SUBGROUP_OPS 1
#endif

struct RMSNormParams
{
    KernelHeader header;
    uint rows;
    uint cols;
    float epsilon;
    float reserved;
};

[[vk::push_constant]]
ConstantBuffer<KernelPushConstants> smlRMSNormPush;

groupshared float smlRMSNormScratch[256];

float sml_rmsnorm_reduce(float value, uint lane)
{
#if SLANGML_ENABLE_SUBGROUP_OPS
    float subgroupValue = slangml_subgroup_add_f32(value);
    uint subgroup = lane / SLANGML_SUBGROUP_SIZE;
    if (WaveGetLaneIndex() == 0u)
        smlRMSNormScratch[subgroup] = subgroupValue;
    GroupMemoryBarrierWithGroupSync();
    uint subgroupCount = (256u + SLANGML_SUBGROUP_SIZE - 1u) / SLANGML_SUBGROUP_SIZE;
    if (subgroup == 0u)
    {
        float v = lane < subgroupCount ? smlRMSNormScratch[lane] : 0.0f;
        v = slangml_subgroup_add_f32(v);
        if (WaveGetLaneIndex() == 0u)
            smlRMSNormScratch[0] = v;
    }
    GroupMemoryBarrierWithGroupSync();
    return smlRMSNormScratch[0];
#else
    return slangml_workgroup_reduce_add_f32_256(value, smlRMSNormScratch, lane);
#endif
}

float sml_rmsnorm_load(TensorDesc tensor, uint row, uint col, uint cols)
{
    return slangml_bda_load_f32(slangml_final_byte_address(tensor, int(row * cols + col), 4u));
}

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_norm_rmsnorm_fwd_f32_subgroup(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID)
{
    RMSNormParams* params = reinterpret<RMSNormParams*>(smlRMSNormPush.headerAddress);
    TensorDesc input = slangml_tensor_desc(params.header, 0u);
    TensorDesc gamma = slangml_tensor_desc(params.header, 1u);
    TensorDesc output = slangml_tensor_desc(params.header, 2u);
    uint row = params.header.dispatchBase.x + gid.x;
    uint lane = tid.x;
    if (row >= params.rows)
        return;

    float localSquares = 0.0f;
    for (uint col = lane; col < params.cols; col += 256u)
    {
        float x = sml_rmsnorm_load(input, row, col, params.cols);
        localSquares += x * x;
    }
    float invRms = rsqrt(sml_rmsnorm_reduce(localSquares, lane) / float(params.cols) + params.epsilon);
    for (uint col = lane; col < params.cols; col += 256u)
    {
        float x = sml_rmsnorm_load(input, row, col, params.cols);
        float g = slangml_bda_load_f32(slangml_final_byte_address(gamma, int(col), 4u));
        slangml_bda_store_f32(slangml_final_byte_address(output, int(row * params.cols + col), 4u), x * invRms * g);
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_norm_rmsnorm_bwd_f32_subgroup(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID)
{
    RMSNormParams* params = reinterpret<RMSNormParams*>(smlRMSNormPush.headerAddress);
    TensorDesc input = slangml_tensor_desc(params.header, 0u);
    TensorDesc gamma = slangml_tensor_desc(params.header, 1u);
    TensorDesc upstream = slangml_tensor_desc(params.header, 2u);
    TensorDesc dx = slangml_tensor_desc(params.header, 3u);
    TensorDesc dGammaPartial = slangml_tensor_desc(params.header, 4u);
    uint row = params.header.dispatchBase.x + gid.x;
    uint lane = tid.x;
    if (row >= params.rows)
        return;

    float localSquares = 0.0f;
    for (uint col = lane; col < params.cols; col += 256u)
    {
        float x = sml_rmsnorm_load(input, row, col, params.cols);
        localSquares += x * x;
    }
    float invRms = rsqrt(sml_rmsnorm_reduce(localSquares, lane) / float(params.cols) + params.epsilon);
    float localCorrection = 0.0f;
    for (uint col = lane; col < params.cols; col += 256u)
    {
        float x = sml_rmsnorm_load(input, row, col, params.cols);
        float g = slangml_bda_load_f32(slangml_final_byte_address(gamma, int(col), 4u));
        float dy = sml_rmsnorm_load(upstream, row, col, params.cols);
        localCorrection += dy * g * x;
    }
    float correction = sml_rmsnorm_reduce(localCorrection, lane);
    for (uint col = lane; col < params.cols; col += 256u)
    {
        float x = sml_rmsnorm_load(input, row, col, params.cols);
        float g = slangml_bda_load_f32(slangml_final_byte_address(gamma, int(col), 4u));
        float dy = sml_rmsnorm_load(upstream, row, col, params.cols);
        float dxValue = dy * g * invRms - x * invRms * invRms * invRms * correction / float(params.cols);
        uint linear = row * params.cols + col;
        slangml_bda_store_f32(slangml_final_byte_address(dx, int(linear), 4u), dxValue);
        slangml_bda_store_f32(slangml_final_byte_address(dGammaPartial, int(linear), 4u), dy * x * invRms);
    }
}
