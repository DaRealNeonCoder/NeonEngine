import slangml_common;

#ifndef SLANGML_ENABLE_SUBGROUP_OPS
#define SLANGML_ENABLE_SUBGROUP_OPS 1
#endif

struct LayerNormParams
{
    KernelHeader header;
    uint rows;
    uint cols;
    float epsilon;
    float reserved;
};

[[vk::push_constant]]
ConstantBuffer<KernelPushConstants> smlLayerNormPush;

groupshared float smlLayerNormScratch[256];

float sml_layernorm_reduce(float value, uint lane)
{
#if SLANGML_ENABLE_SUBGROUP_OPS
    float subgroupValue = slangml_subgroup_add_f32(value);
    uint subgroup = lane / SLANGML_SUBGROUP_SIZE;
    if (WaveGetLaneIndex() == 0u)
        smlLayerNormScratch[subgroup] = subgroupValue;
    GroupMemoryBarrierWithGroupSync();
    uint subgroupCount = (256u + SLANGML_SUBGROUP_SIZE - 1u) / SLANGML_SUBGROUP_SIZE;
    if (subgroup == 0u)
    {
        float v = lane < subgroupCount ? smlLayerNormScratch[lane] : 0.0f;
        v = slangml_subgroup_add_f32(v);
        if (WaveGetLaneIndex() == 0u)
            smlLayerNormScratch[0] = v;
    }
    GroupMemoryBarrierWithGroupSync();
    return smlLayerNormScratch[0];
#else
    return slangml_workgroup_reduce_add_f32_256(value, smlLayerNormScratch, lane);
#endif
}

float sml_layernorm_load(TensorDesc tensor, uint row, uint col, uint cols)
{
    return slangml_bda_load_f32(slangml_final_byte_address(tensor, int(row * cols + col), 4u));
}

void sml_layernorm_stats(
    TensorDesc input,
    LayerNormParams* params,
    uint row,
    uint lane,
    out float mean,
    out float invStd)
{
    float sum = 0.0f;
    for (uint col = lane; col < params.cols; col += 256u)
        sum += sml_layernorm_load(input, row, col, params.cols);
    mean = sml_layernorm_reduce(sum, lane) / float(params.cols);

    float sumSquares = 0.0f;
    for (uint col = lane; col < params.cols; col += 256u)
    {
        float centered = sml_layernorm_load(input, row, col, params.cols) - mean;
        sumSquares += centered * centered;
    }
    float variance = sml_layernorm_reduce(sumSquares, lane) / float(params.cols);
    invStd = rsqrt(variance + params.epsilon);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_norm_layernorm_fwd_f32_subgroup(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID)
{
    LayerNormParams* params = reinterpret<LayerNormParams*>(smlLayerNormPush.headerAddress);
    TensorDesc input = slangml_tensor_desc(params.header, 0u);
    TensorDesc gamma = slangml_tensor_desc(params.header, 1u);
    TensorDesc beta = slangml_tensor_desc(params.header, 2u);
    TensorDesc output = slangml_tensor_desc(params.header, 3u);
    uint row = params.header.dispatchBase.x + gid.x;
    uint lane = tid.x;
    if (row >= params.rows)
        return;

    float mean;
    float invStd;
    sml_layernorm_stats(input, params, row, lane, mean, invStd);
    for (uint col = lane; col < params.cols; col += 256u)
    {
        float x = sml_layernorm_load(input, row, col, params.cols);
        float g = slangml_bda_load_f32(slangml_final_byte_address(gamma, int(col), 4u));
        float b = slangml_bda_load_f32(slangml_final_byte_address(beta, int(col), 4u));
        float normalized = (x - mean) * invStd;
        slangml_bda_store_f32(
            slangml_final_byte_address(output, int(row * params.cols + col), 4u),
            normalized * g + b);
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_norm_layernorm_bwd_f32_subgroup(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID)
{
    LayerNormParams* params = reinterpret<LayerNormParams*>(smlLayerNormPush.headerAddress);
    TensorDesc input = slangml_tensor_desc(params.header, 0u);
    TensorDesc gamma = slangml_tensor_desc(params.header, 1u);
    TensorDesc upstream = slangml_tensor_desc(params.header, 2u);
    TensorDesc dx = slangml_tensor_desc(params.header, 3u);
    TensorDesc dGammaPartial = slangml_tensor_desc(params.header, 4u);
    TensorDesc dBetaPartial = slangml_tensor_desc(params.header, 5u);
    uint row = params.header.dispatchBase.x + gid.x;
    uint lane = tid.x;
    if (row >= params.rows)
        return;

    float mean;
    float invStd;
    sml_layernorm_stats(input, params, row, lane, mean, invStd);

    float sumDyGamma = 0.0f;
    float sumDyGammaX = 0.0f;
    for (uint col = lane; col < params.cols; col += 256u)
    {
        float x = sml_layernorm_load(input, row, col, params.cols);
        float dy = sml_layernorm_load(upstream, row, col, params.cols);
        float g = slangml_bda_load_f32(slangml_final_byte_address(gamma, int(col), 4u));
        float normalized = (x - mean) * invStd;
        sumDyGamma += dy * g;
        sumDyGammaX += dy * g * normalized;
    }
    float reducedDyGamma = sml_layernorm_reduce(sumDyGamma, lane);
    float reducedDyGammaX = sml_layernorm_reduce(sumDyGammaX, lane);

    for (uint col = lane; col < params.cols; col += 256u)
    {
        float x = sml_layernorm_load(input, row, col, params.cols);
        float dy = sml_layernorm_load(upstream, row, col, params.cols);
        float g = slangml_bda_load_f32(slangml_final_byte_address(gamma, int(col), 4u));
        float normalized = (x - mean) * invStd;
        float dxValue = (g * invStd / float(params.cols))
            * (float(params.cols) * dy - reducedDyGamma - normalized * reducedDyGammaX);
        uint linear = row * params.cols + col;
        slangml_bda_store_f32(slangml_final_byte_address(dx, int(linear), 4u), dxValue);
        slangml_bda_store_f32(slangml_final_byte_address(dGammaPartial, int(linear), 4u), dy * normalized);
        slangml_bda_store_f32(slangml_final_byte_address(dBetaPartial, int(linear), 4u), dy);
    }
}
