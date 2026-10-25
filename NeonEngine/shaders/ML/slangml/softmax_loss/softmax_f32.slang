import slangml_common;

#ifndef SLANGML_ENABLE_SUBGROUP_OPS
#define SLANGML_ENABLE_SUBGROUP_OPS 1
#endif

struct SoftmaxParams
{
    KernelHeader header;
    uint rows;
    uint cols;
    float scale;
    float logitCap;
    uint maskMode;
    uint reserved;
};

[[vk::push_constant]]
ConstantBuffer<KernelPushConstants> smlSoftmaxPush;

groupshared float smlSoftmaxScratch[256];

float sml_softmax_load(TensorDesc tensor, uint row, uint col, uint cols)
{
    return slangml_bda_load_f32(slangml_final_byte_address(tensor, int(row * cols + col), 4u));
}

float sml_softmax_reduce_max(float value, uint lane)
{
#if SLANGML_ENABLE_SUBGROUP_OPS
    float subgroupValue = slangml_subgroup_max_f32(value);
    uint subgroup = lane / SLANGML_SUBGROUP_SIZE;
    if (WaveGetLaneIndex() == 0u)
        smlSoftmaxScratch[subgroup] = subgroupValue;
    GroupMemoryBarrierWithGroupSync();
    uint subgroupCount = (256u + SLANGML_SUBGROUP_SIZE - 1u) / SLANGML_SUBGROUP_SIZE;
    if (subgroup == 0u)
    {
        float v = lane < subgroupCount ? smlSoftmaxScratch[lane] : -asfloat(0x7F800000u);
        v = slangml_subgroup_max_f32(v);
        if (WaveGetLaneIndex() == 0u)
            smlSoftmaxScratch[0] = v;
    }
    GroupMemoryBarrierWithGroupSync();
    return smlSoftmaxScratch[0];
#else
    smlSoftmaxScratch[lane] = value;
    GroupMemoryBarrierWithGroupSync();
    for (uint active = 256u; active > 1u; active = (active + 1u) / 2u)
    {
        uint half = active / 2u;
        if (lane < half)
            smlSoftmaxScratch[lane] = max(smlSoftmaxScratch[lane], smlSoftmaxScratch[lane + half]);
        if ((active & 1u) != 0u && lane == 0u)
            smlSoftmaxScratch[0] = max(smlSoftmaxScratch[0], smlSoftmaxScratch[active - 1u]);
        GroupMemoryBarrierWithGroupSync();
    }
    return smlSoftmaxScratch[0];
#endif
}

float sml_softmax_reduce_sum(float value, uint lane)
{
#if SLANGML_ENABLE_SUBGROUP_OPS
    float subgroupValue = slangml_subgroup_add_f32(value);
    uint subgroup = lane / SLANGML_SUBGROUP_SIZE;
    if (WaveGetLaneIndex() == 0u)
        smlSoftmaxScratch[subgroup] = subgroupValue;
    GroupMemoryBarrierWithGroupSync();
    uint subgroupCount = (256u + SLANGML_SUBGROUP_SIZE - 1u) / SLANGML_SUBGROUP_SIZE;
    if (subgroup == 0u)
    {
        float v = lane < subgroupCount ? smlSoftmaxScratch[lane] : 0.0f;
        v = slangml_subgroup_add_f32(v);
        if (WaveGetLaneIndex() == 0u)
            smlSoftmaxScratch[0] = v;
    }
    GroupMemoryBarrierWithGroupSync();
    return smlSoftmaxScratch[0];
#else
    return slangml_workgroup_reduce_add_f32_256(value, smlSoftmaxScratch, lane);
#endif
}

float sml_softmax_logit(float value, SoftmaxParams* params)
{
    float scaled = value * params.scale;
    if (params.logitCap > 0.0f)
        scaled = params.logitCap * tanh(scaled / params.logitCap);
    return scaled;
}

float sml_softmax_masked_logit(
    TensorDesc input,
    TensorDesc mask,
    uint row,
    uint col,
    SoftmaxParams* params)
{
    float logit = sml_softmax_logit(sml_softmax_load(input, row, col, params.cols), params);
    float maskValue = sml_softmax_load(mask, row, col, params.cols);
    if (params.maskMode == 1u)
        return logit + maskValue;
    return maskValue != 0.0f ? logit : -asfloat(0x7F800000u);
}

void sml_softmax_forward(uint mode, uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID)
{
    SoftmaxParams* params = reinterpret<SoftmaxParams*>(smlSoftmaxPush.headerAddress);
    TensorDesc input = slangml_tensor_desc(params.header, 0u);
    TensorDesc output = slangml_tensor_desc(params.header, 1u);
    uint row = params.header.dispatchBase.x + gid.x;
    uint lane = tid.x;
    if (row >= params.rows)
        return;

    float localMax = -asfloat(0x7F800000u);
    for (uint col = lane; col < params.cols; col += 256u)
        localMax = max(localMax, sml_softmax_logit(sml_softmax_load(input, row, col, params.cols), params));
    float rowMax = sml_softmax_reduce_max(localMax, lane);

    float localSum = 0.0f;
    for (uint col = lane; col < params.cols; col += 256u)
        localSum += exp(sml_softmax_logit(sml_softmax_load(input, row, col, params.cols), params) - rowMax);
    float rowSum = sml_softmax_reduce_sum(localSum, lane);
    float logSum = log(rowSum);

    for (uint col = lane; col < params.cols; col += 256u)
    {
        float logit = sml_softmax_logit(sml_softmax_load(input, row, col, params.cols), params);
        float result = mode == 0u ? exp(logit - rowMax) / rowSum : logit - rowMax - logSum;
        slangml_bda_store_f32(slangml_final_byte_address(output, int(row * params.cols + col), 4u), result);
    }
}

void sml_softmax_forward_masked(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID)
{
    SoftmaxParams* params = reinterpret<SoftmaxParams*>(smlSoftmaxPush.headerAddress);
    TensorDesc input = slangml_tensor_desc(params.header, 0u);
    TensorDesc mask = slangml_tensor_desc(params.header, 1u);
    TensorDesc output = slangml_tensor_desc(params.header, 2u);
    uint row = params.header.dispatchBase.x + gid.x;
    uint lane = tid.x;
    if (row >= params.rows)
        return;
    float localMax = -asfloat(0x7F800000u);
    for (uint col = lane; col < params.cols; col += 256u)
        localMax = max(localMax, sml_softmax_masked_logit(input, mask, row, col, params));
    float rowMax = sml_softmax_reduce_max(localMax, lane);
    float localSum = 0.0f;
    for (uint col = lane; col < params.cols; col += 256u)
        localSum += exp(sml_softmax_masked_logit(input, mask, row, col, params) - rowMax);
    float rowSum = sml_softmax_reduce_sum(localSum, lane);
    for (uint col = lane; col < params.cols; col += 256u)
    {
        float logit = sml_softmax_masked_logit(input, mask, row, col, params);
        float result = exp(logit - rowMax) / rowSum;
        slangml_bda_store_f32(slangml_final_byte_address(output, int(row * params.cols + col), 4u), result);
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_softmax_softmax_fwd_f32_online(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID)
{
    sml_softmax_forward(0u, gid, tid);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_softmax_log_softmax_fwd_f32_online(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID)
{
    sml_softmax_forward(1u, gid, tid);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_softmax_softmax_bwd_f32_online(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID)
{
    SoftmaxParams* params = reinterpret<SoftmaxParams*>(smlSoftmaxPush.headerAddress);
    TensorDesc probability = slangml_tensor_desc(params.header, 0u);
    TensorDesc upstream = slangml_tensor_desc(params.header, 1u);
    TensorDesc output = slangml_tensor_desc(params.header, 2u);
    uint row = params.header.dispatchBase.x + gid.x;
    uint lane = tid.x;
    if (row >= params.rows)
        return;

    float localDot = 0.0f;
    for (uint col = lane; col < params.cols; col += 256u)
    {
        float p = sml_softmax_load(probability, row, col, params.cols);
        float dy = sml_softmax_load(upstream, row, col, params.cols);
        localDot += p * dy;
    }
    float dot = sml_softmax_reduce_sum(localDot, lane);
    for (uint col = lane; col < params.cols; col += 256u)
    {
        float p = sml_softmax_load(probability, row, col, params.cols);
        float dy = sml_softmax_load(upstream, row, col, params.cols);
        float dx = p * (dy - dot) * params.scale;
        slangml_bda_store_f32(slangml_final_byte_address(output, int(row * params.cols + col), 4u), dx);
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_softmax_softmax_masked_fwd_f32_online(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID)
{
    sml_softmax_forward_masked(gid, tid);
}
