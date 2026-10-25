import slangml_common;

struct ElementwiseExtraParams
{
    KernelHeader header;
    uint outputRank;
    uint opFlags;
    float alpha;
    float beta;
};

[[vk::push_constant]]
ConstantBuffer<KernelPushConstants> smlElementwiseExtraPush;

float sml_extra_load(TensorDesc tensor, uint linear)
{
    return slangml_bda_load_f32(slangml_final_byte_address(tensor, int(linear), 4u));
}

float sml_extra_load_broadcast(TensorDesc tensor, uint outputRank, uint indices[SLANGML_MAX_RANK])
{
    int relative = slangml_broadcast_relative_offset32(tensor, outputRank, indices);
    return slangml_bda_load_f32(slangml_final_byte_address(tensor, relative, 4u));
}

void sml_extra_store(TensorDesc tensor, uint linear, float value)
{
    slangml_bda_store_f32(slangml_final_byte_address(tensor, int(linear), 4u), value);
}

float sml_extra_binary(uint op, float a, float b)
{
    switch (op)
    {
    case 0u: return pow(a, b);
    case 1u: return atan2(a, b);
    case 2u: return a - floor(a / b) * b;
    case 3u: return a < b ? 1.0f : 0.0f;
    case 4u: return a <= b ? 1.0f : 0.0f;
    case 5u: return a > b ? 1.0f : 0.0f;
    case 6u: return a >= b ? 1.0f : 0.0f;
    case 7u: return a == b ? 1.0f : 0.0f;
    case 8u: return a != b ? 1.0f : 0.0f;
    case 9u: return (a != 0.0f && b != 0.0f) ? 1.0f : 0.0f;
    case 10u: return (a != 0.0f || b != 0.0f) ? 1.0f : 0.0f;
    case 11u: return (a != 0.0f) == (b != 0.0f) ? 1.0f : 0.0f;
    default: return a + b;
    }
}

void sml_extra_binary_forward(uint op, uint3 dtid : SV_DispatchThreadID)
{
    ElementwiseExtraParams* params = reinterpret<ElementwiseExtraParams*>(smlElementwiseExtraPush.headerAddress);
    TensorDesc a = slangml_tensor_desc(params.header, 0u);
    TensorDesc b = slangml_tensor_desc(params.header, 1u);
    TensorDesc output = slangml_tensor_desc(params.header, 2u);
    uint linear = params.header.dispatchBase.x + dtid.x;
    if (linear >= params.header.numel)
        return;
    uint indices[SLANGML_MAX_RANK];
    slangml_linear_to_indices32(output, linear, indices);
    float av = sml_extra_load_broadcast(a, params.outputRank, indices);
    float bv = sml_extra_load_broadcast(b, params.outputRank, indices);
    sml_extra_store(output, linear, sml_extra_binary(op, av, bv));
}

void sml_extra_binary_backward(uint op, uint3 dtid : SV_DispatchThreadID)
{
    ElementwiseExtraParams* params = reinterpret<ElementwiseExtraParams*>(smlElementwiseExtraPush.headerAddress);
    TensorDesc a = slangml_tensor_desc(params.header, 0u);
    TensorDesc b = slangml_tensor_desc(params.header, 1u);
    TensorDesc upstream = slangml_tensor_desc(params.header, 2u);
    TensorDesc da = slangml_tensor_desc(params.header, 3u);
    TensorDesc db = slangml_tensor_desc(params.header, 4u);
    uint linear = params.header.dispatchBase.x + dtid.x;
    if (linear >= params.header.numel)
        return;
    float av = sml_extra_load(a, linear);
    float bv = sml_extra_load(b, linear);
    float dy = sml_extra_load(upstream, linear);
    float daValue = 0.0f;
    float dbValue = 0.0f;
    if (op == 0u)
    {
        float y = pow(av, bv);
        daValue = dy * bv * pow(av, bv - 1.0f);
        dbValue = dy * y * log(max(abs(av), 1.0e-20f));
    }
    else if (op == 1u)
    {
        float denominator = av * av + bv * bv;
        daValue = dy * bv / denominator;
        dbValue = -dy * av / denominator;
    }
    sml_extra_store(da, linear, daValue);
    sml_extra_store(db, linear, dbValue);
}

struct GatedParams
{
    KernelHeader header;
    uint rows;
    uint channels;
    float alpha;
    float reserved;
};

[[vk::push_constant]]
ConstantBuffer<KernelPushConstants> smlGatedPush;

float sml_gated_load(TensorDesc tensor, uint linear)
{
    return slangml_bda_load_f32(slangml_final_byte_address(tensor, int(linear), 4u));
}

void sml_gated_forward(uint mode, uint3 dtid : SV_DispatchThreadID)
{
    GatedParams* params = reinterpret<GatedParams*>(smlGatedPush.headerAddress);
    TensorDesc input = slangml_tensor_desc(params.header, 0u);
    TensorDesc output = slangml_tensor_desc(params.header, 1u);
    uint linear = params.header.dispatchBase.x + dtid.x;
    uint count = params.rows * params.channels;
    if (linear >= count)
        return;
    uint row = linear / params.channels;
    uint col = linear - row * params.channels;
    uint gateOffset = row * params.channels * 2u + col;
    float value = sml_gated_load(input, row * params.channels * 2u + col);
    float gate = sml_gated_load(input, gateOffset + params.channels);
    if (mode == 0u)
        value *= slangml_sigmoid(gate);
    else if (mode == 1u)
        value *= slangml_gelu_tanh(gate);
    else
        value *= gate;
    slangml_bda_store_f32(slangml_final_byte_address(output, int(linear), 4u), value);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_elementwise_pow_fwd_f32_broadcast(uint3 dtid : SV_DispatchThreadID) { sml_extra_binary_forward(0u, dtid); }

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_elementwise_pow_bwd_f32_same(uint3 dtid : SV_DispatchThreadID) { sml_extra_binary_backward(0u, dtid); }

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_elementwise_atan2_fwd_f32_broadcast(uint3 dtid : SV_DispatchThreadID) { sml_extra_binary_forward(1u, dtid); }

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_elementwise_mod_fwd_f32_broadcast(uint3 dtid : SV_DispatchThreadID) { sml_extra_binary_forward(2u, dtid); }

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_elementwise_lt_fwd_f32_broadcast(uint3 dtid : SV_DispatchThreadID) { sml_extra_binary_forward(3u, dtid); }

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_elementwise_le_fwd_f32_broadcast(uint3 dtid : SV_DispatchThreadID) { sml_extra_binary_forward(4u, dtid); }

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_elementwise_gt_fwd_f32_broadcast(uint3 dtid : SV_DispatchThreadID) { sml_extra_binary_forward(5u, dtid); }

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_elementwise_ge_fwd_f32_broadcast(uint3 dtid : SV_DispatchThreadID) { sml_extra_binary_forward(6u, dtid); }

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_elementwise_eq_fwd_f32_broadcast(uint3 dtid : SV_DispatchThreadID) { sml_extra_binary_forward(7u, dtid); }

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_elementwise_ne_fwd_f32_broadcast(uint3 dtid : SV_DispatchThreadID) { sml_extra_binary_forward(8u, dtid); }

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_elementwise_logical_and_fwd_f32_broadcast(uint3 dtid : SV_DispatchThreadID) { sml_extra_binary_forward(9u, dtid); }

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_elementwise_logical_or_fwd_f32_broadcast(uint3 dtid : SV_DispatchThreadID) { sml_extra_binary_forward(10u, dtid); }

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_elementwise_logical_xor_fwd_f32_broadcast(uint3 dtid : SV_DispatchThreadID) { sml_extra_binary_forward(11u, dtid); }

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_elementwise_swiglu_fwd_f32_contig(uint3 dtid : SV_DispatchThreadID) { sml_gated_forward(0u, dtid); }

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_elementwise_geglu_fwd_f32_contig(uint3 dtid : SV_DispatchThreadID) { sml_gated_forward(1u, dtid); }

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_elementwise_glu_fwd_f32_contig(uint3 dtid : SV_DispatchThreadID) { sml_gated_forward(2u, dtid); }
