import slangml_common;

// Phase 2 fast elementwise kernels.  The BDA descriptor-table path is the
// baseline.  Strided/broadcast cases are intentionally separate from the
// contiguous entry points so the common path stays branch-light.

struct ElementwiseParams
{
    KernelHeader header;
    uint outputRank;
    uint opFlags;
    float alpha;
    float beta;
};

[[vk::push_constant]]
ConstantBuffer<KernelPushConstants> smlElementwisePush;

interface IElementwiseOp
{
    float apply(float x);
    float derivative(float x, float y);
};

public struct ReluOp : IElementwiseOp
{
    public float apply(float x) { return max(x, 0.0f); }
    public float derivative(float x, float y) { return x > 0.0f ? y : 0.0f; }
};

public struct SiluOp : IElementwiseOp
{
    public float apply(float x)
    {
        return x * slangml_sigmoid(x);
    }
    public float derivative(float x, float y)
    {
        float s = slangml_sigmoid(x);
        return y * (s + x * s * (1.0f - s));
    }
};

public struct GeluTanhOp : IElementwiseOp
{
    public float apply(float x) { return slangml_gelu_tanh(x); }
    public float derivative(float x, float y)
    {
        float u = 0.7978845608028654f * (x + 0.044715f * x * x * x);
        float t = tanh(u);
        float du = 0.7978845608028654f * (1.0f + 3.0f * 0.044715f * x * x);
        return y * (0.5f * (1.0f + t) + 0.5f * x * (1.0f - t * t) * du);
    }
};

public struct ReluSiluChain : IElementwiseOp
{
    public float apply(float x)
    {
        ReluOp relu;
        SiluOp silu;
        return silu.apply(relu.apply(x));
    }
    public float derivative(float x, float y)
    {
        ReluOp relu;
        SiluOp silu;
        float r = relu.apply(x);
        return x > 0.0f ? silu.derivative(r, y) : 0.0f;
    }
};

float sml_elementwise_load(TensorDesc tensor, uint linear)
{
    if ((tensor.flags & SLANGML_FLAG_CONTIGUOUS) != 0u)
        return slangml_bda_load_f32(slangml_final_byte_address(tensor, int(linear), 4u));

    uint indices[SLANGML_MAX_RANK];
    slangml_linear_to_indices32(tensor, linear, indices);
    int relative = slangml_indices_to_relative_offset32(tensor, indices);
    return slangml_bda_load_f32(slangml_final_byte_address(tensor, relative, 4u));
}

float sml_elementwise_load_broadcast(TensorDesc tensor, uint outputRank, uint indices[SLANGML_MAX_RANK])
{
    int relative = slangml_broadcast_relative_offset32(tensor, outputRank, indices);
    return slangml_bda_load_f32(slangml_final_byte_address(tensor, relative, 4u));
}

void sml_elementwise_store(TensorDesc tensor, uint linear, float value)
{
    if ((tensor.flags & SLANGML_FLAG_CONTIGUOUS) != 0u)
    {
        slangml_bda_store_f32(slangml_final_byte_address(tensor, int(linear), 4u), value);
        return;
    }

    uint indices[SLANGML_MAX_RANK];
    slangml_linear_to_indices32(tensor, linear, indices);
    int relative = slangml_indices_to_relative_offset32(tensor, indices);
    slangml_bda_store_f32(slangml_final_byte_address(tensor, relative, 4u), value);
}

float sml_unary_apply(uint op, float x)
{
    switch (op)
    {
    case 0u: return -x;
    case 1u: return abs(x);
    case 2u: return exp(x);
    case 3u: return log(x);
    case 4u: return sqrt(x);
    case 5u: return rsqrt(x);
    case 6u: return 1.0f / x;
    case 7u: return sin(x);
    case 8u: return cos(x);
    case 9u: return tanh(x);
    case 10u: return slangml_sigmoid(x);
    case 11u: return max(x, 0.0f);
    case 12u: return slangml_gelu_tanh(x);
    case 13u: return x * slangml_sigmoid(x);
    case 14u: return slangml_softplus(x, 1.0f, 20.0f);
    case 15u: return slangml_mish(x);
    default: return x;
    }
}

float sml_unary_apply_params(uint op, float x, ElementwiseParams* params)
{
    float alpha = params.alpha == 0.0f ? 0.01f : params.alpha;
    float beta = params.beta == 0.0f ? 1.0f : params.beta;
    switch (op)
    {
    case 16u: return x >= 0.0f ? x : alpha * x; // leaky_relu
    case 17u: return x >= 0.0f ? x : alpha * x; // prelu, alpha is supplied by host
    case 18u: return x >= 0.0f ? x : alpha * (exp(x) - 1.0f); // elu
    case 19u: return 1.050700987f * (x >= 0.0f ? x : 1.673263242f * (exp(x) - 1.0f)); // selu
    case 20u: return slangml_softsign(x);
    case 21u: return slangml_hardswish(x);
    case 22u: return slangml_hardsigmoid(x);
    case 23u: return slangml_gelu_exact(x);
    case 24u: return x * x;
    case 25u: return x > 0.0f ? 1.0f : (x < 0.0f ? -1.0f : 0.0f);
    case 26u: return clamp(x, params.alpha, beta);
    case 27u: return pow(x, beta);
    case 28u: return isnan(x) ? 1.0f : 0.0f;
    case 29u: return isinf(x) ? 1.0f : 0.0f;
    case 30u: return isnan(x) || isinf(x) ? params.alpha : x;
    case 31u: return slangml_log1p(x);
    case 32u: return exp(x) - 1.0f;
    case 33u: return floor(x);
    case 34u: return ceil(x);
    case 35u: return round(x);
    default: return sml_unary_apply(op, x);
    }
}

float sml_unary_derivative(uint op, float x, float upstream)
{
    switch (op)
    {
    case 0u: return -upstream;
    case 1u: return x > 0.0f ? upstream : (x < 0.0f ? -upstream : 0.0f);
    case 2u: return upstream * exp(x);
    case 3u: return upstream / x;
    case 4u: return upstream * 0.5f / sqrt(x);
    case 5u: return upstream * -0.5f * rsqrt(x) / x;
    case 6u: return -upstream / (x * x);
    case 9u: { float t = tanh(x); return upstream * (1.0f - t * t); }
    case 10u: { float s = slangml_sigmoid(x); return upstream * s * (1.0f - s); }
    case 11u: return x > 0.0f ? upstream : 0.0f;
    case 12u: { GeluTanhOp opImpl; return opImpl.derivative(x, upstream); }
    case 13u: { SiluOp opImpl; return opImpl.derivative(x, upstream); }
    case 14u: return upstream * slangml_sigmoid(x);
    case 15u:
    {
        float softplus = slangml_softplus(x, 1.0f, 20.0f);
        float t = tanh(softplus);
        return upstream * (t + x * slangml_sigmoid(x) * (1.0f - t * t));
    }
    default: return upstream;
    }
}

float sml_unary_derivative_params(uint op, float x, float upstream, ElementwiseParams* params)
{
    float alpha = params.alpha == 0.0f ? 0.01f : params.alpha;
    float beta = params.beta == 0.0f ? 1.0f : params.beta;
    switch (op)
    {
    case 16u: return upstream * (x >= 0.0f ? 1.0f : alpha);
    case 17u: return upstream * (x >= 0.0f ? 1.0f : alpha);
    case 18u: return upstream * (x >= 0.0f ? 1.0f : alpha * exp(x));
    case 19u: return upstream * 1.050700987f * (x >= 0.0f ? 1.0f : 1.673263242f * exp(x));
    case 20u: return upstream / ((1.0f + abs(x)) * (1.0f + abs(x)));
    case 21u: return upstream * (x <= -3.0f ? 0.0f : (x >= 3.0f ? 1.0f : x / 3.0f + 0.5f));
    case 22u: return x > -3.0f && x < 3.0f ? upstream / 6.0f : 0.0f;
    case 23u: return slangml_gelu_exact(x) * 0.0f + upstream * (0.5f * (1.0f + slangml_erf(x * SLANGML_INV_SQRT_2))
        + x * exp(-0.5f * x * x) * 0.3989422804014327f);
    case 24u: return upstream * 2.0f * x;
    case 25u: return 0.0f;
    case 26u: return x > params.alpha && x < beta ? upstream : 0.0f;
    case 27u: return upstream * beta * pow(x, beta - 1.0f);
    case 31u: return upstream / (1.0f + x);
    case 32u: return upstream * exp(x);
    default: return sml_unary_derivative(op, x, upstream);
    }
}

void sml_unary_fwd(uint op, uint3 dtid : SV_DispatchThreadID)
{
    ElementwiseParams* params = reinterpret<ElementwiseParams*>(smlElementwisePush.headerAddress);
    TensorDesc input = slangml_tensor_desc(params.header, 0u);
    TensorDesc output = slangml_tensor_desc(params.header, 1u);
    uint linear = params.header.dispatchBase.x + dtid.x;
    if (linear >= params.header.numel)
        return;
    sml_elementwise_store(output, linear, sml_unary_apply_params(op, sml_elementwise_load(input, linear), params));
}

void sml_unary_bwd(uint op, uint3 dtid : SV_DispatchThreadID)
{
    ElementwiseParams* params = reinterpret<ElementwiseParams*>(smlElementwisePush.headerAddress);
    TensorDesc input = slangml_tensor_desc(params.header, 0u);
    TensorDesc upstream = slangml_tensor_desc(params.header, 1u);
    TensorDesc output = slangml_tensor_desc(params.header, 2u);
    uint linear = params.header.dispatchBase.x + dtid.x;
    if (linear >= params.header.numel)
        return;
    float x = sml_elementwise_load(input, linear);
    float dy = sml_elementwise_load(upstream, linear);
    sml_elementwise_store(output, linear, sml_unary_derivative_params(op, x, dy, params));
}

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_elementwise_neg_fwd_f32_contig(uint3 dtid : SV_DispatchThreadID) { sml_unary_fwd(0u, dtid); }

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_elementwise_neg_bwd_f32_contig(uint3 dtid : SV_DispatchThreadID) { sml_unary_bwd(0u, dtid); }

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_elementwise_exp_fwd_f32_contig(uint3 dtid : SV_DispatchThreadID) { sml_unary_fwd(2u, dtid); }

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_elementwise_exp_bwd_f32_contig(uint3 dtid : SV_DispatchThreadID) { sml_unary_bwd(2u, dtid); }

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_elementwise_log_fwd_f32_contig(uint3 dtid : SV_DispatchThreadID) { sml_unary_fwd(3u, dtid); }

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_elementwise_log_bwd_f32_contig(uint3 dtid : SV_DispatchThreadID) { sml_unary_bwd(3u, dtid); }

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_elementwise_sqrt_fwd_f32_contig(uint3 dtid : SV_DispatchThreadID) { sml_unary_fwd(4u, dtid); }

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_elementwise_sqrt_bwd_f32_contig(uint3 dtid : SV_DispatchThreadID) { sml_unary_bwd(4u, dtid); }

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_elementwise_rsqrt_fwd_f32_contig(uint3 dtid : SV_DispatchThreadID) { sml_unary_fwd(5u, dtid); }

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_elementwise_rsqrt_bwd_f32_contig(uint3 dtid : SV_DispatchThreadID) { sml_unary_bwd(5u, dtid); }

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_elementwise_reciprocal_fwd_f32_contig(uint3 dtid : SV_DispatchThreadID) { sml_unary_fwd(6u, dtid); }

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_elementwise_reciprocal_bwd_f32_contig(uint3 dtid : SV_DispatchThreadID) { sml_unary_bwd(6u, dtid); }

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_elementwise_sin_fwd_f32_contig(uint3 dtid : SV_DispatchThreadID) { sml_unary_fwd(7u, dtid); }

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_elementwise_sin_bwd_f32_contig(uint3 dtid : SV_DispatchThreadID) { sml_unary_bwd(7u, dtid); }

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_elementwise_cos_fwd_f32_contig(uint3 dtid : SV_DispatchThreadID) { sml_unary_fwd(8u, dtid); }

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_elementwise_cos_bwd_f32_contig(uint3 dtid : SV_DispatchThreadID) { sml_unary_bwd(8u, dtid); }

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_elementwise_abs_fwd_f32_contig(uint3 dtid : SV_DispatchThreadID) { sml_unary_fwd(1u, dtid); }

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_elementwise_abs_bwd_f32_contig(uint3 dtid : SV_DispatchThreadID) { sml_unary_bwd(1u, dtid); }

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_elementwise_tanh_fwd_f32_contig(uint3 dtid : SV_DispatchThreadID) { sml_unary_fwd(9u, dtid); }

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_elementwise_tanh_bwd_f32_contig(uint3 dtid : SV_DispatchThreadID) { sml_unary_bwd(9u, dtid); }

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_elementwise_sigmoid_fwd_f32_contig(uint3 dtid : SV_DispatchThreadID) { sml_unary_fwd(10u, dtid); }

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_elementwise_sigmoid_bwd_f32_contig(uint3 dtid : SV_DispatchThreadID) { sml_unary_bwd(10u, dtid); }

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_elementwise_softplus_fwd_f32_contig(uint3 dtid : SV_DispatchThreadID) { sml_unary_fwd(14u, dtid); }

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_elementwise_softplus_bwd_f32_contig(uint3 dtid : SV_DispatchThreadID) { sml_unary_bwd(14u, dtid); }

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_elementwise_mish_fwd_f32_contig(uint3 dtid : SV_DispatchThreadID) { sml_unary_fwd(15u, dtid); }

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_elementwise_mish_bwd_f32_contig(uint3 dtid : SV_DispatchThreadID) { sml_unary_bwd(15u, dtid); }

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_elementwise_relu_fwd_f32_contig(uint3 dtid : SV_DispatchThreadID) { sml_unary_fwd(11u, dtid); }

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_elementwise_relu_bwd_f32_contig(uint3 dtid : SV_DispatchThreadID) { sml_unary_bwd(11u, dtid); }

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_elementwise_silu_fwd_f32_contig(uint3 dtid : SV_DispatchThreadID) { sml_unary_fwd(13u, dtid); }

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_elementwise_silu_bwd_f32_contig(uint3 dtid : SV_DispatchThreadID) { sml_unary_bwd(13u, dtid); }

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_elementwise_gelu_fwd_f32_contig(uint3 dtid : SV_DispatchThreadID) { sml_unary_fwd(12u, dtid); }

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_elementwise_gelu_bwd_f32_contig(uint3 dtid : SV_DispatchThreadID) { sml_unary_bwd(12u, dtid); }

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_elementwise_leaky_relu_fwd_f32_contig(uint3 dtid : SV_DispatchThreadID) { sml_unary_fwd(16u, dtid); }

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_elementwise_leaky_relu_bwd_f32_contig(uint3 dtid : SV_DispatchThreadID) { sml_unary_bwd(16u, dtid); }

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_elementwise_prelu_fwd_f32_contig(uint3 dtid : SV_DispatchThreadID) { sml_unary_fwd(17u, dtid); }

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_elementwise_prelu_bwd_f32_contig(uint3 dtid : SV_DispatchThreadID) { sml_unary_bwd(17u, dtid); }

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_elementwise_elu_fwd_f32_contig(uint3 dtid : SV_DispatchThreadID) { sml_unary_fwd(18u, dtid); }

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_elementwise_elu_bwd_f32_contig(uint3 dtid : SV_DispatchThreadID) { sml_unary_bwd(18u, dtid); }

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_elementwise_selu_fwd_f32_contig(uint3 dtid : SV_DispatchThreadID) { sml_unary_fwd(19u, dtid); }

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_elementwise_selu_bwd_f32_contig(uint3 dtid : SV_DispatchThreadID) { sml_unary_bwd(19u, dtid); }

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_elementwise_softsign_fwd_f32_contig(uint3 dtid : SV_DispatchThreadID) { sml_unary_fwd(20u, dtid); }

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_elementwise_softsign_bwd_f32_contig(uint3 dtid : SV_DispatchThreadID) { sml_unary_bwd(20u, dtid); }

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_elementwise_hardswish_fwd_f32_contig(uint3 dtid : SV_DispatchThreadID) { sml_unary_fwd(21u, dtid); }

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_elementwise_hardswish_bwd_f32_contig(uint3 dtid : SV_DispatchThreadID) { sml_unary_bwd(21u, dtid); }

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_elementwise_hardsigmoid_fwd_f32_contig(uint3 dtid : SV_DispatchThreadID) { sml_unary_fwd(22u, dtid); }

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_elementwise_hardsigmoid_bwd_f32_contig(uint3 dtid : SV_DispatchThreadID) { sml_unary_bwd(22u, dtid); }

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_elementwise_gelu_exact_fwd_f32_contig(uint3 dtid : SV_DispatchThreadID) { sml_unary_fwd(23u, dtid); }

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_elementwise_gelu_exact_bwd_f32_contig(uint3 dtid : SV_DispatchThreadID) { sml_unary_bwd(23u, dtid); }

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_elementwise_square_fwd_f32_contig(uint3 dtid : SV_DispatchThreadID) { sml_unary_fwd(24u, dtid); }

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_elementwise_square_bwd_f32_contig(uint3 dtid : SV_DispatchThreadID) { sml_unary_bwd(24u, dtid); }

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_elementwise_sign_fwd_f32_contig(uint3 dtid : SV_DispatchThreadID) { sml_unary_fwd(25u, dtid); }

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_elementwise_clamp_fwd_f32_contig(uint3 dtid : SV_DispatchThreadID) { sml_unary_fwd(26u, dtid); }

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_elementwise_clamp_bwd_f32_contig(uint3 dtid : SV_DispatchThreadID) { sml_unary_bwd(26u, dtid); }

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_elementwise_pow_scalar_fwd_f32_contig(uint3 dtid : SV_DispatchThreadID) { sml_unary_fwd(27u, dtid); }

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_elementwise_pow_scalar_bwd_f32_contig(uint3 dtid : SV_DispatchThreadID) { sml_unary_bwd(27u, dtid); }

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_elementwise_isnan_fwd_f32_contig(uint3 dtid : SV_DispatchThreadID) { sml_unary_fwd(28u, dtid); }

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_elementwise_isinf_fwd_f32_contig(uint3 dtid : SV_DispatchThreadID) { sml_unary_fwd(29u, dtid); }

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_elementwise_nan_to_num_fwd_f32_contig(uint3 dtid : SV_DispatchThreadID) { sml_unary_fwd(30u, dtid); }

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_elementwise_log1p_fwd_f32_contig(uint3 dtid : SV_DispatchThreadID) { sml_unary_fwd(31u, dtid); }

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_elementwise_log1p_bwd_f32_contig(uint3 dtid : SV_DispatchThreadID) { sml_unary_bwd(31u, dtid); }

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_elementwise_expm1_fwd_f32_contig(uint3 dtid : SV_DispatchThreadID) { sml_unary_fwd(32u, dtid); }

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_elementwise_expm1_bwd_f32_contig(uint3 dtid : SV_DispatchThreadID) { sml_unary_bwd(32u, dtid); }

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_elementwise_floor_fwd_f32_contig(uint3 dtid : SV_DispatchThreadID) { sml_unary_fwd(33u, dtid); }

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_elementwise_ceil_fwd_f32_contig(uint3 dtid : SV_DispatchThreadID) { sml_unary_fwd(34u, dtid); }

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_elementwise_round_fwd_f32_contig(uint3 dtid : SV_DispatchThreadID) { sml_unary_fwd(35u, dtid); }

float sml_apply_chain<Op>(Op op, float x)
    where Op : IElementwiseOp
{
    return op.apply(x);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_elementwise_fused_relu_silu_fwd_f32_contig(uint3 dtid : SV_DispatchThreadID)
{
    ElementwiseParams* params = reinterpret<ElementwiseParams*>(smlElementwisePush.headerAddress);
    TensorDesc input = slangml_tensor_desc(params.header, 0u);
    TensorDesc output = slangml_tensor_desc(params.header, 1u);
    uint linear = params.header.dispatchBase.x + dtid.x;
    if (linear >= params.header.numel)
        return;
    ReluSiluChain chain;
    sml_elementwise_store(output, linear, sml_apply_chain(chain, sml_elementwise_load(input, linear)));
}

float sml_binary_apply(uint op, float a, float b)
{
    switch (op)
    {
    case 0u: return a + b;
    case 1u: return a - b;
    case 2u: return a * b;
    case 3u: return a / b;
    case 4u: return min(a, b);
    case 5u: return max(a, b);
    default: return a + b;
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_elementwise_add_fwd_f32_broadcast(uint3 dtid : SV_DispatchThreadID)
{
    ElementwiseParams* params = reinterpret<ElementwiseParams*>(smlElementwisePush.headerAddress);
    TensorDesc a = slangml_tensor_desc(params.header, 0u);
    TensorDesc b = slangml_tensor_desc(params.header, 1u);
    TensorDesc output = slangml_tensor_desc(params.header, 2u);
    uint linear = params.header.dispatchBase.x + dtid.x;
    if (linear >= params.header.numel)
        return;
    uint indices[SLANGML_MAX_RANK];
    slangml_linear_to_indices32(output, linear, indices);
    float av = sml_elementwise_load_broadcast(a, params.outputRank, indices);
    float bv = sml_elementwise_load_broadcast(b, params.outputRank, indices);
    sml_elementwise_store(output, linear, av + bv);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_elementwise_mul_fwd_f32_broadcast(uint3 dtid : SV_DispatchThreadID)
{
    ElementwiseParams* params = reinterpret<ElementwiseParams*>(smlElementwisePush.headerAddress);
    TensorDesc a = slangml_tensor_desc(params.header, 0u);
    TensorDesc b = slangml_tensor_desc(params.header, 1u);
    TensorDesc output = slangml_tensor_desc(params.header, 2u);
    uint linear = params.header.dispatchBase.x + dtid.x;
    if (linear >= params.header.numel)
        return;
    uint indices[SLANGML_MAX_RANK];
    slangml_linear_to_indices32(output, linear, indices);
    float av = sml_elementwise_load_broadcast(a, params.outputRank, indices);
    float bv = sml_elementwise_load_broadcast(b, params.outputRank, indices);
    sml_elementwise_store(output, linear, av * bv);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_elementwise_sub_fwd_f32_broadcast(uint3 dtid : SV_DispatchThreadID)
{
    ElementwiseParams* params = reinterpret<ElementwiseParams*>(smlElementwisePush.headerAddress);
    TensorDesc a = slangml_tensor_desc(params.header, 0u);
    TensorDesc b = slangml_tensor_desc(params.header, 1u);
    TensorDesc output = slangml_tensor_desc(params.header, 2u);
    uint linear = params.header.dispatchBase.x + dtid.x;
    if (linear >= params.header.numel)
        return;
    uint indices[SLANGML_MAX_RANK];
    slangml_linear_to_indices32(output, linear, indices);
    sml_elementwise_store(output, linear, sml_elementwise_load_broadcast(a, params.outputRank, indices)
        - sml_elementwise_load_broadcast(b, params.outputRank, indices));
}

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_elementwise_div_fwd_f32_broadcast(uint3 dtid : SV_DispatchThreadID)
{
    ElementwiseParams* params = reinterpret<ElementwiseParams*>(smlElementwisePush.headerAddress);
    TensorDesc a = slangml_tensor_desc(params.header, 0u);
    TensorDesc b = slangml_tensor_desc(params.header, 1u);
    TensorDesc output = slangml_tensor_desc(params.header, 2u);
    uint linear = params.header.dispatchBase.x + dtid.x;
    if (linear >= params.header.numel)
        return;
    uint indices[SLANGML_MAX_RANK];
    slangml_linear_to_indices32(output, linear, indices);
    sml_elementwise_store(output, linear, sml_elementwise_load_broadcast(a, params.outputRank, indices)
        / sml_elementwise_load_broadcast(b, params.outputRank, indices));
}

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_elementwise_min_fwd_f32_broadcast(uint3 dtid : SV_DispatchThreadID)
{
    ElementwiseParams* params = reinterpret<ElementwiseParams*>(smlElementwisePush.headerAddress);
    TensorDesc a = slangml_tensor_desc(params.header, 0u);
    TensorDesc b = slangml_tensor_desc(params.header, 1u);
    TensorDesc output = slangml_tensor_desc(params.header, 2u);
    uint linear = params.header.dispatchBase.x + dtid.x;
    if (linear >= params.header.numel)
        return;
    uint indices[SLANGML_MAX_RANK];
    slangml_linear_to_indices32(output, linear, indices);
    float av = sml_elementwise_load_broadcast(a, params.outputRank, indices);
    float bv = sml_elementwise_load_broadcast(b, params.outputRank, indices);
    sml_elementwise_store(output, linear, min(av, bv));
}

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_elementwise_max_fwd_f32_broadcast(uint3 dtid : SV_DispatchThreadID)
{
    ElementwiseParams* params = reinterpret<ElementwiseParams*>(smlElementwisePush.headerAddress);
    TensorDesc a = slangml_tensor_desc(params.header, 0u);
    TensorDesc b = slangml_tensor_desc(params.header, 1u);
    TensorDesc output = slangml_tensor_desc(params.header, 2u);
    uint linear = params.header.dispatchBase.x + dtid.x;
    if (linear >= params.header.numel)
        return;
    uint indices[SLANGML_MAX_RANK];
    slangml_linear_to_indices32(output, linear, indices);
    float av = sml_elementwise_load_broadcast(a, params.outputRank, indices);
    float bv = sml_elementwise_load_broadcast(b, params.outputRank, indices);
    sml_elementwise_store(output, linear, max(av, bv));
}

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_elementwise_where_fwd_f32_broadcast(uint3 dtid : SV_DispatchThreadID)
{
    ElementwiseParams* params = reinterpret<ElementwiseParams*>(smlElementwisePush.headerAddress);
    TensorDesc condition = slangml_tensor_desc(params.header, 0u);
    TensorDesc onTrue = slangml_tensor_desc(params.header, 1u);
    TensorDesc onFalse = slangml_tensor_desc(params.header, 2u);
    TensorDesc output = slangml_tensor_desc(params.header, 3u);
    uint linear = params.header.dispatchBase.x + dtid.x;
    if (linear >= params.header.numel)
        return;
    uint indices[SLANGML_MAX_RANK];
    slangml_linear_to_indices32(output, linear, indices);
    float predicate = sml_elementwise_load_broadcast(condition, params.outputRank, indices);
    float tv = sml_elementwise_load_broadcast(onTrue, params.outputRank, indices);
    float fv = sml_elementwise_load_broadcast(onFalse, params.outputRank, indices);
    sml_elementwise_store(output, linear, predicate != 0.0f ? tv : fv);
}

void sml_binary_bwd(uint op, uint3 dtid : SV_DispatchThreadID)
{
    ElementwiseParams* params = reinterpret<ElementwiseParams*>(smlElementwisePush.headerAddress);
    TensorDesc a = slangml_tensor_desc(params.header, 0u);
    TensorDesc b = slangml_tensor_desc(params.header, 1u);
    TensorDesc upstream = slangml_tensor_desc(params.header, 2u);
    TensorDesc da = slangml_tensor_desc(params.header, 3u);
    TensorDesc db = slangml_tensor_desc(params.header, 4u);
    uint linear = params.header.dispatchBase.x + dtid.x;
    if (linear >= params.header.numel)
        return;
    float av = sml_elementwise_load(a, linear);
    float bv = sml_elementwise_load(b, linear);
    float dy = sml_elementwise_load(upstream, linear);
    float gradA = dy;
    float gradB = dy;
    if (op == 1u)
        gradB = -dy;
    else if (op == 2u)
    {
        gradA = dy * bv;
        gradB = dy * av;
    }
    else if (op == 3u)
    {
        gradA = dy / bv;
        gradB = -dy * av / (bv * bv);
    }
    sml_elementwise_store(da, linear, gradA);
    sml_elementwise_store(db, linear, gradB);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_elementwise_add_bwd_f32_same(uint3 dtid : SV_DispatchThreadID) { sml_binary_bwd(0u, dtid); }

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_elementwise_mul_bwd_f32_same(uint3 dtid : SV_DispatchThreadID) { sml_binary_bwd(2u, dtid); }
