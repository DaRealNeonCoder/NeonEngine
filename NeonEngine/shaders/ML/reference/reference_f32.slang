import slangml_common;

struct ReferenceParams
{
    KernelHeader header;
    uint rows;
    uint cols;
    float scale;
    float epsilon;
    uint mode;
    uint maskMode;
};

[[vk::push_constant]]
ConstantBuffer<KernelPushConstants> smlReferencePush;

float sml_reference_load(TensorDesc tensor, int relative)
{
    return slangml_bda_load_f32(slangml_final_byte_address(tensor, relative, 4u));
}

[shader("compute")]
[numthreads(64, 1, 1)]
void sml_reference_elementwise_add_fwd_f32(uint3 dtid : SV_DispatchThreadID)
{
    ReferenceParams* params = reinterpret<ReferenceParams*>(smlReferencePush.headerAddress);
    TensorDesc a = slangml_tensor_desc(params.header, 0u);
    TensorDesc b = slangml_tensor_desc(params.header, 1u);
    TensorDesc output = slangml_tensor_desc(params.header, 2u);
    uint linear = params.header.dispatchBase.x + dtid.x;
    if (linear >= params.header.numel)
        return;
    float av = sml_reference_load(a, int(linear));
    float bv = sml_reference_load(b, int(linear));
    slangml_bda_store_f32(slangml_final_byte_address(output, int(linear), 4u), av + bv);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void sml_reference_softmax_fwd_f32(uint3 dtid : SV_DispatchThreadID)
{
    ReferenceParams* params = reinterpret<ReferenceParams*>(smlReferencePush.headerAddress);
    TensorDesc input = slangml_tensor_desc(params.header, 0u);
    TensorDesc output = slangml_tensor_desc(params.header, 1u);
    uint row = params.header.dispatchBase.x + dtid.x;
    if (row >= params.rows)
        return;

    float rowMax = -asfloat(0x7F800000u);
    for (uint col = 0u; col < params.cols; ++col)
        rowMax = max(rowMax, sml_reference_load(input, int(row * params.cols + col)) * params.scale);
    float rowSum = 0.0f;
    for (uint col = 0u; col < params.cols; ++col)
        rowSum += exp(sml_reference_load(input, int(row * params.cols + col)) * params.scale - rowMax);
    for (uint col = 0u; col < params.cols; ++col)
    {
        float value = exp(sml_reference_load(input, int(row * params.cols + col)) * params.scale - rowMax) / rowSum;
        slangml_bda_store_f32(slangml_final_byte_address(output, int(row * params.cols + col), 4u), value);
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void sml_reference_norm_layernorm_fwd_f32(uint3 dtid : SV_DispatchThreadID)
{
    ReferenceParams* params = reinterpret<ReferenceParams*>(smlReferencePush.headerAddress);
    TensorDesc input = slangml_tensor_desc(params.header, 0u);
    TensorDesc gamma = slangml_tensor_desc(params.header, 1u);
    TensorDesc beta = slangml_tensor_desc(params.header, 2u);
    TensorDesc output = slangml_tensor_desc(params.header, 3u);
    uint row = params.header.dispatchBase.x + dtid.x;
    if (row >= params.rows)
        return;

    float mean = 0.0f;
    for (uint col = 0u; col < params.cols; ++col)
        mean += sml_reference_load(input, int(row * params.cols + col));
    mean /= float(params.cols);
    float variance = 0.0f;
    for (uint col = 0u; col < params.cols; ++col)
    {
        float centered = sml_reference_load(input, int(row * params.cols + col)) - mean;
        variance += centered * centered;
    }
    float invStd = rsqrt(variance / float(params.cols) + params.epsilon);
    for (uint col = 0u; col < params.cols; ++col)
    {
        float x = sml_reference_load(input, int(row * params.cols + col));
        float g = sml_reference_load(gamma, int(col));
        float b = sml_reference_load(beta, int(col));
        slangml_bda_store_f32(
            slangml_final_byte_address(output, int(row * params.cols + col), 4u),
            (x - mean) * invStd * g + b);
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void sml_reference_elementwise_unary_fwd_f32(uint3 dtid : SV_DispatchThreadID)
{
    ReferenceParams* params = reinterpret<ReferenceParams*>(smlReferencePush.headerAddress);
    TensorDesc input = slangml_tensor_desc(params.header, 0u);
    TensorDesc output = slangml_tensor_desc(params.header, 1u);
    uint linear = params.header.dispatchBase.x + dtid.x;
    if (linear >= params.header.numel)
        return;
    float x = sml_reference_load(input, int(linear));
    float value = x;
    switch (params.mode)
    {
    case 0u: value = -x; break;
    case 1u: value = abs(x); break;
    case 2u: value = exp(x); break;
    case 3u: value = log(x); break;
    case 4u: value = sqrt(x); break;
    case 5u: value = tanh(x); break;
    case 6u: value = slangml_sigmoid(x); break;
    case 7u: value = max(x, 0.0f); break;
    case 8u: value = slangml_gelu_tanh(x); break;
    case 9u: value = x * slangml_sigmoid(x); break;
    default: break;
    }
    slangml_bda_store_f32(slangml_final_byte_address(output, int(linear), 4u), value);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void sml_reference_reduce_fwd_f32(uint3 dtid : SV_DispatchThreadID)
{
    ReferenceParams* params = reinterpret<ReferenceParams*>(smlReferencePush.headerAddress);
    TensorDesc input = slangml_tensor_desc(params.header, 0u);
    TensorDesc output = slangml_tensor_desc(params.header, 1u);
    uint row = params.header.dispatchBase.x + dtid.x;
    if (row >= params.rows)
        return;

    float result = 0.0f;
    if (params.mode == 0u || params.mode == 1u)
    {
        for (uint col = 0u; col < params.cols; ++col)
            result += sml_reference_load(input, int(row * params.cols + col));
        if (params.mode == 1u)
            result /= float(params.cols);
    }
    else if (params.mode == 2u || params.mode == 3u)
    {
        result = params.mode == 2u ? asfloat(0x7F800000u) : -asfloat(0x7F800000u);
        for (uint col = 0u; col < params.cols; ++col)
        {
            float x = sml_reference_load(input, int(row * params.cols + col));
            result = params.mode == 2u ? min(result, x) : max(result, x);
        }
    }
    else if (params.mode == 4u)
    {
        float maximum = -asfloat(0x7F800000u);
        for (uint col = 0u; col < params.cols; ++col)
            maximum = max(maximum, sml_reference_load(input, int(row * params.cols + col)));
        float sum = 0.0f;
        for (uint col = 0u; col < params.cols; ++col)
            sum += exp(sml_reference_load(input, int(row * params.cols + col)) - maximum);
        result = maximum + log(sum);
    }
    else
    {
        float mean = 0.0f;
        for (uint col = 0u; col < params.cols; ++col)
            mean += sml_reference_load(input, int(row * params.cols + col));
        mean /= float(params.cols);
        for (uint col = 0u; col < params.cols; ++col)
        {
            float centered = sml_reference_load(input, int(row * params.cols + col)) - mean;
            result += centered * centered;
        }
        result /= float(max(1u, params.cols - 1u));
        if (params.mode == 6u)
            result = sqrt(result + params.epsilon);
    }
    slangml_bda_store_f32(slangml_final_byte_address(output, int(row), 4u), result);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void sml_reference_softmax_masked_fwd_f32(uint3 dtid : SV_DispatchThreadID)
{
    ReferenceParams* params = reinterpret<ReferenceParams*>(smlReferencePush.headerAddress);
    TensorDesc input = slangml_tensor_desc(params.header, 0u);
    TensorDesc mask = slangml_tensor_desc(params.header, 1u);
    TensorDesc output = slangml_tensor_desc(params.header, 2u);
    uint row = params.header.dispatchBase.x + dtid.x;
    if (row >= params.rows)
        return;
    float maximum = -asfloat(0x7F800000u);
    for (uint col = 0u; col < params.cols; ++col)
    {
        float maskValue = sml_reference_load(mask, int(row * params.cols + col));
        float value = maskValue != 0.0f ? sml_reference_load(input, int(row * params.cols + col)) * params.scale : -asfloat(0x7F800000u);
        maximum = max(maximum, value);
    }
    float sum = 0.0f;
    for (uint col = 0u; col < params.cols; ++col)
    {
        float maskValue = sml_reference_load(mask, int(row * params.cols + col));
        float value = maskValue != 0.0f ? sml_reference_load(input, int(row * params.cols + col)) * params.scale : -asfloat(0x7F800000u);
        sum += exp(value - maximum);
    }
    for (uint col = 0u; col < params.cols; ++col)
    {
        float maskValue = sml_reference_load(mask, int(row * params.cols + col));
        float value = maskValue != 0.0f ? sml_reference_load(input, int(row * params.cols + col)) * params.scale : -asfloat(0x7F800000u);
        slangml_bda_store_f32(slangml_final_byte_address(output, int(row * params.cols + col), 4u), exp(value - maximum) / sum);
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void sml_reference_norm_rmsnorm_fwd_f32(uint3 dtid : SV_DispatchThreadID)
{
    ReferenceParams* params = reinterpret<ReferenceParams*>(smlReferencePush.headerAddress);
    TensorDesc input = slangml_tensor_desc(params.header, 0u);
    TensorDesc gamma = slangml_tensor_desc(params.header, 1u);
    TensorDesc output = slangml_tensor_desc(params.header, 2u);
    uint row = params.header.dispatchBase.x + dtid.x;
    if (row >= params.rows)
        return;
    float squares = 0.0f;
    for (uint col = 0u; col < params.cols; ++col)
    {
        float x = sml_reference_load(input, int(row * params.cols + col));
        squares += x * x;
    }
    float invRms = rsqrt(squares / float(params.cols) + params.epsilon);
    for (uint col = 0u; col < params.cols; ++col)
    {
        float x = sml_reference_load(input, int(row * params.cols + col));
        float g = sml_reference_load(gamma, int(col));
        slangml_bda_store_f32(slangml_final_byte_address(output, int(row * params.cols + col), 4u), x * invRms * g);
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void sml_reference_norm_l2_normalize_fwd_f32(uint3 dtid : SV_DispatchThreadID)
{
    ReferenceParams* params = reinterpret<ReferenceParams*>(smlReferencePush.headerAddress);
    TensorDesc input = slangml_tensor_desc(params.header, 0u);
    TensorDesc output = slangml_tensor_desc(params.header, 1u);
    uint row = params.header.dispatchBase.x + dtid.x;
    if (row >= params.rows)
        return;
    float squares = 0.0f;
    for (uint col = 0u; col < params.cols; ++col)
    {
        float x = sml_reference_load(input, int(row * params.cols + col));
        squares += x * x;
    }
    float invNorm = rsqrt(squares + params.epsilon);
    for (uint col = 0u; col < params.cols; ++col)
    {
        float x = sml_reference_load(input, int(row * params.cols + col));
        slangml_bda_store_f32(slangml_final_byte_address(output, int(row * params.cols + col), 4u), x * invNorm);
    }
}

struct ReferenceGemmParams
{
    KernelHeader header;
    uint M;
    uint N;
    uint K;
    int strideA0;
    int strideA1;
    int strideB0;
    int strideB1;
    int strideC0;
    int strideC1;
    float alpha;
    float beta;
    uint reserved0;
    uint reserved1;
};

[shader("compute")]
[numthreads(64, 1, 1)]
void sml_reference_gemm_fwd_f32(uint3 dtid : SV_DispatchThreadID)
{
    ReferenceGemmParams* params = reinterpret<ReferenceGemmParams*>(smlReferencePush.headerAddress);
    TensorDesc a = slangml_tensor_desc(params.header, 0u);
    TensorDesc b = slangml_tensor_desc(params.header, 1u);
    TensorDesc c = slangml_tensor_desc(params.header, 2u);
    uint linear = params.header.dispatchBase.x + dtid.x;
    uint total = params.M * params.N;
    if (linear >= total)
        return;
    uint row = linear / params.N;
    uint col = linear - row * params.N;
    float value = 0.0f;
    for (uint k = 0u; k < params.K; ++k)
    {
        float av = slangml_bda_load_f32(slangml_final_byte_address(a, int(row) * params.strideA0 + int(k) * params.strideA1, 4u));
        float bv = slangml_bda_load_f32(slangml_final_byte_address(b, int(k) * params.strideB0 + int(col) * params.strideB1, 4u));
        value = fma(av, bv, value);
    }
    int outputOffset = int(row) * params.strideC0 + int(col) * params.strideC1;
    if (params.beta != 0.0f)
        value += params.beta * slangml_bda_load_f32(slangml_final_byte_address(c, outputOffset, 4u));
    slangml_bda_store_f32(slangml_final_byte_address(c, outputOffset, 4u), params.alpha * value);
}
