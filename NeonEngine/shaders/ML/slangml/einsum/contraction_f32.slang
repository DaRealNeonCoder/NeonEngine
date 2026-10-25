import slangml_common;

// These kernels intentionally expose the contractions used by pair/triangle
// blocks instead of hiding them behind a general rank-N einsum interpreter.
// The host canonicalizes descriptors and supplies hot strides for each logical
// index, so the inner loops contain only address arithmetic and FMAs.

struct MatrixContractionParams
{
    KernelHeader header;
    uint I;
    uint J;
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

struct TriangleContractionParams
{
    KernelHeader header;
    uint I;
    uint J;
    uint K;
    uint C;
    int strideA0;
    int strideA1;
    int strideA2;
    int strideB0;
    int strideB1;
    int strideB2;
    int strideO0;
    int strideO1;
    int strideO2;
    float alpha;
    float beta;
};

[[vk::push_constant]]
ConstantBuffer<KernelPushConstants> smlEinsumPush;

float sml_contraction_load(TensorDesc tensor, int relative)
{
    return slangml_bda_load_f32(slangml_final_byte_address(tensor, relative, 4u));
}

void sml_contraction_store(TensorDesc tensor, int relative, float value)
{
    slangml_bda_store_f32(slangml_final_byte_address(tensor, relative, 4u), value);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_einsum_contract_ik_kj_fwd_f32(uint3 dtid : SV_DispatchThreadID)
{
    MatrixContractionParams* p = reinterpret<MatrixContractionParams*>(smlEinsumPush.headerAddress);
    TensorDesc a = slangml_tensor_desc(p.header, 0u);
    TensorDesc b = slangml_tensor_desc(p.header, 1u);
    TensorDesc output = slangml_tensor_desc(p.header, 2u);
    uint linear = p.header.dispatchBase.x + dtid.x;
    uint total = p.I * p.J;
    if (linear >= total)
        return;

    uint i = linear / p.J;
    uint j = linear - i * p.J;
    float value = 0.0f;
    for (uint k = 0u; k < p.K; ++k)
    {
        float av = sml_contraction_load(a, int(i) * p.strideA0 + int(k) * p.strideA1);
        float bv = sml_contraction_load(b, int(k) * p.strideB0 + int(j) * p.strideB1);
        value = fma(av, bv, value);
    }

    int outputOffset = int(i) * p.strideC0 + int(j) * p.strideC1;
    value *= p.alpha;
    if (p.beta != 0.0f)
        value += p.beta * sml_contraction_load(output, outputOffset);
    sml_contraction_store(output, outputOffset, value);
}

float sml_triangle_outgoing_a(TensorDesc tensor, TriangleContractionParams* p, uint i, uint k, uint c)
{
    return sml_contraction_load(tensor,
        int(i) * p.strideA0 + int(k) * p.strideA1 + int(c) * p.strideA2);
}

float sml_triangle_outgoing_b(TensorDesc tensor, TriangleContractionParams* p, uint j, uint k, uint c)
{
    return sml_contraction_load(tensor,
        int(j) * p.strideB0 + int(k) * p.strideB1 + int(c) * p.strideB2);
}

float sml_triangle_incoming_a(TensorDesc tensor, TriangleContractionParams* p, uint i, uint k, uint c)
{
    return sml_contraction_load(tensor,
        int(k) * p.strideA0 + int(i) * p.strideA1 + int(c) * p.strideA2);
}

float sml_triangle_incoming_b(TensorDesc tensor, TriangleContractionParams* p, uint j, uint k, uint c)
{
    return sml_contraction_load(tensor,
        int(k) * p.strideB0 + int(j) * p.strideB1 + int(c) * p.strideB2);
}

void sml_triangle_store(
    TensorDesc output,
    TriangleContractionParams* p,
    uint i,
    uint j,
    uint c,
    float value)
{
    int outputOffset = int(i) * p.strideO0 + int(j) * p.strideO1 + int(c) * p.strideO2;
    value *= p.alpha;
    if (p.beta != 0.0f)
        value += p.beta * sml_contraction_load(output, outputOffset);
    sml_contraction_store(output, outputOffset, value);
}

void sml_triangle_outgoing_tile(uint3 dtid)
{
    TriangleContractionParams* p = reinterpret<TriangleContractionParams*>(smlEinsumPush.headerAddress);
    TensorDesc a = slangml_tensor_desc(p.header, 0u);
    TensorDesc b = slangml_tensor_desc(p.header, 1u);
    TensorDesc output = slangml_tensor_desc(p.header, 2u);
    uint linear = p.header.dispatchBase.x + dtid.x;
    uint total = p.I * p.J * p.C;
    if (linear >= total)
        return;

    uint c = linear % p.C;
    uint pair = linear / p.C;
    uint j = pair % p.J;
    uint i = pair / p.J;
    float value = 0.0f;
    for (uint k = 0u; k < p.K; ++k)
        value = fma(sml_triangle_outgoing_a(a, p, i, k, c),
                    sml_triangle_outgoing_b(b, p, j, k, c), value);
    sml_triangle_store(output, p, i, j, c, value);
}

void sml_triangle_incoming_tile(uint3 dtid)
{
    TriangleContractionParams* p = reinterpret<TriangleContractionParams*>(smlEinsumPush.headerAddress);
    TensorDesc a = slangml_tensor_desc(p.header, 0u);
    TensorDesc b = slangml_tensor_desc(p.header, 1u);
    TensorDesc output = slangml_tensor_desc(p.header, 2u);
    uint linear = p.header.dispatchBase.x + dtid.x;
    uint total = p.I * p.J * p.C;
    if (linear >= total)
        return;

    uint c = linear % p.C;
    uint pair = linear / p.C;
    uint j = pair % p.J;
    uint i = pair / p.J;
    float value = 0.0f;
    for (uint k = 0u; k < p.K; ++k)
        value = fma(sml_triangle_incoming_a(a, p, i, k, c),
                    sml_triangle_incoming_b(b, p, j, k, c), value);
    sml_triangle_store(output, p, i, j, c, value);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_einsum_triangle_mul_outgoing_fwd_f32(uint3 dtid : SV_DispatchThreadID)
{
    sml_triangle_outgoing_tile(dtid);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_einsum_triangle_mul_incoming_fwd_f32(uint3 dtid : SV_DispatchThreadID)
{
    sml_triangle_incoming_tile(dtid);
}
