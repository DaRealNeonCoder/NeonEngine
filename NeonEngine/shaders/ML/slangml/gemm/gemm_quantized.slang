import slangml_common;

struct QuantGemmParams
{
    KernelHeader header;
    uint M;
    uint N;
    uint K;
    uint groupSize;
    int strideA0;
    int strideA1;
    int strideW0;
    int strideW1;
    int strideC0;
    int strideC1;
    float alpha;
    uint quantMode;
    uint zeroPointMode;
    uint reserved0;
    uint reserved1;
};

[[vk::push_constant]]
ConstantBuffer<KernelPushConstants> smlQuantGemmPush;

int sml_quant_weight(TensorDesc weight, QuantGemmParams* p, uint k, uint n)
{
    if (p.quantMode == 0u)
    {
        int offset = int(k) * p.strideW0 + int(n) * p.strideW1;
        uint raw = slangml_bda_load_u32(slangml_final_byte_address(weight, offset, 1u));
        return int(raw << 24u) >> 24;
    }
    int logical = int(k) * p.strideW0 + int(n) * p.strideW1;
    uint packed = slangml_bda_load_u32(slangml_final_byte_address(weight, logical / 8, 4u));
    return slangml_unpack_signed_int4(packed, uint(logical) & 7u);
}

float sml_quant_scale(TensorDesc scale, QuantGemmParams* p, uint k, uint n)
{
    uint group = p.groupSize == 0u ? 0u : k / p.groupSize;
    uint index = p.groupSize == 0u ? n : group * p.N + n;
    return slangml_bda_load_f32(slangml_final_byte_address(scale, int(index), 4u));
}

float sml_quant_zero(TensorDesc zero, QuantGemmParams* p, uint k, uint n)
{
    if (p.zeroPointMode == 0u)
        return 0.0f;
    uint group = p.groupSize == 0u ? 0u : k / p.groupSize;
    uint index = p.groupSize == 0u ? n : group * p.N + n;
    return slangml_bda_load_f32(slangml_final_byte_address(zero, int(index), 4u));
}

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_gemm_quantized_fwd_f32_i8(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID)
{
    QuantGemmParams* p = reinterpret<QuantGemmParams*>(smlQuantGemmPush.headerAddress);
    TensorDesc a = slangml_tensor_desc(p.header, 0u);
    TensorDesc weight = slangml_tensor_desc(p.header, 1u);
    TensorDesc scale = slangml_tensor_desc(p.header, 2u);
    TensorDesc zero = slangml_tensor_desc(p.header, 3u);
    TensorDesc c = slangml_tensor_desc(p.header, 4u);
    uint row = p.header.dispatchBase.y + gid.y * 16u + tid.x / 16u;
    uint col = p.header.dispatchBase.x + gid.x * 16u + tid.x % 16u;
    if (row >= p.M || col >= p.N)
        return;
    float value = 0.0f;
    for (uint k = 0u; k < p.K; ++k)
    {
        float av = slangml_bda_load_f32(slangml_final_byte_address(a, int(row) * p.strideA0 + int(k) * p.strideA1, 4u));
        float w = (float(sml_quant_weight(weight, p, k, col)) - sml_quant_zero(zero, p, k, col)) * sml_quant_scale(scale, p, k, col);
        value = fma(av, w, value);
    }
    slangml_bda_store_f32(slangml_final_byte_address(c, int(row) * p.strideC0 + int(col) * p.strideC1, 4u), p.alpha * value);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_gemm_quantized_fwd_f32_i4(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID)
{
    sml_gemm_quantized_fwd_f32_i8(gid, tid);
}
