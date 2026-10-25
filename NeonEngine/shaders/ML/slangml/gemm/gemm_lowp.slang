import slangml_common;

#ifndef SML_GEMM_BM
#define SML_GEMM_BM 16
#endif

#ifndef SML_GEMM_BN
#define SML_GEMM_BN 16
#endif

#ifndef SML_GEMM_BK
#define SML_GEMM_BK 16
#endif

// Keep this ABI byte-for-byte compatible with GemmParams in gemm_portable_f32.
// A and B use fp16 or bf16 storage; C, bias, and residual remain fp32.
struct LowpGemmParams
{
    KernelHeader header;
    uint M;
    uint N;
    uint K;
    uint batchCount;
    int strideA0;
    int strideA1;
    int strideB0;
    int strideB1;
    int strideC0;
    int strideC1;
    int strideABatch;
    int strideBBatch;
    int strideCBatch;
    float alpha;
    float beta;
    float outputScale;
    uint biasMode;
    uint splitCount;
    uint tileCountM;
    uint tileCountN;
    uint reserved;
};

public static const uint SML_GEMM_FLAG_TRANS_A = 1u << 0;
public static const uint SML_GEMM_FLAG_TRANS_B = 1u << 1;
public static const uint SML_GEMM_FLAG_BIAS = 1u << 2;
public static const uint SML_GEMM_FLAG_RESIDUAL = 1u << 3;
public static const uint SML_GEMM_FLAG_RELU = 1u << 4;
public static const uint SML_GEMM_FLAG_GELU = 1u << 5;
public static const uint SML_GEMM_FLAG_SILU = 1u << 6;
public static const uint SML_GEMM_FLAG_SCALE = 1u << 7;

[[vk::push_constant]]
ConstantBuffer<KernelPushConstants> smlLowpGemmPush;

groupshared float smlLowpGemmA[SML_GEMM_BM * SML_GEMM_BK];
groupshared float smlLowpGemmB[SML_GEMM_BK * SML_GEMM_BN];

uint sml_lowp_storage_bits(uint64_t address)
{
    // A 16-bit element can start in either half of a naturally aligned word.
    // The tensor base offset is included in address before this helper is called.
    uint word = slangml_bda_load_u32(address & ~uint64_t(3));
    uint shift = uint(address & uint64_t(2)) * 8u;
    return (word >> shift) & 0xFFFFu;
}

float sml_gemm_load_fp16(TensorDesc tensor, int relative)
{
    uint bits = sml_lowp_storage_bits(
        slangml_final_byte_address(tensor, relative, 2u));
    return slangml_fp16_to_float(bits);
}

float sml_gemm_load_bf16(TensorDesc tensor, int relative)
{
    uint bits = sml_lowp_storage_bits(
        slangml_final_byte_address(tensor, relative, 2u));
    return slangml_bf16_to_float(bits);
}

int sml_lowp_a_offset(LowpGemmParams* p, uint batch, uint m, uint k)
{
    bool transposed = (p.header.flags & SML_GEMM_FLAG_TRANS_A) != 0u;
    return (transposed
        ? int(k) * p.strideA0 + int(m) * p.strideA1
        : int(m) * p.strideA0 + int(k) * p.strideA1)
        + int(batch) * p.strideABatch;
}

int sml_lowp_b_offset(LowpGemmParams* p, uint batch, uint k, uint n)
{
    bool transposed = (p.header.flags & SML_GEMM_FLAG_TRANS_B) != 0u;
    return (transposed
        ? int(n) * p.strideB0 + int(k) * p.strideB1
        : int(k) * p.strideB0 + int(n) * p.strideB1)
        + int(batch) * p.strideBBatch;
}

float sml_lowp_load_a(TensorDesc tensor, LowpGemmParams* p, uint batch, uint m, uint k, uint dtype)
{
    int offset = sml_lowp_a_offset(p, batch, m, k);
    return dtype == SLANGML_DTYPE_BF16
        ? sml_gemm_load_bf16(tensor, offset)
        : sml_gemm_load_fp16(tensor, offset);
}

float sml_lowp_load_b(TensorDesc tensor, LowpGemmParams* p, uint batch, uint k, uint n, uint dtype)
{
    int offset = sml_lowp_b_offset(p, batch, k, n);
    return dtype == SLANGML_DTYPE_BF16
        ? sml_gemm_load_bf16(tensor, offset)
        : sml_gemm_load_fp16(tensor, offset);
}

float sml_lowp_load_c(TensorDesc tensor, LowpGemmParams* p, uint batch, uint m, uint n)
{
    int offset = int(m) * p.strideC0 + int(n) * p.strideC1 + int(batch) * p.strideCBatch;
    return slangml_bda_load_f32(slangml_final_byte_address(tensor, offset, 4u));
}

void sml_lowp_store_c(TensorDesc tensor, LowpGemmParams* p, uint batch, uint m, uint n, float value)
{
    int offset = int(m) * p.strideC0 + int(n) * p.strideC1 + int(batch) * p.strideCBatch;
    slangml_bda_store_f32(slangml_final_byte_address(tensor, offset, 4u), value);
}

float sml_lowp_load_bias(TensorDesc tensor, LowpGemmParams* p, uint m, uint n)
{
    int offset = p.biasMode == 1u ? int(n) : p.biasMode == 2u ? int(m) : 0;
    return slangml_bda_load_f32(slangml_final_byte_address(tensor, offset, 4u));
}

float sml_lowp_activation(float value, uint flags)
{
    if ((flags & SML_GEMM_FLAG_RELU) != 0u)
        value = max(value, 0.0f);
    if ((flags & SML_GEMM_FLAG_GELU) != 0u)
        value = slangml_gelu_tanh(value);
    if ((flags & SML_GEMM_FLAG_SILU) != 0u)
        value *= slangml_sigmoid(value);
    return value;
}

void sml_lowp_epilogue(
    TensorDesc bias,
    TensorDesc residual,
    TensorDesc output,
    LowpGemmParams* p,
    uint batch,
    uint m,
    uint n,
    inout float value)
{
    uint flags = p.header.flags;
    if ((flags & SML_GEMM_FLAG_BIAS) != 0u)
        value += sml_lowp_load_bias(bias, p, m, n);
    if ((flags & SML_GEMM_FLAG_RESIDUAL) != 0u)
        value += sml_lowp_load_c(residual, p, batch, m, n);
    value = sml_lowp_activation(value, flags);
    if ((flags & SML_GEMM_FLAG_SCALE) != 0u)
        value *= p.outputScale;
    sml_lowp_store_c(output, p, batch, m, n, value);
}

void sml_lowp_gemm_tile(uint3 gid, uint lane, uint storageDtype)
{
    LowpGemmParams* p = reinterpret<LowpGemmParams*>(smlLowpGemmPush.headerAddress);
    TensorDesc a = slangml_tensor_desc(p.header, 0u);
    TensorDesc b = slangml_tensor_desc(p.header, 1u);
    TensorDesc c = slangml_tensor_desc(p.header, 2u);
    TensorDesc bias = slangml_tensor_desc(p.header, 3u);
    TensorDesc residual = slangml_tensor_desc(p.header, 4u);

    uint tileX = gid.x;
    uint tileY = gid.y;
    uint batch = p.header.dispatchBase.z + gid.z;
    uint rowInTile = lane / SML_GEMM_BN;
    uint colInTile = lane - rowInTile * SML_GEMM_BN;
    uint row = p.header.dispatchBase.y + tileY * SML_GEMM_BM + rowInTile;
    uint col = p.header.dispatchBase.x + tileX * SML_GEMM_BN + colInTile;
    float accumulator = 0.0f;

    for (uint tileK = 0u; tileK < p.K; tileK += SML_GEMM_BK)
    {
        for (uint index = lane; index < SML_GEMM_BM * SML_GEMM_BK; index += 256u)
        {
            uint localM = index / SML_GEMM_BK;
            uint localK = index - localM * SML_GEMM_BK;
            uint globalM = p.header.dispatchBase.y + tileY * SML_GEMM_BM + localM;
            uint globalK = tileK + localK;
            smlLowpGemmA[index] = globalM < p.M && globalK < p.K
                ? sml_lowp_load_a(a, p, batch, globalM, globalK, storageDtype)
                : 0.0f;
        }
        for (uint index = lane; index < SML_GEMM_BK * SML_GEMM_BN; index += 256u)
        {
            uint localK = index / SML_GEMM_BN;
            uint localN = index - localK * SML_GEMM_BN;
            uint globalK = tileK + localK;
            uint globalN = p.header.dispatchBase.x + tileX * SML_GEMM_BN + localN;
            smlLowpGemmB[index] = globalK < p.K && globalN < p.N
                ? sml_lowp_load_b(b, p, batch, globalK, globalN, storageDtype)
                : 0.0f;
        }
        GroupMemoryBarrierWithGroupSync();
        if (row < p.M && col < p.N)
        {
            [ForceUnroll]
            for (uint k = 0u; k < SML_GEMM_BK; ++k)
                accumulator = fma(
                    smlLowpGemmA[rowInTile * SML_GEMM_BK + k],
                    smlLowpGemmB[k * SML_GEMM_BN + colInTile],
                    accumulator);
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (row < p.M && col < p.N)
    {
        float value = p.alpha * accumulator;
        if (p.beta != 0.0f)
            value += p.beta * sml_lowp_load_c(c, p, batch, row, col);
        sml_lowp_epilogue(bias, residual, c, p, batch, row, col, value);
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_gemm_gemm_fwd_f16_accum_f32(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID)
{
    sml_lowp_gemm_tile(gid, tid.x, SLANGML_DTYPE_FP16);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_gemm_gemm_fwd_bf16_accum_f32(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID)
{
    sml_lowp_gemm_tile(gid, tid.x, SLANGML_DTYPE_BF16);
}
