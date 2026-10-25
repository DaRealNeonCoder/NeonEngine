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

#ifndef SML_GEMM_STAGES
#define SML_GEMM_STAGES 1
#endif

public static const uint SML_GEMM_FLAG_TRANS_A = 1u << 0;
public static const uint SML_GEMM_FLAG_TRANS_B = 1u << 1;
public static const uint SML_GEMM_FLAG_BIAS = 1u << 2;
public static const uint SML_GEMM_FLAG_RESIDUAL = 1u << 3;
public static const uint SML_GEMM_FLAG_RELU = 1u << 4;
public static const uint SML_GEMM_FLAG_GELU = 1u << 5;
public static const uint SML_GEMM_FLAG_SILU = 1u << 6;
public static const uint SML_GEMM_FLAG_SCALE = 1u << 7;
public static const uint SML_GEMM_FLAG_SPLIT_K = 1u << 8;

public struct GemmParams
{
    public KernelHeader header;
    public uint M;
    public uint N;
    public uint K;
    public uint batchCount;
    public int strideA0;
    public int strideA1;
    public int strideB0;
    public int strideB1;
    public int strideC0;
    public int strideC1;
    public int strideABatch;
    public int strideBBatch;
    public int strideCBatch;
    public float alpha;
    public float beta;
    public float outputScale;
    public uint biasMode;
    public uint splitCount;
    public uint tileCountM;
    public uint tileCountN;
    public uint reserved;
};

[[vk::push_constant]]
ConstantBuffer<KernelPushConstants> smlGemmPush;

groupshared float smlGemmA[SML_GEMM_BM * SML_GEMM_BK];
groupshared float smlGemmB[SML_GEMM_BK * SML_GEMM_BN];

float sml_gemm_load_a(TensorDesc tensor, GemmParams* params, uint batch, uint m, uint k)
{
    bool transposed = (params.header.flags & SML_GEMM_FLAG_TRANS_A) != 0u;
    int logical = transposed
        ? int(k) * params.strideA0 + int(m) * params.strideA1 + int(batch) * params.strideABatch
        : int(m) * params.strideA0 + int(k) * params.strideA1 + int(batch) * params.strideABatch;
    return slangml_bda_load_f32(slangml_final_byte_address(tensor, logical, 4u));
}

float sml_gemm_load_b(TensorDesc tensor, GemmParams* params, uint batch, uint k, uint n)
{
    bool transposed = (params.header.flags & SML_GEMM_FLAG_TRANS_B) != 0u;
    int logical = transposed
        ? int(n) * params.strideB0 + int(k) * params.strideB1 + int(batch) * params.strideBBatch
        : int(k) * params.strideB0 + int(n) * params.strideB1 + int(batch) * params.strideBBatch;
    return slangml_bda_load_f32(slangml_final_byte_address(tensor, logical, 4u));
}

float sml_gemm_load_c(TensorDesc tensor, GemmParams* params, uint batch, uint m, uint n)
{
    int logical = int(m) * params.strideC0 + int(n) * params.strideC1 + int(batch) * params.strideCBatch;
    return slangml_bda_load_f32(slangml_final_byte_address(tensor, logical, 4u));
}

void sml_gemm_store_c(TensorDesc tensor, GemmParams* params, uint batch, uint m, uint n, float value)
{
    int logical = int(m) * params.strideC0 + int(n) * params.strideC1 + int(batch) * params.strideCBatch;
    slangml_bda_store_f32(slangml_final_byte_address(tensor, logical, 4u), value);
}

float sml_gemm_load_bias(TensorDesc tensor, GemmParams* params, uint m, uint n)
{
    if (params.biasMode == 1u)
        return slangml_bda_load_f32(slangml_final_byte_address(tensor, int(n), 4u));
    if (params.biasMode == 2u)
        return slangml_bda_load_f32(slangml_final_byte_address(tensor, int(m), 4u));
    return slangml_bda_load_f32(slangml_final_byte_address(tensor, 0, 4u));
}

float sml_gemm_activation(float value, uint flags)
{
    if ((flags & SML_GEMM_FLAG_RELU) != 0u)
        value = max(value, 0.0f);
    if ((flags & SML_GEMM_FLAG_GELU) != 0u)
        value = slangml_gelu_tanh(value);
    if ((flags & SML_GEMM_FLAG_SILU) != 0u)
        value *= slangml_sigmoid(value);
    return value;
}

void sml_gemm_epilogue(
    TensorDesc bias,
    TensorDesc residual,
    TensorDesc output,
    GemmParams* params,
    uint batch,
    uint m,
    uint n,
    inout float value)
{
    uint flags = params.header.flags;
    if ((flags & SML_GEMM_FLAG_BIAS) != 0u)
        value += sml_gemm_load_bias(bias, params, m, n);
    if ((flags & SML_GEMM_FLAG_RESIDUAL) != 0u)
        value += sml_gemm_load_c(residual, params, batch, m, n);
    value = sml_gemm_activation(value, flags);
    if ((flags & SML_GEMM_FLAG_SCALE) != 0u)
        value *= params.outputScale;
    sml_gemm_store_c(output, params, batch, m, n, value);
}

void sml_gemm_tile(
    uint3 gid,
    uint lane,
    bool writeOutput,
    uint splitIndex)
{
    GemmParams* params = reinterpret<GemmParams*>(smlGemmPush.headerAddress);
    TensorDesc a = slangml_tensor_desc(params.header, 0u);
    TensorDesc b = slangml_tensor_desc(params.header, 1u);
    TensorDesc c = slangml_tensor_desc(params.header, 2u);
    TensorDesc bias = slangml_tensor_desc(params.header, 3u);
    TensorDesc residual = slangml_tensor_desc(params.header, 4u);

    uint tileX = gid.x;
    uint tileY = gid.y;
    uint batch = params.header.dispatchBase.z + gid.z;
    uint rowInTile = lane / SML_GEMM_BN;
    uint colInTile = lane - rowInTile * SML_GEMM_BN;
    uint row = params.header.dispatchBase.y + tileY * SML_GEMM_BM + rowInTile;
    uint col = params.header.dispatchBase.x + tileX * SML_GEMM_BN + colInTile;
    float accumulator = 0.0f;

    uint kBegin = 0u;
    uint kEnd = params.K;
    if (splitIndex != 0xFFFFFFFFu)
    {
        kBegin = (params.K * splitIndex) / params.splitCount;
        kEnd = (params.K * (splitIndex + 1u)) / params.splitCount;
    }

    for (uint tileK = kBegin; tileK < kEnd; tileK += SML_GEMM_BK)
    {
        for (uint index = lane; index < SML_GEMM_BM * SML_GEMM_BK; index += 256u)
        {
            uint localM = index / SML_GEMM_BK;
            uint localK = index - localM * SML_GEMM_BK;
            uint globalM = params.header.dispatchBase.y + tileY * SML_GEMM_BM + localM;
            uint globalK = tileK + localK;
            smlGemmA[index] = globalM < params.M && globalK < kEnd
                ? sml_gemm_load_a(a, params, batch, globalM, globalK)
                : 0.0f;
        }
        for (uint index = lane; index < SML_GEMM_BK * SML_GEMM_BN; index += 256u)
        {
            uint localK = index / SML_GEMM_BN;
            uint localN = index - localK * SML_GEMM_BN;
            uint globalK = tileK + localK;
            uint globalN = params.header.dispatchBase.x + tileX * SML_GEMM_BN + localN;
            smlGemmB[index] = globalK < kEnd && globalN < params.N
                ? sml_gemm_load_b(b, params, batch, globalK, globalN)
                : 0.0f;
        }
        GroupMemoryBarrierWithGroupSync();
        if (row < params.M && col < params.N)
        {
            [ForceUnroll]
            for (uint k = 0u; k < SML_GEMM_BK; ++k)
                accumulator = fma(smlGemmA[rowInTile * SML_GEMM_BK + k], smlGemmB[k * SML_GEMM_BN + colInTile], accumulator);
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (row < params.M && col < params.N && writeOutput)
    {
        float value = params.alpha * accumulator;
        if (params.beta != 0.0f)
            value += params.beta * sml_gemm_load_c(c, params, batch, row, col);
        sml_gemm_epilogue(bias, residual, c, params, batch, row, col, value);
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_gemm_gemm_fwd_f32_portable(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID)
{
    sml_gemm_tile(gid, tid.x, true, 0xFFFFFFFFu);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_gemm_gemm_bwd_f32_portable(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID)
{
    // The engine supplies transposed logical views through the same GEMM
    // parameters: dA = dC * B^T or dB = A^T * dC.
    sml_gemm_tile(gid, tid.x, true, 0xFFFFFFFFu);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_gemm_gemm_fwd_f32_epilogue(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID)
{
    sml_gemm_tile(gid, tid.x, true, 0xFFFFFFFFu);
}
