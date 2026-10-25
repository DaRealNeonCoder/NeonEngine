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

struct SplitKGemmParams
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
    uint splitCount;
    uint tileCountM;
    uint tileCountN;
    uint reserved;
};

[[vk::push_constant]]
ConstantBuffer<KernelPushConstants> smlSplitKGemmPush;

groupshared float smlSplitKA[SML_GEMM_BM * SML_GEMM_BK];
groupshared float smlSplitKB[SML_GEMM_BK * SML_GEMM_BN];

float sml_splitk_load_a(TensorDesc t, SplitKGemmParams* p, uint batch, uint m, uint k)
{
    int offset = int(m) * p.strideA0 + int(k) * p.strideA1 + int(batch) * p.strideABatch;
    return slangml_bda_load_f32(slangml_final_byte_address(t, offset, 4u));
}

float sml_splitk_load_b(TensorDesc t, SplitKGemmParams* p, uint batch, uint k, uint n)
{
    int offset = int(k) * p.strideB0 + int(n) * p.strideB1 + int(batch) * p.strideBBatch;
    return slangml_bda_load_f32(slangml_final_byte_address(t, offset, 4u));
}

uint64_t sml_splitk_workspace_offset(SplitKGemmParams* p, uint split, uint batch, uint row, uint col)
{
    uint64_t element = ((uint64_t(split) * uint64_t(p.batchCount) + uint64_t(batch))
        * uint64_t(p.M) + uint64_t(row)) * uint64_t(p.N) + uint64_t(col);
    return smlSplitKGemmPush.workspaceAddress + element * uint64_t(4u);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_gemm_gemm_splitk_fwd_f32_workspace(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID)
{
    SplitKGemmParams* p = reinterpret<SplitKGemmParams*>(smlSplitKGemmPush.headerAddress);
    TensorDesc a = slangml_tensor_desc(p.header, 0u);
    TensorDesc b = slangml_tensor_desc(p.header, 1u);
    uint split = gid.z % p.splitCount;
    uint batch = p.header.dispatchBase.z + gid.z / p.splitCount;
    uint row = p.header.dispatchBase.y + gid.y * SML_GEMM_BM + tid.x / SML_GEMM_BN;
    uint col = p.header.dispatchBase.x + gid.x * SML_GEMM_BN + tid.x % SML_GEMM_BN;
    uint begin = (p.K * split) / p.splitCount;
    uint end = (p.K * (split + 1u)) / p.splitCount;
    float value = 0.0f;
    for (uint tileK = begin; tileK < end; tileK += SML_GEMM_BK)
    {
        for (uint index = tid.x; index < SML_GEMM_BM * SML_GEMM_BK; index += 256u)
        {
            uint lm = index / SML_GEMM_BK;
            uint lk = index - lm * SML_GEMM_BK;
            uint gm = p.header.dispatchBase.y + gid.y * SML_GEMM_BM + lm;
            uint gk = tileK + lk;
            smlSplitKA[index] = gm < p.M && gk < end ? sml_splitk_load_a(a, p, batch, gm, gk) : 0.0f;
        }
        for (uint index = tid.x; index < SML_GEMM_BK * SML_GEMM_BN; index += 256u)
        {
            uint lk = index / SML_GEMM_BN;
            uint ln = index - lk * SML_GEMM_BN;
            uint gk = tileK + lk;
            uint gn = p.header.dispatchBase.x + gid.x * SML_GEMM_BN + ln;
            smlSplitKB[index] = gk < end && gn < p.N ? sml_splitk_load_b(b, p, batch, gk, gn) : 0.0f;
        }
        GroupMemoryBarrierWithGroupSync();
        if (row < p.M && col < p.N)
        {
            [ForceUnroll]
            for (uint k = 0u; k < SML_GEMM_BK; ++k)
                value = fma(smlSplitKA[(tid.x / SML_GEMM_BN) * SML_GEMM_BK + k], smlSplitKB[k * SML_GEMM_BN + tid.x % SML_GEMM_BN], value);
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (row < p.M && col < p.N)
        slangml_bda_store_f32(sml_splitk_workspace_offset(p, split, batch, row, col), p.alpha * value);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_gemm_gemm_splitk_reduce_f32_workspace(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID)
{
    SplitKGemmParams* p = reinterpret<SplitKGemmParams*>(smlSplitKGemmPush.headerAddress);
    TensorDesc c = slangml_tensor_desc(p.header, 0u);
    uint batch = p.header.dispatchBase.z + gid.z;
    uint row = p.header.dispatchBase.y + gid.y * SML_GEMM_BM + tid.x / SML_GEMM_BN;
    uint col = p.header.dispatchBase.x + gid.x * SML_GEMM_BN + tid.x % SML_GEMM_BN;
    if (row >= p.M || col >= p.N || batch >= p.batchCount)
        return;
    float result = 0.0f;
    for (uint split = 0u; split < p.splitCount; ++split)
        result += slangml_bda_load_f32(sml_splitk_workspace_offset(p, split, batch, row, col));
    int outputOffset = int(row) * p.strideC0 + int(col) * p.strideC1 + int(batch) * p.strideCBatch;
    if (p.beta != 0.0f)
        result += p.beta * slangml_bda_load_f32(slangml_final_byte_address(c, outputOffset, 4u));
    slangml_bda_store_f32(slangml_final_byte_address(c, outputOffset, 4u), result);
}
