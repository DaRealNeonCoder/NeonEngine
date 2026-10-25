import slangml_common;

#ifndef SLANGML_ENABLE_COOPMAT
#define SLANGML_ENABLE_COOPMAT 0
#endif

// The public Slang 2025.17.2 compiler bundled with the local Vulkan SDK does
// not expose a stable BDA overload for CoopMat.Load/Store. Keep this entry
// point capability-gated and correct: it is the fp16/fp32 fallback used until
// the engine selects a compiler/device combination with a supported public
// BDA cooperative-matrix load interface. The common module still owns the
// actual CoopMat aliases and capability contract.
struct CoopGemmParams
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

[[vk::push_constant]]
ConstantBuffer<KernelPushConstants> smlCoopGemmPush;

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_gemm_gemm_fwd_f16_coopmat(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID)
{
    CoopGemmParams* p = reinterpret<CoopGemmParams*>(smlCoopGemmPush.headerAddress);
    TensorDesc aDesc = slangml_tensor_desc(p.header, 0u);
    TensorDesc bDesc = slangml_tensor_desc(p.header, 1u);
    TensorDesc cDesc = slangml_tensor_desc(p.header, 2u);
    half* a = reinterpret<half*>(aDesc.dataAddress);
    half* b = reinterpret<half*>(bDesc.dataAddress);
    float* c = reinterpret<float*>(cDesc.dataAddress);
    uint row = p.header.dispatchBase.y + gid.y * 16u + tid.x / 16u;
    uint col = p.header.dispatchBase.x + gid.x * 16u + tid.x % 16u;
    if (row >= p.M || col >= p.N)
        return;
    float value = 0.0f;
    for (uint k = 0u; k < p.K; ++k)
    {
        half av = a[int(row) * p.strideA0 + int(k) * p.strideA1];
        half bv = b[int(k) * p.strideB0 + int(col) * p.strideB1];
        value = fma(float(av), float(bv), value);
    }
    c[int(row) * p.strideC0 + int(col) * p.strideC1] = p.alpha * value
        + p.beta * c[int(row) * p.strideC0 + int(col) * p.strideC1];
}
