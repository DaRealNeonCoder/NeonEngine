import slangml_common;

struct GemvParams
{
    KernelHeader header;
    uint M;
    uint K;
    int strideA0;
    int strideA1;
    int strideB0;
    int strideB1;
    int strideC0;
    float alpha;
    float beta;
    uint reserved0;
    uint reserved1;
};

struct GroupedGemmProblem
{
    uint M;
    uint N;
    uint K;
    uint flags;
    uint64_t addressA;
    uint64_t addressB;
    uint64_t addressC;
    int strideA0;
    int strideA1;
    int strideB0;
    int strideB1;
    int strideC0;
    int strideC1;
    float alpha;
    float beta;
};

[[vk::push_constant]]
ConstantBuffer<KernelPushConstants> smlGemvPush;

groupshared float smlGemvScratch[256];

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_gemm_gemv_fwd_f32_portable(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID)
{
    GemvParams* p = reinterpret<GemvParams*>(smlGemvPush.headerAddress);
    TensorDesc a = slangml_tensor_desc(p.header, 0u);
    TensorDesc b = slangml_tensor_desc(p.header, 1u);
    TensorDesc c = slangml_tensor_desc(p.header, 2u);
    uint row = p.header.dispatchBase.x + gid.x;
    uint lane = tid.x;
    if (row >= p.M)
        return;
    float value = 0.0f;
    for (uint k = lane; k < p.K; k += 256u)
    {
        float av = slangml_bda_load_f32(slangml_final_byte_address(a, int(row) * p.strideA0 + int(k) * p.strideA1, 4u));
        float bv = slangml_bda_load_f32(slangml_final_byte_address(b, int(k) * p.strideB0, 4u));
        value = fma(av, bv, value);
    }
    value = slangml_workgroup_reduce_add_f32_256(value, smlGemvScratch, lane);
    if (lane == 0u)
    {
        int outputOffset = int(row) * p.strideC0;
        float result = p.alpha * value;
        if (p.beta != 0.0f)
            result += p.beta * slangml_bda_load_f32(slangml_final_byte_address(c, outputOffset, 4u));
        slangml_bda_store_f32(slangml_final_byte_address(c, outputOffset, 4u), result);
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void sml_gemm_grouped_fwd_f32_small(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID)
{
    GroupedGemmProblem* problems = reinterpret<GroupedGemmProblem*>(smlGemvPush.headerAddress + 48u);
    GroupedGemmProblem problem = problems[gid.x];
    float* a = reinterpret<float*>(problem.addressA);
    float* b = reinterpret<float*>(problem.addressB);
    float* c = reinterpret<float*>(problem.addressC);
    uint total = problem.M * problem.N;
    for (uint linear = tid.x; linear < total; linear += 64u)
    {
        uint m = linear / problem.N;
        uint n = linear - m * problem.N;
        float value = 0.0f;
        for (uint k = 0u; k < problem.K; ++k)
            value = fma(a[int(m) * problem.strideA0 + int(k) * problem.strideA1], b[int(k) * problem.strideB0 + int(n) * problem.strideB1], value);
        c[int(m) * problem.strideC0 + int(n) * problem.strideC1] = problem.alpha * value + problem.beta * c[int(m) * problem.strideC0 + int(n) * problem.strideC1];
    }
}
