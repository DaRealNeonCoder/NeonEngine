import slangml_common;

// Backward pass for outer product mean [SI Alg 9].  Forward is
// outer_product_mean_f32.slang:
//
//   o_p[c1*32 + c2] = (1/S) sum_s a[s,i,c1] b[s,j,c2]     p = (i, j)
//   z_p[z]          = zIn_p[z] + bias[z] + sum_cc w[z,cc] o_p[cc]
//
// Given dZ_p (the gradient of z_p; the identity path to zIn is the caller's job):
//
//   dO_p[cc]    = sum_z w[z,cc] dZ_p[z]
//   dbias[z]    = sum_p dZ_p[z]
//   dW[z,cc]    = sum_p dZ_p[z] o_p[cc]
//   da[s,i,c1]  = (1/S) sum_j sum_c2 dO_{ij}[c1*32+c2] b[s,j,c2]
//   db[s,j,c2]  = (1/S) sum_i sum_c1 dO_{ij}[c1*32+c2] a[s,i,c1]
//
// Entries, in order:
//   sml_msa_outer_product_mean_bwd_f32_pairs  workgroup per pair (256): writes oWs, dOWs
//   sml_msa_outer_product_mean_bwd_f32_dbias  thread per z
//   sml_msa_outer_product_mean_bwd_f32_dw     thread per (z, cc)
//   sml_msa_outer_product_mean_bwd_f32_da     thread per (s, i, c1)
//   sml_msa_outer_product_mean_bwd_f32_db     thread per (s, j, c2)
//
// Workspace: oWs and dOWs are [N*N, 1024] fp32, which is N^2 * 4 KiB.  At N = 384
// that is 604 MiB each, so this version is for correctness testing.  The production
// version should process pairs in chunks and replace dW with a GEMM over the chunk.
//
// Layout (fp32, canonical):
//   a, b, da, db : [S, N, 32]     index (s*N + i)*32 + c
//   w            : [128, 1024]    index z*1024 + cc
//   bias, dbias  : [128]
//   zIn-free:      dZ : [N, N, 128]  index p*128 + z
//   oWs, dOWs    : [N*N, 1024]    index p*1024 + cc
//   dW           : [128, 1024]

static const uint SML_OPM_C = 32u;
static const uint SML_OPM_CC = 1024u;
static const uint SML_OPM_CZ = 128u;

struct OuterProductMeanBwdParams
{
    KernelHeader header;
    uint S;
    uint N;
    uint reserved0;
    uint reserved1;
};

[[vk::push_constant]]
ConstantBuffer<KernelPushConstants> smlOuterProductBwdPush;

float sml_opmb_load(TensorDesc t, uint index)
{
    return slangml_bda_load_f32(slangml_final_byte_address(t, int(index), 4u));
}

void sml_opmb_store(TensorDesc t, uint index, float value)
{
    slangml_bda_store_f32(slangml_final_byte_address(t, int(index), 4u), value);
}

// Workgroup per pair p = (i, j).  Recomputes o_p into oWs and computes dO_p into dOWs.
[shader("compute")]
[numthreads(256, 1, 1)]
void sml_msa_outer_product_mean_bwd_f32_pairs(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID)
{
    OuterProductMeanBwdParams* p = reinterpret<OuterProductMeanBwdParams*>(smlOuterProductBwdPush.headerAddress);
    TensorDesc a = slangml_tensor_desc(p.header, 0u);
    TensorDesc b = slangml_tensor_desc(p.header, 1u);
    TensorDesc w = slangml_tensor_desc(p.header, 2u);
    TensorDesc dZ = slangml_tensor_desc(p.header, 3u);
    TensorDesc oWs = slangml_tensor_desc(p.header, 4u);
    TensorDesc dOWs = slangml_tensor_desc(p.header, 5u);

    uint pairIndex = p.header.dispatchBase.x + gid.x;
    if (pairIndex >= p.N * p.N)
        return;
    uint i = pairIndex / p.N;
    uint j = pairIndex - i * p.N;
    float invS = 1.0f / float(p.S);

    for (uint cc = tid.x; cc < SML_OPM_CC; cc += 256u)
    {
        uint c1 = cc / SML_OPM_C;
        uint c2 = cc - c1 * SML_OPM_C;
        float acc = 0.0f;
        for (uint s = 0u; s < p.S; ++s)
        {
            acc = fma(sml_opmb_load(a, (s * p.N + i) * SML_OPM_C + c1),
                      sml_opmb_load(b, (s * p.N + j) * SML_OPM_C + c2), acc);
        }
        sml_opmb_store(oWs, pairIndex * SML_OPM_CC + cc, acc * invS);
    }

    // dO[cc] = sum_z w[z, cc] dZ[z].  Threads own consecutive cc, so w reads coalesce.
    uint dzBase = pairIndex * SML_OPM_CZ;
    for (uint cc = tid.x; cc < SML_OPM_CC; cc += 256u)
    {
        float acc = 0.0f;
        for (uint z = 0u; z < SML_OPM_CZ; ++z)
            acc = fma(sml_opmb_load(w, z * SML_OPM_CC + cc), sml_opmb_load(dZ, dzBase + z), acc);
        sml_opmb_store(dOWs, pairIndex * SML_OPM_CC + cc, acc);
    }
}

[shader("compute")]
[numthreads(128, 1, 1)]
void sml_msa_outer_product_mean_bwd_f32_dbias(uint3 dtid : SV_DispatchThreadID)
{
    OuterProductMeanBwdParams* p = reinterpret<OuterProductMeanBwdParams*>(smlOuterProductBwdPush.headerAddress);
    TensorDesc dZ = slangml_tensor_desc(p.header, 0u);
    TensorDesc dbias = slangml_tensor_desc(p.header, 1u);

    uint z = (p.header.dispatchBase.x + dtid.x);
    if (z >= SML_OPM_CZ)
        return;
    uint pairs = p.N * p.N;
    float acc = 0.0f;
    for (uint pIdx = 0u; pIdx < pairs; ++pIdx)
        acc += sml_opmb_load(dZ, pIdx * SML_OPM_CZ + z);
    sml_opmb_store(dbias, z, acc);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_msa_outer_product_mean_bwd_f32_dw(uint3 dtid : SV_DispatchThreadID)
{
    OuterProductMeanBwdParams* p = reinterpret<OuterProductMeanBwdParams*>(smlOuterProductBwdPush.headerAddress);
    TensorDesc dZ = slangml_tensor_desc(p.header, 0u);
    TensorDesc oWs = slangml_tensor_desc(p.header, 1u);
    TensorDesc dW = slangml_tensor_desc(p.header, 2u);

    uint idx = (p.header.dispatchBase.x + dtid.x);
    if (idx >= SML_OPM_CZ * SML_OPM_CC)
        return;
    uint z = idx / SML_OPM_CC;
    uint cc = idx - z * SML_OPM_CC;
    uint pairs = p.N * p.N;
    float acc = 0.0f;
    for (uint pIdx = 0u; pIdx < pairs; ++pIdx)
        acc = fma(sml_opmb_load(dZ, pIdx * SML_OPM_CZ + z), sml_opmb_load(oWs, pIdx * SML_OPM_CC + cc), acc);
    sml_opmb_store(dW, idx, acc);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void sml_msa_outer_product_mean_bwd_f32_da(uint3 dtid : SV_DispatchThreadID)
{
    OuterProductMeanBwdParams* p = reinterpret<OuterProductMeanBwdParams*>(smlOuterProductBwdPush.headerAddress);
    TensorDesc b = slangml_tensor_desc(p.header, 0u);
    TensorDesc dOWs = slangml_tensor_desc(p.header, 1u);
    TensorDesc da = slangml_tensor_desc(p.header, 2u);

    uint idx = (p.header.dispatchBase.x + dtid.x);
    if (idx >= p.S * p.N * SML_OPM_C)
        return;
    uint c1 = idx % SML_OPM_C;
    uint si = idx / SML_OPM_C;
    uint i = si % p.N;
    uint s = si / p.N;
    float invS = 1.0f / float(p.S);

    float acc = 0.0f;
    for (uint j = 0u; j < p.N; ++j)
    {
        uint pairIndex = i * p.N + j;
        for (uint c2 = 0u; c2 < SML_OPM_C; ++c2)
            acc = fma(sml_opmb_load(dOWs, pairIndex * SML_OPM_CC + c1 * SML_OPM_C + c2),
                      sml_opmb_load(b, (s * p.N + j) * SML_OPM_C + c2), acc);
    }
    sml_opmb_store(da, idx, acc * invS);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void sml_msa_outer_product_mean_bwd_f32_db(uint3 dtid : SV_DispatchThreadID)
{
    OuterProductMeanBwdParams* p = reinterpret<OuterProductMeanBwdParams*>(smlOuterProductBwdPush.headerAddress);
    TensorDesc a = slangml_tensor_desc(p.header, 0u);
    TensorDesc dOWs = slangml_tensor_desc(p.header, 1u);
    TensorDesc db = slangml_tensor_desc(p.header, 2u);

    uint idx = (p.header.dispatchBase.x + dtid.x);
    if (idx >= p.S * p.N * SML_OPM_C)
        return;
    uint c2 = idx % SML_OPM_C;
    uint sj = idx / SML_OPM_C;
    uint j = sj % p.N;
    uint s = sj / p.N;
    float invS = 1.0f / float(p.S);

    float acc = 0.0f;
    for (uint i = 0u; i < p.N; ++i)
    {
        uint pairIndex = i * p.N + j;
        for (uint c1 = 0u; c1 < SML_OPM_C; ++c1)
            acc = fma(sml_opmb_load(dOWs, pairIndex * SML_OPM_CC + c1 * SML_OPM_C + c2),
                      sml_opmb_load(a, (s * p.N + i) * SML_OPM_C + c1), acc);
    }
    sml_opmb_store(db, idx, acc * invS);
}
