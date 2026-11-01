import slangml_common;

// Outer product mean communication from the MSA stream into the pair stream
// [SI Alg 9, Alg 8 line 6].
//
//   o_ij   = mean_s ( a_si ⊗ b_sj )           flattened to 32*32 = 1024
//   z_ij  += Linear(o_ij)                      Linear(1024 -> 128), with bias
//
// The 1024-channel intermediate o is never written to global memory.  Each
// workgroup owns one (i, j) pair, builds o in shared memory, then reduces it
// through the 128 x 1024 projection.  This turns an N^2 * 1024 floats tensor
// (1 GiB at N = 512 in fp32) into zero extra traffic.
//
// Inputs (fp32, canonical):
//   a, b : [S, N, 32]          output of LayerNorm(m) followed by LinearNoBias
//   w    : [128, 1024]         row-major, w[z*1024 + (c1*32 + c2)]
//   bias : [128]
//   zIn  : [N, N, 128]         pair representation before this block
// Output:
//   zOut : [N, N, 128]         zIn + bias + w · o
//
// Dispatch: one workgroup per pair, numthreads(256).  Pair index is
// dispatchBase.x + groupId.x, so large N splits across dispatches.

static const uint SML_OPM_C = 32u;
static const uint SML_OPM_CC = 1024u;
static const uint SML_OPM_CZ = 128u;

struct OuterProductMeanParams
{
    KernelHeader header;
    uint S;
    uint N;
    uint reserved0;
    uint reserved1;
};

[[vk::push_constant]]
ConstantBuffer<KernelPushConstants> smlOuterProductPush;

groupshared float smlOuterProductMean[SML_OPM_CC];

float sml_opm_load(TensorDesc t, uint index)
{
    return slangml_bda_load_f32(slangml_final_byte_address(t, int(index), 4u));
}

void sml_opm_store(TensorDesc t, uint index, float value)
{
    slangml_bda_store_f32(slangml_final_byte_address(t, int(index), 4u), value);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_msa_outer_product_mean_fwd_f32(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID)
{
    OuterProductMeanParams* p = reinterpret<OuterProductMeanParams*>(smlOuterProductPush.headerAddress);
    TensorDesc a = slangml_tensor_desc(p.header, 0u);
    TensorDesc b = slangml_tensor_desc(p.header, 1u);
    TensorDesc w = slangml_tensor_desc(p.header, 2u);
    TensorDesc bias = slangml_tensor_desc(p.header, 3u);
    TensorDesc zIn = slangml_tensor_desc(p.header, 4u);
    TensorDesc zOut = slangml_tensor_desc(p.header, 5u);

    uint pairIndex = p.header.dispatchBase.x + gid.x;
    if (pairIndex >= p.N * p.N)
        return;
    uint i = pairIndex / p.N;
    uint j = pairIndex - i * p.N;
    float invS = 1.0f / float(p.S);

    // Stage 1: o[c1*32 + c2] = mean_s a[s,i,c1] * b[s,j,c2].
    for (uint cc = tid.x; cc < SML_OPM_CC; cc += 256u)
    {
        uint c1 = cc / SML_OPM_C;
        uint c2 = cc - c1 * SML_OPM_C;
        float acc = 0.0f;
        for (uint s = 0u; s < p.S; ++s)
        {
            float av = sml_opm_load(a, (s * p.N + i) * SML_OPM_C + c1);
            float bv = sml_opm_load(b, (s * p.N + j) * SML_OPM_C + c2);
            acc = fma(av, bv, acc);
        }
        smlOuterProductMean[cc] = acc * invS;
    }
    GroupMemoryBarrierWithGroupSync();

    // Stage 2: z_out[z] = z_in[z] + bias[z] + sum_cc w[z, cc] * o[cc].
    uint pairBase = pairIndex * SML_OPM_CZ;
    for (uint z = tid.x; z < SML_OPM_CZ; z += 256u)
    {
        float acc = sml_opm_load(bias, z) + sml_opm_load(zIn, pairBase + z);
        uint wBase = z * SML_OPM_CC;
        for (uint cc = 0u; cc < SML_OPM_CC; ++cc)
            acc = fma(sml_opm_load(w, wBase + cc), smlOuterProductMean[cc], acc);
        sml_opm_store(zOut, pairBase + z, acc);
    }
}
