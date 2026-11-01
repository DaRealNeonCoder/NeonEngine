import slangml_common;

// MSA row-wise gated self-attention using only pair bias [SI Alg 10].
//
//   weights_ij^h = softmax_j( bias[i,j,h] )            (no query-key term)
//   out[s,i,h,c] = sigmoid(gate[s,i,h,c]) * sum_j weights_ij^h * v[s,j,h,c]
//
// The attention weights come from the pair representation only, so each MSA
// row combines information through the same weights.  bias is the host
// projection LinearNoBias(LayerNorm(z)) and v, gate are host GEMMs.
//
// Layout (fp32, canonical):
//   v, gate, out : [S, N, H*C]     index (s*N + i)*H*C + h*C + c
//   bias         : [N, N, H]       index (i*N + j)*H + h
//
// Paper config (Alg 8): H = 8, C = 8 (c = 8 at the call site, so H*C = c_m = 64).
//
// Thread mapping: one thread per (s, i, h), online softmax over j.  The weights
// do not depend on s, so a precomputed-weights variant (one thread per i, h
// writing [N, N, H] weights once) is the first optimization in docs/af3_ops.md.

static const uint SML_MSA_MAX_C = 8u;

struct MsaPairWeightedAvgParams
{
    KernelHeader header;
    uint S;
    uint N;
    uint H;
    uint C;
};

[[vk::push_constant]]
ConstantBuffer<KernelPushConstants> smlMsaPairAvgPush;

float sml_msa_load(TensorDesc t, uint index)
{
    return slangml_bda_load_f32(slangml_final_byte_address(t, int(index), 4u));
}

[shader("compute")]
[numthreads(64, 1, 1)]
void sml_msa_pair_weighted_avg_fwd_f32(uint3 dtid : SV_DispatchThreadID)
{
    MsaPairWeightedAvgParams* p = reinterpret<MsaPairWeightedAvgParams*>(smlMsaPairAvgPush.headerAddress);
    TensorDesc v = slangml_tensor_desc(p.header, 0u);
    TensorDesc gate = slangml_tensor_desc(p.header, 1u);
    TensorDesc bias = slangml_tensor_desc(p.header, 2u);
    TensorDesc output = slangml_tensor_desc(p.header, 3u);

    uint S = p.S;
    uint N = p.N;
    uint H = p.H;
    uint C = p.C;
    uint t = (p.header.dispatchBase.x + dtid.x);
    if (t >= S * N * H)
        return;

    uint h = t % H;
    uint si = t / H;
    uint i = si % N;
    uint s = si / N;
    uint rowStride = H * C;

    float acc[SML_MSA_MAX_C];
    for (uint c = 0u; c < C; ++c)
        acc[c] = 0.0f;

    float runningMax = -1.0e30f;
    float runningSum = 0.0f;
    for (uint j = 0u; j < N; ++j)
    {
        float logit = sml_msa_load(bias, (i * N + j) * H + h);
        float newMax = max(runningMax, logit);
        float correction = exp(runningMax - newMax);
        float weight = exp(logit - newMax);
        runningSum = runningSum * correction + weight;

        uint vBase = (s * N + j) * rowStride + h * C;
        for (uint c = 0u; c < C; ++c)
            acc[c] = fma(weight, sml_msa_load(v, vBase + c), acc[c] * correction);
        runningMax = newMax;
    }

    float invSum = 1.0f / runningSum;
    uint outBase = (s * N + i) * rowStride + h * C;
    for (uint c = 0u; c < C; ++c)
    {
        float g = slangml_sigmoid(sml_msa_load(gate, outBase + c));
        slangml_bda_store_f32(slangml_final_byte_address(output, int(outBase + c), 4u), g * acc[c] * invSum);
    }
}
