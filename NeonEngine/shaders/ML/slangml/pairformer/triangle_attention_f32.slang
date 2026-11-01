import slangml_common;

// Triangular gated self-attention around the starting or ending node
// [SI Alg 14, Alg 15].  One thread owns one (i, j, head) output row and runs
// an online softmax over k, so no attention matrix is materialized.
//
// Layout (all fp32, row-major, canonical):
//   q, k, v, gate : [N, N, H, C]   index ((i*N + j)*H + h)*C + c
//   bias          : [N, N, H]      index (i*N + j)*H + h
//   out           : [N, N, H, C]
//
// Starting node (Alg 14):  logit_k = q[i,j]·k[i,k]/sqrt(C) + bias[j,k]
//                          out[i,j] = sigmoid(gate[i,j]) * sum_k softmax_k(logit) v[i,k]
// Ending node   (Alg 15):  logit_k = q[i,j]·k[k,j]/sqrt(C) + bias[k,i]
//                          out[i,j] = sigmoid(gate[i,j]) * sum_k softmax_k(logit) v[k,j]

static const uint SML_TRI_MAX_C = 32u;

struct TriangleAttentionParams
{
    KernelHeader header;
    uint N;
    uint H;
    uint C;
    float scale;
};

[[vk::push_constant]]
ConstantBuffer<KernelPushConstants> smlTriangleAttentionPush;

float sml_tri_attn_load(TensorDesc t, uint index)
{
    return slangml_bda_load_f32(slangml_final_byte_address(t, int(index), 4u));
}

void sml_tri_attn_store(TensorDesc t, uint index, float value)
{
    slangml_bda_store_f32(slangml_final_byte_address(t, int(index), 4u), value);
}

void sml_tri_attn_run(uint t, bool ending)
{
    TriangleAttentionParams* p = reinterpret<TriangleAttentionParams*>(smlTriangleAttentionPush.headerAddress);
    t += p.header.dispatchBase.x;
    TensorDesc q = slangml_tensor_desc(p.header, 0u);
    TensorDesc k = slangml_tensor_desc(p.header, 1u);
    TensorDesc v = slangml_tensor_desc(p.header, 2u);
    TensorDesc bias = slangml_tensor_desc(p.header, 3u);
    TensorDesc gate = slangml_tensor_desc(p.header, 4u);
    TensorDesc output = slangml_tensor_desc(p.header, 5u);

    uint N = p.N;
    uint H = p.H;
    uint C = p.C;
    if (t >= N * N * H)
        return;

    uint h = t % H;
    uint ij = t / H;
    uint j = ij % N;
    uint i = ij / N;

    float qv[SML_TRI_MAX_C];
    float acc[SML_TRI_MAX_C];
    uint qBase = ((i * N + j) * H + h) * C;
    for (uint c = 0u; c < C; ++c)
    {
        qv[c] = sml_tri_attn_load(q, qBase + c);
        acc[c] = 0.0f;
    }

    float runningMax = -1.0e30f;
    float runningSum = 0.0f;
    for (uint kk = 0u; kk < N; ++kk)
    {
        // Starting node reads row i of k/v and bias[j,k]; ending node reads
        // column j of k/v and bias[k,i].
        uint rowA = ending ? kk : i;
        uint rowB = ending ? j : kk;
        uint kvBase = ((rowA * N + rowB) * H + h) * C;
        uint biasIndex = ending ? ((kk * N + i) * H + h) : ((j * N + kk) * H + h);

        float dot = 0.0f;
        for (uint c = 0u; c < C; ++c)
            dot = fma(qv[c], sml_tri_attn_load(k, kvBase + c), dot);
        float logit = fma(dot, p.scale, sml_tri_attn_load(bias, biasIndex));

        float newMax = max(runningMax, logit);
        float correction = exp(runningMax - newMax);
        float weight = exp(logit - newMax);
        runningSum = runningSum * correction + weight;
        for (uint c = 0u; c < C; ++c)
            acc[c] = fma(weight, sml_tri_attn_load(v, kvBase + c), acc[c] * correction);
        runningMax = newMax;
    }

    float invSum = 1.0f / runningSum;
    for (uint c = 0u; c < C; ++c)
    {
        float g = slangml_sigmoid(sml_tri_attn_load(gate, qBase + c));
        sml_tri_attn_store(output, qBase + c, g * acc[c] * invSum);
    }
}

[shader("compute")]
[numthreads(128, 1, 1)]
void sml_pairformer_triangle_attention_start_fwd_f32(uint3 dtid : SV_DispatchThreadID)
{
    sml_tri_attn_run(dtid.x, false);
}

[shader("compute")]
[numthreads(128, 1, 1)]
void sml_pairformer_triangle_attention_end_fwd_f32(uint3 dtid : SV_DispatchThreadID)
{
    sml_tri_attn_run(dtid.x, true);
}
