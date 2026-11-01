import slangml_common;

// Backward pass for MSA row-wise gated attention with pair bias only [SI Alg 10].
// Forward is msa_pair_weighted_avg_f32.slang:
//
//   A_ij^h        = softmax_j(bias[i,j,h])          (independent of s)
//   oAcc[s,i,h,c] = sum_j A_ij^h v[s,j,h,c]
//   out           = G * oAcc,   G = sigmoid(gate)
//
// Gradients, with dOAcc = dOut * G and delta[s,i,h] = dOAcc[s,i,h,:] . oAcc[s,i,h,:]:
//
//   dGate[s,i,h,c] = dOut * oAcc * G * (1 - G)
//   dV[s,j,h,c]    = sum_i A_ij^h dOAcc[s,i,h,c]
//   dA_s,ij^h      = dOAcc[s,i,h,:] . v[s,j,h,:]
//   dBias[i,j,h]   = sum_s A_ij^h (dA_s,ij^h - delta[s,i,h])
//
// Using A = exp(bias - lse[i,h]) avoids a second softmax pass.  The weights depend
// only on (i, h), so lse is computed once per (i, h).
//
// Entries, in order:
//   sml_msa_pair_weighted_avg_bwd_f32_lse    thread per (i, h)
//   sml_msa_pair_weighted_avg_bwd_f32_rows   thread per (s, i, h): oAcc, dOAcc, delta, dGate
//   sml_msa_pair_weighted_avg_bwd_f32_dv     thread per (s, j, h)
//   sml_msa_pair_weighted_avg_bwd_f32_dbias  thread per (i, j, h)
//
// Layout (fp32):
//   v, gate, dOut, oAcc, dOAcc, dV, dGate : [S, N, H*C]  index (s*N + i)*H*C + h*C + c
//   bias, dBias                           : [N, N, H]    index (i*N + j)*H + h
//   lse                                   : [N, H]       index i*H + h
//   delta                                 : [S, N, H]    index (s*N + i)*H + h
//
// Performance: the dbias pass reads every row for every (i, j, h), so it is O(S N^2 H C).
// Acceptable for the test size.  Tiling over s is the first optimization.

static const uint SML_MSA_MAX_C = 8u;

struct MsaPairWeightedAvgBwdParams
{
    KernelHeader header;
    uint S;
    uint N;
    uint H;
    uint C;
};

[[vk::push_constant]]
ConstantBuffer<KernelPushConstants> smlMsaPairAvgBwdPush;

float sml_msab_load(TensorDesc t, uint index)
{
    return slangml_bda_load_f32(slangml_final_byte_address(t, int(index), 4u));
}

void sml_msab_store(TensorDesc t, uint index, float value)
{
    slangml_bda_store_f32(slangml_final_byte_address(t, int(index), 4u), value);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void sml_msa_pair_weighted_avg_bwd_f32_lse(uint3 dtid : SV_DispatchThreadID)
{
    MsaPairWeightedAvgBwdParams* p = reinterpret<MsaPairWeightedAvgBwdParams*>(smlMsaPairAvgBwdPush.headerAddress);
    TensorDesc bias = slangml_tensor_desc(p.header, 0u);
    TensorDesc lse = slangml_tensor_desc(p.header, 1u);

    uint N = p.N;
    uint H = p.H;
    uint t = (p.header.dispatchBase.x + dtid.x);
    if (t >= N * H)
        return;
    uint h = t % H;
    uint i = t / H;

    float m = -1.0e30f;
    for (uint j = 0u; j < N; ++j)
        m = max(m, sml_msab_load(bias, (i * N + j) * H + h));
    float sumExp = 0.0f;
    for (uint j = 0u; j < N; ++j)
        sumExp += exp(sml_msab_load(bias, (i * N + j) * H + h) - m);
    sml_msab_store(lse, i * H + h, m + log(sumExp));
}

[shader("compute")]
[numthreads(64, 1, 1)]
void sml_msa_pair_weighted_avg_bwd_f32_rows(uint3 dtid : SV_DispatchThreadID)
{
    MsaPairWeightedAvgBwdParams* p = reinterpret<MsaPairWeightedAvgBwdParams*>(smlMsaPairAvgBwdPush.headerAddress);
    TensorDesc bias = slangml_tensor_desc(p.header, 0u);
    TensorDesc v = slangml_tensor_desc(p.header, 1u);
    TensorDesc gate = slangml_tensor_desc(p.header, 2u);
    TensorDesc lse = slangml_tensor_desc(p.header, 3u);
    TensorDesc dOut = slangml_tensor_desc(p.header, 4u);
    TensorDesc oAcc = slangml_tensor_desc(p.header, 5u);
    TensorDesc dOAcc = slangml_tensor_desc(p.header, 6u);
    TensorDesc delta = slangml_tensor_desc(p.header, 7u);
    TensorDesc dGate = slangml_tensor_desc(p.header, 8u);

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
    uint rowBase = (s * N + i) * rowStride + h * C;

    float acc[SML_MSA_MAX_C];
    for (uint c = 0u; c < C; ++c)
        acc[c] = 0.0f;
    float lseI = sml_msab_load(lse, i * H + h);
    for (uint j = 0u; j < N; ++j)
    {
        float weight = exp(sml_msab_load(bias, (i * N + j) * H + h) - lseI);
        uint vBase = (s * N + j) * rowStride + h * C;
        for (uint c = 0u; c < C; ++c)
            acc[c] = fma(weight, sml_msab_load(v, vBase + c), acc[c]);
    }

    float deltaValue = 0.0f;
    for (uint c = 0u; c < C; ++c)
    {
        float g = slangml_sigmoid(sml_msab_load(gate, rowBase + c));
        float dO = sml_msab_load(dOut, rowBase + c);
        float dOa = dO * g;
        sml_msab_store(oAcc, rowBase + c, acc[c]);
        sml_msab_store(dOAcc, rowBase + c, dOa);
        sml_msab_store(dGate, rowBase + c, dO * acc[c] * g * (1.0f - g));
        deltaValue = fma(dOa, acc[c], deltaValue);
    }
    sml_msab_store(delta, si * H + h, deltaValue);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void sml_msa_pair_weighted_avg_bwd_f32_dv(uint3 dtid : SV_DispatchThreadID)
{
    MsaPairWeightedAvgBwdParams* p = reinterpret<MsaPairWeightedAvgBwdParams*>(smlMsaPairAvgBwdPush.headerAddress);
    TensorDesc bias = slangml_tensor_desc(p.header, 0u);
    TensorDesc lse = slangml_tensor_desc(p.header, 1u);
    TensorDesc dOAcc = slangml_tensor_desc(p.header, 2u);
    TensorDesc dV = slangml_tensor_desc(p.header, 3u);

    uint S = p.S;
    uint N = p.N;
    uint H = p.H;
    uint C = p.C;
    uint t = (p.header.dispatchBase.x + dtid.x);
    if (t >= S * N * H)
        return;
    uint h = t % H;
    uint sj = t / H;
    uint j = sj % N;
    uint s = sj / N;
    uint rowStride = H * C;

    float acc[SML_MSA_MAX_C];
    for (uint c = 0u; c < C; ++c)
        acc[c] = 0.0f;
    for (uint i = 0u; i < N; ++i)
    {
        float weight = exp(sml_msab_load(bias, (i * N + j) * H + h) - sml_msab_load(lse, i * H + h));
        uint rowBase = (s * N + i) * rowStride + h * C;
        for (uint c = 0u; c < C; ++c)
            acc[c] = fma(weight, sml_msab_load(dOAcc, rowBase + c), acc[c]);
    }
    uint outBase = (s * N + j) * rowStride + h * C;
    for (uint c = 0u; c < C; ++c)
        sml_msab_store(dV, outBase + c, acc[c]);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void sml_msa_pair_weighted_avg_bwd_f32_dbias(uint3 dtid : SV_DispatchThreadID)
{
    MsaPairWeightedAvgBwdParams* p = reinterpret<MsaPairWeightedAvgBwdParams*>(smlMsaPairAvgBwdPush.headerAddress);
    TensorDesc bias = slangml_tensor_desc(p.header, 0u);
    TensorDesc lse = slangml_tensor_desc(p.header, 1u);
    TensorDesc v = slangml_tensor_desc(p.header, 2u);
    TensorDesc dOAcc = slangml_tensor_desc(p.header, 3u);
    TensorDesc delta = slangml_tensor_desc(p.header, 4u);
    TensorDesc dBias = slangml_tensor_desc(p.header, 5u);

    uint S = p.S;
    uint N = p.N;
    uint H = p.H;
    uint C = p.C;
    uint t = (p.header.dispatchBase.x + dtid.x);
    if (t >= N * N * H)
        return;
    uint h = t % H;
    uint ij = t / H;
    uint j = ij % N;
    uint i = ij / N;
    uint rowStride = H * C;
    float weight = exp(sml_msab_load(bias, t) - sml_msab_load(lse, i * H + h));

    float acc = 0.0f;
    for (uint s = 0u; s < S; ++s)
    {
        uint dRow = (s * N + i) * rowStride + h * C;
        uint vRow = (s * N + j) * rowStride + h * C;
        float dA = 0.0f;
        for (uint c = 0u; c < C; ++c)
            dA = fma(sml_msab_load(dOAcc, dRow + c), sml_msab_load(v, vRow + c), dA);
        acc += weight * (dA - sml_msab_load(delta, (s * N + i) * H + h));
    }
    sml_msab_store(dBias, t, acc);
}
