import slangml_common;

// Backward pass for sequence-local atom attention [SI Alg 5, 6, 7].  Forward is
// sequence_local_attention_f32.slang.  Query l reads band slot t in [0, 128), with
// key m = 32*(l/32) - 48 + t.  Slots with m outside [0, N) are masked.
//
//   logit_{l,t,h} = scale * q_l . k_m + bias[l,t,h]
//   A_{l,t,h}     = softmax_t(logit)           (over valid slots)
//   oAcc_l        = sum_t A v_m
//   out_l         = G_l * oAcc_l,   G = sigmoid(gate)
//
// Gradients, with dOAcc = dOut * G and delta_l = dOAcc_l . oAcc_l:
//
//   dGate_l       = dOut_l * oAcc_l * G_l * (1 - G_l)
//   dlogit_{l,t}  = A_{l,t} * (dOAcc_l . v_m - delta_l)
//   dQ_l          = scale * sum_t dlogit_{l,t} k_m
//   dBias_{l,t,h} = dlogit_{l,t}                         (each bias element has one reader)
//   dK_m          = scale * sum over queries l whose band holds m of dlogit_{l,t} q_l
//   dV_m          = sum over those queries of A_{l,t} dOAcc_l
//
// Entries:
//   sml_atom_sequence_local_attention_bwd_f32_stats  thread per (l, h)
//   sml_atom_sequence_local_attention_bwd_f32_dq     thread per (l, h)
//   sml_atom_sequence_local_attention_bwd_f32_dbias  thread per (l, t, h)
//   sml_atom_sequence_local_attention_bwd_f32_dkdv   thread per (m, h)
//
// Layout (fp32):
//   q, k, v, gate, dOut, dQ, dK, dV, dGate, oAcc, dOAcc : [N, H, D]   index (l*H + h)*D + d
//   bias, dBias                                         : [N, 128, H] index (l*128 + t)*H + h
//   lse, delta                                          : [N, H]      index l*H + h
//
// Performance: the dkdv pass visits only the 32-query blocks whose band can hold
// key m (at most 4 blocks of 32 queries), so it does 128 queries per key, not N.

static const uint SML_ALB_BAND = 128u;
static const uint SML_ALB_BLOCK = 32u;
static const uint SML_ALB_MAX_D = 64u;

struct AtomAttentionBwdParams
{
    KernelHeader header;
    uint N;
    uint H;
    uint D;
    float scale;
};

[[vk::push_constant]]
ConstantBuffer<KernelPushConstants> smlAtomBwdPush;

float sml_alb_load(TensorDesc t, uint index)
{
    return slangml_bda_load_f32(slangml_final_byte_address(t, int(index), 4u));
}

void sml_alb_store(TensorDesc t, uint index, float value)
{
    slangml_bda_store_f32(slangml_final_byte_address(t, int(index), 4u), value);
}

int sml_alb_key_start(uint l)
{
    return int(SML_ALB_BLOCK * (l / SML_ALB_BLOCK)) - 48;
}

// Stats for one (l, h): lse, oAcc, dOAcc, delta, dGate.
void sml_alb_stats_run(uint t)
{
    AtomAttentionBwdParams* p = reinterpret<AtomAttentionBwdParams*>(smlAtomBwdPush.headerAddress);
    t += p.header.dispatchBase.x;
    TensorDesc q = slangml_tensor_desc(p.header, 0u);
    TensorDesc k = slangml_tensor_desc(p.header, 1u);
    TensorDesc v = slangml_tensor_desc(p.header, 2u);
    TensorDesc bias = slangml_tensor_desc(p.header, 3u);
    TensorDesc gate = slangml_tensor_desc(p.header, 4u);
    TensorDesc dOut = slangml_tensor_desc(p.header, 5u);
    TensorDesc lse = slangml_tensor_desc(p.header, 6u);
    TensorDesc oAcc = slangml_tensor_desc(p.header, 7u);
    TensorDesc dOAcc = slangml_tensor_desc(p.header, 8u);
    TensorDesc delta = slangml_tensor_desc(p.header, 9u);
    TensorDesc dGate = slangml_tensor_desc(p.header, 10u);

    uint N = p.N;
    uint H = p.H;
    uint D = p.D;
    if (t >= N * H)
        return;
    uint h = t % H;
    uint l = t / H;
    int keyStart = sml_alb_key_start(l);

    float qv[SML_ALB_MAX_D];
    float acc[SML_ALB_MAX_D];
    uint rowBase = (l * H + h) * D;
    for (uint d = 0u; d < D; ++d)
    {
        qv[d] = sml_alb_load(q, rowBase + d);
        acc[d] = 0.0f;
    }

    float runningMax = -1.0e30f;
    float runningSum = 0.0f;
    for (uint band = 0u; band < SML_ALB_BAND; ++band)
    {
        int key = keyStart + int(band);
        if (key < 0 || key >= int(N))
            continue;
        uint keyBase = (uint(key) * H + h) * D;
        float dot = 0.0f;
        for (uint d = 0u; d < D; ++d)
            dot = fma(qv[d], sml_alb_load(k, keyBase + d), dot);
        float logit = fma(dot, p.scale, sml_alb_load(bias, (l * SML_ALB_BAND + band) * H + h));

        float newMax = max(runningMax, logit);
        float correction = exp(runningMax - newMax);
        float weight = exp(logit - newMax);
        runningSum = runningSum * correction + weight;
        for (uint d = 0u; d < D; ++d)
            acc[d] = fma(weight, sml_alb_load(v, keyBase + d), acc[d] * correction);
        runningMax = newMax;
    }

    float invSum = 1.0f / runningSum;
    sml_alb_store(lse, l * H + h, runningMax + log(runningSum));
    float deltaValue = 0.0f;
    for (uint d = 0u; d < D; ++d)
    {
        float g = slangml_sigmoid(sml_alb_load(gate, rowBase + d));
        float o = acc[d] * invSum;
        float dO = sml_alb_load(dOut, rowBase + d);
        float dOa = dO * g;
        sml_alb_store(oAcc, rowBase + d, o);
        sml_alb_store(dOAcc, rowBase + d, dOa);
        sml_alb_store(dGate, rowBase + d, dO * o * g * (1.0f - g));
        deltaValue = fma(dOa, o, deltaValue);
    }
    sml_alb_store(delta, l * H + h, deltaValue);
}

// dQ for one (l, h).
void sml_alb_dq_run(uint t)
{
    AtomAttentionBwdParams* p = reinterpret<AtomAttentionBwdParams*>(smlAtomBwdPush.headerAddress);
    t += p.header.dispatchBase.x;
    TensorDesc q = slangml_tensor_desc(p.header, 0u);
    TensorDesc k = slangml_tensor_desc(p.header, 1u);
    TensorDesc v = slangml_tensor_desc(p.header, 2u);
    TensorDesc bias = slangml_tensor_desc(p.header, 3u);
    TensorDesc lse = slangml_tensor_desc(p.header, 4u);
    TensorDesc dOAcc = slangml_tensor_desc(p.header, 5u);
    TensorDesc delta = slangml_tensor_desc(p.header, 6u);
    TensorDesc dQ = slangml_tensor_desc(p.header, 7u);

    uint N = p.N;
    uint H = p.H;
    uint D = p.D;
    if (t >= N * H)
        return;
    uint h = t % H;
    uint l = t / H;
    int keyStart = sml_alb_key_start(l);

    float qv[SML_ALB_MAX_D];
    float dO[SML_ALB_MAX_D];
    float dq[SML_ALB_MAX_D];
    uint rowBase = (l * H + h) * D;
    for (uint d = 0u; d < D; ++d)
    {
        qv[d] = sml_alb_load(q, rowBase + d);
        dO[d] = sml_alb_load(dOAcc, rowBase + d);
        dq[d] = 0.0f;
    }
    float lseI = sml_alb_load(lse, l * H + h);
    float deltaI = sml_alb_load(delta, l * H + h);

    for (uint band = 0u; band < SML_ALB_BAND; ++band)
    {
        int key = keyStart + int(band);
        if (key < 0 || key >= int(N))
            continue;
        uint keyBase = (uint(key) * H + h) * D;
        float dot = 0.0f;
        float dA = 0.0f;
        for (uint d = 0u; d < D; ++d)
        {
            dot = fma(qv[d], sml_alb_load(k, keyBase + d), dot);
            dA = fma(dO[d], sml_alb_load(v, keyBase + d), dA);
        }
        float logit = fma(dot, p.scale, sml_alb_load(bias, (l * SML_ALB_BAND + band) * H + h));
        float prob = exp(logit - lseI);
        float scaled = p.scale * prob * (dA - deltaI);
        for (uint d = 0u; d < D; ++d)
            dq[d] = fma(scaled, sml_alb_load(k, keyBase + d), dq[d]);
    }

    for (uint d = 0u; d < D; ++d)
        sml_alb_store(dQ, rowBase + d, dq[d]);
}

// dBias for one (l, band, h).  Each bias element has exactly one reader, so this is
// a direct write.  Masked slots get zero.
void sml_alb_dbias_run(uint t)
{
    AtomAttentionBwdParams* p = reinterpret<AtomAttentionBwdParams*>(smlAtomBwdPush.headerAddress);
    t += p.header.dispatchBase.x;
    TensorDesc q = slangml_tensor_desc(p.header, 0u);
    TensorDesc k = slangml_tensor_desc(p.header, 1u);
    TensorDesc v = slangml_tensor_desc(p.header, 2u);
    TensorDesc bias = slangml_tensor_desc(p.header, 3u);
    TensorDesc lse = slangml_tensor_desc(p.header, 4u);
    TensorDesc dOAcc = slangml_tensor_desc(p.header, 5u);
    TensorDesc delta = slangml_tensor_desc(p.header, 6u);
    TensorDesc dBias = slangml_tensor_desc(p.header, 7u);

    uint N = p.N;
    uint H = p.H;
    uint D = p.D;
    if (t >= N * SML_ALB_BAND * H)
        return;
    uint h = t % H;
    uint lt = t / H;
    uint band = lt % SML_ALB_BAND;
    uint l = lt / SML_ALB_BAND;
    int key = sml_alb_key_start(l) + int(band);
    if (key < 0 || key >= int(N))
    {
        sml_alb_store(dBias, t, 0.0f);
        return;
    }

    uint rowBase = (l * H + h) * D;
    uint keyBase = (uint(key) * H + h) * D;
    float dot = 0.0f;
    float dA = 0.0f;
    for (uint d = 0u; d < D; ++d)
    {
        dot = fma(sml_alb_load(q, rowBase + d), sml_alb_load(k, keyBase + d), dot);
        dA = fma(sml_alb_load(dOAcc, rowBase + d), sml_alb_load(v, keyBase + d), dA);
    }
    float logit = fma(dot, p.scale, sml_alb_load(bias, t));
    float prob = exp(logit - sml_alb_load(lse, l * H + h));
    sml_alb_store(dBias, t, prob * (dA - sml_alb_load(delta, l * H + h)));
}

// dK and dV for one (m, h).  Key m is read by query l when m = keyStart(l) + band
// for some band in [0, 128).  Only the 32-query blocks that can contain m are visited.
void sml_alb_dkdv_run(uint t)
{
    AtomAttentionBwdParams* p = reinterpret<AtomAttentionBwdParams*>(smlAtomBwdPush.headerAddress);
    t += p.header.dispatchBase.x;
    TensorDesc q = slangml_tensor_desc(p.header, 0u);
    TensorDesc k = slangml_tensor_desc(p.header, 1u);
    TensorDesc v = slangml_tensor_desc(p.header, 2u);
    TensorDesc bias = slangml_tensor_desc(p.header, 3u);
    TensorDesc lse = slangml_tensor_desc(p.header, 4u);
    TensorDesc dOAcc = slangml_tensor_desc(p.header, 5u);
    TensorDesc delta = slangml_tensor_desc(p.header, 6u);
    TensorDesc dK = slangml_tensor_desc(p.header, 7u);
    TensorDesc dV = slangml_tensor_desc(p.header, 8u);

    uint N = p.N;
    uint H = p.H;
    uint D = p.D;
    if (t >= N * H)
        return;
    uint h = t % H;
    uint m = t / H;

    float kv[SML_ALB_MAX_D];
    float vv[SML_ALB_MAX_D];
    float dk[SML_ALB_MAX_D];
    float dv[SML_ALB_MAX_D];
    uint keyBase = (m * H + h) * D;
    for (uint d = 0u; d < D; ++d)
    {
        kv[d] = sml_alb_load(k, keyBase + d);
        vv[d] = sml_alb_load(v, keyBase + d);
        dk[d] = 0.0f;
        dv[d] = 0.0f;
    }

    uint numBlocks = (N + SML_ALB_BLOCK - 1u) / SML_ALB_BLOCK;
    for (uint block = 0u; block < numBlocks; ++block)
    {
        // Block start is 32*block; key m is in its band iff 0 <= m - (32*block - 48) < 128.
        int blockStart = int(block * SML_ALB_BLOCK);
        int offset = int(m) - (blockStart - 48);
        if (offset < 0 || offset >= int(SML_ALB_BAND))
            continue;
        uint lEnd = min(block * SML_ALB_BLOCK + SML_ALB_BLOCK, N);
        for (uint l = block * SML_ALB_BLOCK; l < lEnd; ++l)
        {
            uint band = uint(offset);
            uint rowBase = (l * H + h) * D;
            float dot = 0.0f;
            float dA = 0.0f;
            for (uint d = 0u; d < D; ++d)
            {
                dot = fma(sml_alb_load(q, rowBase + d), kv[d], dot);
                dA = fma(sml_alb_load(dOAcc, rowBase + d), vv[d], dA);
            }
            float logit = fma(dot, p.scale, sml_alb_load(bias, (l * SML_ALB_BAND + band) * H + h));
            float prob = exp(logit - sml_alb_load(lse, l * H + h));
            float scaled = p.scale * prob * (dA - sml_alb_load(delta, l * H + h));
            for (uint d = 0u; d < D; ++d)
            {
                dv[d] = fma(prob, sml_alb_load(dOAcc, rowBase + d), dv[d]);
                dk[d] = fma(scaled, sml_alb_load(q, rowBase + d), dk[d]);
            }
        }
    }

    for (uint d = 0u; d < D; ++d)
    {
        sml_alb_store(dK, keyBase + d, dk[d]);
        sml_alb_store(dV, keyBase + d, dv[d]);
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void sml_atom_sequence_local_attention_bwd_f32_stats(uint3 dtid : SV_DispatchThreadID)
{
    sml_alb_stats_run(dtid.x);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void sml_atom_sequence_local_attention_bwd_f32_dq(uint3 dtid : SV_DispatchThreadID)
{
    sml_alb_dq_run(dtid.x);
}

[shader("compute")]
[numthreads(128, 1, 1)]
void sml_atom_sequence_local_attention_bwd_f32_dbias(uint3 dtid : SV_DispatchThreadID)
{
    sml_alb_dbias_run(dtid.x);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void sml_atom_sequence_local_attention_bwd_f32_dkdv(uint3 dtid : SV_DispatchThreadID)
{
    sml_alb_dkdv_run(dtid.x);
}
