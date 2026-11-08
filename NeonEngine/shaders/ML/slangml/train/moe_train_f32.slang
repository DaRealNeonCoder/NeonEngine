// Mixture-of-experts training kernels for gpt-oss (docs/gpt_oss_training.md).
//
// Routing uses the inference router (llm/moe_router_f32.slang, NORM_TOPK) for
// the forward top-k ids and weights. Then the K = topK (token, slot) entries
// are sorted by expert into a padded row space: expert g owns rows
// [seg[g], seg[g+1]), each segment rounded up to `pad` (= the GEMM tile
// height, 128) so every 128-row tile belongs to one expert. Entry e = n * K + s
// lives in row slotPos[e]; rowEntry[row] maps back (SML_INVALID = padding).
// The expert products then run as grouped GEMMs (train/gemm_train_f32.slang).
//
// Elementwise kernels use grid-stride loops: grid x = min(ceil(n / 256), 65535)
// and `stride` = grid x * 256.
import slangml_common;
import slangml_llm;
import slangml_train;

[[vk::push_constant]]
ConstantBuffer<KernelPushConstants> smlTMoePush;

// ---------------------------------------------------------------------------
// Route build. One workgroup of 1024 threads.
// Tensor table: 0 ids [N*K], 1 slotPos [N*K] (out), 2 rowEntry [maxRows] (out),
// 3 tileGroup [maxTiles] (out), 4 seg [E+1] (out), 5 counts [E] (out, uint).
// ---------------------------------------------------------------------------
struct RouteParams
{
    KernelHeader header;
    uint N;         // 48
    uint K;         // 52
    uint E;         // 56
    uint pad;       // 60
    uint maxRows;   // 64
    uint maxTiles;  // 68
    uint reserved0; // 72
    uint reserved1; // 76
};

groupshared uint smlRouteCnt[512];
groupshared uint smlRouteOff[513];

[shader("compute")]
[numthreads(1024, 1, 1)]
void sml_train_moe_route_f32(uint3 tid3 : SV_GroupThreadID)
{
    RouteParams* p = reinterpret<RouteParams*>(smlTMoePush.headerAddress);
    uint tid = tid3.x;
    uint* ids = sml_tu(p.header, 0u);
    uint* slotPos = sml_tu(p.header, 1u);
    uint* rowEntry = sml_tu(p.header, 2u);
    uint* tileGroup = sml_tu(p.header, 3u);
    uint* seg = sml_tu(p.header, 4u);
    uint* counts = sml_tu(p.header, 5u);
    uint entries = p.N * p.K;

    for (uint e = tid; e < p.E; e += 1024u)
        smlRouteCnt[e] = 0u;
    for (uint r = tid; r < p.maxRows; r += 1024u)
        rowEntry[r] = SML_INVALID;
    AllMemoryBarrierWithGroupSync();

    for (uint e = tid; e < entries; e += 1024u)
    {
        uint old;
        InterlockedAdd(smlRouteCnt[ids[e]], 1u, old);
        slotPos[e] = old;
    }
    AllMemoryBarrierWithGroupSync();

    if (tid == 0u)
    {
        uint acc = 0u;
        for (uint g = 0; g < p.E; ++g)
        {
            smlRouteOff[g] = acc;
            seg[g] = acc;
            counts[g] = smlRouteCnt[g];
            acc += (smlRouteCnt[g] + p.pad - 1u) / p.pad * p.pad;
        }
        smlRouteOff[p.E] = acc;
        seg[p.E] = acc;
    }
    AllMemoryBarrierWithGroupSync();

    for (uint e = tid; e < entries; e += 1024u)
    {
        uint r = slotPos[e] + smlRouteOff[ids[e]];
        slotPos[e] = r;
        rowEntry[r] = e;
    }
    for (uint t = tid; t < p.maxTiles; t += 1024u)
    {
        uint row = t * p.pad;
        uint g = SML_INVALID;
        for (uint x = 0; x < p.E; ++x)
        {
            if (row >= smlRouteOff[x] && row < smlRouteOff[x + 1u])
                g = x;
        }
        tileGroup[t] = g;
    }
}

// ---------------------------------------------------------------------------
// Xe[r] = t[rowEntry[r] / K] (0 for padding). Over maxRows * H / 4 float4.
// Tensor table: 0 rowEntry, 1 t [N][H], 2 Xe [maxRows][H].
// ---------------------------------------------------------------------------
struct MoeEltParams
{
    KernelHeader header;
    uint rows;    // 48
    uint cols;    // 52  (H, or I for swiglu)
    uint K;       // 56
    uint stride;  // 60  grid-stride in threads
    float limit;  // 64
    float alpha;  // 68
    uint flags;   // 72
    uint reserved0;
};

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_train_moe_gather_f32(uint3 dtid : SV_DispatchThreadID)
{
    MoeEltParams* p = reinterpret<MoeEltParams*>(smlTMoePush.headerAddress);
    uint* rowEntry = sml_tu(p.header, 0u);
    float4* t = sml_tf4(p.header, 1u);
    float4* xe = sml_tf4(p.header, 2u);
    uint q = p.cols >> 2u;
    uint total = p.rows * q;
    for (uint idx = dtid.x; idx < total; idx += p.stride)
    {
        uint r = idx / q;
        uint c = idx - r * q;
        uint e = rowEntry[r];
        float4 v = e == SML_INVALID ? float4(0.0f) : sml_ld_f4(t + (e / p.K) * q + c);
        sml_st_f4(xe + idx, v);
    }
}

// h[r][i] = swiglu_oai(H1[r][2i], H1[r][2i+1]). Tensor table: 0 H1, 1 h.
[shader("compute")]
[numthreads(256, 1, 1)]
void sml_train_moe_swiglu_fwd_f32(uint3 dtid : SV_DispatchThreadID)
{
    MoeEltParams* p = reinterpret<MoeEltParams*>(smlTMoePush.headerAddress);
    float2* h1 = reinterpret<float2*>(sml_tf(p.header, 0u));
    float* h = sml_tf(p.header, 1u);
    uint total = p.rows * p.cols;
    for (uint idx = dtid.x; idx < total; idx += p.stride)
    {
        float2 gu = h1[idx];
        h[idx] = sml_glu(SML_ACT_SWIGLU_OAI, gu.x, gu.y, p.limit, p.alpha);
    }
}

// In place: H1 (pre-activation) -> dH1, from dh. torch.clamp passes the
// gradient where min <= x <= max (inclusive). Tensor table: 0 H1, 1 dh.
[shader("compute")]
[numthreads(256, 1, 1)]
void sml_train_moe_swiglu_bwd_f32(uint3 dtid : SV_DispatchThreadID)
{
    MoeEltParams* p = reinterpret<MoeEltParams*>(smlTMoePush.headerAddress);
    float2* h1 = reinterpret<float2*>(sml_tf(p.header, 0u));
    float* dh = sml_tf(p.header, 1u);
    uint total = p.rows * p.cols;
    for (uint idx = dtid.x; idx < total; idx += p.stride)
    {
        float2 gu = h1[idx];
        float d = dh[idx];
        float g = min(gu.x, p.limit);
        float u = clamp(gu.y, -p.limit, p.limit);
        float sg = sml_sigmoid(p.alpha * g);
        float dg = gu.x <= p.limit ? d * (u + 1.0f) * (sg + p.alpha * g * sg * (1.0f - sg)) : 0.0f;
        float du = (gu.y >= -p.limit && gu.y <= p.limit) ? d * g * sg : 0.0f;
        h1[idx] = float2(dg, du);
    }
}

// out[n] = res[n] + sum_s w[n][s] Y2[slotPos[n*K+s]]. Over N * H / 4.
// Tensor table: 0 res, 1 Y2, 2 slotPos, 3 w, 4 out.
[shader("compute")]
[numthreads(256, 1, 1)]
void sml_train_moe_combine_fwd_f32(uint3 dtid : SV_DispatchThreadID)
{
    MoeEltParams* p = reinterpret<MoeEltParams*>(smlTMoePush.headerAddress);
    float4* res = sml_tf4(p.header, 0u);
    float4* y2 = sml_tf4(p.header, 1u);
    uint* slotPos = sml_tu(p.header, 2u);
    float* w = sml_tf(p.header, 3u);
    float4* out = sml_tf4(p.header, 4u);
    uint q = p.cols >> 2u;
    uint total = p.rows * q;
    for (uint idx = dtid.x; idx < total; idx += p.stride)
    {
        uint n = idx / q;
        uint c = idx - n * q;
        float4 acc = sml_ld_f4(res + idx);
        for (uint s = 0; s < p.K; ++s)
        {
            uint e = n * p.K + s;
            acc += w[e] * sml_ld_f4(y2 + slotPos[e] * q + c);
        }
        sml_st_f4(out + idx, acc);
    }
}

// dY2[r] = w[e] * dx[e / K] with e = rowEntry[r] (0 for padding). Over maxRows * H / 4.
// Tensor table: 0 rowEntry, 1 w, 2 dx, 3 dY2.
[shader("compute")]
[numthreads(256, 1, 1)]
void sml_train_moe_combine_bwd_f32(uint3 dtid : SV_DispatchThreadID)
{
    MoeEltParams* p = reinterpret<MoeEltParams*>(smlTMoePush.headerAddress);
    uint* rowEntry = sml_tu(p.header, 0u);
    float* w = sml_tf(p.header, 1u);
    float4* dx = sml_tf4(p.header, 2u);
    float4* dy2 = sml_tf4(p.header, 3u);
    uint q = p.cols >> 2u;
    uint total = p.rows * q;
    for (uint idx = dtid.x; idx < total; idx += p.stride)
    {
        uint r = idx / q;
        uint c = idx - r * q;
        uint e = rowEntry[r];
        float4 v = e == SML_INVALID ? float4(0.0f) : w[e] * sml_ld_f4(dx + (e / p.K) * q + c);
        sml_st_f4(dy2 + idx, v);
    }
}

// dt[n] (+)= sum_s dXe[slotPos[n*K+s]]. Over N * H / 4. flags & 1 = accumulate.
// Tensor table: 0 slotPos, 1 dXe, 2 dt.
[shader("compute")]
[numthreads(256, 1, 1)]
void sml_train_moe_gather_add_f32(uint3 dtid : SV_DispatchThreadID)
{
    MoeEltParams* p = reinterpret<MoeEltParams*>(smlTMoePush.headerAddress);
    uint* slotPos = sml_tu(p.header, 0u);
    float4* dxe = sml_tf4(p.header, 1u);
    float4* dt = sml_tf4(p.header, 2u);
    uint q = p.cols >> 2u;
    uint total = p.rows * q;
    for (uint idx = dtid.x; idx < total; idx += p.stride)
    {
        uint n = idx / q;
        uint c = idx - n * q;
        float4 acc = float4(0.0f);
        for (uint s = 0; s < p.K; ++s)
            acc += sml_ld_f4(dxe + slotPos[n * p.K + s] * q + c);
        if ((p.flags & 1u) != 0u)
            acc += sml_ld_f4(dt + idx);
        sml_st_f4(dt + idx, acc);
    }
}

// ---------------------------------------------------------------------------
// dw[e] = dx[e / K] . Y2[slotPos[e]]: 32 lanes per entry, 8 entries per
// workgroup. Grid x = ceil(N*K / 8) (at most 65535; larger counts loop).
// Tensor table: 0 slotPos, 1 dx, 2 Y2, 3 dw.
// ---------------------------------------------------------------------------
struct MoeDotParams
{
    KernelHeader header;
    uint entries;  // 48
    uint H;        // 52
    uint K;        // 56
    uint groups;   // 60  grid x
};

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_train_moe_dweight_f32(uint3 gid : SV_GroupID, uint3 tid3 : SV_GroupThreadID)
{
    MoeDotParams* p = reinterpret<MoeDotParams*>(smlTMoePush.headerAddress);
    uint lane = tid3.x & 31u;
    uint* slotPos = sml_tu(p.header, 0u);
    float* dx = sml_tf(p.header, 1u);
    float* y2 = sml_tf(p.header, 2u);
    float* dw = sml_tf(p.header, 3u);
    for (uint base = gid.x * 8u; base < p.entries; base += p.groups * 8u)
    {
        uint e = base + (tid3.x >> 5u);
        uint ec = min(e, p.entries - 1u);  // dead lanes mirror a live entry so the shuffle stays convergent
        float* a = dx + (ec / p.K) * p.H;
        float* b = y2 + slotPos[ec] * p.H;
        float s = 0.0f;
        for (uint c = lane; c < p.H; c += 32u)
            s += a[c] * b[c];
        s = sml_cluster_sum(s, 32u);
        if (lane == 0u && e < p.entries)
            dw[e] = s;
    }
}

// ---------------------------------------------------------------------------
// Router backward, one 32-lane group per token (8 tokens per workgroup).
//   selected experts (softmax over the top-k logits):  dz_sel = w (dw - sum w dw)
//   load-balancing loss (Switch Transformer), P = softmax over all E logits,
//   f_e = counts_e / (N K):  L_aux = auxScale * sum_n sum_e f_e P_ne, auxScale = coef * E / N
//   dz_e += P_e (a_e - sum_e' P_e' a_e'), a_e = auxScale f_e
// Grid x = ceil(N / 8). Tensor table: 0 logits [N][E], 1 ids [N][K], 2 w,
// 3 dw, 4 counts [E] (uint), 5 dLogits [N][E] (out), 6 aux [N] (out).
// ---------------------------------------------------------------------------
struct RouterBwdParams
{
    KernelHeader header;
    uint N;          // 48
    uint E;          // 52
    uint K;          // 56
    float auxScale;  // 60
};

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_train_moe_router_bwd_f32(uint3 gid : SV_GroupID, uint3 tid3 : SV_GroupThreadID)
{
    RouterBwdParams* p = reinterpret<RouterBwdParams*>(smlTMoePush.headerAddress);
    uint lane = tid3.x & 31u;
    uint n = gid.x * 8u + (tid3.x >> 5u);
    bool live = n < p.N;
    uint nc = min(n, p.N - 1u);
    float* logits = sml_tf(p.header, 0u) + nc * p.E;
    uint* ids = sml_tu(p.header, 1u) + nc * p.K;
    float* w = sml_tf(p.header, 2u) + nc * p.K;
    float* dw = sml_tf(p.header, 3u) + nc * p.K;
    uint* counts = sml_tu(p.header, 4u);
    float* dz = sml_tf(p.header, 5u) + nc * p.E;

    float sd = 0.0f;
    for (uint s = lane; s < p.K; s += 32u)
        sd += w[s] * dw[s];
    sd = sml_cluster_sum(sd, 32u);

    float mx = -3.0e38f;
    for (uint e = lane; e < p.E; e += 32u)
        mx = max(mx, logits[e]);
    mx = sml_cluster_max(mx, 32u);
    float den = 0.0f;
    for (uint e = lane; e < p.E; e += 32u)
        den += exp(logits[e] - mx);
    den = sml_cluster_sum(den, 32u);
    float fScale = 1.0f / float(p.N * p.K);
    float pa = 0.0f;
    for (uint e = lane; e < p.E; e += 32u)
        pa += exp(logits[e] - mx) / den * (p.auxScale * float(counts[e]) * fScale);
    pa = sml_cluster_sum(pa, 32u);

    for (uint e = lane; e < p.E; e += 32u)
    {
        float P = exp(logits[e] - mx) / den;
        float a = p.auxScale * float(counts[e]) * fScale;
        float g = P * (a - pa);
        for (uint s = 0; s < p.K; ++s)
        {
            if (ids[s] == e)
                g += w[s] * (dw[s] - sd);
        }
        if (live)
            dz[e] = g;
    }
    if (live && lane == 0u)
        sml_tf(p.header, 6u)[n] = pa;
}
