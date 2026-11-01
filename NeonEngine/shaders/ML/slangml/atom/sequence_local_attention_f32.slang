import slangml_common;

// Sequence-local atom attention [SI Alg 5, Alg 6, Alg 7, Suppl. Fig 1].
//
// Atoms are a flat list of N atoms.  Queries are grouped into blocks of 32
// consecutive atoms.  Block B (queries l in [32B, 32B+31]) attends to keys
// m in [32B - 48, 32B + 79], a band of 128 atoms.  This is exactly the mask
// of Alg 7 with block centres c = 32B + 15.5:
//   |l - c| < 16  and  |m - c| < 64.
// Keys outside [0, N) are masked out.
//
// The band is addressed by t in [0, 128): key m = 32*(l/32) - 48 + t.
//
// Layout (fp32, canonical):
//   q, k, v, gate, out : [N, H, D]         index (l*H + h)*D + d
//   bias               : [N, 128, H]       index (l*128 + t)*H + h   (pair bias, host-projected)
//
// out[l,h,:] = sigmoid(gate[l,h,:]) * sum_t softmax_t(q·k / sqrt(D) + bias) v
//
// Thread mapping: one thread per (l, h).  Each thread touches 128 keys, which
// is the reuse window of its 32-query block.  The sm/L2 traffic per block is
// the 128 K/V rows, shared by 32 queries.  See docs/af3_ops.md for the tiling
// plan that stages K/V in shared memory.
//
// Also provides the atom-pair geometry band features of Alg 5 (line 2-6), so
// the pair embedding can be a single LinearNoBias over 5 channels.

static const uint SML_ATOM_BAND = 128u;
static const uint SML_ATOM_BLOCK = 32u;
static const uint SML_ATOM_MAX_D = 64u;

struct AtomAttentionParams
{
    KernelHeader header;
    uint N;
    uint H;
    uint D;
    float scale;
};

struct AtomBandParams
{
    KernelHeader header;
    uint N;
    uint reserved0;
    uint reserved1;
    uint reserved2;
};

[[vk::push_constant]]
ConstantBuffer<KernelPushConstants> smlAtomPush;

float sml_atom_load(TensorDesc t, uint index)
{
    return slangml_bda_load_f32(slangml_final_byte_address(t, int(index), 4u));
}

void sml_atom_store(TensorDesc t, uint index, float value)
{
    slangml_bda_store_f32(slangml_final_byte_address(t, int(index), 4u), value);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void sml_atom_sequence_local_attention_fwd_f32(uint3 dtid : SV_DispatchThreadID)
{
    AtomAttentionParams* p = reinterpret<AtomAttentionParams*>(smlAtomPush.headerAddress);
    TensorDesc q = slangml_tensor_desc(p.header, 0u);
    TensorDesc k = slangml_tensor_desc(p.header, 1u);
    TensorDesc v = slangml_tensor_desc(p.header, 2u);
    TensorDesc bias = slangml_tensor_desc(p.header, 3u);
    TensorDesc gate = slangml_tensor_desc(p.header, 4u);
    TensorDesc output = slangml_tensor_desc(p.header, 5u);

    uint N = p.N;
    uint H = p.H;
    uint D = p.D;
    uint t = (p.header.dispatchBase.x + dtid.x);
    if (t >= N * H)
        return;

    uint h = t % H;
    uint l = t / H;
    int keyStart = int(SML_ATOM_BLOCK * (l / SML_ATOM_BLOCK)) - 48;

    float qv[SML_ATOM_MAX_D];
    float acc[SML_ATOM_MAX_D];
    uint rowBase = (l * H + h) * D;
    for (uint d = 0u; d < D; ++d)
    {
        qv[d] = sml_atom_load(q, rowBase + d);
        acc[d] = 0.0f;
    }

    float runningMax = -1.0e30f;
    float runningSum = 0.0f;
    for (uint band = 0u; band < SML_ATOM_BAND; ++band)
    {
        int key = keyStart + int(band);
        if (key < 0 || key >= int(N))
            continue;

        uint keyBase = (uint(key) * H + h) * D;
        float dot = 0.0f;
        for (uint d = 0u; d < D; ++d)
            dot = fma(qv[d], sml_atom_load(k, keyBase + d), dot);
        float logit = fma(dot, p.scale, sml_atom_load(bias, (l * SML_ATOM_BAND + band) * H + h));

        float newMax = max(runningMax, logit);
        float correction = exp(runningMax - newMax);
        float weight = exp(logit - newMax);
        runningSum = runningSum * correction + weight;
        for (uint d = 0u; d < D; ++d)
            acc[d] = fma(weight, sml_atom_load(v, keyBase + d), acc[d] * correction);
        runningMax = newMax;
    }

    float invSum = 1.0f / runningSum;
    for (uint d = 0u; d < D; ++d)
    {
        float g = slangml_sigmoid(sml_atom_load(gate, rowBase + d));
        sml_atom_store(output, rowBase + d, g * acc[d] * invSum);
    }
}

// Band geometry features, one thread per (l, t).
//   refPos   : [N, 3] float   reference conformer positions
//   refUid   : [N]    uint    (chain id, residue index) encoded per atom
//   out      : [N, 128, 5]    channels = [d.x*v, d.y*v, d.z*v, v/(1+|d|^2), v]
// where d = ref_l - ref_m and v = [uid_l == uid_m] * [m is a valid atom].
// Because LinearNoBias(d) * v == LinearNoBias(d * v), feeding these five
// channels into one GEMM reproduces Alg 5 lines 2-6 exactly.
[shader("compute")]
[numthreads(128, 1, 1)]
void sml_atom_band_geometry_fwd_f32(uint3 dtid : SV_DispatchThreadID)
{
    AtomBandParams* p = reinterpret<AtomBandParams*>(smlAtomPush.headerAddress);
    uint total = p.N * SML_ATOM_BAND;
    if ((p.header.dispatchBase.x + dtid.x) >= total)
        return;
    TensorDesc refPos = slangml_tensor_desc(p.header, 0u);
    TensorDesc refUid = slangml_tensor_desc(p.header, 1u);
    TensorDesc output = slangml_tensor_desc(p.header, 2u);

    uint l = (p.header.dispatchBase.x + dtid.x) / SML_ATOM_BAND;
    uint band = (p.header.dispatchBase.x + dtid.x) - l * SML_ATOM_BAND;
    int key = int(SML_ATOM_BLOCK * (l / SML_ATOM_BLOCK)) - 48 + int(band);

    float4 result = float4(0.0f);
    float fifth = 0.0f;
    if (key >= 0 && key < int(p.N))
    {
        uint m = uint(key);
        uint uidL = slangml_bda_load_u32(slangml_final_byte_address(refUid, int(l), 4u));
        uint uidM = slangml_bda_load_u32(slangml_final_byte_address(refUid, int(m), 4u));
        float v = uidL == uidM ? 1.0f : 0.0f;
        float3 d = float3(
            sml_atom_load(refPos, l * 3u + 0u) - sml_atom_load(refPos, m * 3u + 0u),
            sml_atom_load(refPos, l * 3u + 1u) - sml_atom_load(refPos, m * 3u + 1u),
            sml_atom_load(refPos, l * 3u + 2u) - sml_atom_load(refPos, m * 3u + 2u));
        float inv = 1.0f / (1.0f + dot(d, d));
        result = float4(d * v, v * inv);
        fifth = v;
    }

    uint base = (p.header.dispatchBase.x + dtid.x) * 5u;
    sml_atom_store(output, base + 0u, result.x);
    sml_atom_store(output, base + 1u, result.y);
    sml_atom_store(output, base + 2u, result.z);
    sml_atom_store(output, base + 3u, result.w);
    sml_atom_store(output, base + 4u, fifth);
}
