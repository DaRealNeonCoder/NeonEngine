import slangml_common;

// Smooth LDDT structural loss [SI Alg 27].
//
//   delta_lm = |d_gt_lm - d_pred_lm|
//   l_lm     = 1/4 * sum_{t in {0.5, 1, 2, 4}} sigmoid(t - delta_lm)
//   c_lm     = [d_gt_lm < R_l] with R_l = 30 A if atom l is nucleotide, else 15 A
//   lddt     = sum_{l != m} c_lm l_lm / sum_{l != m} c_lm
//   loss     = 1 - lddt
//
// The pair weight c depends on the first index l only, so c_lm and c_ml can differ.
// Forward is split so the host can form the ratio:
//   rowNum[l] = sum_m c_lm l_lm,   rowDen[l] = sum_m c_lm
//   loss = 1 - sum(rowNum) / sum(rowDen)
//
// Backward is analytic and O(N^2) total.  With B = sum(rowDen) and
// dl/dA = -upstream / B, define f_km = l'(delta_km) * sign(p_km - g_km) / p_km,
// where p is the predicted distance, g the ground-truth distance, and
//   l'(delta) = -1/4 * sum_t sigmoid(t - delta) * (1 - sigmoid(t - delta)).
// Then (symmetric p, g, delta):
//   dA/dx_k = sum_m (c_km + c_mk) * f_km * (x_k - x_m)
// and the host passes gradScale = -upstream / B to the backward entry.
//
// Layout (fp32):
//   xPred, xGt : [N, 3]
//   nucFlag    : [N]   1.0 if atom is DNA or RNA, else 0.0
//   rowNum, rowDen : [N]
//   gradX      : [N, 3]
//
// Cost: N^2 pair evaluations with four sigmoids each.  Compute-bound on the
// sigmoids; the dominant optimization is a tiled shared-memory variant for
// N > ~4096 (see docs/af3_ops.md).

struct SmoothLDDTParams
{
    KernelHeader header;
    uint N;
    uint reserved0;
    uint reserved1;
    uint reserved2;
    float gradScale;
    float reserved3;
    float reserved4;
    float reserved5;
};

[[vk::push_constant]]
ConstantBuffer<KernelPushConstants> smlSmoothLDDTPush;

float sml_slddt_load(TensorDesc t, uint index)
{
    return slangml_bda_load_f32(slangml_final_byte_address(t, int(index), 4u));
}

void sml_slddt_store(TensorDesc t, uint index, float value)
{
    slangml_bda_store_f32(slangml_final_byte_address(t, int(index), 4u), value);
}

float3 sml_slddt_position(TensorDesc t, uint atom)
{
    return float3(sml_slddt_load(t, atom * 3u), sml_slddt_load(t, atom * 3u + 1u), sml_slddt_load(t, atom * 3u + 2u));
}

// Sum over the four thresholds of sigmoid(t - delta).
float sml_slddt_sigmoid_sum(float delta)
{
    return slangml_sigmoid(0.5f - delta) + slangml_sigmoid(1.0f - delta)
        + slangml_sigmoid(2.0f - delta) + slangml_sigmoid(4.0f - delta);
}

// Derivative of l(delta) = 1/4 * sum_t sigmoid(t - delta) w.r.t. delta.
float sml_slddt_dl_ddelta(float delta)
{
    float total = 0.0f;
    float thresholds[4] = { 0.5f, 1.0f, 2.0f, 4.0f };
    for (uint t = 0u; t < 4u; ++t)
    {
        float s = slangml_sigmoid(thresholds[t] - delta);
        total += s * (1.0f - s);
    }
    return -0.25f * total;
}

float sml_slddt_inclusion(float gtDistance, float nucFlag)
{
    float radius = nucFlag > 0.5f ? 30.0f : 15.0f;
    return gtDistance < radius ? 1.0f : 0.0f;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void sml_loss_smooth_lddt_fwd_f32(uint3 dtid : SV_DispatchThreadID)
{
    SmoothLDDTParams* p = reinterpret<SmoothLDDTParams*>(smlSmoothLDDTPush.headerAddress);
    if ((p.header.dispatchBase.x + dtid.x) >= p.N)
        return;
    TensorDesc xPred = slangml_tensor_desc(p.header, 0u);
    TensorDesc xGt = slangml_tensor_desc(p.header, 1u);
    TensorDesc nucFlag = slangml_tensor_desc(p.header, 2u);
    TensorDesc rowNum = slangml_tensor_desc(p.header, 3u);
    TensorDesc rowDen = slangml_tensor_desc(p.header, 4u);

    uint l = (p.header.dispatchBase.x + dtid.x);
    float3 pl = sml_slddt_position(xPred, l);
    float3 gl = sml_slddt_position(xGt, l);
    float nuc = sml_slddt_load(nucFlag, l);

    float num = 0.0f;
    float den = 0.0f;
    for (uint m = 0u; m < p.N; ++m)
    {
        if (m == l)
            continue;
        float dGt = length(gl - sml_slddt_position(xGt, m));
        float c = sml_slddt_inclusion(dGt, nuc);
        if (c == 0.0f)
            continue;
        float dPred = length(pl - sml_slddt_position(xPred, m));
        num += c * 0.25f * sml_slddt_sigmoid_sum(abs(dGt - dPred));
        den += c;
    }
    sml_slddt_store(rowNum, l, num);
    sml_slddt_store(rowDen, l, den);
}

// gradX[k] = gradScale * sum_{m != k} (c_km + c_mk) * f_km * (x_k - x_m).
[shader("compute")]
[numthreads(64, 1, 1)]
void sml_loss_smooth_lddt_bwd_f32(uint3 dtid : SV_DispatchThreadID)
{
    SmoothLDDTParams* p = reinterpret<SmoothLDDTParams*>(smlSmoothLDDTPush.headerAddress);
    if ((p.header.dispatchBase.x + dtid.x) >= p.N)
        return;
    TensorDesc xPred = slangml_tensor_desc(p.header, 0u);
    TensorDesc xGt = slangml_tensor_desc(p.header, 1u);
    TensorDesc nucFlag = slangml_tensor_desc(p.header, 2u);
    TensorDesc gradX = slangml_tensor_desc(p.header, 3u);

    uint k = (p.header.dispatchBase.x + dtid.x);
    float3 pk = sml_slddt_position(xPred, k);
    float3 gk = sml_slddt_position(xGt, k);
    float nucK = sml_slddt_load(nucFlag, k);

    float3 accum = float3(0.0f);
    for (uint m = 0u; m < p.N; ++m)
    {
        if (m == k)
            continue;
        float3 pm = sml_slddt_position(xPred, m);
        float3 gm = sml_slddt_position(xGt, m);
        float nucM = sml_slddt_load(nucFlag, m);

        float dGt = length(gk - gm);
        float cKM = sml_slddt_inclusion(dGt, nucK);
        float cMK = sml_slddt_inclusion(dGt, nucM);
        float cSum = cKM + cMK;
        if (cSum == 0.0f)
            continue;

        float3 diffPred = pk - pm;
        float dPred = length(diffPred);
        if (dPred < 1.0e-8f)
            continue;
        float delta = dGt - dPred;
        float sgn = (dPred - dGt) >= 0.0f ? 1.0f : -1.0f;
        float f = sml_slddt_dl_ddelta(abs(delta)) * sgn / dPred;
        accum += cSum * f * diffPred;
    }

    float3 grad = p.gradScale * accum;
    sml_slddt_store(gradX, k * 3u + 0u, grad.x);
    sml_slddt_store(gradX, k * 3u + 1u, grad.y);
    sml_slddt_store(gradX, k * 3u + 2u, grad.z);
}
