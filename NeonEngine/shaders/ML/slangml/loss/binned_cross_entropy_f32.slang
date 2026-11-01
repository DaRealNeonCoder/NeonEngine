import slangml_common;

// Binned cross-entropy against soft targets from logits [SI Eq. 9, 10, 12, 14; 4.4].
//
//   loss[r]     = - sum_b t[r,b] * log_softmax(logits[r])_b
//   dLogits[r,b] = upstream[r] * (softmax_b(logits[r]) * sum_b' t[r,b'] - t[r,b])
//
// Used for: pLDDT (50 bins), PAE (64 bins), PDE (64 bins), experimentally
// resolved (2 bins), and the distogram.  Targets are one-hot for the binned
// ground truth, or any soft distribution.  The per-term normalization
// (1/N_atom, 1/N_token^2) is applied by the host.
//
// Layout (fp32): logits, targets, dLogits : [R, B].  loss, upstream : [R].
// Thread mapping: one thread per row; two passes over B (max, then sum).  B is at
// most 64 in AF3, so a row fits in registers and L1.

struct BinnedCEParams
{
    KernelHeader header;
    uint rows;
    uint bins;
    uint reserved0;
    uint reserved1;
};

[[vk::push_constant]]
ConstantBuffer<KernelPushConstants> smlBinnedCEPush;

float sml_bce_load(TensorDesc t, uint index)
{
    return slangml_bda_load_f32(slangml_final_byte_address(t, int(index), 4u));
}

void sml_bce_store(TensorDesc t, uint index, float value)
{
    slangml_bda_store_f32(slangml_final_byte_address(t, int(index), 4u), value);
}

float sml_bce_row_max(TensorDesc logits, uint base, uint bins)
{
    float m = -1.0e30f;
    for (uint b = 0u; b < bins; ++b)
        m = max(m, sml_bce_load(logits, base + b));
    return m;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void sml_loss_binned_cross_entropy_fwd_f32(uint3 dtid : SV_DispatchThreadID)
{
    BinnedCEParams* p = reinterpret<BinnedCEParams*>(smlBinnedCEPush.headerAddress);
    uint row = (p.header.dispatchBase.x + dtid.x);
    if (row >= p.rows)
        return;
    TensorDesc logits = slangml_tensor_desc(p.header, 0u);
    TensorDesc targets = slangml_tensor_desc(p.header, 1u);
    TensorDesc loss = slangml_tensor_desc(p.header, 2u);

    uint base = row * p.bins;
    float m = sml_bce_row_max(logits, base, p.bins);
    float sumExp = 0.0f;
    for (uint b = 0u; b < p.bins; ++b)
        sumExp += exp(sml_bce_load(logits, base + b) - m);
    float logSumExp = m + log(sumExp);

    float total = 0.0f;
    for (uint b = 0u; b < p.bins; ++b)
        total = fma(-sml_bce_load(targets, base + b), sml_bce_load(logits, base + b) - logSumExp, total);
    sml_bce_store(loss, row, total);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void sml_loss_binned_cross_entropy_bwd_f32(uint3 dtid : SV_DispatchThreadID)
{
    BinnedCEParams* p = reinterpret<BinnedCEParams*>(smlBinnedCEPush.headerAddress);
    uint row = (p.header.dispatchBase.x + dtid.x);
    if (row >= p.rows)
        return;
    TensorDesc logits = slangml_tensor_desc(p.header, 0u);
    TensorDesc targets = slangml_tensor_desc(p.header, 1u);
    TensorDesc upstream = slangml_tensor_desc(p.header, 2u);
    TensorDesc dLogits = slangml_tensor_desc(p.header, 3u);

    uint base = row * p.bins;
    float m = sml_bce_row_max(logits, base, p.bins);
    float sumExp = 0.0f;
    float sumTarget = 0.0f;
    for (uint b = 0u; b < p.bins; ++b)
    {
        sumExp += exp(sml_bce_load(logits, base + b) - m);
        sumTarget += sml_bce_load(targets, base + b);
    }
    float up = sml_bce_load(upstream, row);
    float invSum = 1.0f / sumExp;
    for (uint b = 0u; b < p.bins; ++b)
    {
        float softmax = exp(sml_bce_load(logits, base + b) - m) * invSum;
        sml_bce_store(dLogits, base + b, up * fma(softmax, sumTarget, -sml_bce_load(targets, base + b)));
    }
}
