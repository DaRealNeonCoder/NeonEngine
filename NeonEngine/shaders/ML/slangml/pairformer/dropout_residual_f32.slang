import slangml_common;

// Shared-axis dropout fused with the residual add [SI Alg 8, 9-12, 16, 17].
//
//   out = resid + mask * y / (1 - p)
//
// The mask is shared along one axis of a rank-3 tensor [A, B, C]:
//   sharedAxis = 0  ->  mask[b, c] is reused for every a   (DropoutRowwise on pairs:
//                       rows i share the mask, columns j and channels c vary)
//   sharedAxis = 1  ->  mask[a, c] is reused for every b   (DropoutColumnwise on pairs)
// For the pair tensor z[i, j, c] this gives rowwise = shared over i and
// columnwise = shared over j, matching the AF2 shared_dim convention.
// VERIFY against the AF2 reference: the axis mapping is an assumption carried
// over from AF2, and the AF3 supplement does not restate it.
//
// Randomness is Philox keyed by (seed, stream) and the mask counter is the
// shared-key index.  The backward pass recomputes the same mask (same seed,
// stream, counter), so no mask is stored.
//
// Layout: y, resid, out : [A, B, C] fp32, canonical.
//
// Entries:
//   sml_pairformer_dropout_residual_fwd_f32   out = resid + mask*y*scale
//   sml_pairformer_dropout_bwd_f32            dy = dOut * mask * scale
//                                             (dResid = dOut is passed through by the host)

struct DropoutParams
{
    KernelHeader header;
    uint A;
    uint B;
    uint C;
    uint sharedAxis;
    float dropProbability;
    uint seed;
    uint stream;
    uint reserved0;
};

[[vk::push_constant]]
ConstantBuffer<KernelPushConstants> smlDropoutPush;

float sml_dropout_load(TensorDesc t, uint index)
{
    return slangml_bda_load_f32(slangml_final_byte_address(t, int(index), 4u));
}

void sml_dropout_store(TensorDesc t, uint index, float value)
{
    slangml_bda_store_f32(slangml_final_byte_address(t, int(index), 4u), value);
}

// Returns mask * 1/(1-p) for element (a, b, c).
float sml_dropout_scale(DropoutParams* p, uint a, uint b, uint c)
{
    uint key = p.sharedAxis == 0u ? (b * p.C + c) : (a * p.C + c);
    bool keep = slangml_random_bernoulli(uint64_t(key), p.seed, p.stream, 1.0f - p.dropProbability);
    return keep ? 1.0f / (1.0f - p.dropProbability) : 0.0f;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_pairformer_dropout_residual_fwd_f32(uint3 dtid : SV_DispatchThreadID)
{
    DropoutParams* p = reinterpret<DropoutParams*>(smlDropoutPush.headerAddress);
    uint total = p.A * p.B * p.C;
    if ((p.header.dispatchBase.x + dtid.x) >= total)
        return;
    TensorDesc y = slangml_tensor_desc(p.header, 0u);
    TensorDesc resid = slangml_tensor_desc(p.header, 1u);
    TensorDesc output = slangml_tensor_desc(p.header, 2u);

    uint c = (p.header.dispatchBase.x + dtid.x) % p.C;
    uint ab = (p.header.dispatchBase.x + dtid.x) / p.C;
    uint b = ab % p.B;
    uint a = ab / p.B;
    float scale = sml_dropout_scale(p, a, b, c);
    sml_dropout_store(output, (p.header.dispatchBase.x + dtid.x), fma(scale, sml_dropout_load(y, (p.header.dispatchBase.x + dtid.x)), sml_dropout_load(resid, (p.header.dispatchBase.x + dtid.x))));
}

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_pairformer_dropout_bwd_f32(uint3 dtid : SV_DispatchThreadID)
{
    DropoutParams* p = reinterpret<DropoutParams*>(smlDropoutPush.headerAddress);
    uint total = p.A * p.B * p.C;
    if ((p.header.dispatchBase.x + dtid.x) >= total)
        return;
    TensorDesc dOut = slangml_tensor_desc(p.header, 0u);
    TensorDesc dy = slangml_tensor_desc(p.header, 1u);

    uint c = (p.header.dispatchBase.x + dtid.x) % p.C;
    uint ab = (p.header.dispatchBase.x + dtid.x) / p.C;
    uint b = ab % p.B;
    uint a = ab / p.B;
    float scale = sml_dropout_scale(p, a, b, c);
    sml_dropout_store(dy, (p.header.dispatchBase.x + dtid.x), sml_dropout_load(dOut, (p.header.dispatchBase.x + dtid.x)) * scale);
}
