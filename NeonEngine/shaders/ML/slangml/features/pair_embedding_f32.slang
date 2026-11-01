import slangml_common;

// Pair-embedding features that are one-hot encodings followed by a
// LinearNoBias.  A one-hot LinearNoBias is a gather of weight rows, so these
// kernels never materialize the one-hot tensors [SI Alg 1, Alg 3, Alg 4, Alg 31].
//
//   sml_features_relative_position_encoding_fwd_f32
//       z_ij = LinearNoBias(concat(onehot(rel_pos), onehot(rel_token),
//                                  same_entity, onehot(rel_chain)))  + bonds_ij * Wb
//   sml_features_distance_bin_embed_fwd_f32
//       z_ij = LinearNoBias(onehot_nearest(d_ij, v_bins))
//
// Relative position encoding (Alg 3, rmax = 32, smax = 2):
//   rel_pos   : same chain ? clip(res_i - res_j + 32, 0, 64) : 65        (66 bins)
//   rel_token : same chain and same residue ? clip(tok_i - tok_j + 32, 0, 64) : 65
//   same_entity : [entity_i == entity_j]                                  (1 channel)
//   rel_chain : same chain ? 5 : clip(sym_i - sym_j + 2, 0, 4)           (6 bins)
// The concatenated one-hot length is 66 + 66 + 1 + 6 = 139 rows.
// Interpretation note: the Alg 3 text layer does not fully fix which branch
// each index takes when chains differ; this follows the AF2-multimer
// convention.  Verify against the released pseudocode before training.
//
// Weight layouts (fp32, row-major):
//   wRelPos : [139, CZ]     rows 0..65 pos, 66..131 token, 132 entity, 133..138 chain
//   wBond   : [CZ]          token-bond projection (Alg 1 line 5)
// Output: [N, N, CZ].  Thread per (i, j, c).

static const uint SML_REL_RMAX = 32u;
static const uint SML_REL_SMAX = 2u;
static const uint SML_REL_TOKEN_OFFSET = 66u;
static const uint SML_REL_ENTITY_ROW = 132u;
static const uint SML_REL_CHAIN_OFFSET = 133u;

struct RelPosParams
{
    KernelHeader header;
    uint N;
    uint CZ;
    uint hasBonds;
    uint reserved;
};

struct DistanceBinParams
{
    KernelHeader header;
    uint N;
    uint CZ;
    uint bins;
    uint reserved;
};

[[vk::push_constant]]
ConstantBuffer<KernelPushConstants> smlPairEmbeddingPush;

float sml_pair_load(TensorDesc t, uint index)
{
    return slangml_bda_load_f32(slangml_final_byte_address(t, int(index), 4u));
}

int sml_pair_load_i32(TensorDesc t, uint index)
{
    return asint(slangml_bda_load_u32(slangml_final_byte_address(t, int(index), 4u)));
}

void sml_pair_store(TensorDesc t, uint index, float value)
{
    slangml_bda_store_f32(slangml_final_byte_address(t, int(index), 4u), value);
}

int sml_pair_clip(int value, int low, int high)
{
    return clamp(value, low, high);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_features_relative_position_encoding_fwd_f32(uint3 dtid : SV_DispatchThreadID)
{
    RelPosParams* p = reinterpret<RelPosParams*>(smlPairEmbeddingPush.headerAddress);
    uint total = p.N * p.N * p.CZ;
    if ((p.header.dispatchBase.x + dtid.x) >= total)
        return;
    TensorDesc residueIndex = slangml_tensor_desc(p.header, 0u);
    TensorDesc tokenIndex = slangml_tensor_desc(p.header, 1u);
    TensorDesc asymId = slangml_tensor_desc(p.header, 2u);
    TensorDesc symId = slangml_tensor_desc(p.header, 3u);
    TensorDesc entityId = slangml_tensor_desc(p.header, 4u);
    TensorDesc wRelPos = slangml_tensor_desc(p.header, 5u);
    TensorDesc bonds = slangml_tensor_desc(p.header, 6u);
    TensorDesc wBond = slangml_tensor_desc(p.header, 7u);
    TensorDesc output = slangml_tensor_desc(p.header, 8u);

    uint c = (p.header.dispatchBase.x + dtid.x) % p.CZ;
    uint ij = (p.header.dispatchBase.x + dtid.x) / p.CZ;
    uint j = ij % p.N;
    uint i = ij / p.N;

    bool sameChain = sml_pair_load_i32(asymId, i) == sml_pair_load_i32(asymId, j);
    bool sameResidue = sml_pair_load_i32(residueIndex, i) == sml_pair_load_i32(residueIndex, j);
    bool sameEntity = sml_pair_load_i32(entityId, i) == sml_pair_load_i32(entityId, j);

    int rmax = int(SML_REL_RMAX);
    int relPos = int(2u * SML_REL_RMAX + 1u);
    if (sameChain)
    {
        int diff = sml_pair_load_i32(residueIndex, i) - sml_pair_load_i32(residueIndex, j);
        relPos = sml_pair_clip(diff + rmax, 0, 2 * rmax);
    }

    int relToken = int(2u * SML_REL_RMAX + 1u);
    if (sameChain && sameResidue)
    {
        int diff = sml_pair_load_i32(tokenIndex, i) - sml_pair_load_i32(tokenIndex, j);
        relToken = sml_pair_clip(diff + rmax, 0, 2 * rmax);
    }

    int smax = int(SML_REL_SMAX);
    int relChain = 2 * smax + 1;
    if (!sameChain)
    {
        int diff = sml_pair_load_i32(symId, i) - sml_pair_load_i32(symId, j);
        relChain = sml_pair_clip(diff + smax, 0, 2 * smax);
    }

    float value = sml_pair_load(wRelPos, uint(relPos) * p.CZ + c)
        + sml_pair_load(wRelPos, (SML_REL_TOKEN_OFFSET + uint(relToken)) * p.CZ + c)
        + sml_pair_load(wRelPos, (SML_REL_CHAIN_OFFSET + uint(relChain)) * p.CZ + c);
    if (sameEntity)
        value += sml_pair_load(wRelPos, SML_REL_ENTITY_ROW * p.CZ + c);

    if (p.hasBonds != 0u)
        value = fma(sml_pair_load(bonds, ij), sml_pair_load(wBond, c), value);

    sml_pair_store(output, (p.header.dispatchBase.x + dtid.x), value);
}

// Nearest-bin one-hot followed by LinearNoBias: out = W[argmin_b |d - v_b|, c].
//   d      : [N, N]           distance per pair
//   vBins  : [bins]           bin centres
//   w      : [bins, CZ]
[shader("compute")]
[numthreads(256, 1, 1)]
void sml_features_distance_bin_embed_fwd_f32(uint3 dtid : SV_DispatchThreadID)
{
    DistanceBinParams* p = reinterpret<DistanceBinParams*>(smlPairEmbeddingPush.headerAddress);
    uint total = p.N * p.N * p.CZ;
    if ((p.header.dispatchBase.x + dtid.x) >= total)
        return;
    TensorDesc d = slangml_tensor_desc(p.header, 0u);
    TensorDesc vBins = slangml_tensor_desc(p.header, 1u);
    TensorDesc w = slangml_tensor_desc(p.header, 2u);
    TensorDesc output = slangml_tensor_desc(p.header, 3u);

    uint c = (p.header.dispatchBase.x + dtid.x) % p.CZ;
    uint ij = (p.header.dispatchBase.x + dtid.x) / p.CZ;
    float dv = sml_pair_load(d, ij);

    uint best = 0u;
    float bestDistance = abs(dv - sml_pair_load(vBins, 0u));
    for (uint b = 1u; b < p.bins; ++b)
    {
        float distance = abs(dv - sml_pair_load(vBins, b));
        if (distance < bestDistance)
        {
            bestDistance = distance;
            best = b;
        }
    }
    sml_pair_store(output, (p.header.dispatchBase.x + dtid.x), sml_pair_load(w, best * p.CZ + c));
}
