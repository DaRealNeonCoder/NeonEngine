import slangml_common;

// pTM and ipTM aggregation from the PAE distribution [SI Eq. 17, Eq. 18].
//
//   pTM(D)  = max_{i valid} (1/|D|)        sum_{j in D}        sum_b p_ij^b / (1 + (e_b/d0)^2)
//   ipTM(D) = max_{i valid} (1/|D - chain(i)|) sum_{j in D - chain(i)} sum_b p_ij^b / (1 + (e_b/d0)^2)
//
// d0 = 1.24 * cbrt(max(|D|, 19) - 15) - 1.8 is computed on the host from |D|.
// The per-row score is produced by sml_confidence_ptm_row_fwd_f32; the max over
// rows is sml_confidence_ptm_max_fwd_f32, a single workgroup reduction.  Rows
// whose frame is invalid get score -1 and never win the max.
//
// Layout (fp32 unless noted):
//   pae        : [N, N, NB]   softmax probabilities over PAE bins
//   binCenters : [NB]         e_b
//   inD        : [N]          1.0 if token j is in the set D, else 0.0
//   chainId    : [N]  uint    used only when interfaceOnly != 0
//   validFrame : [N]          1.0 if token i has a valid frame
//   rowScore   : [N]          output per-row score
//
// Thread mapping: one thread per row i.  The inner loop over j reads one PAE row
// (N * NB floats), which is streamed once.  Bandwidth-bound: N^2 * NB reads.

struct PTMRowParams
{
    KernelHeader header;
    uint N;
    uint NB;
    uint interfaceOnly;
    uint reserved;
    float d0;
    float reserved1;
    float reserved2;
    float reserved3;
};

struct PTMMaxParams
{
    KernelHeader header;
    uint N;
    uint reserved0;
    uint reserved1;
    uint reserved2;
};

[[vk::push_constant]]
ConstantBuffer<KernelPushConstants> smlPTMPush;

groupshared float smlPTMScratch[256];

float sml_ptm_load(TensorDesc t, uint index)
{
    return slangml_bda_load_f32(slangml_final_byte_address(t, int(index), 4u));
}

[shader("compute")]
[numthreads(64, 1, 1)]
void sml_confidence_ptm_row_fwd_f32(uint3 dtid : SV_DispatchThreadID)
{
    PTMRowParams* p = reinterpret<PTMRowParams*>(smlPTMPush.headerAddress);
    if ((p.header.dispatchBase.x + dtid.x) >= p.N)
        return;
    TensorDesc pae = slangml_tensor_desc(p.header, 0u);
    TensorDesc binCenters = slangml_tensor_desc(p.header, 1u);
    TensorDesc inD = slangml_tensor_desc(p.header, 2u);
    TensorDesc chainId = slangml_tensor_desc(p.header, 3u);
    TensorDesc validFrame = slangml_tensor_desc(p.header, 4u);
    TensorDesc rowScore = slangml_tensor_desc(p.header, 5u);

    uint i = (p.header.dispatchBase.x + dtid.x);
    uint chainI = slangml_bda_load_u32(slangml_final_byte_address(chainId, int(i), 4u));
    if (sml_ptm_load(validFrame, i) < 0.5f)
    {
        slangml_bda_store_f32(slangml_final_byte_address(rowScore, int(i), 4u), -1.0f);
        return;
    }

    float sum = 0.0f;
    float count = 0.0f;
    for (uint j = 0u; j < p.N; ++j)
    {
        if (sml_ptm_load(inD, j) < 0.5f)
            continue;
        uint chainJ = slangml_bda_load_u32(slangml_final_byte_address(chainId, int(j), 4u));
        if (p.interfaceOnly != 0u && chainJ == chainI)
            continue;

        float weighted = 0.0f;
        uint base = (i * p.N + j) * p.NB;
        for (uint b = 0u; b < p.NB; ++b)
        {
            float e = sml_ptm_load(binCenters, b) / p.d0;
            weighted = fma(sml_ptm_load(pae, base + b), 1.0f / (1.0f + e * e), weighted);
        }
        sum += weighted;
        count += 1.0f;
    }

    float score = count > 0.0f ? sum / count : 0.0f;
    slangml_bda_store_f32(slangml_final_byte_address(rowScore, int(i), 4u), score);
}

// Max over rowScore[0..N).  Single workgroup of 256 threads.
[shader("compute")]
[numthreads(256, 1, 1)]
void sml_confidence_ptm_max_fwd_f32(uint3 tid : SV_GroupThreadID)
{
    PTMMaxParams* p = reinterpret<PTMMaxParams*>(smlPTMPush.headerAddress);
    TensorDesc rowScore = slangml_tensor_desc(p.header, 0u);
    TensorDesc output = slangml_tensor_desc(p.header, 1u);

    float best = -1.0f;
    for (uint i = tid.x; i < p.N; i += 256u)
        best = max(best, sml_ptm_load(rowScore, i));
    smlPTMScratch[tid.x] = best;
    GroupMemoryBarrierWithGroupSync();
    for (uint active = 128u; active > 0u; active >>= 1u)
    {
        if (tid.x < active)
            smlPTMScratch[tid.x] = max(smlPTMScratch[tid.x], smlPTMScratch[tid.x + active]);
        GroupMemoryBarrierWithGroupSync();
    }
    if (tid.x == 0u)
        slangml_bda_store_f32(slangml_final_byte_address(output, 0, 4u), smlPTMScratch[0]);
}
