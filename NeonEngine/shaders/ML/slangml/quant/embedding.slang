// Token embedding lookup from a quantized / 16-bit table (docs/quant.md).
//
//   out[t][k] = scale * dequant(table[tokens[t]][k])
//
// The table uses the same [V][K] row layouts as the GEMV weights, so a tied
// LM head can share the buffer. One thread per 4 output columns.
// Grid: x = ceil(K / 4 / 256), y = T.
// Tensor table: 0 codes, 1 meta (Q4/Q8), 2 tokens i32 [T], 3 out fp32 [T][ldOut].
import slangml_common;
import slangml_llm;

struct EmbedParams
{
    KernelHeader header;
    uint T;           // 48
    uint K;           // 52  embedding width (multiple of 32)
    uint groupShift;  // 56
    uint ldOut;       // 60
    float scale;      // 64  e.g. sqrt(hidden) for Gemma, else 1
    uint reserved0;   // 68
    uint reserved1;   // 72
    uint reserved2;   // 76
};

[[vk::push_constant]]
ConstantBuffer<KernelPushConstants> smlEmbedPush;

[ForceInline]
void sml_embed_main<let FMT : int>(uint3 gid, uint3 tid)
{
    EmbedParams* p = reinterpret<EmbedParams*>(smlEmbedPush.headerAddress);
    uint t = p.header.dispatchBase.y + gid.y;
    uint quad = (p.header.dispatchBase.x + gid.x) * 256u + tid.x;
    if (t >= p.T || quad >= (p.K >> 2u))
        return;
    uint token = reinterpret<uint*>(sml_base(slangml_tensor_desc(p.header, 2u), 4u))[t];
    uint64_t codes = sml_base(slangml_tensor_desc(p.header, 0u), 1u);
    float4 v;
    if (FMT == SML_WFMT_Q4 || FMT == SML_WFMT_Q8)
    {
        uint nb = p.K >> 5u;
        uint block = quad >> 3u;
        uint* meta = reinterpret<uint*>(sml_base(slangml_tensor_desc(p.header, 1u), 4u));
        float2 dm = sml_half2_to_float2(meta[token * (nb >> p.groupShift) + (block >> p.groupShift)]);
        if (FMT == SML_WFMT_Q4)
        {
            uint e0 = (quad & 7u) * 4u;  // element within the block
            uint word = reinterpret<uint*>(codes)[(token * nb + block) * 4u + ((e0 & 15u) >> 2u)];
            uint nib = (word >> (e0 < 16u ? 0u : 4u)) & 0x0F0F0F0Fu;
            v = float4(float(nib & 0xFFu), float((nib >> 8u) & 0xFFu),
                       float((nib >> 16u) & 0xFFu), float(nib >> 24u)) * dm.x + dm.y;
        }
        else
        {
            uint word = reinterpret<uint*>(codes)[token * (p.K >> 2u) + quad];
            v = sml_q8_word_to_float(word, dm);
        }
    }
    else
    {
        uint2 w = sml_ld_u2(reinterpret<uint2*>(codes) + (token * (p.K >> 2u) + quad));
        float2 a = FMT == SML_WFMT_F16 ? sml_half2_to_float2(w.x) : sml_bf16x2_to_float2(w.x);
        float2 b = FMT == SML_WFMT_F16 ? sml_half2_to_float2(w.y) : sml_bf16x2_to_float2(w.y);
        v = float4(a, b);
    }
    float4* out = reinterpret<float4*>(sml_base(slangml_tensor_desc(p.header, 3u), 4u) + uint64_t(t * p.ldOut) * 4u);
    sml_st_f4(out + quad, v * p.scale);
}

[shader("compute")] [numthreads(256, 1, 1)]
void sml_quant_embedding_fwd_q4(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID) { sml_embed_main<SML_WFMT_Q4>(gid, tid); }
[shader("compute")] [numthreads(256, 1, 1)]
void sml_quant_embedding_fwd_q8(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID) { sml_embed_main<SML_WFMT_Q8>(gid, tid); }
[shader("compute")] [numthreads(256, 1, 1)]
void sml_quant_embedding_fwd_f16(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID) { sml_embed_main<SML_WFMT_F16>(gid, tid); }
[shader("compute")] [numthreads(256, 1, 1)]
void sml_quant_embedding_fwd_bf16(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID) { sml_embed_main<SML_WFMT_BF16>(gid, tid); }
