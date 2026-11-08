// Activation quantization for the int8-dot GEMV path (docs/quant.md).
//
//   X  fp32 [rows][ldX]          (only the first `cols` columns are used)
//   Xq int8 [rows][cols]         packed 4 per uint, row stride cols bytes
//   Xs float2 [rows][cols / 32]  (d, s): d = amax / 127, s = d * sum(q)
//
// Dequantized x ~= d * q. s is the sum of the *dequantized* block, so the GEMV's
// affine term (m * s) and its int8 dot (d_w * d * sum(q_w * q)) together equal
// the exact dot of dequantized weights with dequantized activations. Using the
// exact fp32 sum instead would leave the rounding error scaled by the
// un-centred Q4 codes (measured 2-6x worse). cols must be a multiple of 32.
//
// Thread layout: one thread per 4 columns, so a 32-column block is 8
// consecutive lanes and the block reductions are 3 shuffle steps.
// Dispatch: x = ceil(cols / 4 / 256), y = rows.
import slangml_common;
import slangml_llm;

struct QuantizeActParams
{
    KernelHeader header;
    uint rows;
    uint cols;
    uint ldX;
    uint reserved;
};

[[vk::push_constant]]
ConstantBuffer<KernelPushConstants> smlQuantizePush;

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_quant_quantize_q8act_fwd_f32(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID)
{
    QuantizeActParams* p = reinterpret<QuantizeActParams*>(smlQuantizePush.headerAddress);
    uint row = p.header.dispatchBase.y + gid.y;
    uint quad = (p.header.dispatchBase.x + gid.x) * 256u + tid.x;  // index of 4-column group
    uint quads = p.cols >> 2u;
    // Blocks are 8 aligned lanes and quads is a multiple of 8, so a block is
    // either fully inside or fully outside; keep lanes alive for the shuffles.
    bool active = row < p.rows && quad < quads;
    uint safeRow = min(row, p.rows - 1u);
    uint safeQuad = min(quad, quads - 1u);

    TensorDesc xd = slangml_tensor_desc(p.header, 0u);
    TensorDesc qd = slangml_tensor_desc(p.header, 1u);
    TensorDesc sd = slangml_tensor_desc(p.header, 2u);
    float4* x = reinterpret<float4*>(sml_base(xd, 4u) + uint64_t(safeRow) * uint64_t(p.ldX) * 4u);
    float4 v = sml_ld_f4(x + safeQuad);

    float amax = max(max(abs(v.x), abs(v.y)), max(abs(v.z), abs(v.w)));
    amax = sml_cluster_max(amax, 8u);
    float d = amax * (1.0f / 127.0f);
    float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
    int4 q = int4(round(v * inv));
    q = clamp(q, int4(-127), int4(127));
    float qsum = sml_cluster_sum(float(q.x + q.y + q.z + q.w), 8u);  // exact: |qsum| <= 4064
    float sum = d * qsum;

    if (!active)
        return;
    uint* xq = reinterpret<uint*>(sml_base(qd, 1u) + uint64_t(row) * uint64_t(p.cols));
    xq[quad] = sml_pack_i8x4(q);
    if ((quad & 7u) == 0u)
    {
        float2* xs = reinterpret<float2*>(sml_base(sd, 8u) + uint64_t(row) * uint64_t(p.cols >> 5u) * 8u);
        xs[quad >> 3u] = float2(d, sum);
    }
}
