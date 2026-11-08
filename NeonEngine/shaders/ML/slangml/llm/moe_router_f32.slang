// MoE router: scores -> top-k experts and combine weights (docs/llm_ops.md).
//
// Softmax mode (default; Mixtral, Qwen-MoE, gpt-oss):
//   p = softmax(logits); pick the top-k p; w = p, or p / sum(top-k p) with
//   SML_ROUTER_NORM_TOPK (equivalent to softmax over the selected logits).
// Sigmoid mode (SML_ROUTER_SIGMOID; DeepSeek-V3 style, without group limits):
//   s = sigmoid(logits); pick the top-k of s + bias (SML_ROUTER_SELECT_BIAS);
//   w = s, or s / sum(top-k s) with SML_ROUTER_NORM_TOPK.
// Finally w *= scaleOut (routed scaling factor). Ties pick the lower index.
// Experts are emitted in descending score order.
//
// One 32-lane group per token (4 tokens per 128-thread workgroup); E <= 512,
// topK <= 32. Grid: x = ceil(T / 4).
// Tensor table: 0 logits [T][ldLogits], 1 selection bias [E],
//               2 expert ids i32 [T][topK], 3 weights fp32 [T][topK].
import slangml_common;
import slangml_llm;

static const uint SML_ROUTER_PER_LANE = 16u;
static const uint SML_ROUTER_SIGMOID = 1u;
static const uint SML_ROUTER_NORM_TOPK = 2u;
static const uint SML_ROUTER_SELECT_BIAS = 4u;

struct RouterParams
{
    KernelHeader header;
    uint T;         // 48
    uint E;         // 52
    uint topK;      // 56
    uint flags;     // 60
    float scaleOut; // 64
    uint ldLogits;  // 68
    uint reserved0; // 72
    uint reserved1; // 76
};

[[vk::push_constant]]
ConstantBuffer<KernelPushConstants> smlRouterPush;

[shader("compute")]
[numthreads(128, 1, 1)]
void sml_llm_moe_router_fwd_f32(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID)
{
    RouterParams* p = reinterpret<RouterParams*>(smlRouterPush.headerAddress);
    uint lane = tid.x & 31u;
    uint t = (p.header.dispatchBase.x + gid.x) * 4u + (tid.x >> 5u);
    bool live = t < p.T;
    uint tt = min(t, p.T - 1u);  // dead groups mirror a live row; 64-wide subgroups stay convergent
    float* logits = reinterpret<float*>(sml_base(slangml_tensor_desc(p.header, 0u), 4u)) + tt * p.ldLogits;
    float* selBias = reinterpret<float*>(sml_base(slangml_tensor_desc(p.header, 1u), 4u));
    bool sigmoidMode = (p.flags & SML_ROUTER_SIGMOID) != 0u;
    bool useBias = (p.flags & SML_ROUTER_SELECT_BIAS) != 0u;

    float score[SML_ROUTER_PER_LANE];  // routing weight before normalisation
    float sel[SML_ROUTER_PER_LANE];    // selection key
    float mx = -3.0e38f;
    [ForceUnroll]
    for (uint j = 0; j < SML_ROUTER_PER_LANE; ++j)
    {
        uint e = lane + 32u * j;
        float l = e < p.E ? logits[e] : -3.0e38f;
        score[j] = l;
        mx = max(mx, l);
    }
    mx = sml_cluster_max(mx, 32u);
    float denom = 0.0f;
    [ForceUnroll]
    for (uint j = 0; j < SML_ROUTER_PER_LANE; ++j)
    {
        uint e = lane + 32u * j;
        if (sigmoidMode)
        {
            score[j] = e < p.E ? sml_sigmoid(score[j]) : 0.0f;
            sel[j] = e < p.E ? score[j] + (useBias ? selBias[e] : 0.0f) : -3.0e38f;
        }
        else
        {
            float ex = e < p.E ? exp(score[j] - mx) : 0.0f;
            sel[j] = e < p.E ? score[j] : -3.0e38f;  // logits order == probability order
            score[j] = ex;
            denom += ex;
        }
    }
    float invDenom = sigmoidMode ? 1.0f : 1.0f / sml_cluster_sum(denom, 32u);

    uint* idsOut = reinterpret<uint*>(sml_base(slangml_tensor_desc(p.header, 2u), 4u)) + tt * p.topK;
    float* wOut = reinterpret<float*>(sml_base(slangml_tensor_desc(p.header, 3u), 4u)) + tt * p.topK;
    float picked = 0.0f;  // lane r holds the weight of the r-th pick
    float pickedSum = 0.0f;
    uint pickedId = 0u;
    for (uint r = 0; r < p.topK; ++r)
    {
        float best = -3.0e38f;
        uint bestE = 0xFFFFFFFFu;
        [ForceUnroll]
        for (uint j = 0; j < SML_ROUTER_PER_LANE; ++j)
        {
            if (sel[j] > best)
            {
                best = sel[j];
                bestE = lane + 32u * j;
            }
        }
        float gBest = sml_cluster_max(best, 32u);
        uint cand = best == gBest ? bestE : 0xFFFFFFFFu;
        for (uint off = 1u; off < 32u; off <<= 1u)
            cand = min(cand, WaveReadLaneAt(cand, WaveGetLaneIndex() ^ off));
        uint winner = cand;
        float wv = 0.0f;
        [ForceUnroll]
        for (uint j = 0; j < SML_ROUTER_PER_LANE; ++j)
        {
            if (lane + 32u * j == winner)
            {
                wv = score[j];
                sel[j] = -3.0e38f;
            }
        }
        wv = sml_cluster_sum(wv, 32u) * invDenom;  // only the owner contributed
        pickedSum += wv;
        if (lane == r)
        {
            picked = wv;
            pickedId = winner;
        }
    }
    float norm = (p.flags & SML_ROUTER_NORM_TOPK) != 0u ? 1.0f / max(pickedSum, 1.0e-20f) : 1.0f;
    if (live && lane < p.topK)
    {
        idsOut[lane] = pickedId;
        wOut[lane] = picked * norm * p.scaleOut;
    }
}
