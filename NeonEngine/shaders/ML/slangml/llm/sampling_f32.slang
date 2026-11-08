// Token selection over large vocabularies (docs/llm_ops.md).
//
// Greedy:   sml_llm_argmax_partial_fwd_f32 -> sml_llm_argmax_final_fwd_f32
// Sampling: sml_llm_topk_partial_fwd_f32 -> sml_llm_topk_merge_fwd_f32
//           -> sml_llm_sample_topk_fwd_f32
// Penalties (before either path): sml_llm_logits_penalty_fwd_f32
//
// Rows are independent (batch / MTP draft positions). Logits are fp32 [R][ldLogits].
// Ties resolve to the lower index (torch.argmax / topk order).
//
// Chunking: a partial workgroup covers SML_SAMPLE_CHUNK = 256 * SML_SAMPLE_PER_THREAD
// logits (4096 by default). Grid for partial passes: x = numChunks, y = R.
// The top-k merge needs numChunks * K <= SML_SAMPLE_CHUNK (V <= 262144 at K = 64).
// Top-k extraction is iterative (K workgroup-argmax rounds); a radix-select
// variant is a known speed-up (docs/LLM_ROADMAP.md).
import slangml_common;
import slangml_llm;

#ifndef SML_SAMPLE_PER_THREAD
#define SML_SAMPLE_PER_THREAD 16
#endif
static const uint SML_SAMPLE_CHUNK = 256u * SML_SAMPLE_PER_THREAD;
static const uint SML_TOPK_MAX = 256u;
static const float SML_SAMPLE_NEG = -3.0e38f;

static const uint SML_SAMPLE_RANDOM_TENSOR = 1u;  // read u from tensor 4 instead of Philox

struct SampleParams
{
    KernelHeader header;
    uint R;            // 48 rows
    uint V;            // 52 vocabulary (or candidate count for merge)
    uint ldLogits;     // 56
    uint K;            // 60 top-k (<= 256 for sampling, <= SML_SAMPLE_CHUNK / numChunks for merge)
    uint numChunks;    // 64
    float temperature; // 68
    float topP;        // 72 1 = off
    float minP;        // 76 0 = off
    uint seed;         // 80
    uint step;         // 84 Philox counter (decode step)
    uint flags;        // 88
    uint reserved;     // 92
};

[[vk::push_constant]]
ConstantBuffer<KernelPushConstants> smlSamplePush;

groupshared float smlSampleV[32];
groupshared uint smlSampleI[32];

// Workgroup-wide (value, index) max over 256 threads; ties -> lower index.
[ForceInline]
void sml_wg_argmax(inout float v, inout uint idx, uint tidx)
{
    float wm = WaveActiveMax(v);
    uint wi = WaveActiveMin(v == wm ? idx : 0xFFFFFFFFu);
    uint laneCount = WaveGetLaneCount();
    GroupMemoryBarrierWithGroupSync();  // previous round's readers are done
    if (WaveGetLaneIndex() == 0u)
    {
        smlSampleV[tidx / laneCount] = wm;
        smlSampleI[tidx / laneCount] = wi;
    }
    GroupMemoryBarrierWithGroupSync();
    float bv = smlSampleV[0];
    uint bi = smlSampleI[0];
    for (uint s = 1u; s < 256u / laneCount; ++s)
    {
        float sv = smlSampleV[s];
        uint si = smlSampleI[s];
        if (sv > bv || (sv == bv && si < bi))
        {
            bv = sv;
            bi = si;
        }
    }
    v = bv;
    idx = bi;
}

// ---------------------------------------------------------------------------
// Greedy argmax. Workspace: float values [R][numChunks], uint indices [R][numChunks].
// Tensor table: 0 logits, 1 ws values, 2 ws indices, 3 out tokens i32 [R].
// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(256, 1, 1)]
void sml_llm_argmax_partial_fwd_f32(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID)
{
    SampleParams* p = reinterpret<SampleParams*>(smlSamplePush.headerAddress);
    uint row = gid.y;
    uint chunk = p.header.dispatchBase.x + gid.x;
    float* logits = reinterpret<float*>(sml_base(slangml_tensor_desc(p.header, 0u), 4u)) + row * p.ldLogits;
    float bv = SML_SAMPLE_NEG;
    uint bi = 0xFFFFFFFFu;
    [ForceUnroll]
    for (uint j = 0; j < SML_SAMPLE_PER_THREAD; ++j)
    {
        uint i = chunk * SML_SAMPLE_CHUNK + j * 256u + tid.x;  // coalesced
        if (i < p.V)
        {
            float v = logits[i];
            if (v > bv)
            {
                bv = v;
                bi = i;
            }
        }
    }
    sml_wg_argmax(bv, bi, tid.x);
    if (tid.x == 0u)
    {
        reinterpret<float*>(sml_base(slangml_tensor_desc(p.header, 1u), 4u))[row * p.numChunks + chunk] = bv;
        reinterpret<uint*>(sml_base(slangml_tensor_desc(p.header, 2u), 4u))[row * p.numChunks + chunk] = bi;
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_llm_argmax_final_fwd_f32(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID)
{
    SampleParams* p = reinterpret<SampleParams*>(smlSamplePush.headerAddress);
    uint row = gid.x;
    float* vals = reinterpret<float*>(sml_base(slangml_tensor_desc(p.header, 1u), 4u)) + row * p.numChunks;
    uint* idxs = reinterpret<uint*>(sml_base(slangml_tensor_desc(p.header, 2u), 4u)) + row * p.numChunks;
    float bv = SML_SAMPLE_NEG;
    uint bi = 0xFFFFFFFFu;
    for (uint c = tid.x; c < p.numChunks; c += 256u)
    {
        float v = vals[c];
        uint i = idxs[c];
        if (v > bv || (v == bv && i < bi))
        {
            bv = v;
            bi = i;
        }
    }
    sml_wg_argmax(bv, bi, tid.x);
    if (tid.x == 0u)
        reinterpret<uint*>(sml_base(slangml_tensor_desc(p.header, 3u), 4u))[row] = bi;
}

// ---------------------------------------------------------------------------
// Top-k. Each thread keeps SML_SAMPLE_PER_THREAD candidates in registers; K
// rounds of workgroup argmax extract them in descending order.
// partial: tensor 0 logits -> 1 values [R][numChunks][K], 2 indices.
// merge:   tensors 1, 2 -> 3 values [R][K], 4 indices [R][K]; V = numChunks * K.
// ---------------------------------------------------------------------------
[ForceInline]
void sml_topk_extract(float v[SML_SAMPLE_PER_THREAD], uint ix[SML_SAMPLE_PER_THREAD], uint K,
                      float* outV, uint* outI, uint tidx)
{
    float lv = SML_SAMPLE_NEG;
    uint li = 0xFFFFFFFFu;
    [ForceUnroll]
    for (uint j = 0; j < SML_SAMPLE_PER_THREAD; ++j)
        if (v[j] > lv || (v[j] == lv && ix[j] < li))
        {
            lv = v[j];
            li = ix[j];
        }
    for (uint r = 0; r < K; ++r)
    {
        float gv = lv;
        uint gi = li;
        sml_wg_argmax(gv, gi, tidx);
        if (tidx == 0u)
        {
            outV[r] = gv;
            outI[r] = gi;
        }
        if (li == gi && gi != 0xFFFFFFFFu)
        {
            // This thread owned the winner: drop it and rescan.
            lv = SML_SAMPLE_NEG;
            li = 0xFFFFFFFFu;
            [ForceUnroll]
            for (uint j = 0; j < SML_SAMPLE_PER_THREAD; ++j)
            {
                if (ix[j] == gi)
                {
                    v[j] = SML_SAMPLE_NEG;
                    ix[j] = 0xFFFFFFFFu;
                }
                if (v[j] > lv || (v[j] == lv && ix[j] < li))
                {
                    lv = v[j];
                    li = ix[j];
                }
            }
        }
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_llm_topk_partial_fwd_f32(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID)
{
    SampleParams* p = reinterpret<SampleParams*>(smlSamplePush.headerAddress);
    uint row = gid.y;
    uint chunk = p.header.dispatchBase.x + gid.x;
    float* logits = reinterpret<float*>(sml_base(slangml_tensor_desc(p.header, 0u), 4u)) + row * p.ldLogits;
    float v[SML_SAMPLE_PER_THREAD];
    uint ix[SML_SAMPLE_PER_THREAD];
    [ForceUnroll]
    for (uint j = 0; j < SML_SAMPLE_PER_THREAD; ++j)
    {
        uint i = chunk * SML_SAMPLE_CHUNK + j * 256u + tid.x;
        bool ok = i < p.V;
        v[j] = ok ? logits[i] : SML_SAMPLE_NEG;
        ix[j] = ok ? i : 0xFFFFFFFFu;
    }
    uint base = (row * p.numChunks + chunk) * p.K;
    sml_topk_extract(v, ix, p.K,
                     reinterpret<float*>(sml_base(slangml_tensor_desc(p.header, 1u), 4u)) + base,
                     reinterpret<uint*>(sml_base(slangml_tensor_desc(p.header, 2u), 4u)) + base, tid.x);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_llm_topk_merge_fwd_f32(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID)
{
    SampleParams* p = reinterpret<SampleParams*>(smlSamplePush.headerAddress);
    uint row = gid.x;
    uint n = p.numChunks * p.K;
    float* cv = reinterpret<float*>(sml_base(slangml_tensor_desc(p.header, 1u), 4u)) + row * n;
    uint* ci = reinterpret<uint*>(sml_base(slangml_tensor_desc(p.header, 2u), 4u)) + row * n;
    float v[SML_SAMPLE_PER_THREAD];
    uint ix[SML_SAMPLE_PER_THREAD];
    [ForceUnroll]
    for (uint j = 0; j < SML_SAMPLE_PER_THREAD; ++j)
    {
        uint i = j * 256u + tid.x;
        bool ok = i < n;
        v[j] = ok ? cv[i] : SML_SAMPLE_NEG;
        ix[j] = ok ? ci[i] : 0xFFFFFFFFu;
    }
    sml_topk_extract(v, ix, p.K,
                     reinterpret<float*>(sml_base(slangml_tensor_desc(p.header, 3u), 4u)) + row * p.K,
                     reinterpret<uint*>(sml_base(slangml_tensor_desc(p.header, 4u), 4u)) + row * p.K, tid.x);
}

// ---------------------------------------------------------------------------
// Sample from sorted top-k candidates: temperature, min-p, top-p, then inverse
// CDF with u ~ U[0,1) (Philox(seed, step, row) or tensor 4).
// Tensor table: 0 values [R][K] (descending), 1 indices [R][K], 2 out tokens
// i32 [R], 3 out probs [R][K] (filtered, renormalised; 0 for dropped), 4 u [R].
// Grid: x = R. temperature <= 0 picks the first candidate.
// ---------------------------------------------------------------------------
groupshared float smlSampleP[SML_TOPK_MAX];

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_llm_sample_topk_fwd_f32(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID)
{
    SampleParams* p = reinterpret<SampleParams*>(smlSamplePush.headerAddress);
    uint row = gid.x;
    uint K = min(p.K, SML_TOPK_MAX);
    float* vals = reinterpret<float*>(sml_base(slangml_tensor_desc(p.header, 0u), 4u)) + row * p.K;
    uint* idxs = reinterpret<uint*>(sml_base(slangml_tensor_desc(p.header, 1u), 4u)) + row * p.K;
    float* probsOut = reinterpret<float*>(sml_base(slangml_tensor_desc(p.header, 3u), 4u)) + row * p.K;
    bool greedy = p.temperature <= 0.0f;
    float invT = greedy ? 1.0f : 1.0f / p.temperature;
    float top = vals[0];
    if (tid.x < K)
    {
        float v = vals[tid.x];
        smlSampleP[tid.x] = (greedy ? (tid.x == 0u ? 1.0f : 0.0f) : exp((v - top) * invT));
    }
    GroupMemoryBarrierWithGroupSync();
    if (tid.x == 0u)
    {
        // K <= 256 sequential steps: cheap next to the vocabulary passes.
        float total = 0.0f;
        for (uint i = 0; i < K; ++i)
            total += smlSampleP[i];
        float pmax = smlSampleP[0] / total;
        uint keep = K;
        float cum = 0.0f;
        for (uint i = 0; i < K; ++i)
        {
            float pi = smlSampleP[i] / total;
            if (p.minP > 0.0f && pi < p.minP * pmax)
            {
                keep = i;
                break;
            }
            cum += pi;
            if (cum >= p.topP)
            {
                keep = i + 1u;
                break;
            }
        }
        keep = max(keep, 1u);
        float kept = 0.0f;
        for (uint i = 0; i < keep; ++i)
            kept += smlSampleP[i];
        float u = (p.flags & SML_SAMPLE_RANDOM_TENSOR) != 0u
            ? reinterpret<float*>(sml_base(slangml_tensor_desc(p.header, 4u), 4u))[row]
            : slangml_random_uniform(uint64_t(p.step), p.seed, row);
        float target = u * kept;
        uint choice = keep - 1u;
        float acc = 0.0f;
        for (uint i = 0; i < keep; ++i)
        {
            acc += smlSampleP[i];
            if (target < acc)
            {
                choice = i;
                break;
            }
        }
        reinterpret<uint*>(sml_base(slangml_tensor_desc(p.header, 2u), 4u))[row] = idxs[choice];
        for (uint i = 0; i < K; ++i)
            probsOut[i] = i < keep ? smlSampleP[i] / kept : 0.0f;
    }
}

// ---------------------------------------------------------------------------
// Penalties for tokens already generated, given unique (token, count) pairs:
//   repetition (HF): l = l > 0 ? l / rep : l * rep
//   frequency/presence (OpenAI): l -= freq * count + presence
// Tensor table: 0 logits [R][ldLogits] (in place), 1 tokens i32 [R][numPairs],
//   2 counts i32 [R][numPairs]. Grid: x = ceil(numPairs / 256), y = R.
// ---------------------------------------------------------------------------
struct PenaltyParams
{
    KernelHeader header;
    uint R;              // 48
    uint numPairs;       // 52
    uint ldLogits;       // 56
    float repetition;    // 60 1 = off
    float frequency;     // 64
    float presence;      // 68
    uint reserved0;      // 72
    uint reserved1;      // 76
};

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_llm_logits_penalty_fwd_f32(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID)
{
    PenaltyParams* p = reinterpret<PenaltyParams*>(smlSamplePush.headerAddress);
    uint row = gid.y;
    uint i = (p.header.dispatchBase.x + gid.x) * 256u + tid.x;
    if (row >= p.R || i >= p.numPairs)
        return;
    uint token = reinterpret<uint*>(sml_base(slangml_tensor_desc(p.header, 1u), 4u))[row * p.numPairs + i];
    int count = reinterpret<int*>(sml_base(slangml_tensor_desc(p.header, 2u), 4u))[row * p.numPairs + i];
    if (count <= 0)
        return;
    float* logits = reinterpret<float*>(sml_base(slangml_tensor_desc(p.header, 0u), 4u)) + row * p.ldLogits;
    float l = logits[token];
    if (p.repetition != 1.0f)
        l = l > 0.0f ? l / p.repetition : l * p.repetition;
    l -= p.frequency * float(count) + p.presence;
    logits[token] = l;
}
