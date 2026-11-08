// Training GEMM (docs/gpt_oss_training.md): fp32 in, fp32 accumulate.
//
//   C[m][n] = alpha * sum_k A(m, k) B(k, n)  (+ bias[n]) (+ R[m][n]) (+ C[m][n])
//
// A and B are addressed through element strides, so one entry point covers
// every product a linear layer needs:
//   forward   Y  = X W^T   A = X  (sAm = ldX, sAk = 1)   B(k, n) = W[n][k]  (sBk = 1, sBn = ldW)
//   dX        dX = dY W    A = dY (sAm = ldY, sAk = 1)   B(k, n) = W[k][n]  (sBk = ldW, sBn = 1)
//   dW        dW = dY^T X  A(m, k) = dY[k][m] (sAm = 1, sAk = ldY)   B(k, n) = X[k][n]
//
// Modes (p.mode):
//   0 plain. Grid z = split-K index: split s covers k in [s * kChunk, (s+1) * kChunk)
//     and writes C + s * strideCG (a partial buffer reduced by
//     sml_train_splitk_reduce_f32). With one split, C is the real output.
//   1 M-grouped (MoE expert products). Rows are expert-sorted and each 128-row
//     tile belongs to one expert: g = table[tileM] (SML_INVALID = unused tile).
//     B += g * strideBG, bias += g * strideBiasG.
//   2 K-grouped (MoE weight gradients). Grid z = expert g; k runs over that
//     expert's rows [table[g], table[g+1]); C += g * strideCG.
//
// Tiles: 128 x 128 x 8, 256 threads, 8 x 8 outputs per thread (two 4 x 4
// quads 64 apart in each direction so shared reads are float4 and
// conflict-free). The next k-slab is prefetched into registers while the
// current one is multiplied. Grid: x = ceil(N / 128), y = ceil(M / 128).
//
// Tensor table: 0 A, 1 B, 2 C, 3 bias [N], 4 R (ldC), 5 table (uint).
import slangml_common;
import slangml_llm;
import slangml_train;

static const uint SML_TGEMM_BIAS = 1u;
static const uint SML_TGEMM_RESID = 2u;
static const uint SML_TGEMM_ACCUM = 4u;

static const uint BM = 128u;
static const uint BN = 128u;
static const uint BK = 8u;
static const uint PAD = 4u;
static const uint LDS = BM + PAD;  // 132 floats per k-row of a shared tile

struct TrainGemmParams
{
    KernelHeader header;
    uint M;            // 48
    uint N;            // 52
    uint K;            // 56
    uint flags;        // 60
    uint sAm;          // 64
    uint sAk;          // 68
    uint sBk;          // 72
    uint sBn;          // 76
    uint ldC;          // 80
    uint mode;         // 84
    uint kChunk;       // 88
    uint strideBG;     // 92
    uint strideBiasG;  // 96
    uint strideCG;     // 100
    float alpha;       // 104
    uint reserved;     // 108
};

[[vk::push_constant]]
ConstantBuffer<KernelPushConstants> smlTGemmPush;

groupshared float smlTgA[BK * LDS];
groupshared float smlTgB[BK * LDS];

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_train_gemm_fwd_f32(uint3 gid : SV_GroupID, uint3 tid3 : SV_GroupThreadID)
{
    TrainGemmParams* p = reinterpret<TrainGemmParams*>(smlTGemmPush.headerAddress);
    uint tid = tid3.x;
    uint tileN = gid.x;
    uint tileM = gid.y;
    uint z = gid.z;

    float* A = sml_tf(p.header, 0u);
    float* B = sml_tf(p.header, 1u);
    float* C = sml_tf(p.header, 2u);
    float* bias = sml_tf(p.header, 3u);
    float* R = sml_tf(p.header, 4u);
    uint* table = sml_tu(p.header, 5u);

    uint kBegin = 0u;
    uint kEnd = p.K;
    if (p.mode == 0u)
    {
        kBegin = z * p.kChunk;
        kEnd = min(p.K, kBegin + p.kChunk);
        C = C + (z * p.strideCG);
    }
    else if (p.mode == 1u)
    {
        uint g = table[tileM];
        if (g == SML_INVALID)
            return;  // uniform per workgroup
        B = B + (g * p.strideBG);
        bias = bias + (g * p.strideBiasG);
    }
    else
    {
        kBegin = table[z];
        kEnd = table[z + 1u];
        C = C + (z * p.strideCG);
        if (kBegin >= kEnd && (p.flags & SML_TGEMM_ACCUM) != 0u)
            return;
    }

    uint m0 = tileM * BM;
    uint n0 = tileN * BN;
    bool aMContig = p.sAm == 1u;  // else k-contiguous (or general strides)
    bool bNContig = p.sBn == 1u;  // else k-contiguous (weights stored [N][K])

    // Global -> register prefetch of one 128 x 8 slab of A and 8 x 128 slab of B.
    float ra[4];
    float rb[4];
    uint aM[4], aK[4], bK[4], bN[4];
    [ForceUnroll]
    for (uint i = 0; i < 4u; ++i)
    {
        uint l = tid + 256u * i;
        aM[i] = aMContig ? (l & 127u) : (l >> 3u);
        aK[i] = aMContig ? (l >> 7u) : (l & 7u);
        bN[i] = bNContig ? (l & 127u) : (l >> 3u);
        bK[i] = bNContig ? (l >> 7u) : (l & 7u);
    }

    float acc[8][8];
    [ForceUnroll]
    for (uint i = 0; i < 8u; ++i)
        [ForceUnroll]
        for (uint j = 0; j < 8u; ++j)
            acc[i][j] = 0.0f;

    uint ty = tid >> 4u;  // 0..15
    uint tx = tid & 15u;  // 0..15

    // Prologue load.
    [ForceUnroll]
    for (uint i = 0; i < 4u; ++i)
    {
        uint m = m0 + aM[i];
        uint k = kBegin + aK[i];
        ra[i] = (m < p.M && k < kEnd) ? A[m * p.sAm + k * p.sAk] : 0.0f;
        uint n = n0 + bN[i];
        uint kb = kBegin + bK[i];
        rb[i] = (n < p.N && kb < kEnd) ? B[kb * p.sBk + n * p.sBn] : 0.0f;
    }

    for (uint k0 = kBegin; k0 < kEnd; k0 += BK)
    {
        [ForceUnroll]
        for (uint i = 0; i < 4u; ++i)
        {
            smlTgA[aK[i] * LDS + aM[i]] = ra[i];
            smlTgB[bK[i] * LDS + bN[i]] = rb[i];
        }
        GroupMemoryBarrierWithGroupSync();

        uint kn = k0 + BK;
        if (kn < kEnd)
        {
            [ForceUnroll]
            for (uint i = 0; i < 4u; ++i)
            {
                uint m = m0 + aM[i];
                uint k = kn + aK[i];
                ra[i] = (m < p.M && k < kEnd) ? A[m * p.sAm + k * p.sAk] : 0.0f;
                uint n = n0 + bN[i];
                uint kb = kn + bK[i];
                rb[i] = (n < p.N && kb < kEnd) ? B[kb * p.sBk + n * p.sBn] : 0.0f;
            }
        }

        [ForceUnroll]
        for (uint k = 0; k < BK; ++k)
        {
            float av[8];
            float bv[8];
            [ForceUnroll]
            for (uint c = 0; c < 4u; ++c)
            {
                av[c] = smlTgA[k * LDS + ty * 4u + c];
                av[c + 4u] = smlTgA[k * LDS + 64u + ty * 4u + c];
                bv[c] = smlTgB[k * LDS + tx * 4u + c];
                bv[c + 4u] = smlTgB[k * LDS + 64u + tx * 4u + c];
            }
            [ForceUnroll]
            for (uint i = 0; i < 8u; ++i)
                [ForceUnroll]
                for (uint j = 0; j < 8u; ++j)
                    acc[i][j] = fma(av[i], bv[j], acc[i][j]);
        }
        GroupMemoryBarrierWithGroupSync();
    }

    bool useBias = (p.flags & SML_TGEMM_BIAS) != 0u;
    bool useResid = (p.flags & SML_TGEMM_RESID) != 0u;
    bool useAccum = (p.flags & SML_TGEMM_ACCUM) != 0u;
    [ForceUnroll]
    for (uint i = 0; i < 8u; ++i)
    {
        uint m = m0 + (i < 4u ? ty * 4u + i : 64u + ty * 4u + (i - 4u));
        if (m >= p.M)
            continue;
        [ForceUnroll]
        for (uint j = 0; j < 8u; ++j)
        {
            uint n = n0 + (j < 4u ? tx * 4u + j : 64u + tx * 4u + (j - 4u));
            if (n >= p.N)
                continue;
            uint idx = m * p.ldC + n;
            float v = p.alpha * acc[i][j];
            if (useBias)
                v += bias[n];
            if (useResid)
                v += R[idx];
            if (useAccum)
                v += C[idx];
            C[idx] = v;
        }
    }
}

// C[i] (+)= sum_s P[s * stride + i]  for i < n. Grid: x = ceil(n / 256).
// Tensor table: 0 P, 1 C.
struct SplitReduceParams
{
    KernelHeader header;
    uint n;        // 48
    uint splits;   // 52
    uint stride;   // 56
    uint accum;    // 60
};

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_train_splitk_reduce_f32(uint3 dtid : SV_DispatchThreadID)
{
    SplitReduceParams* p = reinterpret<SplitReduceParams*>(smlTGemmPush.headerAddress);
    uint i = (p.header.dispatchBase.x * 256u) + dtid.x;
    if (i >= p.n)
        return;
    float* P = sml_tf(p.header, 0u);
    float* C = sml_tf(p.header, 1u);
    float s = 0.0f;
    for (uint k = 0; k < p.splits; ++k)
        s += P[k * p.stride + i];
    C[i] = p.accum != 0u ? C[i] + s : s;
}
