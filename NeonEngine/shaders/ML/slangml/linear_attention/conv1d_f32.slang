// Causal depthwise conv1d with a rolling state, for Gated DeltaNet / Mamba-style
// short convolutions (docs/linear_attention.md).
//
//   y[t][c] = act( sum_{j<W} w[c][j] * in(t - (W-1) + j)[c] + bias[c] )
//   in(tau) = x[tau] for tau >= 0, else stateIn[(W-1) + tau][c]   (zeros without SML_CONV_HAS_STATE)
//   stateOut[i][c] = in(T - (W-1) + i), i < W-1                   (SML_CONV_WRITE_STATE)
//
// HF layout mapping: conv1d.weight [C, 1, W] -> w [C][W]. The state is stored
// channel-minor ([W-1][C]) so every access is coalesced; HF's conv_state
// [C][W] must be transposed when importing.
//
// One thread per channel walks `tokensPerThread` tokens with the window in
// registers, so x is read once and y written once. Grid: x = ceil(C / 128),
// y = ceil(T / tokensPerThread). stateOut may alias stateIn only when
// y == 1 (decode); with several token chunks the first chunk still reads the
// old state while the last one writes the new state.
//
// Tensor table: 0 x [T][ldX], 1 w [C][W], 2 bias [C], 3 stateIn [W-1][C],
//               4 y [T][ldY], 5 stateOut [W-1][C].
import slangml_common;
import slangml_llm;

static const uint SML_CONV_MAX_W = 8u;
static const uint SML_CONV_SILU = 1u;
static const uint SML_CONV_BIAS = 2u;
static const uint SML_CONV_HAS_STATE = 4u;
static const uint SML_CONV_WRITE_STATE = 8u;

struct Conv1dParams
{
    KernelHeader header;
    uint T;                // 48
    uint C;                // 52
    uint W;                // 56 (<= 8)
    uint flags;            // 60
    uint ldX;              // 64
    uint ldY;              // 68
    uint tokensPerThread;  // 72
    uint reserved;         // 76
};

[[vk::push_constant]]
ConstantBuffer<KernelPushConstants> smlConvPush;

[shader("compute")]
[numthreads(128, 1, 1)]
void sml_linattn_conv1d_fwd_f32(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID)
{
    Conv1dParams* p = reinterpret<Conv1dParams*>(smlConvPush.headerAddress);
    uint c = (p.header.dispatchBase.x + gid.x) * 128u + tid.x;
    if (c >= p.C)
        return;
    uint tBegin = (p.header.dispatchBase.y + gid.y) * p.tokensPerThread;
    if (tBegin >= p.T)
        return;
    uint tEnd = min(tBegin + p.tokensPerThread, p.T);
    uint W = p.W;
    uint hist = W - 1u;

    float* x = reinterpret<float*>(sml_base(slangml_tensor_desc(p.header, 0u), 4u));
    float* w = reinterpret<float*>(sml_base(slangml_tensor_desc(p.header, 1u), 4u)) + c * W;
    float* stateIn = reinterpret<float*>(sml_base(slangml_tensor_desc(p.header, 3u), 4u));
    float* y = reinterpret<float*>(sml_base(slangml_tensor_desc(p.header, 4u), 4u));
    float* stateOut = reinterpret<float*>(sml_base(slangml_tensor_desc(p.header, 5u), 4u));
    float bias = (p.flags & SML_CONV_BIAS) != 0u
        ? reinterpret<float*>(sml_base(slangml_tensor_desc(p.header, 2u), 4u))[c] : 0.0f;
    bool hasState = (p.flags & SML_CONV_HAS_STATE) != 0u;

    float wk[SML_CONV_MAX_W];
    float win[SML_CONV_MAX_W];  // win[j] = in(t - hist + j) for the current t
    [ForceUnroll]
    for (uint j = 0; j < SML_CONV_MAX_W; ++j)
    {
        wk[j] = j < W ? w[j] : 0.0f;
        win[j] = 0.0f;
    }
    // Preload the history window for tBegin.
    [ForceUnroll]
    for (uint j = 0; j < SML_CONV_MAX_W - 1u; ++j)
    {
        if (j < hist)
        {
            int tau = int(tBegin) - int(hist) + int(j);
            win[j] = tau >= 0 ? x[uint(tau) * p.ldX + c]
                              : (hasState ? stateIn[uint(int(hist) + tau) * p.C + c] : 0.0f);
        }
    }

    for (uint t = tBegin; t < tEnd; ++t)
    {
        float xt = x[t * p.ldX + c];
        float acc = bias;
        [ForceUnroll]
        for (uint j = 0; j < SML_CONV_MAX_W; ++j)
        {
            float v = j < hist ? win[j] : (j == hist ? xt : 0.0f);
            acc = fma(wk[j], v, acc);
        }
        y[t * p.ldY + c] = (p.flags & SML_CONV_SILU) != 0u ? sml_silu(acc) : acc;
        // Shift the window left by one and append xt.
        [ForceUnroll]
        for (uint j = 0; j + 1u < SML_CONV_MAX_W; ++j)
            win[j] = (j + 1u < hist) ? win[j + 1u] : (j + 1u == hist ? xt : win[j]);
    }

    if ((p.flags & SML_CONV_WRITE_STATE) != 0u && tEnd == p.T)
    {
        // win now holds in(T - hist + j) for j < hist.
        [ForceUnroll]
        for (uint j = 0; j < SML_CONV_MAX_W - 1u; ++j)
            if (j < hist)
                stateOut[j * p.C + c] = win[j];
    }
}
