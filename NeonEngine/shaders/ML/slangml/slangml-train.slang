// SlangML training helper module (docs/gpt_oss_training.md).
//
// Small helpers shared by the train/ family: typed tensor-table pointers and
// the MXFP4 rounding rule used for quantization-aware training. Import with:
//     import slangml_common;
//     import slangml_llm;
//     import slangml_train;
module slangml_train;

import slangml_common;
import slangml_llm;

// Tensor-table entry `i` as a typed pointer (dataAddress + baseOffset).
[ForceInline] public float* sml_tf(KernelHeader h, uint i) { return reinterpret<float*>(sml_base(slangml_tensor_desc(h, i), 4u)); }
[ForceInline] public uint* sml_tu(KernelHeader h, uint i) { return reinterpret<uint*>(sml_base(slangml_tensor_desc(h, i), 4u)); }
[ForceInline] public float4* sml_tf4(KernelHeader h, uint i) { return reinterpret<float4*>(sml_base(slangml_tensor_desc(h, i), 4u)); }

public static const uint SML_INVALID = 0xFFFFFFFFu;

// ---------------------------------------------------------------------------
// MXFP4 rounding (must match the exporter in src/gpt_oss_train.cpp and
// tests/llm_formats.py quantize_mxfp4_checkpoint): per block of 32, the scale
// is 2^e with e the smallest integer such that amax / 2^e <= 6 (clamped to
// [-127, 127]); each value rounds to the nearest e2m1 magnitude
// {0, .5, 1, 1.5, 2, 3, 4, 6}, ties to the smaller magnitude.
// ---------------------------------------------------------------------------
[ForceInline]
public float sml_pow2(int e)
{
    // 2^e for e in [-126, 127] as an exact float; smaller exponents flush to 0.
    return e < -126 ? 0.0f : asfloat(uint(e + 127) << 23u);
}

[ForceInline]
public int sml_mxfp4_exponent(float amax)
{
    if (!(amax > 0.0f))
        return -127;
    // amax = m * 2^ex with m in [1, 2): exact from the bits (amax is normal here).
    int ex = int((asuint(amax) >> 23u) & 0xFFu) - 127;
    // Smallest e with amax <= 6 * 2^e. Start from a close guess and fix it up.
    int e = ex - 2;
    [ForceUnroll]
    for (int i = 0; i < 3; ++i)
    {
        if (amax > 6.0f * sml_pow2(e))
            e += 1;
    }
    return clamp(e, -127, 127);
}

[ForceInline]
public float sml_e2m1_round(float a)  // a >= 0, already divided by the block scale
{
    if (a <= 0.25f) return 0.0f;
    if (a <= 0.75f) return 0.5f;
    if (a <= 1.25f) return 1.0f;
    if (a <= 1.75f) return 1.5f;
    if (a <= 2.5f) return 2.0f;
    if (a <= 3.5f) return 3.0f;
    if (a <= 5.0f) return 4.0f;
    return 6.0f;
}
