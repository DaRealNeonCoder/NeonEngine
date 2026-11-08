// Shared inner loops of the decode GEMV kernels (gemv_quant.slang,
// gemv_moe.slang). Textually included; see gemv_quant.slang for the layouts.
//
// One lane accumulates partial dot products for ROWS consecutive weight rows
// and MC activation rows; the caller reduces across the subgroup.

struct GemvShape
{
    uint N;           // weight rows
    uint K;           // input dim, multiple of 32
    uint M;           // live activation rows (<= MC)
    uint groupShift;  // log2(group / 32)
};

// Partial dot products of one lane for ROWS rows x MC activation rows.
struct GemvAcc<let MC : int, let ROWS : int>
{
    float v[ROWS][MC];
};

[ForceInline]
void sml_gemv_quant_rows<let FMT : int, let MC : int, let ROWS : int>(
    GemvShape p, uint64_t codesBase, uint64_t metaBase, uint64_t xqBase, uint64_t xsBase,
    uint rowBase, uint lane, uint laneCount, inout GemvAcc<MC, ROWS> acc)
{
    uint nb = p.K >> 5u;
    uint ngroups = nb >> p.groupShift;
    uint rows[ROWS];
    [ForceUnroll]
    for (uint r = 0; r < ROWS; ++r)
        rows[r] = min(rowBase + r, p.N - 1u);

    uint4* xq = reinterpret<uint4*>(xqBase);
    float2* xs = reinterpret<float2*>(xsBase);
    uint* meta = reinterpret<uint*>(metaBase);

    for (uint b = lane; b < nb; b += laneCount)
    {
        uint4 xa[MC];
        uint4 xb[MC];
        float2 xsv[MC];
        [ForceUnroll]
        for (uint m = 0; m < MC; ++m)
        {
            uint mm = min(m, p.M - 1u);
            xa[m] = sml_ld_u4(xq + (mm * nb + b) * 2u);
            xb[m] = sml_ld_u4(xq + ((mm * nb + b) * 2u + 1u));
            xsv[m] = sml_ld_f2(xs + (mm * nb + b));
        }
        uint g = b >> p.groupShift;
        [ForceUnroll]
        for (uint r = 0; r < ROWS; ++r)
        {
            if (FMT == SML_WFMT_MXFP4)
            {
                // meta = one e8m0 byte per 32-block; group is always 32 (groupShift ignored).
                float dw = sml_e8m0_half_scale(sml_e8m0_load(meta, rows[r] * nb + b));
                uint4 w = sml_ld_u4(reinterpret<uint4*>(codesBase) + (rows[r] * nb + b));
                [ForceUnroll]
                for (uint m = 0; m < MC; ++m)
                {
                    int dot = sml_mxfp4_dot(w, xa[m], xb[m]);
                    acc.v[r][m] = fma(dw * xsv[m].x, float(dot), acc.v[r][m]);
                }
            }
            else if (FMT == SML_WFMT_Q4)
            {
                float2 dm = sml_half2_to_float2(meta[rows[r] * ngroups + g]);
                uint4 w = sml_ld_u4(reinterpret<uint4*>(codesBase) + (rows[r] * nb + b));
                [ForceUnroll]
                for (uint m = 0; m < MC; ++m)
                {
                    int dot = sml_q4_dot(w, xa[m], xb[m]);
                    acc.v[r][m] = fma(dm.x * xsv[m].x, float(dot), fma(dm.y, xsv[m].y, acc.v[r][m]));
                }
            }
            else
            {
                float2 dm = sml_half2_to_float2(meta[rows[r] * ngroups + g]);
                uint4* wp = reinterpret<uint4*>(codesBase) + (rows[r] * nb + b) * 2u;
                uint4 w0 = sml_ld_u4(wp);
                uint4 w1 = sml_ld_u4(wp + 1);
                [ForceUnroll]
                for (uint m = 0; m < MC; ++m)
                {
                    int dot = sml_q8_dot(w0, w1, xa[m], xb[m]);
                    acc.v[r][m] = fma(dm.x * xsv[m].x, float(dot), fma(dm.y, xsv[m].y, acc.v[r][m]));
                }
            }
        }
    }
}

[ForceInline]
void sml_gemv_f16_rows<let FMT : int, let MC : int, let ROWS : int>(
    GemvShape p, uint64_t codesBase, uint64_t xBase,
    uint rowBase, uint lane, uint laneCount, inout GemvAcc<MC, ROWS> acc)
{
    uint chunks = p.K >> 3u;  // 8 halves per uint4
    uint rows[ROWS];
    [ForceUnroll]
    for (uint r = 0; r < ROWS; ++r)
        rows[r] = min(rowBase + r, p.N - 1u);
    float4* x = reinterpret<float4*>(xBase);
    uint4* w = reinterpret<uint4*>(codesBase);

    for (uint c = lane; c < chunks; c += laneCount)
    {
        float4 x0[MC];
        float4 x1[MC];
        [ForceUnroll]
        for (uint m = 0; m < MC; ++m)
        {
            uint mm = min(m, p.M - 1u);
            x0[m] = sml_ld_f4(x + (mm * chunks + c) * 2u);
            x1[m] = sml_ld_f4(x + ((mm * chunks + c) * 2u + 1u));
        }
        [ForceUnroll]
        for (uint r = 0; r < ROWS; ++r)
        {
            uint4 wv = sml_ld_u4(w + (rows[r] * chunks + c));
            float4 lo, hi;
            if (FMT == SML_WFMT_F16)
                sml_unpack_half8(wv, lo, hi);
            else
                sml_unpack_bf16x8(wv, lo, hi);
            [ForceUnroll]
            for (uint m = 0; m < MC; ++m)
                acc.v[r][m] += dot(lo, x0[m]) + dot(hi, x1[m]);
        }
    }
}

