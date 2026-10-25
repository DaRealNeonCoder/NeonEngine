implementing slangml_common;

public static const uint SLANGML_DTYPE_MASK = 0xFFu;
public static const uint SLANGML_CAST_SATURATE = 1u << 8;
public static const uint SLANGML_CAST_ROUND_TO_NEAREST = 1u << 9;

public uint slangml_dtype_base(uint dtypeAndFlags)
{
    return dtypeAndFlags & SLANGML_DTYPE_MASK;
}

public float slangml_bf16_to_float(uint bits)
{
    return asfloat(bits << 16u);
}

public float slangml_fp16_to_float(uint bits)
{
    uint sign = bits >> 15u;
    uint exponent = (bits >> 10u) & 0x1Fu;
    uint mantissa = bits & 0x3FFu;
    if (exponent == 0u)
    {
        float subnormal = float(mantissa) * exp2(-24.0f);
        return sign != 0u ? -subnormal : subnormal;
    }
    if (exponent == 31u)
    {
        if (mantissa == 0u)
            return sign != 0u ? -asfloat(0x7F800000u) : asfloat(0x7F800000u);
        return asfloat(0x7FC00000u);
    }
    float magnitude = (1.0f + float(mantissa) * (1.0f / 1024.0f)) * exp2(float(exponent) - 15.0f);
    return sign != 0u ? -magnitude : magnitude;
}

public uint2 slangml_int64_to_uint2(int64_t value)
{
    uint64_t bits = uint64_t(value);
    return uint2(uint(bits), uint(bits >> 32u));
}

public int64_t slangml_uint2_to_int64(uint2 bits)
{
    uint64_t value = (uint64_t(bits.y) << 32u) | uint64_t(bits.x);
    return int64_t(value);
}

public uint slangml_float_to_bf16(float value)
{
    uint bits = asuint(value);
    uint rounding = 0x7FFFu + ((bits >> 16u) & 1u);
    return (bits + rounding) >> 16u;
}

public uint slangml_unpack_int4(uint packed, uint lane)
{
    uint nibble = (packed >> ((lane & 7u) * 4u)) & 0xFu;
    return nibble;
}

public int slangml_unpack_signed_int4(uint packed, uint lane)
{
    uint nibble = slangml_unpack_int4(packed, lane);
    return (nibble & 8u) != 0u ? int(nibble) - 16 : int(nibble);
}

public uint slangml_pack_int4(uint packed, uint lane, uint value)
{
    uint shift = (lane & 7u) * 4u;
    uint mask = 0xFu << shift;
    return (packed & ~mask) | ((value & 0xFu) << shift);
}

public int slangml_round_to_int(float value)
{
    return int(value >= 0.0f ? floor(value + 0.5f) : ceil(value - 0.5f));
}

public int slangml_saturate_int8(float value)
{
    return slangml_round_to_int(clamp(value, -128.0f, 127.0f));
}

public uint slangml_saturate_uint8(float value)
{
    return uint(clamp(slangml_round_to_int(value), 0, 255));
}

public int slangml_saturate_int32(float value)
{
    return slangml_round_to_int(clamp(value, -2147483648.0f, 2147483520.0f));
}

// IEEE-like decode helpers for the two optional 8-bit formats.  E4M3 uses the
// finite-only FN encoding; E5M2 preserves Inf/NaN.  Callers may choose a
// different policy in a specialized kernel if their model requires it.
public float slangml_fp8_e4m3_to_float(uint bits)
{
    uint sign = bits >> 7u;
    uint exponent = (bits >> 3u) & 0xFu;
    uint mantissa = bits & 0x7u;
    if (exponent == 0u)
        return (sign != 0u ? -1.0f : 1.0f) * (float(mantissa) * exp2(-9.0f));
    float magnitude = (1.0f + float(mantissa) * (1.0f / 8.0f)) * exp2(float(exponent) - 8.0f);
    return sign != 0u ? -magnitude : magnitude;
}

public float slangml_fp8_e5m2_to_float(uint bits)
{
    uint sign = bits >> 7u;
    uint exponent = (bits >> 2u) & 0x1Fu;
    uint mantissa = bits & 0x3u;
    if (exponent == 0u)
        return (sign != 0u ? -1.0f : 1.0f) * (float(mantissa) * exp2(-16.0f));
    if (exponent == 31u)
        return mantissa == 0u ? (sign != 0u ? -asfloat(0x7F800000u) : asfloat(0x7F800000u)) : asfloat(0x7FC00000u);
    float magnitude = (1.0f + float(mantissa) * 0.25f) * exp2(float(exponent) - 15.0f);
    return sign != 0u ? -magnitude : magnitude;
}

public uint slangml_float_to_fp8_e4m3(float value)
{
    if (value == 0.0f)
        return 0u;
    uint sign = value < 0.0f ? 0x80u : 0u;
    float magnitude = abs(value);
    int exponent = int(floor(log2(magnitude)));
    int encodedExponent = exponent + 8;
    if (encodedExponent <= 0)
        return sign | uint(clamp(slangml_round_to_int(magnitude * exp2(9.0f)), 0, 7));
    if (encodedExponent >= 15)
        return sign | 0x77u;
    float normalized = magnitude * exp2(float(-exponent)) - 1.0f;
    uint mantissa = uint(clamp(slangml_round_to_int(normalized * 8.0f), 0, 7));
    if (mantissa == 8u)
    {
        mantissa = 0u;
        encodedExponent++;
    }
    return sign | (uint(clamp(encodedExponent, 1, 15)) << 3u) | mantissa;
}

public uint slangml_float_to_fp8_e5m2(float value)
{
    if (value == 0.0f)
        return 0u;
    uint sign = value < 0.0f ? 0x80u : 0u;
    float magnitude = abs(value);
    if (isnan(magnitude))
        return sign | 0x7Fu;
    if (isinf(magnitude))
        return sign | 0x7Cu;
    int exponent = int(floor(log2(magnitude)));
    int encodedExponent = exponent + 15;
    if (encodedExponent <= 0)
        return sign | uint(clamp(slangml_round_to_int(magnitude * exp2(16.0f)), 0, 3));
    if (encodedExponent >= 31)
        return sign | 0x7Cu;
    float normalized = magnitude * exp2(float(-exponent)) - 1.0f;
    uint mantissa = uint(clamp(slangml_round_to_int(normalized * 4.0f), 0, 3));
    if (mantissa == 4u)
    {
        mantissa = 0u;
        encodedExponent++;
    }
    return sign | (uint(clamp(encodedExponent, 1, 30)) << 2u) | mantissa;
}

public float slangml_load_as_float(uint rawBits, uint dtype)
{
    switch (dtype)
    {
    case SLANGML_DTYPE_FP32: return asfloat(rawBits);
    case SLANGML_DTYPE_FP16: return slangml_fp16_to_float(rawBits & 0xFFFFu);
    case SLANGML_DTYPE_BF16: return slangml_bf16_to_float(rawBits & 0xFFFFu);
    case SLANGML_DTYPE_INT8: return float(int(rawBits << 24u) >> 24);
    case SLANGML_DTYPE_UINT8: return float(rawBits & 0xFFu);
    case SLANGML_DTYPE_INT32: return float(asint(rawBits));
    case SLANGML_DTYPE_BOOL: return (rawBits & 1u) != 0u ? 1.0f : 0.0f;
    case SLANGML_DTYPE_FP8_E4M3: return slangml_fp8_e4m3_to_float(rawBits);
    case SLANGML_DTYPE_FP8_E5M2: return slangml_fp8_e5m2_to_float(rawBits);
    default: return 0.0f;
    }
}

public float slangml_load_as_float64(uint2 rawBits, uint dtype)
{
    if (dtype == SLANGML_DTYPE_INT64)
        return float(slangml_uint2_to_int64(rawBits));
    return slangml_load_as_float(rawBits.x, dtype);
}
