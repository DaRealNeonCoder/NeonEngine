implementing slangml_common;

public struct PhiloxResult
{
    public uint4 value;
};

public static const uint SLANGML_PHILOX_M0 = 0xD2511F53u;
public static const uint SLANGML_PHILOX_M1 = 0xCD9E8D57u;
public static const uint SLANGML_PHILOX_W0 = 0x9E3779B9u;
public static const uint SLANGML_PHILOX_W1 = 0xBB67AE85u;

public PhiloxResult slangml_philox4x32_10(uint4 counter, uint2 key)
{
    uint4 state = counter;
    uint2 roundKey = key;
    for (uint round = 0; round < 10u; ++round)
    {
        uint hi0 = slangml_mul_hi(SLANGML_PHILOX_M0, state.x);
        uint lo0 = slangml_mul_lo(SLANGML_PHILOX_M0, state.x);
        uint hi1 = slangml_mul_hi(SLANGML_PHILOX_M1, state.z);
        uint lo1 = slangml_mul_lo(SLANGML_PHILOX_M1, state.z);
        state = uint4(hi1 ^ state.y ^ roundKey.x, lo1, hi0 ^ state.w ^ roundKey.y, lo0);
        roundKey.x += SLANGML_PHILOX_W0;
        roundKey.y += SLANGML_PHILOX_W1;
    }
    PhiloxResult result;
    result.value = state;
    return result;
}

// Portable 32x32 -> 64-bit multiply helpers.  They avoid relying on a target
// extension for uint64 arithmetic in the PRNG hot path.
public uint slangml_mul_lo(uint a, uint b)
{
    return uint((uint64_t(a) * uint64_t(b)) & uint64_t(0xFFFFFFFFu));
}

public uint slangml_mul_hi(uint a, uint b)
{
    return uint((uint64_t(a) * uint64_t(b)) >> 32u);
}

public uint4 slangml_random_u32x4(uint64_t counter, uint seed, uint stream)
{
    uint4 ctr = uint4(uint(counter), uint(counter >> 32u), stream, 0u);
    PhiloxResult generated = slangml_philox4x32_10(ctr, uint2(seed, seed ^ 0x9E3779B9u));
    return generated.value;
}

public float slangml_uint_to_uniform01(uint value)
{
    return (float(value >> 8u) + 0.5f) * (1.0f / 16777216.0f);
}

public float slangml_random_uniform(uint64_t counter, uint seed, uint stream)
{
    return slangml_uint_to_uniform01(slangml_random_u32x4(counter, seed, stream).x);
}

public float slangml_random_normal(uint64_t counter, uint seed, uint stream)
{
    uint4 randoms = slangml_random_u32x4(counter, seed, stream);
    float u0 = max(slangml_uint_to_uniform01(randoms.x), 1.0e-7f);
    float u1 = slangml_uint_to_uniform01(randoms.y);
    return sqrt(-2.0f * log(u0)) * cos(6.283185307179586f * u1);
}

public bool slangml_random_bernoulli(uint64_t counter, uint seed, uint stream, float probability)
{
    return slangml_random_uniform(counter, seed, stream) < probability;
}
