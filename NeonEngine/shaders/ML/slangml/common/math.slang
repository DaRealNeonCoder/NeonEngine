implementing slangml_common;

public static const float SLANGML_LOG2_E = 1.4426950408889634f;
public static const float SLANGML_INV_SQRT_2 = 0.7071067811865475f;
public static const float SLANGML_SQRT_2 = 1.4142135623730951f;
public static const float SLANGML_EPSILON = 1.1920928955078125e-7f;

public float slangml_square(float x) { return x * x; }

public float slangml_sigmoid(float x)
{
    float e = exp(-abs(x));
    float result = 1.0f / (1.0f + e);
    return x >= 0.0f ? result : result * e;
}

public float slangml_log1pexp(float x)
{
    if (x > 20.0f)
        return x;
    if (x < -20.0f)
        return exp(x);
    return log(1.0f + exp(x));
}

public float slangml_log1p(float x)
{
    // The polynomial keeps the small-|x| path accurate without relying on a
    // target-specific log1p intrinsic.
    if (abs(x) < 1.0e-4f)
        return x - 0.5f * x * x + (x * x * x) / 3.0f;
    return log(1.0f + x);
}

public float slangml_erf(float x)
{
    // Abramowitz-Stegun 7.1.26; max error is suitable for activation kernels.
    float sign = x < 0.0f ? -1.0f : 1.0f;
    float ax = abs(x);
    float t = 1.0f / (1.0f + 0.3275911f * ax);
    float poly = (((((1.061405429f * t - 1.453152027f) * t)
        + 1.421413741f) * t - 0.284496736f) * t + 0.254829592f) * t;
    return sign * (1.0f - poly * exp(-ax * ax));
}

public float slangml_softplus(float x, float beta, float threshold)
{
    float scaled = beta * x;
    return scaled > threshold ? x : slangml_log1p(exp(scaled)) / beta;
}

public float slangml_softsign(float x)
{
    return x / (1.0f + abs(x));
}

public float slangml_gelu_exact(float x)
{
    return 0.5f * x * (1.0f + slangml_erf(x * SLANGML_INV_SQRT_2));
}

public float slangml_gelu_tanh(float x)
{
    float cubic = x * x * x;
    float inner = 0.7978845608028654f * (x + 0.044715f * cubic);
    return 0.5f * x * (1.0f + tanh(inner));
}

public float slangml_mish(float x)
{
    return x * tanh(slangml_softplus(x, 1.0f, 20.0f));
}

public float slangml_hardsigmoid(float x)
{
    return clamp(x * (1.0f / 6.0f) + 0.5f, 0.0f, 1.0f);
}

public float slangml_hardswish(float x)
{
    return x * slangml_hardsigmoid(x);
}

public float slangml_stable_logsumexp(float a, float b)
{
    float m = max(a, b);
    return m + log(exp(a - m) + exp(b - m));
}

public float slangml_safe_rsqrt(float x, float epsilon)
{
    return rsqrt(max(x, epsilon));
}

public float slangml_lerp(float a, float b, float t)
{
    return a + t * (b - a);
}

public float slangml_smoothstep(float edge0, float edge1, float x)
{
    float t = clamp((x - edge0) / (edge1 - edge0), 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}
