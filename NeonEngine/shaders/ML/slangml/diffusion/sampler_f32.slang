import slangml_common;

// Diffusion sampling loop pieces [SI Alg 18, Alg 19, Eq. 7].
//
//   sml_diffusion_sampler_init_fwd_f32       x = c0 * N(0, I)
//   sml_diffusion_sampler_step_fwd_f32       one update of the Alg 18 loop
//   sml_diffusion_centre_augment_fwd_f32     CentreRandomAugmentation (Alg 19)
//
// The schedule (Eq. 7) is computed on the host and passed as cPrev / c per step:
//   c_t = sigma * (s_max^(1/p) + t*(s_min^(1/p) - s_max^(1/p)))^p,  t = k/200
//
// Step (Alg 18), with cPrev = c_{tau-1} and c = c_tau:
//   gamma  = gamma0 if c > gammaMin else 0
//   tHat   = cPrev * (gamma + 1)
//   xTilde = x + noiseScale * sqrt(tHat^2 - cPrev^2) * N(0, I)
//   d      = (xTilde - xDenoised) / tHat
//   x      = xTilde + stepScale * (c - tHat) * d
//
// Randomness is Philox keyed by (seed, stream) with a per-element counter, so
// results do not depend on workgroup scheduling.

struct SamplerParams
{
    KernelHeader header;
    uint elements;
    uint stepIndex;
    uint seed;
    uint stream;
    float cPrev;
    float c;
    float gamma0;
    float gammaMin;
    float noiseScale;
    float stepScale;
    float reserved0;
    float reserved1;
};

struct AugmentParams
{
    KernelHeader header;
    uint structures;
    uint atomsPerStructure;
    uint seed;
    uint stream;
    float sTrans;
    float reserved0;
    float reserved1;
    float reserved2;
};

[[vk::push_constant]]
ConstantBuffer<KernelPushConstants> smlSamplerPush;

groupshared float smlSamplerScratch[256];

float sml_sampler_load(TensorDesc t, uint index)
{
    return slangml_bda_load_f32(slangml_final_byte_address(t, int(index), 4u));
}

void sml_sampler_store(TensorDesc t, uint index, float value)
{
    slangml_bda_store_f32(slangml_final_byte_address(t, int(index), 4u), value);
}

// Reduce, then barrier, so the next reduction cannot overwrite scratch[0]
// while a slower lane is still reading the previous result.
float sml_sampler_reduce(float value, uint lane)
{
    float result = slangml_workgroup_reduce_add_f32_256(value, smlSamplerScratch, lane);
    GroupMemoryBarrierWithGroupSync();
    return result;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_diffusion_sampler_init_fwd_f32(uint3 dtid : SV_DispatchThreadID)
{
    SamplerParams* p = reinterpret<SamplerParams*>(smlSamplerPush.headerAddress);
    if ((p.header.dispatchBase.x + dtid.x) >= p.elements)
        return;
    TensorDesc output = slangml_tensor_desc(p.header, 0u);
    float noise = slangml_random_normal(uint64_t((p.header.dispatchBase.x + dtid.x)), p.seed, p.stream);
    sml_sampler_store(output, (p.header.dispatchBase.x + dtid.x), p.cPrev * noise);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_diffusion_sampler_step_fwd_f32(uint3 dtid : SV_DispatchThreadID)
{
    SamplerParams* p = reinterpret<SamplerParams*>(smlSamplerPush.headerAddress);
    if ((p.header.dispatchBase.x + dtid.x) >= p.elements)
        return;
    TensorDesc x = slangml_tensor_desc(p.header, 0u);
    TensorDesc xDenoised = slangml_tensor_desc(p.header, 1u);
    TensorDesc output = slangml_tensor_desc(p.header, 2u);

    float gamma = p.c > p.gammaMin ? p.gamma0 : 0.0f;
    float tHat = p.cPrev * (gamma + 1.0f);
    float noise = slangml_random_normal(uint64_t(p.stepIndex) * uint64_t(p.elements) + uint64_t((p.header.dispatchBase.x + dtid.x)),
                                        p.seed, p.stream);
    float xv = sml_sampler_load(x, (p.header.dispatchBase.x + dtid.x));
    float xTilde = fma(p.noiseScale * sqrt(max(tHat * tHat - p.cPrev * p.cPrev, 0.0f)), noise, xv);
    float d = (xTilde - sml_sampler_load(xDenoised, (p.header.dispatchBase.x + dtid.x))) / tHat;
    sml_sampler_store(output, (p.header.dispatchBase.x + dtid.x), fma(p.stepScale * (p.c - tHat), d, xTilde));
}

// One workgroup per structure.  Subtract the atom mean, apply a uniform random
// rotation (quaternion from four normals), then add t = sTrans * N(0, I).
[shader("compute")]
[numthreads(256, 1, 1)]
void sml_diffusion_centre_augment_fwd_f32(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID)
{
    AugmentParams* p = reinterpret<AugmentParams*>(smlSamplerPush.headerAddress);
    uint structure = p.header.dispatchBase.x + gid.x;
    if (structure >= p.structures)
        return;
    TensorDesc x = slangml_tensor_desc(p.header, 0u);
    TensorDesc output = slangml_tensor_desc(p.header, 1u);

    uint base = structure * p.atomsPerStructure * 3u;
    uint lane = tid.x;

    float sum0 = 0.0f;
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    for (uint a = lane; a < p.atomsPerStructure; a += 256u)
    {
        sum0 += sml_sampler_load(x, base + a * 3u + 0u);
        sum1 += sml_sampler_load(x, base + a * 3u + 1u);
        sum2 += sml_sampler_load(x, base + a * 3u + 2u);
    }
    float3 mean = float3(
        sml_sampler_reduce(sum0, lane),
        sml_sampler_reduce(sum1, lane),
        sml_sampler_reduce(sum2, lane)) / float(p.atomsPerStructure);

    // Quaternion from four standard normals; the counter is per structure.
    uint64_t key = uint64_t(structure) * uint64_t(8);
    float4 quat = float4(
        slangml_random_normal(key + 0ul, p.seed, p.stream),
        slangml_random_normal(key + 1ul, p.seed, p.stream),
        slangml_random_normal(key + 2ul, p.seed, p.stream),
        slangml_random_normal(key + 3ul, p.seed, p.stream));
    quat *= rsqrt(max(dot(quat, quat), 1.0e-12f));
    float qw = quat.x;
    float qx = quat.y;
    float qy = quat.z;
    float qz = quat.w;
    float3x3 rot = float3x3(
        1.0f - 2.0f * (qy * qy + qz * qz), 2.0f * (qx * qy - qw * qz), 2.0f * (qx * qz + qw * qy),
        2.0f * (qx * qy + qw * qz), 1.0f - 2.0f * (qx * qx + qz * qz), 2.0f * (qy * qz - qw * qx),
        2.0f * (qx * qz - qw * qy), 2.0f * (qy * qz + qw * qx), 1.0f - 2.0f * (qx * qx + qy * qy));

    float3 translation = p.sTrans * float3(
        slangml_random_normal(key + 4ul, p.seed, p.stream),
        slangml_random_normal(key + 5ul, p.seed, p.stream),
        slangml_random_normal(key + 6ul, p.seed, p.stream));

    for (uint a = lane; a < p.atomsPerStructure; a += 256u)
    {
        float3 centred = float3(
            sml_sampler_load(x, base + a * 3u + 0u),
            sml_sampler_load(x, base + a * 3u + 1u),
            sml_sampler_load(x, base + a * 3u + 2u)) - mean;
        float3 moved = mul(rot, centred) + translation;
        sml_sampler_store(output, base + a * 3u + 0u, moved.x);
        sml_sampler_store(output, base + a * 3u + 1u, moved.y);
        sml_sampler_store(output, base + a * 3u + 2u, moved.z);
    }
}
