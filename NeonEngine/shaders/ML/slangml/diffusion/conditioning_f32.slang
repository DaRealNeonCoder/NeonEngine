import slangml_common;

// Diffusion conditioning and denoiser preconditioning [SI Alg 20, Alg 22].
//
//   sml_diffusion_fourier_fwd_f32          FourierEmbedding(1/4 log(t/sigma), D)
//   sml_diffusion_input_scale_fwd_f32      r_noisy = x_noisy / sqrt(t^2 + sigma^2)
//   sml_diffusion_output_combine_fwd_f32   x_out = c_skip * x_noisy + c_out * r_update
//
// The Fourier frequencies w and phases b are sampled once on the host
// (N(0, I)) and are frozen parameters, so the embedding has no backward pass.
//
// The output combine uses the EDM preconditioning form
//   c_skip = sigma^2 / (sigma^2 + t^2),   c_out = sigma * t / sqrt(sigma^2 + t^2)
// confirmed against the rendered SI Alg 20 step 8 (see docs/af3_ops.md).
//
// t is a per-row or per-atom noise level array (the host broadcasts the
// per-structure t_hat to atoms), which lets one dispatch cover a batch of
// diffusion copies that each have their own noise level.

struct FourierParams
{
    KernelHeader header;
    uint rows;
    uint dims;
    float sigmaData;
    float reserved;
};

struct PrecondParams
{
    KernelHeader header;
    uint atoms;
    uint reserved0;
    float sigmaData;
    float reserved1;
};

[[vk::push_constant]]
ConstantBuffer<KernelPushConstants> smlConditioningPush;

float sml_cond_load(TensorDesc t, uint index)
{
    return slangml_bda_load_f32(slangml_final_byte_address(t, int(index), 4u));
}

void sml_cond_store(TensorDesc t, uint index, float value)
{
    slangml_bda_store_f32(slangml_final_byte_address(t, int(index), 4u), value);
}

// Inputs: t[rows], w[dims], b[dims].  Output: [rows, dims].
[shader("compute")]
[numthreads(256, 1, 1)]
void sml_diffusion_fourier_fwd_f32(uint3 dtid : SV_DispatchThreadID)
{
    FourierParams* p = reinterpret<FourierParams*>(smlConditioningPush.headerAddress);
    if ((p.header.dispatchBase.x + dtid.x) >= p.rows * p.dims)
        return;
    TensorDesc t = slangml_tensor_desc(p.header, 0u);
    TensorDesc w = slangml_tensor_desc(p.header, 1u);
    TensorDesc b = slangml_tensor_desc(p.header, 2u);
    TensorDesc output = slangml_tensor_desc(p.header, 3u);

    uint row = (p.header.dispatchBase.x + dtid.x) / p.dims;
    uint d = (p.header.dispatchBase.x + dtid.x) - row * p.dims;
    float x = 0.25f * log(sml_cond_load(t, row) / p.sigmaData);
    float phase = fma(x, sml_cond_load(w, d), sml_cond_load(b, d));
    sml_cond_store(output, (p.header.dispatchBase.x + dtid.x), cos(6.283185307179586f * phase));
}

// r = x / sqrt(t^2 + sigma^2), per coordinate.  Inputs: x[atoms*3], t[atoms].
[shader("compute")]
[numthreads(256, 1, 1)]
void sml_diffusion_input_scale_fwd_f32(uint3 dtid : SV_DispatchThreadID)
{
    PrecondParams* p = reinterpret<PrecondParams*>(smlConditioningPush.headerAddress);
    if ((p.header.dispatchBase.x + dtid.x) >= p.atoms * 3u)
        return;
    TensorDesc x = slangml_tensor_desc(p.header, 0u);
    TensorDesc t = slangml_tensor_desc(p.header, 1u);
    TensorDesc output = slangml_tensor_desc(p.header, 2u);

    uint atom = (p.header.dispatchBase.x + dtid.x) / 3u;
    float tv = sml_cond_load(t, atom);
    float scale = rsqrt(fma(tv, tv, p.sigmaData * p.sigmaData));
    sml_cond_store(output, (p.header.dispatchBase.x + dtid.x), sml_cond_load(x, (p.header.dispatchBase.x + dtid.x)) * scale);
}

// x_out = c_skip * x_noisy + c_out * r_update.  Inputs: x_noisy[atoms*3],
// r_update[atoms*3], t[atoms].
[shader("compute")]
[numthreads(256, 1, 1)]
void sml_diffusion_output_combine_fwd_f32(uint3 dtid : SV_DispatchThreadID)
{
    PrecondParams* p = reinterpret<PrecondParams*>(smlConditioningPush.headerAddress);
    if ((p.header.dispatchBase.x + dtid.x) >= p.atoms * 3u)
        return;
    TensorDesc xNoisy = slangml_tensor_desc(p.header, 0u);
    TensorDesc rUpdate = slangml_tensor_desc(p.header, 1u);
    TensorDesc t = slangml_tensor_desc(p.header, 2u);
    TensorDesc output = slangml_tensor_desc(p.header, 3u);

    uint atom = (p.header.dispatchBase.x + dtid.x) / 3u;
    float sigma2 = p.sigmaData * p.sigmaData;
    float tv = sml_cond_load(t, atom);
    float denom = rsqrt(sigma2 + tv * tv);
    float cSkip = sigma2 / (sigma2 + tv * tv);
    float cOut = p.sigmaData * tv * denom;
    sml_cond_store(output, (p.header.dispatchBase.x + dtid.x),
        fma(cSkip, sml_cond_load(xNoisy, (p.header.dispatchBase.x + dtid.x)), cOut * sml_cond_load(rUpdate, (p.header.dispatchBase.x + dtid.x))));
}
