import slangml_common;

// Adaptive LayerNorm and adaLN-Zero gating for the diffusion conditioning
// [SI Alg 25, Alg 26].  The host computes LN(s) with its gain, then the two
// conditioning GEMMs:
//   scalePre = Linear(LN(s))        (bias term included, Alg 26 line 3)
//   shift    = LinearNoBias(s)
//
// AdaLN (Alg 26):        y = sigmoid(scalePre) * LN_noaffine(a) + shift
// adaLN-Zero gate (Alg 25 line 3, Alg 24 line 13):
//                        y = resid + sigmoid(gatePre) * x
//   gatePre = Linear(s, bias_init = -2.0) is a host GEMM.
//
// Layout: rows R x cols C, fp32, canonical.  AdaLN kernels use one 256-thread
// workgroup per row.  Statistics and all row reductions are fp32.

struct AdaLNParams
{
    KernelHeader header;
    uint rows;
    uint cols;
    float epsilon;
    float reserved;
};

struct AdaLNGateParams
{
    KernelHeader header;
    uint numel;
    uint reserved0;
    uint reserved1;
    uint reserved2;
};

[[vk::push_constant]]
ConstantBuffer<KernelPushConstants> smlAdaLNPush;

groupshared float smlAdaLNScratch[256];

// Reduce, then barrier, so the next reduction cannot overwrite scratch[0]
// while a slower lane is still reading the previous result.
float sml_adaln_reduce(float value, uint lane)
{
    float result = slangml_workgroup_reduce_add_f32_256(value, smlAdaLNScratch, lane);
    GroupMemoryBarrierWithGroupSync();
    return result;
}

float sml_adaln_load(TensorDesc t, uint index)
{
    return slangml_bda_load_f32(slangml_final_byte_address(t, int(index), 4u));
}

void sml_adaln_store(TensorDesc t, uint index, float value)
{
    slangml_bda_store_f32(slangml_final_byte_address(t, int(index), 4u), value);
}

// Row mean and inverse standard deviation, two-pass for numerical stability.
void sml_adaln_row_stats(TensorDesc a, uint rowBase, uint cols, float epsilon, uint lane,
                         out float mean, out float invStd)
{
    float sum = 0.0f;
    for (uint c = lane; c < cols; c += 256u)
        sum += sml_adaln_load(a, rowBase + c);
    mean = sml_adaln_reduce(sum, lane) / float(cols);

    float sumSq = 0.0f;
    for (uint c = lane; c < cols; c += 256u)
    {
        float d = sml_adaln_load(a, rowBase + c) - mean;
        sumSq += d * d;
    }
    float variance = sml_adaln_reduce(sumSq, lane) / float(cols);
    invStd = rsqrt(variance + epsilon);
}

// y = sigmoid(scalePre) * LN(a) + shift
[shader("compute")]
[numthreads(256, 1, 1)]
void sml_diffusion_adaln_fwd_f32(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID)
{
    AdaLNParams* p = reinterpret<AdaLNParams*>(smlAdaLNPush.headerAddress);
    TensorDesc a = slangml_tensor_desc(p.header, 0u);
    TensorDesc scalePre = slangml_tensor_desc(p.header, 1u);
    TensorDesc shift = slangml_tensor_desc(p.header, 2u);
    TensorDesc output = slangml_tensor_desc(p.header, 3u);

    uint row = p.header.dispatchBase.x + gid.x;
    if (row >= p.rows)
        return;
    uint rowBase = row * p.cols;

    float mean;
    float invStd;
    sml_adaln_row_stats(a, rowBase, p.cols, p.epsilon, tid.x, mean, invStd);
    for (uint c = tid.x; c < p.cols; c += 256u)
    {
        uint idx = rowBase + c;
        float normalized = (sml_adaln_load(a, idx) - mean) * invStd;
        float gain = slangml_sigmoid(sml_adaln_load(scalePre, idx));
        sml_adaln_store(output, idx, fma(gain, normalized, sml_adaln_load(shift, idx)));
    }
}

// Backward w.r.t. a (input) and scalePre.  dShift is dy and is returned to the
// host unchanged, so it is not written here.
//   g = sigmoid(sp), ahat = LN(a), dhat = dy * g
//   dsp = dy * ahat * g * (1 - g)
//   da  = invStd * (dhat - mean(dhat) - ahat * mean(dhat * ahat))
[shader("compute")]
[numthreads(256, 1, 1)]
void sml_diffusion_adaln_bwd_f32(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID)
{
    AdaLNParams* p = reinterpret<AdaLNParams*>(smlAdaLNPush.headerAddress);
    TensorDesc a = slangml_tensor_desc(p.header, 0u);
    TensorDesc scalePre = slangml_tensor_desc(p.header, 1u);
    TensorDesc dy = slangml_tensor_desc(p.header, 2u);
    TensorDesc da = slangml_tensor_desc(p.header, 3u);
    TensorDesc dScalePre = slangml_tensor_desc(p.header, 4u);

    uint row = p.header.dispatchBase.x + gid.x;
    if (row >= p.rows)
        return;
    uint rowBase = row * p.cols;

    float mean;
    float invStd;
    sml_adaln_row_stats(a, rowBase, p.cols, p.epsilon, tid.x, mean, invStd);

    float sumDhat = 0.0f;
    float sumDhatAhat = 0.0f;
    for (uint c = tid.x; c < p.cols; c += 256u)
    {
        uint idx = rowBase + c;
        float ahat = (sml_adaln_load(a, idx) - mean) * invStd;
        float g = slangml_sigmoid(sml_adaln_load(scalePre, idx));
        float dhat = sml_adaln_load(dy, idx) * g;
        sumDhat += dhat;
        sumDhatAhat += dhat * ahat;
    }
    float reducedDhat = sml_adaln_reduce(sumDhat, tid.x) / float(p.cols);
    float reducedDhatAhat = sml_adaln_reduce(sumDhatAhat, tid.x) / float(p.cols);

    for (uint c = tid.x; c < p.cols; c += 256u)
    {
        uint idx = rowBase + c;
        float ahat = (sml_adaln_load(a, idx) - mean) * invStd;
        float sp = sml_adaln_load(scalePre, idx);
        float g = slangml_sigmoid(sp);
        float dyv = sml_adaln_load(dy, idx);
        float dhat = dyv * g;
        sml_adaln_store(da, idx, invStd * (dhat - reducedDhat - ahat * reducedDhatAhat));
        sml_adaln_store(dScalePre, idx, dyv * ahat * g * (1.0f - g));
    }
}

// adaLN-Zero gated residual: out = resid + sigmoid(gatePre) * x
[shader("compute")]
[numthreads(256, 1, 1)]
void sml_diffusion_adaln_zero_residual_fwd_f32(uint3 dtid : SV_DispatchThreadID)
{
    AdaLNGateParams* p = reinterpret<AdaLNGateParams*>(smlAdaLNPush.headerAddress);
    if ((p.header.dispatchBase.x + dtid.x) >= p.numel)
        return;
    TensorDesc x = slangml_tensor_desc(p.header, 0u);
    TensorDesc gatePre = slangml_tensor_desc(p.header, 1u);
    TensorDesc resid = slangml_tensor_desc(p.header, 2u);
    TensorDesc output = slangml_tensor_desc(p.header, 3u);

    float g = slangml_sigmoid(sml_adaln_load(gatePre, (p.header.dispatchBase.x + dtid.x)));
    sml_adaln_store(output, (p.header.dispatchBase.x + dtid.x), fma(g, sml_adaln_load(x, (p.header.dispatchBase.x + dtid.x)), sml_adaln_load(resid, (p.header.dispatchBase.x + dtid.x))));
}

// Backward of the adaLN-Zero gate.  dResid = dy is passed through by the host.
//   dx = dy * g,   dGatePre = dy * x * g * (1 - g)
[shader("compute")]
[numthreads(256, 1, 1)]
void sml_diffusion_adaln_zero_residual_bwd_f32(uint3 dtid : SV_DispatchThreadID)
{
    AdaLNGateParams* p = reinterpret<AdaLNGateParams*>(smlAdaLNPush.headerAddress);
    if ((p.header.dispatchBase.x + dtid.x) >= p.numel)
        return;
    TensorDesc x = slangml_tensor_desc(p.header, 0u);
    TensorDesc gatePre = slangml_tensor_desc(p.header, 1u);
    TensorDesc dy = slangml_tensor_desc(p.header, 2u);
    TensorDesc dx = slangml_tensor_desc(p.header, 3u);
    TensorDesc dGatePre = slangml_tensor_desc(p.header, 4u);

    float g = slangml_sigmoid(sml_adaln_load(gatePre, (p.header.dispatchBase.x + dtid.x)));
    float dyv = sml_adaln_load(dy, (p.header.dispatchBase.x + dtid.x));
    sml_adaln_store(dx, (p.header.dispatchBase.x + dtid.x), dyv * g);
    sml_adaln_store(dGatePre, (p.header.dispatchBase.x + dtid.x), dyv * sml_adaln_load(x, (p.header.dispatchBase.x + dtid.x)) * g * (1.0f - g));
}
