import slangml_common;

struct GenerateParams
{
    KernelHeader header;
    float start;
    float end;
    float step;
    uint inclusive;
};

[[vk::push_constant]]
ConstantBuffer<KernelPushConstants> smlGeneratePush;

float sml_generate_input(TensorDesc tensor, uint linear)
{
    return slangml_bda_load_f32(slangml_final_byte_address(tensor, int(linear), 4u));
}

void sml_generate_store(TensorDesc tensor, uint linear, float value)
{
    slangml_bda_store_f32(slangml_final_byte_address(tensor, int(linear), 4u), value);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_elementwise_copy_fwd_f32_contig(uint3 dtid : SV_DispatchThreadID)
{
    GenerateParams* params = reinterpret<GenerateParams*>(smlGeneratePush.headerAddress);
    TensorDesc input = slangml_tensor_desc(params.header, 0u);
    TensorDesc output = slangml_tensor_desc(params.header, 1u);
    uint linear = params.header.dispatchBase.x + dtid.x;
    if (linear >= params.header.numel)
        return;
    sml_generate_store(output, linear, sml_generate_input(input, linear));
}

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_elementwise_fill_fwd_f32_contig(uint3 dtid : SV_DispatchThreadID)
{
    GenerateParams* params = reinterpret<GenerateParams*>(smlGeneratePush.headerAddress);
    TensorDesc output = slangml_tensor_desc(params.header, 0u);
    uint linear = params.header.dispatchBase.x + dtid.x;
    if (linear >= params.header.numel)
        return;
    sml_generate_store(output, linear, params.start);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_elementwise_arange_fwd_f32_contig(uint3 dtid : SV_DispatchThreadID)
{
    GenerateParams* params = reinterpret<GenerateParams*>(smlGeneratePush.headerAddress);
    TensorDesc output = slangml_tensor_desc(params.header, 0u);
    uint linear = params.header.dispatchBase.x + dtid.x;
    if (linear >= params.header.numel)
        return;
    sml_generate_store(output, linear, fma(float(linear), params.step, params.start));
}

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_elementwise_linspace_fwd_f32_contig(uint3 dtid : SV_DispatchThreadID)
{
    GenerateParams* params = reinterpret<GenerateParams*>(smlGeneratePush.headerAddress);
    TensorDesc output = slangml_tensor_desc(params.header, 0u);
    uint linear = params.header.dispatchBase.x + dtid.x;
    if (linear >= params.header.numel)
        return;
    uint denominator = params.header.numel > 1u ? params.header.numel - 1u : 1u;
    float t = float(linear) / float(denominator);
    float value = params.header.numel > 1u ? lerp(params.start, params.end, t) : params.start;
    sml_generate_store(output, linear, value);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_elementwise_eye_fwd_f32_contig(uint3 dtid : SV_DispatchThreadID)
{
    GenerateParams* params = reinterpret<GenerateParams*>(smlGeneratePush.headerAddress);
    TensorDesc output = slangml_tensor_desc(params.header, 0u);
    uint linear = params.header.dispatchBase.x + dtid.x;
    uint rows = uint(params.start);
    uint cols = uint(params.end);
    uint total = rows * cols;
    if (linear >= total)
        return;
    uint row = linear / cols;
    uint col = linear - row * cols;
    sml_generate_store(output, linear, row == col ? 1.0f : 0.0f);
}
