import slangml_common;

// Fixed inference shape for the requested 27x27 MNIST-style recognizer.
// Architecture: 729 input pixels -> 128 ReLU features -> 10 classes.
public static const uint SML_DIGIT_INPUT_FEATURES = 27u * 27u;
public static const uint SML_DIGIT_HIDDEN_FEATURES = 128u;
public static const uint SML_DIGIT_CLASSES = 10u;

struct DigitRecognizerParams
{
    KernelHeader header;
    uint batchCount;
    int inputStrideBatch;
    int inputStrideFeature;
    int weight1StrideFeature;
    int weight1StrideHidden;
    int bias1Stride;
    int weight2StrideHidden;
    int weight2StrideClass;
    int bias2Stride;
    int outputStrideBatch;
    int outputStrideClass;
    int predictionStrideBatch;
    float inputScale;
    float inputBias;
    uint reserved0;
    uint reserved1;
};

[[vk::push_constant]]
ConstantBuffer<KernelPushConstants> smlDigitRecognizerPush;

groupshared float smlDigitHidden[SML_DIGIT_HIDDEN_FEATURES];
groupshared float smlDigitLogits[SML_DIGIT_CLASSES];

uint sml_digit_storage_bits(uint64_t address)
{
    uint word = slangml_bda_load_u32(address & ~uint64_t(3));
    uint shift = uint(address & uint64_t(2)) * 8u;
    return (word >> shift) & 0xFFFFu;
}

float sml_digit_load_fp16(TensorDesc tensor, int relative)
{
    return slangml_fp16_to_float(sml_digit_storage_bits(
        slangml_final_byte_address(tensor, relative, 2u)));
}

float sml_digit_load_bf16(TensorDesc tensor, int relative)
{
    return slangml_bf16_to_float(sml_digit_storage_bits(
        slangml_final_byte_address(tensor, relative, 2u)));
}

float sml_digit_load_weight(TensorDesc tensor, int relative, uint storageDtype)
{
    return storageDtype == SLANGML_DTYPE_BF16
        ? sml_digit_load_bf16(tensor, relative)
        : sml_digit_load_fp16(tensor, relative);
}

float sml_digit_load_input(TensorDesc tensor, DigitRecognizerParams* p, uint batch, uint feature)
{
    int relative = int(batch) * p.inputStrideBatch + int(feature) * p.inputStrideFeature;
    float value = slangml_bda_load_f32(slangml_final_byte_address(tensor, relative, 4u));
    return value * p.inputScale + p.inputBias;
}

float sml_digit_load_bias(TensorDesc tensor, int relative)
{
    return slangml_bda_load_f32(slangml_final_byte_address(tensor, relative, 4u));
}

void sml_digit_store_probability(
    TensorDesc tensor,
    DigitRecognizerParams* p,
    uint batch,
    uint classIndex,
    float value)
{
    int relative = int(batch) * p.outputStrideBatch + int(classIndex) * p.outputStrideClass;
    slangml_bda_store_f32(slangml_final_byte_address(tensor, relative, 4u), value);
}

void sml_digit_store_prediction(TensorDesc tensor, DigitRecognizerParams* p, uint batch, uint value)
{
    int relative = int(batch) * p.predictionStrideBatch;
    slangml_bda_store_u32(slangml_final_byte_address(tensor, relative, 4u), value);
}

void sml_digit_recognizer_forward(uint3 gid, uint lane, uint storageDtype)
{
    DigitRecognizerParams* p = reinterpret<DigitRecognizerParams*>(smlDigitRecognizerPush.headerAddress);
    TensorDesc input = slangml_tensor_desc(p.header, 0u);
    TensorDesc weight1 = slangml_tensor_desc(p.header, 1u);
    TensorDesc bias1 = slangml_tensor_desc(p.header, 2u);
    TensorDesc weight2 = slangml_tensor_desc(p.header, 3u);
    TensorDesc bias2 = slangml_tensor_desc(p.header, 4u);
    TensorDesc probabilities = slangml_tensor_desc(p.header, 5u);
    TensorDesc prediction = slangml_tensor_desc(p.header, 6u);
    uint batch = p.header.dispatchBase.x + gid.x;

    // Each active lane owns one hidden neuron. The fixed 729-term dot product
    // is accumulated in fp32 even though both weight matrices use 16-bit
    // storage in the selected pipeline variant.
    if (lane < SML_DIGIT_HIDDEN_FEATURES)
    {
        float value = sml_digit_load_bias(bias1, int(lane) * p.bias1Stride);
        for (uint feature = 0u; feature < SML_DIGIT_INPUT_FEATURES; ++feature)
        {
            int weightOffset = int(feature) * p.weight1StrideFeature
                + int(lane) * p.weight1StrideHidden;
            value = fma(
                sml_digit_load_input(input, p, batch, feature),
                sml_digit_load_weight(weight1, weightOffset, storageDtype),
                value);
        }
        smlDigitHidden[lane] = max(value, 0.0f);
    }
    GroupMemoryBarrierWithGroupSync();

    // Ten lanes compute the class logits from the ReLU hidden layer.
    if (lane < SML_DIGIT_CLASSES)
    {
        float value = sml_digit_load_bias(bias2, int(lane) * p.bias2Stride);
        for (uint hidden = 0u; hidden < SML_DIGIT_HIDDEN_FEATURES; ++hidden)
        {
            int weightOffset = int(hidden) * p.weight2StrideHidden
                + int(lane) * p.weight2StrideClass;
            value = fma(
                smlDigitHidden[hidden],
                sml_digit_load_weight(weight2, weightOffset, storageDtype),
                value);
        }
        smlDigitLogits[lane] = value;
    }
    GroupMemoryBarrierWithGroupSync();

    // The output head is only ten values, so one lane performs the stable
    // softmax and argmax. This avoids a second dispatch and keeps the host
    // integration to one inference pipeline.
    if (lane == 0u)
    {
        float maximum = -asfloat(0x7F800000u);
        uint bestClass = 0u;
        for (uint classIndex = 0u; classIndex < SML_DIGIT_CLASSES; ++classIndex)
        {
            if (smlDigitLogits[classIndex] > maximum)
            {
                maximum = smlDigitLogits[classIndex];
                bestClass = classIndex;
            }
        }

        float sum = 0.0f;
        for (uint classIndex = 0u; classIndex < SML_DIGIT_CLASSES; ++classIndex)
            sum += exp(smlDigitLogits[classIndex] - maximum);

        for (uint classIndex = 0u; classIndex < SML_DIGIT_CLASSES; ++classIndex)
            sml_digit_store_probability(
                probabilities,
                p,
                batch,
                classIndex,
                exp(smlDigitLogits[classIndex] - maximum) / sum);
        sml_digit_store_prediction(prediction, p, batch, bestClass);
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_model_digit_recognizer_fwd_f16_f32(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID)
{
    sml_digit_recognizer_forward(gid, tid.x, SLANGML_DTYPE_FP16);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void sml_model_digit_recognizer_fwd_bf16_f32(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID)
{
    sml_digit_recognizer_forward(gid, tid.x, SLANGML_DTYPE_BF16);
}
