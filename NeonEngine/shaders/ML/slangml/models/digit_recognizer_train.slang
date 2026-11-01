import slangml_common;

// GPU training kernels for the 729 -> 128 (ReLU) -> 10 (softmax) digit recognizer.
//
// One minibatch is five dispatches, in this order:
//   hidden_fwd -> output_fwd -> hidden_bwd -> grad_output -> grad_hidden -> adam
// with a compute-to-compute memory barrier between them. Every kernel writes each
// element it owns, so no clears are needed between minibatches and no atomics are
// used (the reductions over the batch are sequential per thread, so results are
// deterministic).
//
// All trainable tensors live in one flat fp32 buffer (params, grads, adamM, adamV)
// so a single kernel updates everything. Offsets must match the host trainer.

public static const uint SML_TRAIN_INPUT_FEATURES = 27u * 27u;  // 729
public static const uint SML_TRAIN_HIDDEN_FEATURES = 128u;
public static const uint SML_TRAIN_CLASSES = 10u;

// Flat parameter layout: W1[729,128] | b1[128] | W2[128,10] | b2[10]
public static const uint SML_TRAIN_OFF_W1 = 0u;
public static const uint SML_TRAIN_OFF_B1 = 93312u;   // 729 * 128
public static const uint SML_TRAIN_OFF_W2 = 93440u;   // + 128
public static const uint SML_TRAIN_OFF_B2 = 94720u;   // + 128 * 10
public static const uint SML_TRAIN_PARAM_COUNT = 94730u;

struct DigitTrainPush
{
    uint batchStart;   // first position in the epoch permutation for this minibatch
    uint batchCount;   // samples in this minibatch
    uint statsOffset;  // first row in the stats buffer for this minibatch
    uint reserved;
    float learningRate;
    float beta1;
    float beta2;
    float beta1Pow;    // beta1^t, for Adam bias correction
    float beta2Pow;    // beta2^t
    float invBatch;    // 1 / batchCount, for mean cross-entropy
};

[[vk::push_constant]]
ConstantBuffer<DigitTrainPush> smlDigitTrainPush;

[[vk::binding(0, 0)]] StructuredBuffer<float> smlDigitTrainImages;     // [trainCount, 729]
[[vk::binding(1, 0)]] StructuredBuffer<uint> smlDigitTrainLabels;      // [trainCount]
[[vk::binding(2, 0)]] StructuredBuffer<uint> smlDigitOrder;            // [trainCount] epoch permutation
[[vk::binding(3, 0)]] RWStructuredBuffer<float> smlDigitParams;        // flat, see offsets above
[[vk::binding(4, 0)]] RWStructuredBuffer<float> smlDigitGrads;         // flat, same layout
[[vk::binding(5, 0)]] RWStructuredBuffer<float> smlDigitAdamM;         // flat, same layout
[[vk::binding(6, 0)]] RWStructuredBuffer<float> smlDigitAdamV;         // flat, same layout
[[vk::binding(7, 0)]] RWStructuredBuffer<float> smlDigitHiddenPre;     // [batch, 128]
[[vk::binding(8, 0)]] RWStructuredBuffer<float> smlDigitHiddenAct;     // [batch, 128]
[[vk::binding(9, 0)]] RWStructuredBuffer<float> smlDigitHiddenGrad;    // [batch, 128]
[[vk::binding(10, 0)]] RWStructuredBuffer<float> smlDigitLogitGrad;    // [batch, 10]
[[vk::binding(11, 0)]] RWStructuredBuffer<float> smlDigitStats;        // [steps * batch, 2]: loss, correct

// Hidden layer forward: one thread per (sample, hidden neuron).
[shader("compute")]
[numthreads(256, 1, 1)]
void sml_model_digit_recognizer_train_hidden_fwd(uint3 tid : SV_DispatchThreadID)
{
    uint b = tid.x / SML_TRAIN_HIDDEN_FEATURES;
    uint j = tid.x % SML_TRAIN_HIDDEN_FEATURES;
    if (b >= smlDigitTrainPush.batchCount) return;

    uint sample = smlDigitOrder[smlDigitTrainPush.batchStart + b];
    float value = smlDigitParams[SML_TRAIN_OFF_B1 + j];
    for (uint i = 0u; i < SML_TRAIN_INPUT_FEATURES; ++i)
    {
        value = fma(
            smlDigitTrainImages[sample * SML_TRAIN_INPUT_FEATURES + i],
            smlDigitParams[SML_TRAIN_OFF_W1 + i * SML_TRAIN_HIDDEN_FEATURES + j],
            value);
    }
    smlDigitHiddenPre[b * SML_TRAIN_HIDDEN_FEATURES + j] = value;
    smlDigitHiddenAct[b * SML_TRAIN_HIDDEN_FEATURES + j] = max(value, 0.0f);
}

// Output layer forward, stable softmax, cross-entropy, and dL/dlogits.
// One thread per sample; ten logits per thread is small enough to keep local.
[shader("compute")]
[numthreads(256, 1, 1)]
void sml_model_digit_recognizer_train_output_fwd(uint3 tid : SV_DispatchThreadID)
{
    uint b = tid.x;
    if (b >= smlDigitTrainPush.batchCount) return;

    float logits[SML_TRAIN_CLASSES];
    for (uint c = 0u; c < SML_TRAIN_CLASSES; ++c)
    {
        float value = smlDigitParams[SML_TRAIN_OFF_B2 + c];
        for (uint j = 0u; j < SML_TRAIN_HIDDEN_FEATURES; ++j)
        {
            value = fma(
                smlDigitHiddenAct[b * SML_TRAIN_HIDDEN_FEATURES + j],
                smlDigitParams[SML_TRAIN_OFF_W2 + j * SML_TRAIN_CLASSES + c],
                value);
        }
        logits[c] = value;
    }

    float maximum = logits[0];
    uint bestClass = 0u;
    for (uint c = 1u; c < SML_TRAIN_CLASSES; ++c)
    {
        if (logits[c] > maximum)
        {
            maximum = logits[c];
            bestClass = c;
        }
    }

    float sum = 0.0f;
    float prob[SML_TRAIN_CLASSES];
    for (uint c = 0u; c < SML_TRAIN_CLASSES; ++c)
    {
        prob[c] = exp(logits[c] - maximum);
        sum += prob[c];
    }
    for (uint c = 0u; c < SML_TRAIN_CLASSES; ++c)
        prob[c] /= sum;

    uint label = smlDigitTrainLabels[smlDigitOrder[smlDigitTrainPush.batchStart + b]];
    uint row = smlDigitTrainPush.statsOffset + b;
    smlDigitStats[row * 2u + 0u] = -log(max(prob[label], 1e-12f));
    smlDigitStats[row * 2u + 1u] = bestClass == label ? 1.0f : 0.0f;

    for (uint c = 0u; c < SML_TRAIN_CLASSES; ++c)
    {
        float target = c == label ? 1.0f : 0.0f;
        smlDigitLogitGrad[b * SML_TRAIN_CLASSES + c] = (prob[c] - target) * smlDigitTrainPush.invBatch;
    }
}

// Hidden layer backward: dH = (dLogits * W2^T) * ReLU'(hiddenPre).
[shader("compute")]
[numthreads(256, 1, 1)]
void sml_model_digit_recognizer_train_hidden_bwd(uint3 tid : SV_DispatchThreadID)
{
    uint b = tid.x / SML_TRAIN_HIDDEN_FEATURES;
    uint j = tid.x % SML_TRAIN_HIDDEN_FEATURES;
    if (b >= smlDigitTrainPush.batchCount) return;

    float value = 0.0f;
    for (uint c = 0u; c < SML_TRAIN_CLASSES; ++c)
    {
        value = fma(
            smlDigitLogitGrad[b * SML_TRAIN_CLASSES + c],
            smlDigitParams[SML_TRAIN_OFF_W2 + j * SML_TRAIN_CLASSES + c],
            value);
    }
    uint idx = b * SML_TRAIN_HIDDEN_FEATURES + j;
    smlDigitHiddenGrad[idx] = smlDigitHiddenPre[idx] > 0.0f ? value : 0.0f;
}

// Output layer gradients: gW2[j,c] = sum_b h[b,j] * dL[b,c]; gb2[c] = sum_b dL[b,c].
// Threads [0, 1280) produce gW2, threads [1280, 1290) produce gb2.
[shader("compute")]
[numthreads(256, 1, 1)]
void sml_model_digit_recognizer_train_grad_output(uint3 tid : SV_DispatchThreadID)
{
    uint idx = tid.x;
    const uint w2Count = SML_TRAIN_HIDDEN_FEATURES * SML_TRAIN_CLASSES;
    if (idx < w2Count)
    {
        uint j = idx / SML_TRAIN_CLASSES;
        uint c = idx % SML_TRAIN_CLASSES;
        float value = 0.0f;
        for (uint b = 0u; b < smlDigitTrainPush.batchCount; ++b)
            value = fma(smlDigitHiddenAct[b * SML_TRAIN_HIDDEN_FEATURES + j],
                        smlDigitLogitGrad[b * SML_TRAIN_CLASSES + c], value);
        smlDigitGrads[SML_TRAIN_OFF_W2 + idx] = value;
    }
    else if (idx < w2Count + SML_TRAIN_CLASSES)
    {
        uint c = idx - w2Count;
        float value = 0.0f;
        for (uint b = 0u; b < smlDigitTrainPush.batchCount; ++b)
            value += smlDigitLogitGrad[b * SML_TRAIN_CLASSES + c];
        smlDigitGrads[SML_TRAIN_OFF_B2 + c] = value;
    }
}

// Hidden layer gradients: gW1[i,j] = sum_b x[b,i] * dH[b,j]; gb1[j] = sum_b dH[b,j].
// Threads [0, 93312) produce gW1, threads [93312, 93440) produce gb1.
[shader("compute")]
[numthreads(256, 1, 1)]
void sml_model_digit_recognizer_train_grad_hidden(uint3 tid : SV_DispatchThreadID)
{
    uint idx = tid.x;
    const uint w1Count = SML_TRAIN_INPUT_FEATURES * SML_TRAIN_HIDDEN_FEATURES;
    if (idx < w1Count)
    {
        uint i = idx / SML_TRAIN_HIDDEN_FEATURES;
        uint j = idx % SML_TRAIN_HIDDEN_FEATURES;
        float value = 0.0f;
        for (uint b = 0u; b < smlDigitTrainPush.batchCount; ++b)
        {
            uint sample = smlDigitOrder[smlDigitTrainPush.batchStart + b];
            value = fma(smlDigitTrainImages[sample * SML_TRAIN_INPUT_FEATURES + i],
                        smlDigitHiddenGrad[b * SML_TRAIN_HIDDEN_FEATURES + j], value);
        }
        smlDigitGrads[SML_TRAIN_OFF_W1 + idx] = value;
    }
    else if (idx < w1Count + SML_TRAIN_HIDDEN_FEATURES)
    {
        uint j = idx - w1Count;
        float value = 0.0f;
        for (uint b = 0u; b < smlDigitTrainPush.batchCount; ++b)
            value += smlDigitHiddenGrad[b * SML_TRAIN_HIDDEN_FEATURES + j];
        smlDigitGrads[SML_TRAIN_OFF_B1 + j] = value;
    }
}

// Adam update over the whole flat parameter vector.
[shader("compute")]
[numthreads(256, 1, 1)]
void sml_model_digit_recognizer_train_adam(uint3 tid : SV_DispatchThreadID)
{
    uint idx = tid.x;
    if (idx >= SML_TRAIN_PARAM_COUNT) return;

    const float eps = 1e-8f;
    float grad = smlDigitGrads[idx];
    float m = smlDigitTrainPush.beta1 * smlDigitAdamM[idx] + (1.0f - smlDigitTrainPush.beta1) * grad;
    float v = smlDigitTrainPush.beta2 * smlDigitAdamV[idx] + (1.0f - smlDigitTrainPush.beta2) * grad * grad;
    float mHat = m / (1.0f - smlDigitTrainPush.beta1Pow);
    float vHat = v / (1.0f - smlDigitTrainPush.beta2Pow);

    smlDigitAdamM[idx] = m;
    smlDigitAdamV[idx] = v;
    smlDigitParams[idx] -= smlDigitTrainPush.learningRate * mHat / (sqrt(vHat) + eps);
}
