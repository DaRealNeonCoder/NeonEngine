import slangml_common;

// GPU training kernels for the 729 -> 128 (sigmoid) -> 10 (sigmoid) digit recognizer.
// The algorithm matches Nielsen's network.py: sigmoid activations on both layers,
// quadratic cost C = 0.5 * ||a - y||^2, and plain mini-batch SGD.
//
// One minibatch is six dispatches, in this order:
//   hidden_fwd -> output_fwd -> hidden_bwd -> grad_output -> grad_hidden -> sgd
// with a compute-to-compute memory barrier between them. Every kernel writes each
// element it owns, so no clears are needed between minibatches and no atomics are
// used (the reductions over the batch are sequential per thread, so results are
// deterministic).
//
// All trainable tensors live in one flat fp32 buffer (params, grads) so a single
// kernel updates everything. Offsets must match the host trainer.

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
    float invBatch;    // 1 / batchCount: the SGD step is eta / m * (sum of per-sample gradients)
};

[[vk::push_constant]]
ConstantBuffer<DigitTrainPush> smlDigitTrainPush;

[[vk::binding(0, 0)]] StructuredBuffer<float> smlDigitTrainImages;     // [trainCount, 729]
[[vk::binding(1, 0)]] StructuredBuffer<uint> smlDigitTrainLabels;      // [trainCount]
[[vk::binding(2, 0)]] StructuredBuffer<uint> smlDigitOrder;            // [trainCount] epoch permutation
[[vk::binding(3, 0)]] RWStructuredBuffer<float> smlDigitParams;        // flat, see offsets above
[[vk::binding(4, 0)]] RWStructuredBuffer<float> smlDigitGrads;         // flat, same layout (sums over the batch)
[[vk::binding(5, 0)]] RWStructuredBuffer<float> smlDigitHiddenAct;     // [batch, 128] sigmoid(z1)
[[vk::binding(6, 0)]] RWStructuredBuffer<float> smlDigitHiddenGrad;    // [batch, 128] delta1
[[vk::binding(7, 0)]] RWStructuredBuffer<float> smlDigitOutputGrad;    // [batch, 10] delta2
[[vk::binding(8, 0)]] RWStructuredBuffer<float> smlDigitStats;         // [steps * batch, 2]: loss, correct

float sml_train_sigmoid(float z)
{
    return 1.0f / (1.0f + exp(-z));
}

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
    smlDigitHiddenAct[b * SML_TRAIN_HIDDEN_FEATURES + j] = sml_train_sigmoid(value);
}

// Output layer forward, quadratic cost, and the output delta
// delta2 = (a2 - y) * sigmoid'(z2) = (a2 - y) * a2 * (1 - a2).
// One thread per sample; ten outputs per thread is small enough to keep local.
[shader("compute")]
[numthreads(256, 1, 1)]
void sml_model_digit_recognizer_train_output_fwd(uint3 tid : SV_DispatchThreadID)
{
    uint b = tid.x;
    if (b >= smlDigitTrainPush.batchCount) return;

    uint label = smlDigitTrainLabels[smlDigitOrder[smlDigitTrainPush.batchStart + b]];
    float loss = 0.0f;
    float best = -1.0f;
    uint bestClass = 0u;
    for (uint c = 0u; c < SML_TRAIN_CLASSES; ++c)
    {
        float z = smlDigitParams[SML_TRAIN_OFF_B2 + c];
        for (uint j = 0u; j < SML_TRAIN_HIDDEN_FEATURES; ++j)
        {
            z = fma(
                smlDigitHiddenAct[b * SML_TRAIN_HIDDEN_FEATURES + j],
                smlDigitParams[SML_TRAIN_OFF_W2 + j * SML_TRAIN_CLASSES + c],
                z);
        }
        float a = sml_train_sigmoid(z);
        if (a > best)
        {
            best = a;
            bestClass = c;
        }

        float diff = a - (c == label ? 1.0f : 0.0f);
        loss += 0.5f * diff * diff;
        smlDigitOutputGrad[b * SML_TRAIN_CLASSES + c] = diff * a * (1.0f - a);
    }

    uint row = smlDigitTrainPush.statsOffset + b;
    smlDigitStats[row * 2u + 0u] = loss;
    smlDigitStats[row * 2u + 1u] = bestClass == label ? 1.0f : 0.0f;
}

// Hidden layer backward: delta1 = (W2 * delta2) * sigmoid'(z1), with sigmoid'(z1) = a1 * (1 - a1).
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
            smlDigitOutputGrad[b * SML_TRAIN_CLASSES + c],
            smlDigitParams[SML_TRAIN_OFF_W2 + j * SML_TRAIN_CLASSES + c],
            value);
    }
    uint idx = b * SML_TRAIN_HIDDEN_FEATURES + j;
    float a = smlDigitHiddenAct[idx];
    smlDigitHiddenGrad[idx] = value * a * (1.0f - a);
}

// Output layer gradients: gW2[j,c] = sum_b a1[b,j] * delta2[b,c]; gb2[c] = sum_b delta2[b,c].
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
                        smlDigitOutputGrad[b * SML_TRAIN_CLASSES + c], value);
        smlDigitGrads[SML_TRAIN_OFF_W2 + idx] = value;
    }
    else if (idx < w2Count + SML_TRAIN_CLASSES)
    {
        uint c = idx - w2Count;
        float value = 0.0f;
        for (uint b = 0u; b < smlDigitTrainPush.batchCount; ++b)
            value += smlDigitOutputGrad[b * SML_TRAIN_CLASSES + c];
        smlDigitGrads[SML_TRAIN_OFF_B2 + c] = value;
    }
}

// Hidden layer gradients: gW1[i,j] = sum_b x[b,i] * delta1[b,j]; gb1[j] = sum_b delta1[b,j].
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

// Plain SGD over the whole flat parameter vector: p -= (eta / m) * sum_b grad_b.
[shader("compute")]
[numthreads(256, 1, 1)]
void sml_model_digit_recognizer_train_sgd(uint3 tid : SV_DispatchThreadID)
{
    uint idx = tid.x;
    if (idx >= SML_TRAIN_PARAM_COUNT) return;
    smlDigitParams[idx] -= smlDigitTrainPush.learningRate * smlDigitTrainPush.invBatch * smlDigitGrads[idx];
}
