# 27x27 digit recognizer

This is the Slang-only inference model for a small MNIST-style digit
recognizer. The host owns image preprocessing, weight loading, descriptor
creation, pipeline selection, dispatch, synchronization, and reading the
result. No host-side code or training code is included here.

## Model

The fixed architecture is:

```text
27x27 grayscale input = 729 features
        -> Linear(729, 128)
        -> sigmoid
        -> Linear(128, 10)
        -> sigmoid
```

The two weight matrices are stored as fp16 or bf16 and decoded to fp32 before
each FMA. Biases, probabilities, and the input are fp32. The selected entry
point determines the weight storage format; dtype is not selected in the hot
loop.

## Entry points

```text
sml_model_digit_recognizer_fwd_f16_f32
sml_model_digit_recognizer_fwd_bf16_f32
```

Both use a 256-thread workgroup per image. Dispatch is:

```text
dispatch.x = batchCount - dispatchBase.x
dispatch.y = 1
dispatch.z = 1
```

This is the same feedforward as Nielsen's `network.py`, and training
(`digit_recognizer_train.slang`) uses the same algorithm: quadratic cost, plain
mini-batch SGD, and N(0, 1) initialization of weights and biases.

The kernel writes ten output activations and one integer prediction per image.
The outputs are independent sigmoids in (0, 1), so they do not sum to one; the
prediction is their argmax.

## Tensor table

The host must bind seven descriptors in this order:

| Index | Tensor | Shape | Storage | Access |
| ---: | --- | --- | --- | --- |
| 0 | input | `[batch, 729]` | fp32 | read |
| 1 | weight1 | `[729, 128]` | fp16 or bf16 | read |
| 2 | bias1 | `[128]` | fp32 | read |
| 3 | weight2 | `[128, 10]` | fp16 or bf16 | read |
| 4 | bias2 | `[10]` | fp32 | read |
| 5 | probabilities (sigmoid outputs) | `[batch, 10]` | fp32 | write |
| 6 | prediction | `[batch]` | uint32/int32 bits | write |

Weights use row-major logical indexing. The host supplies strides in elements,
so `weight1[feature, hidden]` and `weight2[hidden, class]` are addressed as:

```text
feature * weight1StrideFeature + hidden * weight1StrideHidden
hidden  * weight2StrideHidden  + class  * weight2StrideClass
```

Input and output batch strides are also supplied explicitly. This keeps the
kernel usable with canonical contiguous descriptors while avoiding descriptor
table reloads inside the dot-product loops.

## Parameter block

`headerAddress` points to `DigitRecognizerParams`:

| Offset | Field |
| ---: | --- |
| 0..47 | `KernelHeader` |
| 48 | `batchCount` |
| 52 | `inputStrideBatch` |
| 56 | `inputStrideFeature` |
| 60 | `weight1StrideFeature` |
| 64 | `weight1StrideHidden` |
| 68 | `bias1Stride` |
| 72 | `weight2StrideHidden` |
| 76 | `weight2StrideClass` |
| 80 | `bias2Stride` |
| 84 | `outputStrideBatch` |
| 88 | `outputStrideClass` |
| 92 | `predictionStrideBatch` |
| 96 | `inputScale` |
| 100 | `inputBias` |

The input value used by the model is `rawInput * inputScale + inputBias`.
For normalized MNIST-style pixels, the host can upload values in `[0,1]` and
use `inputScale=1`, `inputBias=0`; or upload byte-range values and use
`inputScale=1/255`, `inputBias=0`.

## Synchronization and aliasing

The host must issue a storage-write to storage-read/write barrier before any
consumer reads probabilities or prediction. Inputs, weights, biases, and
outputs must not alias. The output tensors are fully overwritten for each
dispatched batch element, so no prior clear is required.

The current kernel intentionally fuses both GEMM layers, both sigmoids, and
argmax into one dispatch. Hidden and output activations stay in workgroup
memory, avoiding intermediate global-memory round trips. The fixed 729x128
and 128x10 shapes make this appropriate for inference latency; larger models
should use the general GEMM and elementwise families.

## Compilation

Compile either entry point with the normal SlangML command:

```text
slangc shaders/ML/slangml/models/digit_recognizer.slang \
  -I shaders/ML/slangml \
  -target spirv -profile spirv_1_5 -stage compute \
  -entry sml_model_digit_recognizer_fwd_f16_f32 \
  -fvk-use-entrypoint-name -fvk-use-scalar-layout -fspv-reflect -O3
```

The fp16 entry point requires fp16 storage support; the bf16 entry point uses
the explicit bf16 decoder in `common/dtypes.slang` and requires 16-bit storage
support for the weight buffers.
