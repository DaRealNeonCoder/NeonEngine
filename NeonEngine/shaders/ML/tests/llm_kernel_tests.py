"""GPU tests for the SlangML LLM kernels against NumPy references.

Usage:
    python llm_kernel_tests.py            # run every test
    python llm_kernel_tests.py gemv       # run tests whose name contains "gemv"

Each test dispatches one or more entry points through llm_harness and compares
with a float64 NumPy reference. Failures print the largest error.
"""

from __future__ import annotations

import struct
import sys
import traceback

import numpy as np

import llm_formats as F
import llm_harness as h

RNG = np.random.default_rng(20261102)
_results: list[tuple[str, bool, str]] = []
_gpu: h.Gpu | None = None


def gpu() -> h.Gpu:
    global _gpu
    if _gpu is None:
        _gpu = h.Gpu()
        print(f"Device: {_gpu.device_name}  fp16={_gpu.has_fp16} int8={_gpu.has_int8} coopmat={_gpu.has_coopmat}")
    return _gpu


def check(name: str, got, want, rtol: float = 1e-4) -> None:
    got = np.asarray(got, dtype=np.float64)
    want = np.asarray(want, dtype=np.float64)
    if got.shape != want.shape:
        _results.append((name, False, f"shape {got.shape} != {want.shape}"))
        return
    err = float(np.max(np.abs(got - want))) if got.size else 0.0
    scale = max(float(np.max(np.abs(want))) if want.size else 0.0, 1e-30)
    ok = bool(np.all(np.isfinite(got))) and err <= rtol * scale
    _results.append((name, ok, f"max abs err {err:.3e} (scale {scale:.3e})"))


def check_rel_l2(name: str, got, want, tol: float) -> None:
    got = np.asarray(got, dtype=np.float64)
    want = np.asarray(want, dtype=np.float64)
    rel = float(np.linalg.norm(got - want) / max(np.linalg.norm(want), 1e-30))
    _results.append((name, bool(np.isfinite(rel)) and rel <= tol, f"rel L2 err {rel:.3e} (tol {tol:.0e})"))


def record(name: str, ok: bool, detail: str) -> None:
    _results.append((name, ok, detail))


# ---------------------------------------------------------------------------
# Shared launch helpers
# ---------------------------------------------------------------------------
QUANT_SRC = "slangml/quant/quantize_f32.slang"
GEMV_SRC = "slangml/quant/gemv_quant.slang"


def quantize_act(g: h.Gpu, x_buf: h.Buffer, rows: int, cols: int, ld: int | None = None):
    """Returns (dispatch, xq_buf, xs_buf)."""
    xq = g.buffer(rows * cols)
    xs = g.buffer(rows * (cols // 32) * 8)
    pipe = g.pipeline(QUANT_SRC, "sml_quant_quantize_q8act_fwd_f32")
    payload = struct.pack("<4I", rows, cols, ld or cols, 0)
    d = g.launch(pipe, [x_buf, (xq, h.DTYPE_INT8, 0), xs], payload, (h.ceil_div(cols // 4, 256), rows, 1))
    return d, xq, xs


def gemv_payload(n, k, m, group, flags=0, alpha=1.0, ld_y=None, ld_r=None, act=0, limit=7.0, act_alpha=1.702):
    return struct.pack("<5If3IffI", n, k, m, F.group_shift(group), flags, alpha,
                       ld_y or n, ld_r or n, act, limit, act_alpha, 0)


GEMV_ROWS = {1: 2, 2: 4, 4: 4, 8: 8}  # rows per subgroup for each _m<MC> variant


def gemv_groups(g: h.Gpu, n: int, mc: int = 1) -> int:
    return h.ceil_div(n, GEMV_ROWS[mc] * 128 // g.subgroup_size)


# ---------------------------------------------------------------------------
# Activation quantization
# ---------------------------------------------------------------------------
def test_quantize_q8act() -> None:
    g = gpu()
    rows, cols = 5, 4096
    x = (RNG.normal(size=(rows, cols)) * RNG.uniform(0.1, 10.0, size=(rows, 1))).astype(np.float32)
    x[2, 64:96] = 0.0  # all-zero block
    xb = g.array(x)
    d, xq, xs = quantize_act(g, xb, rows, cols)
    g.run(d)
    q = g.read(xq, np.int8, (rows, cols))
    s = g.read(xs, np.float32, (rows, cols // 32, 2))
    rq, rd, rs = F.quantize_act_ref(x)
    mismatch = int(np.sum(np.abs(q.astype(int) - rq.astype(int)) > 1))
    record("quantize_q8act codes (|dq|<=1)", mismatch == 0, f"{mismatch} codes off by >1")
    check("quantize_q8act scale", s[:, :, 0], rd, 1e-6)
    check("quantize_q8act block sum", s[:, :, 1], rs, 1e-5)


# ---------------------------------------------------------------------------
# GEMV
# ---------------------------------------------------------------------------
def _upload_weight(g: h.Gpu, fmt: str, w: np.ndarray, group: int):
    codes, meta, deq = F.pack_weight(fmt, w, group)
    cb = g.array(np.ascontiguousarray(codes).view(np.uint8))
    mb = g.array(meta) if meta is not None else None
    return codes, meta, deq, cb, mb


def _gemv_case(fmt: str, mc: int, m: int, n: int, k: int, group: int, flags: int) -> None:
    g = gpu()
    w = RNG.normal(size=(n, k)).astype(np.float32) * 0.05
    x = RNG.normal(size=(m, k)).astype(np.float32)
    bias = RNG.normal(size=n).astype(np.float32)
    res = RNG.normal(size=(m, n)).astype(np.float32)
    alpha = 0.75
    codes, meta, deq, cb, mb = _upload_weight(g, fmt, w, group)
    xb = g.array(x)
    yb = g.zeros(m * n * 4)
    biasb = g.array(bias)
    resb = g.array(res)
    entry = f"sml_quant_gemv_fwd_{fmt}_m{mc}"
    pipe = g.pipeline(GEMV_SRC, entry)
    payload = gemv_payload(n, k, m, group, flags=flags, alpha=alpha)
    epi = np.zeros((m, n))
    if flags & 1:
        epi += bias[None, :]
    if flags & 2:
        epi += res
    if fmt in ("q4", "q8", "mxfp4"):
        qd, xq, xs = quantize_act(g, xb, m, k)
        d = g.launch(pipe, [cb, mb, (xq, h.DTYPE_INT8, 0), xs, yb, biasb, resb], payload, gemv_groups(g, n, mc))
        g.run([qd, d])
        y = g.read(yb, np.float32, (m, n))
        xqv = g.read(xq, np.int8, (m, k))
        xsv = g.read(xs, np.float32, (m, k // 32, 2))
        exact = alpha * F.gemv_q_ref(fmt, codes, meta, xqv, xsv, n, k, group) + epi
        check(f"gemv {fmt} m{m}/MC{mc} N{n} K{k} g{group} flags{flags} (exact)", y, exact, 2e-5)
        check_rel_l2(f"gemv {fmt} m{m} vs fp64 (act-quant error)", y - epi,
                     alpha * x.astype(np.float64) @ deq.T.astype(np.float64), 1e-2)
    else:
        d = g.launch(pipe, [cb, None, xb, None, yb, biasb, resb], payload, gemv_groups(g, n, mc))
        g.run(d)
        y = g.read(yb, np.float32, (m, n))
        want = alpha * x.astype(np.float64) @ deq.T.astype(np.float64) + epi
        check(f"gemv {fmt} m{m}/MC{mc} N{n} K{k} flags{flags}", y, want, 2e-5)


def test_gemv_q4() -> None:
    _gemv_case("q4", 1, 1, 1000, 4096, 64, 0)
    _gemv_case("q4", 2, 2, 777, 2048, 32, 1)
    _gemv_case("q4", 4, 3, 513, 1024, 128, 3)
    _gemv_case("q4", 8, 8, 256, 5120, 32, 2)


def test_gemv_q8() -> None:
    _gemv_case("q8", 1, 1, 1001, 4096, 32, 3)
    _gemv_case("q8", 4, 4, 300, 2048, 64, 0)


def test_gemv_mxfp4() -> None:
    # Host-side layout check: repacked split-nibble codes decode to the official dequant exactly.
    w = RNG.normal(size=(64, 256)).astype(np.float32) * 0.03
    w[3, :32] = 0.0  # all-zero block
    blocks, scales = F.quantize_mxfp4_checkpoint(w)
    codes, meta = F.repack_mxfp4(blocks, scales)
    c = codes.reshape(64, 8, 16)
    nib = np.concatenate([c & 0xF, c >> 4], axis=2)
    ours = (F.FP4_VALUES[nib] * np.exp2(meta.astype(np.float32) - 127.0)[..., None]).reshape(64, 256)
    want = F.dequant_mxfp4_checkpoint(blocks, scales)
    record("mxfp4 repack == official dequant", bool(np.array_equal(ours, want)),
           f"max diff {float(np.max(np.abs(ours - want))):.1e}")
    # gpt-oss expert shapes: down 2880 x 2880, gate/up rows 2880 x 2880 (K = 90 blocks).
    _gemv_case("mxfp4", 1, 1, 2880, 2880, 32, 0)
    _gemv_case("mxfp4", 1, 1, 1000, 4096, 32, 3)
    _gemv_case("mxfp4", 4, 3, 513, 1024, 32, 1)


def test_gemv_f16_bf16() -> None:
    _gemv_case("f16", 1, 1, 999, 2048, 32, 3)
    _gemv_case("f16", 4, 2, 257, 1024, 32, 0)
    _gemv_case("bf16", 1, 1, 640, 4096, 32, 1)
    _gemv_case("bf16", 4, 4, 130, 512, 32, 2)


def _glu_ref(mode: int, gt, up, limit=7.0, alpha=1.702):
    if mode == 0:
        return gt / (1 + np.exp(-gt)) * up
    if mode == 1:
        return 0.5 * gt * (1 + np.tanh(0.7978845608028654 * (gt + 0.044715 * gt ** 3))) * up
    gt = np.minimum(gt, limit)
    up = np.clip(up, -limit, limit)
    return (up + 1) * gt / (1 + np.exp(-alpha * gt))


def _swiglu_case(fmt: str, mc: int, m: int, n: int, k: int, group: int, mode: int, bias: bool) -> None:
    g = gpu()
    wg = RNG.normal(size=(n, k)).astype(np.float32) * 0.05
    wu = RNG.normal(size=(n, k)).astype(np.float32) * 0.05
    x = RNG.normal(size=(m, k)).astype(np.float32) * 2.0
    bg = RNG.normal(size=n).astype(np.float32)
    bu = RNG.normal(size=n).astype(np.float32)
    cg, mg, dg, cgb, mgb = _upload_weight(g, fmt, wg, group)
    cu, mu, du, cub, mub = _upload_weight(g, fmt, wu, group)
    xb = g.array(x)
    yb = g.zeros(m * n * 4)
    entry = f"sml_quant_gemv_swiglu_fwd_{fmt}_m{mc}"
    pipe = g.pipeline(GEMV_SRC, entry)
    payload = gemv_payload(n, k, m, group, flags=1 if bias else 0, act=mode)
    tensors_tail = [g.array(bg), g.array(bu), cub, mub]
    if fmt in ("q4", "q8"):
        qd, xq, xs = quantize_act(g, xb, m, k)
        d = g.launch(pipe, [cgb, mgb, (xq, h.DTYPE_INT8, 0), xs, yb] + tensors_tail, payload, gemv_groups(g, n, mc))
        g.run([qd, d])
        xqv = g.read(xq, np.int8, (m, k))
        xsv = g.read(xs, np.float32, (m, k // 32, 2))
        gt = F.gemv_q_ref(fmt, cg, mg, xqv, xsv, n, k, group)
        up = F.gemv_q_ref(fmt, cu, mu, xqv, xsv, n, k, group)
    else:
        d = g.launch(pipe, [cgb, None, xb, None, yb] + tensors_tail, payload, gemv_groups(g, n, mc))
        g.run(d)
        gt = x.astype(np.float64) @ dg.T.astype(np.float64)
        up = x.astype(np.float64) @ du.T.astype(np.float64)
    if bias:
        gt, up = gt + bg, up + bu
    y = g.read(yb, np.float32, (m, n))
    check(f"gemv_swiglu {fmt} m{m}/MC{mc} mode{mode} bias{int(bias)}", y, _glu_ref(mode, gt, up), 5e-5)


def test_gemv_swiglu() -> None:
    _swiglu_case("q4", 1, 1, 1500, 2048, 64, 0, False)
    _swiglu_case("q4", 4, 3, 333, 1024, 32, 1, True)
    _swiglu_case("q8", 1, 1, 640, 2880, 32, 2, True)
    _swiglu_case("q8", 4, 4, 128, 512, 32, 0, False)
    _swiglu_case("f16", 1, 1, 700, 1024, 32, 2, True)
    _swiglu_case("bf16", 1, 1, 300, 2048, 32, 0, False)


# ---------------------------------------------------------------------------
# Fused RMSNorm
# ---------------------------------------------------------------------------
NORM_SRC = "slangml/llm/rmsnorm_fused_f32.slang"
NORM_ADD, NORM_SUM, NORM_UNIT, NORM_GATE, NORM_F32, NORM_Q8 = 1, 2, 4, 8, 16, 32


def _silu(x):
    return x / (1 + np.exp(-x))


def _norm_case(variant: str, rows: int, cols: int, flags: int, alias_sum_to_x: bool = False) -> None:
    g = gpu()
    eps = 1e-6
    x = RNG.normal(size=(rows, cols)).astype(np.float32) * 3
    r = RNG.normal(size=(rows, cols)).astype(np.float32)
    w = RNG.normal(size=cols).astype(np.float32) * 0.1
    z = RNG.normal(size=(rows, cols)).astype(np.float32)
    xb, rb, wb, zb = g.array(x), g.array(r), g.array(w), g.array(z)
    yb = g.zeros(rows * cols * 4)
    hb = xb if alias_sum_to_x else g.zeros(rows * cols * 4)
    yq = g.zeros(rows * cols)
    ys = g.zeros(rows * cols // 32 * 8)
    pipe = g.pipeline(NORM_SRC, f"sml_llm_rmsnorm_fwd_f32_{variant}")
    payload = struct.pack("<2IfI4I", rows, cols, eps, flags, cols, cols, cols, cols)
    groups = rows if variant == "row" else h.ceil_div(rows, 4)
    g.run(g.launch(pipe, [xb, rb, wb, yb, hb, zb, (yq, h.DTYPE_INT8, 0), ys], payload, groups))
    hh = x.astype(np.float64) + (r if flags & NORM_ADD else 0)
    ww = w + (1.0 if flags & NORM_UNIT else 0.0)
    y = hh / np.sqrt((hh ** 2).mean(axis=1, keepdims=True) + eps) * ww
    if flags & NORM_GATE:
        y = y * _silu(z.astype(np.float64))
    tag = f"rmsnorm_{variant} {rows}x{cols} flags{flags}{' alias' if alias_sum_to_x else ''}"
    if flags & NORM_F32:
        check(tag + " y", g.read(yb, np.float32, (rows, cols)), y, 2e-6)
    if flags & NORM_SUM:
        check(tag + " sum", g.read(hb, np.float32, (rows, cols)), hh, 1e-6)
    if flags & NORM_Q8:
        q = g.read(yq, np.int8, (rows, cols)).astype(np.float64)
        s = g.read(ys, np.float32, (rows, cols // 32, 2)).astype(np.float64)
        deq = (q.reshape(rows, -1, 32) * s[:, :, :1]).reshape(rows, cols)
        err = np.abs(deq - y).reshape(rows, -1, 32).max(axis=2)
        ok_codes = bool(np.all(err <= s[:, :, 0] * 0.5 + 1e-5))
        ok_sum = np.allclose(s[:, :, 1], s[:, :, 0] * q.reshape(rows, -1, 32).sum(axis=2), rtol=1e-6, atol=1e-6)
        record(tag + " q8", ok_codes and ok_sum, f"max block err/d {float(np.max(err / np.maximum(s[:, :, 0], 1e-30))):.3f}")


def test_rmsnorm_fused() -> None:
    _norm_case("row", 3, 5120, NORM_ADD | NORM_SUM | NORM_UNIT | NORM_F32 | NORM_Q8)
    _norm_case("row", 2, 2880, NORM_F32)
    _norm_case("row", 4, 4096, NORM_ADD | NORM_SUM | NORM_Q8, alias_sum_to_x=True)
    _norm_case("small", 7 * 48 + 3, 128, NORM_GATE | NORM_F32 | NORM_Q8)
    _norm_case("small", 9, 256, NORM_UNIT | NORM_F32)


# ---------------------------------------------------------------------------
# QK-norm + RoPE + KV cache append
# ---------------------------------------------------------------------------
ROPE_SRC = "slangml/llm/rope_qknorm_f32.slang"
R_NQ, R_NK, R_UNIT, R_INTER, R_MROPE, R_MROPE_I, R_TABLE, R_V, R_SLOTS = 1, 2, 4, 8, 16, 32, 64, 128, 256


def _rope_ref(x, pos_rows, rd, inv, interleaved, sec_of, attn):
    """x [T][H][D] float64; pos_rows [3][T]; returns rotated copy."""
    out = x.copy()
    half = rd // 2
    for i in range(half):
        p = pos_rows[sec_of(i)].astype(np.float32)
        ang = (p * inv[i]).astype(np.float32).astype(np.float64)
        c, s = np.cos(ang)[:, None] * attn, np.sin(ang)[:, None] * attn
        a, b = (2 * i, 2 * i + 1) if interleaved else (i, i + half)
        xa, xb = x[:, :, a], x[:, :, b]
        out[:, :, a] = xa * c - xb * s
        out[:, :, b] = xb * c + xa * s
    return out


def _rope_case(name, T, Hq, Hkv, D, rd, flags, theta=1e7, sections=(0, 0, 0), attn=1.0, q_head_stride=None,
               big_pos=False, yarn=False):
    g = gpu()
    eps = 1e-6
    qhs = q_head_stride or D
    qsrc = RNG.normal(size=(T, Hq, qhs)).astype(np.float32)
    ksrc = RNG.normal(size=(T, Hkv, D)).astype(np.float32)
    vsrc = RNG.normal(size=(T, Hkv, D)).astype(np.float32)
    qn = RNG.normal(size=D).astype(np.float32) * 0.2
    kn = RNG.normal(size=D).astype(np.float32) * 0.2
    hi = 262000 if big_pos else 5000
    pos = RNG.integers(0, hi, size=(3, T)).astype(np.int32)
    if not flags & R_MROPE:
        pos[1] = pos[2] = pos[0]
    # HF: inv_freq = 1 / theta ** (arange(0, rd, 2) / rd), effectively correctly rounded fp32.
    inv = (1.0 / theta ** (np.arange(0, rd, 2, dtype=np.float64) / rd)).astype(np.float32)
    if yarn:
        inv = (inv * RNG.uniform(0.5, 1.0, size=rd // 2)).astype(np.float32)
    max_ctx = 64
    slots = RNG.permutation(max_ctx)[:T].astype(np.int32)
    slot_base = 7
    qo = g.zeros(T * Hq * D * 4)
    kc = g.zeros(Hkv * max_ctx * D * 2)
    vc = g.zeros(Hkv * max_ctx * D * 2)
    payload = struct.pack("<6Ifff9I4I", T, Hq, Hkv, D, rd, flags, eps, float(np.log2(theta)), attn,
                          Hq * qhs, qhs, Hkv * D, D, Hkv * D, D, max_ctx, slot_base, T,
                          sections[0], sections[1], sections[2], 0)
    pipe = g.pipeline(ROPE_SRC, "sml_llm_rope_qknorm_fwd_f32")
    tensors = [g.array(qsrc), g.array(ksrc), g.array(vsrc), g.array(qn), g.array(kn), g.array(pos),
               qo, kc, vc, g.array(inv), g.array(slots)]
    g.run(g.launch(pipe, tensors, payload, (Hq + 2 * Hkv, T, 1)))

    def norm(v, w):
        return v / np.sqrt((v ** 2).mean(axis=-1, keepdims=True) + eps) * (w + (1.0 if flags & R_UNIT else 0.0))

    if flags & R_MROPE_I:
        def sec_of(i):
            return 1 if (i % 3 == 1 and i < 3 * sections[1]) else (2 if (i % 3 == 2 and i < 3 * sections[2]) else 0)
    elif flags & R_MROPE:
        def sec_of(i):
            return 0 if i < sections[0] else (1 if i < sections[0] + sections[1] else 2)
    else:
        def sec_of(i):
            return 0
    q = qsrc[:, :, :D].astype(np.float64)
    k = ksrc.astype(np.float64)
    if flags & R_NQ:
        q = norm(q, qn)
    if flags & R_NK:
        k = norm(k, kn)
    q = _rope_ref(q, pos, rd, inv, bool(flags & R_INTER), sec_of, attn)
    k = _rope_ref(k, pos, rd, inv, bool(flags & R_INTER), sec_of, attn)
    # The host table reproduces HF exactly; the in-shader exp2 formula has ~3e-7
    # relative inv_freq error, which grows with position (fallback only).
    tol = 1e-5 if flags & R_TABLE else (2e-3 if big_pos else 2e-4)
    check(f"rope {name} q", g.read(qo, np.float32, (T, Hq, D)), q, tol)
    kcache = g.read(kc, np.float16, (Hkv, max_ctx, D)).astype(np.float64)
    vcache = g.read(vc, np.float16, (Hkv, max_ctx, D)).astype(np.float64)
    use = slots if flags & R_SLOTS else slot_base + np.arange(T)
    check(f"rope {name} k cache", kcache[:, use, :].transpose(1, 0, 2), k, 2e-3)
    if flags & R_V:
        check(f"rope {name} v cache", vcache[:, use, :].transpose(1, 0, 2), vsrc, 1e-3)


def test_rope_qknorm() -> None:
    # Qwen3.8 full attention: D=256, 64 rotary dims, (1+w) QK norm, interleaved MRoPE,
    # q rows interleaved with the output gate (head stride 2*D), slot table.
    qwen = R_NQ | R_NK | R_UNIT | R_V | R_TABLE
    _rope_case("qwen3.8", 5, 6, 1, 256, 64, qwen | R_MROPE | R_MROPE_I | R_SLOTS,
               sections=(11, 11, 10), q_head_stride=512)
    _rope_case("qwen3.8 pos~262K", 3, 6, 1, 256, 64, qwen, q_head_stride=512, big_pos=True)
    _rope_case("qwen3.8 formula fallback", 3, 6, 1, 256, 64, R_NQ | R_NK | R_UNIT | R_V, q_head_stride=512)
    # gpt-oss style: D=64, full rotary, YaRN-scaled table + attention factor, no norm.
    _rope_case("gptoss yarn", 4, 8, 2, 64, 64, R_TABLE | R_V, theta=150000.0, attn=1.35, yarn=True)
    # GPT-J interleaved pairs, plain-w norm, contiguous MRoPE sections.
    _rope_case("gptj mrope", 3, 4, 2, 128, 128, R_NQ | R_NK | R_INTER | R_MROPE | R_V | R_TABLE, sections=(16, 24, 24))


# ---------------------------------------------------------------------------
# Flash attention (decode + prefill)
# ---------------------------------------------------------------------------
ATTN_SRC = "slangml/llm/attention_flash_f32.slang"
A_CAUSAL, A_SINKS, A_GATE = 1, 2, 4


def _attn_case(name, Hq, Hkv, D, T, kv_base, chunk, tokens_per_group, flags, window=0, softcap=0.0):
    g = gpu()
    G = Hq // Hkv
    max_ctx = kv_base + T + 5
    limit = [(kv_base + t + 1) if flags & A_CAUSAL else kv_base for t in range(T)]
    n_chunks = h.ceil_div(max(limit), chunk)
    q = RNG.normal(size=(T, Hq, D)).astype(np.float32)
    kc = (RNG.normal(size=(Hkv, max_ctx, D))).astype(np.float16)
    vc = (RNG.normal(size=(Hkv, max_ctx, D))).astype(np.float16)
    sinks = RNG.normal(size=Hq).astype(np.float32)
    gate = RNG.normal(size=(T, Hq, 2 * D)).astype(np.float32)  # gate in the upper half of each head
    scale = 1.0 / np.sqrt(D)
    out = g.zeros(T * Hq * D * 4)
    ws_o = g.zeros(n_chunks * T * Hq * D * 4)
    ws_ml = g.zeros(n_chunks * T * Hq * 8)
    gate_buf = g.array(gate)
    tensors = [g.array(q), g.array(kc), g.array(vc), out, g.array(sinks), (gate_buf, h.DTYPE_FP32, D), ws_o, ws_ml]
    payload = struct.pack("<9Iff5I", Hq, Hkv, T, tokens_per_group, Hq * D, max_ctx, kv_base, chunk, n_chunks,
                          scale, softcap, window, flags, Hq * 2 * D, 2 * D, Hq * D)
    pipe = g.pipeline(ATTN_SRC, f"sml_llm_attention_fwd_f16kv_d{D}")
    disp = [g.launch(pipe, tensors, payload, (n_chunks, h.ceil_div(T, tokens_per_group), Hkv))]
    if n_chunks > 1:
        comb = g.pipeline(ATTN_SRC, "sml_llm_attention_combine_fwd_f32")
        disp.append(g.launch(comb, tensors, payload + struct.pack("<I", D), T * Hq))
    g.run(disp)
    got = g.read(out, np.float32, (T, Hq, D))

    want = np.zeros((T, Hq, D))
    K = kc.astype(np.float64)
    V = vc.astype(np.float64)
    for t in range(T):
        lo = max(0, limit[t] - window) if window else 0
        for hq in range(Hq):
            kh = hq // G
            s = (K[kh, lo:limit[t]] @ q[t, hq].astype(np.float64)) * scale
            if softcap > 0:
                s = softcap * np.tanh(s / softcap)
            mx = s.max()
            e = np.exp(s - mx)
            den = e.sum() + (np.exp(sinks[hq] - mx) if flags & A_SINKS else 0.0)
            o = (e @ V[kh, lo:limit[t]]) / den
            if flags & A_GATE:
                o = o / (1 + np.exp(-gate[t, hq, D:].astype(np.float64)))
            want[t, hq] = o
    check(f"attention {name} (chunks={n_chunks})", got, want, 2e-5)


def test_attention() -> None:
    # Qwen3.8 decode: G=6, D=256, gated output, split-KV and single-chunk.
    _attn_case("qwen3.8 decode split", 12, 2, 256, 1, 1000, 256, 1, A_CAUSAL | A_GATE)
    _attn_case("qwen3.8 decode 1chunk", 12, 2, 256, 1, 1000, 1024, 1, A_CAUSAL | A_GATE)
    # MTP-style verify of 3 tokens (G=6 -> one token per group).
    _attn_case("qwen3.8 verify T=3", 6, 1, 256, 3, 777, 128, 1, A_CAUSAL | A_GATE)
    # Prefill, G=2, 4 tokens per group, causal across the new tokens.
    _attn_case("prefill d128 G2", 4, 2, 128, 37, 50, 64, 4, A_CAUSAL)
    _attn_case("prefill d128 from 0", 4, 2, 128, 70, 0, 4096, 4, A_CAUSAL)
    # gpt-oss: G=8, D=64, sliding window 128 + sinks; softcap variant.
    _attn_case("gptoss window+sinks", 16, 2, 64, 3, 300, 128, 1, A_CAUSAL | A_SINKS, window=128)
    _attn_case("softcap noncausal", 8, 4, 64, 2, 200, 64, 4, 0, softcap=30.0)


# ---------------------------------------------------------------------------
# Gated DeltaNet: causal conv1d + recurrence
# ---------------------------------------------------------------------------
CONV_SRC = "slangml/linear_attention/conv1d_f32.slang"
GDN_SRC = "slangml/linear_attention/gated_delta_rule_f32.slang"


def _conv_case(name, T, C, W, flags, tokens_per_thread, separate_state=True):
    g = gpu()
    x = RNG.normal(size=(T, C)).astype(np.float32)
    w = RNG.normal(size=(C, W)).astype(np.float32) * 0.5
    bias = RNG.normal(size=C).astype(np.float32)
    st = RNG.normal(size=(W - 1, C)).astype(np.float32)
    xb, yb = g.array(x), g.zeros(T * C * 4)
    sb = g.array(st)
    so = g.zeros((W - 1) * C * 4) if separate_state else sb
    payload = struct.pack("<8I", T, C, W, flags, C, C, tokens_per_thread, 0)
    pipe = g.pipeline(CONV_SRC, "sml_linattn_conv1d_fwd_f32")
    g.run(g.launch(pipe, [xb, g.array(w), g.array(bias), sb, yb, so], payload,
                   (h.ceil_div(C, 128), h.ceil_div(T, tokens_per_thread), 1)))
    hist = st.astype(np.float64) if flags & 4 else np.zeros((W - 1, C))
    full = np.concatenate([hist, x.astype(np.float64)], axis=0)
    want = np.zeros((T, C))
    for t in range(T):
        want[t] = (full[t:t + W] * w.T).sum(axis=0) + (bias if flags & 2 else 0.0)
    if flags & 1:
        want = _silu(want)
    check(f"conv1d {name}", g.read(yb, np.float32, (T, C)), want, 2e-6)
    if flags & 8:
        check(f"conv1d {name} state", g.read(so, np.float32, (W - 1, C)), full[-(W - 1):], 1e-7)


def test_conv1d() -> None:
    _conv_case("decode in-place state", 1, 10240, 4, 1 | 4 | 8, 1, separate_state=False)
    _conv_case("prefill chunks", 50, 1000, 4, 1 | 2 | 4 | 8, 16)
    _conv_case("prefill no state", 7, 300, 4, 1 | 8, 3)
    _conv_case("W=2", 9, 129, 2, 4 | 8, 4)


def _l2n(x, eps=1e-6):
    return x / np.sqrt((x * x).sum(axis=-1, keepdims=True) + eps)


def _gdn_case(name, T, Hk, Hv, dk, dv, precomputed=False):
    g = gpu()
    ld = 2 * Hk * dk + Hv * dv
    qkv = RNG.normal(size=(T, ld)).astype(np.float32)
    a = RNG.normal(size=(T, Hv)).astype(np.float32)
    b = RNG.normal(size=(T, Hv)).astype(np.float32)
    if precomputed:
        a = -np.abs(a) * 0.3  # g (log decay)
        b = 1 / (1 + np.exp(-b))  # beta
        a, b = a.astype(np.float32), b.astype(np.float32)
    a_log = RNG.normal(size=Hv).astype(np.float32) * 0.5
    dt_bias = RNG.normal(size=Hv).astype(np.float32) * 0.5
    s0 = RNG.normal(size=(Hv, dk, dv)).astype(np.float32) * 0.1
    sb = g.array(s0)
    ob = g.zeros(T * Hv * dv * 4)
    scale = 1.0 / np.sqrt(dk)
    payload = struct.pack("<10IffII", T, Hk, Hv, dv, ld, 0, Hk * dk, 2 * Hk * dk, Hv, Hv * dv,
                          scale, 1e-6, 1 if precomputed else 0, 0)
    pipe = g.pipeline(GDN_SRC, "sml_linattn_gated_delta_rule_fwd_f32", {"SML_GDN_DK": dk})
    g.run(g.launch(pipe, [g.array(qkv), g.array(a), g.array(b), g.array(a_log), g.array(dt_bias), sb, ob],
                   payload, (dv // 32, Hv, 1)))
    S = s0.astype(np.float64).copy()
    want = np.zeros((T, Hv, dv))
    for t in range(T):
        for hv in range(Hv):
            kh = hv // (Hv // Hk)
            q = _l2n(qkv[t, kh * dk:(kh + 1) * dk].astype(np.float64)) * scale
            k = _l2n(qkv[t, Hk * dk + kh * dk:Hk * dk + (kh + 1) * dk].astype(np.float64))
            v = qkv[t, 2 * Hk * dk + hv * dv:2 * Hk * dk + (hv + 1) * dv].astype(np.float64)
            if precomputed:
                gt, beta = float(a[t, hv]), float(b[t, hv])
            else:
                sp = np.log1p(np.exp(a[t, hv] + dt_bias[hv])) if a[t, hv] + dt_bias[hv] <= 20 else a[t, hv] + dt_bias[hv]
                gt = -np.exp(a_log[hv]) * sp
                beta = 1 / (1 + np.exp(-b[t, hv]))
            S[hv] *= np.exp(gt)
            kv_mem = (S[hv] * k[:, None]).sum(axis=0)
            delta = (v - kv_mem) * beta
            S[hv] += k[:, None] * delta[None, :]
            want[t, hv] = (S[hv] * q[:, None]).sum(axis=0)
    check(f"gated_delta_rule {name} out", g.read(ob, np.float32, (T, Hv, dv)), want, 2e-5)
    check(f"gated_delta_rule {name} state", g.read(sb, np.float32, (Hv, dk, dv)), S, 2e-5)


def test_gated_delta_rule() -> None:
    _gdn_case("decode qwen3.8 heads", 1, 16, 48, 128, 128)
    _gdn_case("prefill T=37", 37, 2, 6, 128, 128)
    _gdn_case("prefill precomputed gates", 20, 1, 2, 128, 64, precomputed=True)


# ---------------------------------------------------------------------------
# Embedding, MoE router + expert GEMVs, sampling
# ---------------------------------------------------------------------------
EMBED_SRC = "slangml/quant/embedding.slang"
MOE_SRC = "slangml/quant/gemv_moe.slang"
ROUTER_SRC = "slangml/llm/moe_router_f32.slang"
SAMPLE_SRC = "slangml/llm/sampling_f32.slang"


def test_embedding() -> None:
    g = gpu()
    V, K, T = 1000, 512, 7
    table = RNG.normal(size=(V, K)).astype(np.float32)
    tokens = RNG.integers(0, V, size=T).astype(np.int32)
    tokens[0] = V - 1
    for fmt, group in (("q4", 64), ("q8", 32), ("f16", 32), ("bf16", 32)):
        codes, meta, deq = F.pack_weight(fmt, table, group)
        out = g.zeros(T * K * 4)
        payload = struct.pack("<4If3I", T, K, F.group_shift(group), K, 1.5, 0, 0, 0)
        pipe = g.pipeline(EMBED_SRC, f"sml_quant_embedding_fwd_{fmt}")
        g.run(g.launch(pipe, [g.array(np.ascontiguousarray(codes).view(np.uint8)),
                              g.array(meta) if meta is not None else None, g.array(tokens), out],
                       payload, (h.ceil_div(K // 4, 256), T, 1)))
        check(f"embedding {fmt}", g.read(out, np.float32, (T, K)), 1.5 * deq[tokens].astype(np.float64), 1e-6)


def _router_ref(logits, k, flags, bias, scale):
    out_ids, out_w = [], []
    for row in logits.astype(np.float64):
        if flags & 1:
            sc = 1 / (1 + np.exp(-row))
            sel = sc + (bias if flags & 4 else 0)
        else:
            e = np.exp(row - row.max())
            sc = e / e.sum()
            sel = row
        ids = np.argsort(-sel, kind="stable")[:k]
        w = sc[ids]
        if flags & 2:
            w = w / w.sum()
        out_ids.append(ids)
        out_w.append(w * scale)
    return np.array(out_ids), np.array(out_w)


def _router_case(name, T, E, k, flags, scale=1.0):
    g = gpu()
    logits = RNG.normal(size=(T, E)).astype(np.float32) * 2
    bias = RNG.normal(size=E).astype(np.float32) * 0.1
    ids, wts = g.zeros(T * k * 4), g.zeros(T * k * 4)
    payload = struct.pack("<4IfI2I", T, E, k, flags, scale, E, 0, 0)
    pipe = g.pipeline(ROUTER_SRC, "sml_llm_moe_router_fwd_f32")
    g.run(g.launch(pipe, [g.array(logits), g.array(bias), ids, wts], payload, h.ceil_div(T, 4)))
    rid, rw = _router_ref(logits, k, flags, bias, scale)
    gid = g.read(ids, np.int32, (T, k))
    record(f"moe_router {name} ids", bool(np.array_equal(gid, rid)), f"{int(np.sum(gid != rid))} mismatches")
    check(f"moe_router {name} weights", g.read(wts, np.float32, (T, k)), rw, 1e-5)


def test_moe_router() -> None:
    _router_case("softmax top8 norm (qwen-moe)", 5, 128, 8, 2)
    _router_case("softmax top4 norm (gpt-oss)", 3, 32, 4, 2)
    _router_case("softmax top2 raw", 6, 60, 2, 0)
    _router_case("sigmoid+bias top8 norm x2.5", 9, 256, 8, 1 | 2 | 4, 2.5)
    _router_case("E=512", 2, 512, 6, 2)


def test_moe_gemv() -> None:
    g = gpu()
    E, H, FF, topk, tokens = 8, 512, 256, 2, 2
    slots = topk * tokens
    for fmt, use_table in (("q4", False), ("q8", False), ("bf16", False), ("mxfp4", False), ("mxfp4", True)):
        group = 32
        wg = RNG.normal(size=(E, FF, H)).astype(np.float32) * 0.05
        wu = RNG.normal(size=(E, FF, H)).astype(np.float32) * 0.05
        wd = RNG.normal(size=(E, H, FF)).astype(np.float32) * 0.05
        bg = RNG.normal(size=(E, FF)).astype(np.float32)
        bu = RNG.normal(size=(E, FF)).astype(np.float32)
        bd = RNG.normal(size=(E, H)).astype(np.float32)
        packs = {n: [F.pack_weight(fmt, w[e], group) for e in range(E)] for n, w in (("g", wg), ("u", wu), ("d", wd))}
        quant = fmt in ("q4", "q8", "mxfp4")

        def codes_of(n):
            return np.stack([np.ascontiguousarray(pk[0]).view(np.uint8) for pk in packs[n]])

        def meta_of(n):
            return np.stack([pk[1] for pk in packs[n]])

        ids = np.array([3, 5, 5, 0], dtype=np.int32)
        rw = np.array([0.7, 0.3, 0.6, 0.4], dtype=np.float32)
        x = RNG.normal(size=(tokens, H)).astype(np.float32)
        res = RNG.normal(size=(tokens, H)).astype(np.float32)
        cg, cu, cd = (g.array(codes_of(n)) for n in "gud")
        mg, mu, md = ((g.array(meta_of(n)) if quant else None) for n in "gud")
        xb = g.array(x)
        y1 = g.zeros(slots * FF * 4)
        out = g.zeros(tokens * H * 4)
        disp = []
        if quant:
            qd, xq, xs = quantize_act(g, xb, tokens, H)
            disp.append(qd)
            xt = [(xq, h.DTYPE_INT8, 0), xs]
        else:
            xt = [xb, None]
        cs_g = codes_of("g")[0].nbytes
        ms_g = meta_of("g")[0].nbytes if quant else 0
        cs_d = codes_of("d")[0].nbytes
        ms_d = meta_of("d")[0].nbytes if quant else 0
        table1 = table2 = None
        tflag = 4 if use_table else 0
        if use_table:
            # Per-expert addresses in reversed order inside the buffers, so a wrong
            # stride/table lookup cannot pass by accident.
            def addrs(buf, stride):
                return [buf.address + (E - 1 - e) * stride for e in range(E)]
            def rev(a):
                return np.ascontiguousarray(a[::-1])
            cg, cu, cd = (g.array(rev(codes_of(n))) for n in "gud")
            mg, mu, md = (g.array(rev(meta_of(n))) for n in "gud")
            t1 = np.array([[a, b, c, d] for a, b, c, d in zip(addrs(cg, cs_g), addrs(mg, ms_g),
                                                               addrs(cu, cs_g), addrs(mu, ms_g))], dtype=np.uint64)
            t2 = np.array([[a, b, 0, 0] for a, b in zip(addrs(cd, cs_d), addrs(md, ms_d))], dtype=np.uint64)
            table1, table2 = g.array(t1), g.array(t2)
        p1 = struct.pack("<5IfIIffIIII", FF, H, slots, 0, 1 | tflag, 1.0, FF, 0, 7.0, 1.702, cs_g, ms_g, FF, topk)
        w_args = [None, None] if use_table else [cg, mg]
        u_args = [None, None] if use_table else [cu, mu]
        disp.append(g.launch(g.pipeline(MOE_SRC, f"sml_quant_gemv_moe_swiglu_fwd_{fmt}"),
                             w_args + xt + [y1, g.array(bg), g.array(bu)] + u_args + [g.array(ids), g.array(rw), table1],
                             p1, (h.ceil_div(FF, 2 * 128 // g.subgroup_size), slots, 1)))
        if quant:
            qd2, yq, ys = quantize_act(g, y1, slots, FF)
            disp.append(qd2)
            yt = [(yq, h.DTYPE_INT8, 0), ys]
        else:
            yt = [y1, None]
        p2 = struct.pack("<5IfIIffIIII", H, FF, slots, 0, 1 | 2 | tflag, 1.0, H, 0, 7.0, 1.702, cs_d, ms_d, H, topk)
        d_args = [None, None] if use_table else [cd, md]
        disp.append(g.launch(g.pipeline(MOE_SRC, f"sml_quant_gemv_moe_down_fwd_{fmt}"),
                             d_args + yt + [out, g.array(bd), g.array(res), None, None, g.array(ids), g.array(rw), table2],
                             p2, (h.ceil_div(H, 2 * 128 // g.subgroup_size), tokens, 1)))
        g.run(disp)
        # Reference with dequantized weights in fp64 (activation int8 error allowed for q4/q8).
        y_ref = np.zeros((slots, FF))
        for s_ in range(slots):
            e = ids[s_]
            xx = x[s_ // topk].astype(np.float64)
            gt = packs["g"][e][2].astype(np.float64) @ xx + bg[e]
            up = packs["u"][e][2].astype(np.float64) @ xx + bu[e]
            y_ref[s_] = _silu(gt) * up
        o_ref = res.astype(np.float64).copy()
        for s_ in range(slots):
            e = ids[s_]
            o_ref[s_ // topk] += rw[s_] * (packs["d"][e][2].astype(np.float64) @ y_ref[s_] + bd[e])
        tol = 1.5e-2 if quant else 1e-5
        tag = f"{fmt}{' table' if use_table else ''}"
        check_rel_l2(f"moe {tag} gate/up swiglu", g.read(y1, np.float32, (slots, FF)), y_ref, tol)
        check_rel_l2(f"moe {tag} down weighted sum + residual", g.read(out, np.float32, (tokens, H)), o_ref, tol)


def _sample_params(R, V, ld, K, chunks, temp=1.0, top_p=1.0, min_p=0.0, seed=1, step=0, flags=0):
    return struct.pack("<5IfffIIII", R, V, ld, K, chunks, temp, top_p, min_p, seed, step, flags, 0)


def test_sampling() -> None:
    g = gpu()
    R, V, K = 3, 248320, 64
    logits = RNG.normal(size=(R, V)).astype(np.float32) * 3
    logits[1, 123456] = 50.0
    lb = g.array(logits)
    chunks = h.ceil_div(V, 4096)
    # argmax
    wv, wi, tok = g.zeros(R * chunks * 4), g.zeros(R * chunks * 4), g.zeros(R * 4)
    pa = _sample_params(R, V, V, 1, chunks)
    g.run([g.launch(g.pipeline(SAMPLE_SRC, "sml_llm_argmax_partial_fwd_f32"), [lb, wv, wi, tok], pa, (chunks, R, 1)),
           g.launch(g.pipeline(SAMPLE_SRC, "sml_llm_argmax_final_fwd_f32"), [lb, wv, wi, tok], pa, R)])
    got = g.read(tok, np.int32, (R,))
    record("argmax V=248320", bool(np.array_equal(got, logits.argmax(axis=1))), f"got {got.tolist()}")
    # top-k
    cv, ci = g.zeros(R * chunks * K * 4), g.zeros(R * chunks * K * 4)
    tv, ti = g.zeros(R * K * 4), g.zeros(R * K * 4)
    pk = _sample_params(R, V, V, K, chunks)
    g.run([g.launch(g.pipeline(SAMPLE_SRC, "sml_llm_topk_partial_fwd_f32"), [lb, cv, ci], pk, (chunks, R, 1)),
           g.launch(g.pipeline(SAMPLE_SRC, "sml_llm_topk_merge_fwd_f32"), [lb, cv, ci, tv, ti], pk, R)])
    ref_i = np.argsort(-logits, axis=1, kind="stable")[:, :K]
    got_i = g.read(ti, np.int32, (R, K))
    got_v = g.read(tv, np.float32, (R, K))
    record("topk K=64 V=248320 indices", bool(np.array_equal(got_i, ref_i)), f"{int(np.sum(got_i != ref_i))} mismatches")
    check("topk K=64 values", got_v, np.take_along_axis(logits, ref_i, axis=1), 0)
    # sampling from the top-k with temperature / top-p / min-p and given u
    u = np.array([0.1, 0.5, 0.97], dtype=np.float32)
    toks, probs = g.zeros(R * 4), g.zeros(R * K * 4)
    temp, top_p, min_p = 0.8, 0.9, 0.02
    ps = _sample_params(R, V, V, K, chunks, temp, top_p, min_p, flags=1)
    g.run(g.launch(g.pipeline(SAMPLE_SRC, "sml_llm_sample_topk_fwd_f32"), [tv, ti, toks, probs, g.array(u)], ps, R))
    got_tok = g.read(toks, np.int32, (R,))
    got_pr = g.read(probs, np.float32, (R, K))
    ok = True
    for r in range(R):
        v = got_v[r].astype(np.float64)
        pr = np.exp((v - v[0]) / temp)
        pr /= pr.sum()
        keep = K
        cum = 0.0
        for i in range(K):
            if pr[i] < min_p * pr[0]:
                keep = i
                break
            cum += pr[i]
            if cum >= top_p:
                keep = i + 1
                break
        keep = max(keep, 1)
        kept = pr[:keep] / pr[:keep].sum()
        choice = min(int(np.searchsorted(np.cumsum(kept), u[r], side="right")), keep - 1)
        ok &= int(got_tok[r]) == int(got_i[r, choice])
        want_p = np.zeros(K)
        want_p[:keep] = kept
        ok &= bool(np.allclose(got_pr[r], want_p, atol=1e-5))
    record("sample top-k temp/top-p/min-p", ok, f"tokens {got_tok.tolist()}")
    # Philox path: deterministic per (seed, step, row) and a valid candidate
    ps2 = _sample_params(R, V, V, K, chunks, 1.0, 0.95, 0.0, seed=42, step=7)
    runs = []
    for _ in range(2):
        g.run(g.launch(g.pipeline(SAMPLE_SRC, "sml_llm_sample_topk_fwd_f32"), [tv, ti, toks, probs, None], ps2, R))
        runs.append(g.read(toks, np.int32, (R,)))
    valid = all(int(runs[0][r]) in set(got_i[r].tolist()) for r in range(R))
    record("sample philox deterministic + valid", bool(np.array_equal(runs[0], runs[1])) and valid, f"{runs[0].tolist()}")
    # penalties
    pl = RNG.normal(size=(2, 1000)).astype(np.float32)
    pairs = np.array([[5, 17, 900], [0, 999, 3]], dtype=np.int32)
    counts = np.array([[1, 3, 0], [2, 1, 1]], dtype=np.int32)
    plb = g.array(pl)
    pp = struct.pack("<3Ifff2I", 2, 3, 1000, 1.3, 0.2, 0.5, 0, 0)
    g.run(g.launch(g.pipeline(SAMPLE_SRC, "sml_llm_logits_penalty_fwd_f32"),
                   [plb, g.array(pairs), g.array(counts)], pp, (1, 2, 1)))
    want = pl.astype(np.float64).copy()
    for r in range(2):
        for t_, c_ in zip(pairs[r], counts[r]):
            if c_ > 0:
                l_ = want[r, t_]
                l_ = l_ / 1.3 if l_ > 0 else l_ * 1.3
                want[r, t_] = l_ - 0.2 * c_ - 0.5
    check("logits penalty", g.read(plb, np.float32, (2, 1000)), want, 1e-6)


# ---------------------------------------------------------------------------
# Prefill GEMM on cooperative matrices
# ---------------------------------------------------------------------------
GEMM_SRC = "slangml/quant/gemm_quant_coopmat.slang"


def gemm_payload(M, N, K, group, lda, ldc, ldr, flags, alpha):
    return struct.pack("<8If3I", M, N, K, F.group_shift(group), lda, ldc, ldr, flags, alpha, 0, 0, 0)


def _gemm_case(fmt, variant, M, N, K, group, flags, alpha=1.0):
    g = gpu()
    if not g.has_coopmat:
        record(f"gemm {fmt}{variant}", True, "skipped: no cooperative matrix support")
        return
    a = RNG.normal(size=(M, K)).astype(np.float32)
    w = RNG.normal(size=(N, K)).astype(np.float32) * 0.05
    bias = RNG.normal(size=N).astype(np.float32)
    res = RNG.normal(size=(M, N)).astype(np.float32)
    codes, meta, deq = F.pack_weight(fmt, w, group)
    cb = g.array(np.ascontiguousarray(codes).view(np.uint8))
    mb = g.array(meta) if meta is not None else None
    cbuf = g.zeros(M * N * 4)
    bm = 64 if variant == "_m64" else 128
    pipe = g.pipeline(GEMM_SRC, f"sml_quant_gemm_fwd_{fmt}_coopmat{variant}")
    d = g.launch(pipe, [cb, mb, g.array(a), cbuf, g.array(bias), g.array(res)],
                 gemm_payload(M, N, K, group, K, N, N, flags, alpha), (h.ceil_div(N, 128), h.ceil_div(M, bm), 1))
    g.run(d)
    a16 = a.astype(np.float16).astype(np.float64)
    w16 = deq.astype(np.float16).astype(np.float64)
    want = alpha * a16 @ w16.T
    if flags & 1:
        want += bias
    if flags & 2:
        want += res
    check(f"gemm coopmat {fmt}{variant} M{M} N{N} K{K} flags{flags}", g.read(cbuf, np.float32, (M, N)), want, 1e-4)


def test_gemm_coopmat() -> None:
    _gemm_case("q4", "", 300, 1000, 512, 64, 3, 0.5)
    _gemm_case("q4", "", 256, 256, 1024, 32, 0)          # full tiles -> direct store path
    _gemm_case("q8", "", 129, 384, 256, 32, 1)
    _gemm_case("f16", "", 128, 384, 256, 32, 0)
    _gemm_case("bf16", "", 200, 130, 96, 32, 2)
    _gemm_case("q4", "_m64", 64, 256, 5120, 32, 0)
    _gemm_case("q8", "_m64", 37, 300, 512, 64, 3, 2.0)
    _gemm_case("bf16", "_m64", 77, 200, 96, 32, 1)


TESTS = [
    test_quantize_q8act,
    test_gemv_q4,
    test_gemv_q8,
    test_gemv_mxfp4,
    test_gemv_f16_bf16,
    test_gemv_swiglu,
    test_rmsnorm_fused,
    test_rope_qknorm,
    test_attention,
    test_conv1d,
    test_gated_delta_rule,
    test_embedding,
    test_moe_router,
    test_moe_gemv,
    test_sampling,
    test_gemm_coopmat,
]


def main(argv: list[str]) -> int:
    pattern = argv[1] if len(argv) > 1 else ""
    for test in TESTS:
        if pattern and pattern not in test.__name__:
            continue
        try:
            test()
        except Exception as exc:  # report and continue with the other kernels
            traceback.print_exc()
            record(test.__name__, False, f"exception: {exc!r}")
    failed = 0
    for name, ok, detail in _results:
        print(f"{'PASS' if ok else 'FAIL'}  {name:58s} {detail}")
        failed += 0 if ok else 1
    print(f"{len(_results) - failed}/{len(_results)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
