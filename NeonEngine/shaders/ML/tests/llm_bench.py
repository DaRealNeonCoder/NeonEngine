"""Benchmarks for the SlangML LLM kernels at Qwen3.8-27B shapes.

Usage:
    python llm_bench.py            # everything
    python llm_bench.py gemv       # benchmarks whose name contains "gemv"

Reports kernel time from GPU timestamps and the effective bandwidth / FLOP rate.
Memory-bound kernels are compared against the measured device copy bandwidth
(read + write), which is the practical DRAM ceiling.
"""

from __future__ import annotations

import struct
import sys

import numpy as np

import llm_formats as F
import llm_harness as h
import llm_kernel_tests as T

RNG = np.random.default_rng(7)
g = None
peak_gbs = 0.0


def setup() -> None:
    global g, peak_gbs
    g = h.Gpu()
    peak_gbs = g.copy_bandwidth_gbs()
    print(f"Device: {g.device_name}")
    print(f"Copy bandwidth (read+write): {peak_gbs:.1f} GB/s")


def row(name: str, ms: float, nbytes: float = 0.0, flops: float = 0.0) -> None:
    parts = [f"{name:52s} {ms * 1e3:9.1f} us"]
    if nbytes:
        gbs = nbytes / (ms * 1e-3) / 1e9
        parts.append(f"{gbs:7.1f} GB/s ({100 * gbs / peak_gbs:5.1f}% of copy)")
    if flops:
        parts.append(f"{flops / (ms * 1e-3) / 1e12:6.2f} TFLOP/s")
    print("  ".join(parts))


# Qwen3.8-27B decode projections (N, K)
QWEN_SHAPES = [
    ("q_proj+gate 5120->12288", 12288, 5120),
    ("in_proj_qkv 5120->10240", 10240, 5120),
    ("in_proj_z / o_proj-ish 5120->6144", 6144, 5120),
    ("out_proj 6144->5120", 5120, 6144),
    ("down_proj 17408->5120", 5120, 17408),
]


def bench_gemv() -> None:
    print("\n# GEMV (decode, M activation rows)")
    for fmt in ("q4", "q8", "f16"):
        for name, n, k in QWEN_SHAPES:
            for mc in ((1, 4) if fmt != "f16" else (1,)):
                w = RNG.normal(size=(n, k)).astype(np.float32) * 0.02
                codes, meta, _ = F.pack_weight(fmt, w, 64 if fmt == "q4" else 32)
                cb = g.array(np.ascontiguousarray(codes).view(np.uint8))
                mb = g.array(meta) if meta is not None else None
                x = RNG.normal(size=(mc, k)).astype(np.float32)
                xb = g.array(x)
                yb = g.zeros(mc * n * 4)
                pipe = g.pipeline(T.GEMV_SRC, f"sml_quant_gemv_fwd_{fmt}_m{mc}")
                payload = T.gemv_payload(n, k, mc, 64 if fmt == "q4" else 32)
                if fmt in ("q4", "q8"):
                    qd, xq, xs = T.quantize_act(g, xb, mc, k)
                    g.run(qd)
                    d = g.launch(pipe, [cb, mb, (xq, h.DTYPE_INT8, 0), xs, yb], payload, T.gemv_groups(g, n, mc))
                else:
                    d = g.launch(pipe, [cb, None, xb, None, yb], payload, T.gemv_groups(g, n, mc))
                ms = g.bench(d, repeat=50)
                nbytes = codes.nbytes + (meta.nbytes if meta is not None else 0)
                row(f"{fmt} M={mc} {name}", ms, nbytes, 2.0 * mc * n * k)
                for b in (cb, mb, xb, yb):
                    if b is not None:
                        g.free(b)

    print("\n# fused gate/up SwiGLU GEMV 5120 -> 2 x 17408")
    n, k = 17408, 5120
    for fmt in ("q4", "q8"):
        wgt = RNG.normal(size=(n, k)).astype(np.float32) * 0.02
        cg, mg, _ = F.pack_weight(fmt, wgt, 64 if fmt == "q4" else 32)
        cu, mu, _ = F.pack_weight(fmt, wgt[::-1].copy(), 64 if fmt == "q4" else 32)
        bufs = [g.array(np.ascontiguousarray(a).view(np.uint8)) for a in (cg, cu)] + [g.array(mg), g.array(mu)]
        xb = g.array(RNG.normal(size=(1, k)).astype(np.float32))
        yb = g.zeros(n * 4)
        qd, xq, xs = T.quantize_act(g, xb, 1, k)
        g.run(qd)
        pipe = g.pipeline(T.GEMV_SRC, f"sml_quant_gemv_swiglu_fwd_{fmt}_m1")
        payload = T.gemv_payload(n, k, 1, 64 if fmt == "q4" else 32)
        d = g.launch(pipe, [bufs[0], bufs[2], (xq, h.DTYPE_INT8, 0), xs, yb, None, None, bufs[1], bufs[3]],
                     payload, T.gemv_groups(g, n))
        ms = g.bench(d, repeat=30)
        row(f"{fmt} swiglu M=1", ms, 2 * (cg.nbytes + mg.nbytes), 4.0 * n * k)
        for b in bufs + [xb, yb]:
            g.free(b)

    print("\n# activation quantize")
    for k in (5120, 17408):
        xb = g.array(RNG.normal(size=(1, k)).astype(np.float32))
        qd, xq, xs = T.quantize_act(g, xb, 1, k)
        row(f"quantize_q8act 1x{k}", g.bench(qd, repeat=100), k * 4 + k + k // 32 * 8)


def bench_gptoss() -> None:
    """gpt-oss-20b decode shapes: MXFP4 experts (vs Q4 at the same shape) and the MoE pair."""
    print()
    print("# gpt-oss-20b: dense GEMV, MXFP4 vs Q4 g32 (2880 x 2880 expert matrix)")
    n = k = 2880
    for fmt in ("mxfp4", "q4"):
        w = RNG.normal(size=(n, k)).astype(np.float32) * 0.02
        codes, meta, _ = F.pack_weight(fmt, w, 32)
        cb, mb = g.array(np.ascontiguousarray(codes).view(np.uint8)), g.array(meta)
        xb = g.array(RNG.normal(size=(1, k)).astype(np.float32))
        yb = g.zeros(n * 4)
        qd, xq, xs = T.quantize_act(g, xb, 1, k)
        g.run(qd)
        d = g.launch(g.pipeline(T.GEMV_SRC, f"sml_quant_gemv_fwd_{fmt}_m1"), [cb, mb, (xq, h.DTYPE_INT8, 0), xs, yb],
                     T.gemv_payload(n, k, 1, 32), T.gemv_groups(g, n))
        row(f"{fmt} M=1 2880->2880", g.bench(d, repeat=100), codes.nbytes + meta.nbytes, 2.0 * n * k)
        for b in (cb, mb, xb, yb):
            g.free(b)

    print()
    print("# gpt-oss-20b: MoE layer (32 experts, top-4, H = I = 2880), MXFP4 via expert table")
    E, H, I, topk = 32, 2880, 2880, 4
    blk = lambda rows, cols: (rows * cols // 2, rows * cols // 32)  # codes, scales bytes per expert
    gc, gs = blk(I, H)
    dc, ds = blk(H, I)
    rnd = lambda nbytes: g.array(RNG.integers(0, 256, size=nbytes, dtype=np.uint8))
    # Random codes are fine for timing; scales near 2^0 keep values finite.
    scales = lambda nbytes: g.array(np.full(nbytes, 120, dtype=np.uint8))
    gate_c, up_c, down_c = rnd(E * gc), rnd(E * gc), rnd(E * dc)
    gate_s, up_s, down_s = scales(E * gs), scales(E * gs), scales(E * ds)
    t1 = np.array([[gate_c.address + e * gc, gate_s.address + e * gs, up_c.address + e * gc, up_s.address + e * gs]
                   for e in range(E)], dtype=np.uint64)
    t2 = np.array([[down_c.address + e * dc, down_s.address + e * ds, 0, 0] for e in range(E)], dtype=np.uint64)
    tb1, tb2 = g.array(t1), g.array(t2)
    ids = g.array(np.array([3, 17, 22, 30], dtype=np.int32))
    rw = g.array(np.full(topk, 0.25, dtype=np.float32))
    bias = g.zeros(E * max(H, I) * 4)
    xb = g.array(RNG.normal(size=(1, H)).astype(np.float32))
    qd, xq, xs = T.quantize_act(g, xb, 1, H)
    g.run(qd)
    y1 = g.zeros(topk * I * 4)
    out = g.zeros(H * 4)
    rows_per_wg = 2 * 128 // g.subgroup_size
    p1 = struct.pack("<5IfIIffIIII", I, H, topk, 0, 1 | 4, 1.0, I, 2, 7.0, 1.702, 0, 0, I, topk)
    up = g.launch(g.pipeline(T.MOE_SRC, "sml_quant_gemv_moe_swiglu_fwd_mxfp4"),
                  [None, None, (xq, h.DTYPE_INT8, 0), xs, y1, bias, bias, None, None, ids, rw, tb1],
                  p1, (h.ceil_div(I, rows_per_wg), topk, 1))
    qd2, yq, ys = T.quantize_act(g, y1, topk, I)
    p2 = struct.pack("<5IfIIffIIII", H, I, topk, 0, 1 | 2 | 4, 1.0, H, 0, 7.0, 1.702, 0, 0, H, topk)
    down = g.launch(g.pipeline(T.MOE_SRC, "sml_quant_gemv_moe_down_fwd_mxfp4"),
                    [None, None, (yq, h.DTYPE_INT8, 0), ys, out, bias, out, None, None, ids, rw, tb2],
                    p2, (h.ceil_div(H, rows_per_wg), 1, 1))
    up_bytes = topk * 2 * (gc + gs)
    down_bytes = topk * (dc + ds)
    row("moe gate/up swiglu (4 experts)", g.bench(up, repeat=50), up_bytes, 4.0 * topk * I * H)
    row("moe down weighted sum (4 experts)", g.bench(down, repeat=50), down_bytes, 2.0 * topk * H * I)
    row("moe layer: up + quantize + down", g.bench([up, qd2, down], repeat=30), up_bytes + down_bytes)
    for b in (gate_c, up_c, down_c, gate_s, up_s, down_s, tb1, tb2, ids, rw, bias, xb, y1, out):
        g.free(b)


def bench_gemm() -> None:
    print("\n# coopmat GEMM (prefill), fp16 tensor cores, fp32 accumulate")
    shapes = [("gate/up 5120->17408", 17408, 5120), ("down 17408->5120", 5120, 17408),
              ("q_proj+gate 5120->12288", 12288, 5120)]
    for fmt in ("q4", "q8", "f16"):
        for name, n, k in shapes:
            w = RNG.normal(size=(n, k)).astype(np.float32) * 0.02
            codes, meta, _ = F.pack_weight(fmt, w, 64 if fmt == "q4" else 32)
            cb = g.array(np.ascontiguousarray(codes).view(np.uint8))
            mb = g.array(meta) if meta is not None else None
            for m, variant in ((512, ""), (2048, ""), (64, "_m64")):
                if variant == "_m64" and fmt == "f16":
                    continue
                ab = g.array(RNG.normal(size=(m, k)).astype(np.float32))
                cbuf = g.zeros(m * n * 4)
                bm = 64 if variant else 128
                pipe = g.pipeline(T.GEMM_SRC, f"sml_quant_gemm_fwd_{fmt}_coopmat{variant}")
                d = g.launch(pipe, [cb, mb, ab, cbuf, None, None],
                             T.gemm_payload(m, n, k, 64 if fmt == "q4" else 32, k, n, n, 0, 1.0),
                             (h.ceil_div(n, 128), h.ceil_div(m, bm), 1))
                ms = g.bench(d, repeat=5 if m >= 2048 else 10)
                row(f"{fmt}{variant} M={m} {name}", ms, flops=2.0 * m * n * k)
                g.free(ab)
                g.free(cbuf)
            g.free(cb)
            if mb is not None:
                g.free(mb)


def bench_attention() -> None:
    print("\n# attention decode, Qwen3.8 full-attention layer (Hq=24, Hkv=4, D=256, gate)")
    Hq, Hkv, D = 24, 4, 256
    for ctx in (1024, 8192, 32768):
        kc = g.array((RNG.normal(size=(Hkv, ctx + 8, D))).astype(np.float16))
        vc = g.array((RNG.normal(size=(Hkv, ctx + 8, D))).astype(np.float16))
        q = g.array(RNG.normal(size=(1, Hq, D)).astype(np.float32))
        gate = g.array(RNG.normal(size=(1, Hq, 2 * D)).astype(np.float32))
        out = g.zeros(Hq * D * 4)
        for chunk in (256, 1024):
            nch = h.ceil_div(ctx, chunk)
            ws_o, ws_ml = g.zeros(nch * Hq * D * 4), g.zeros(nch * Hq * 8)
            tensors = [q, kc, vc, out, None, (gate, h.DTYPE_FP32, D), ws_o, ws_ml]
            payload = struct.pack("<9Iff5I", Hq, Hkv, 1, 1, Hq * D, ctx + 8, ctx - 1, chunk, nch,
                                  1 / 16.0, 0.0, 0, 1 | 4, Hq * 2 * D, 2 * D, Hq * D)
            disp = [g.launch(g.pipeline(T.ATTN_SRC, "sml_llm_attention_fwd_f16kv_d256"), tensors, payload, (nch, 1, Hkv))]
            if nch > 1:
                disp.append(g.launch(g.pipeline(T.ATTN_SRC, "sml_llm_attention_combine_fwd_f32"), tensors,
                                     payload + struct.pack("<I", D), Hq))
            ms = g.bench(disp, repeat=20)
            row(f"ctx={ctx} chunk={chunk} ({nch} splits)", ms, 2 * Hkv * ctx * D * 2)
            g.free(ws_o)
            g.free(ws_ml)
        for b in (kc, vc, q, gate, out):
            g.free(b)

    print("\n# attention prefill (T new tokens on an empty cache), D=256, G=6")
    for t_new in (512, 2048):
        Hq, Hkv = 24, 4
        kc = g.array((RNG.normal(size=(Hkv, t_new, D))).astype(np.float16))
        vc = g.array((RNG.normal(size=(Hkv, t_new, D))).astype(np.float16))
        q = g.array(RNG.normal(size=(t_new, Hq, D)).astype(np.float32))
        out = g.zeros(t_new * Hq * D * 4)
        tensors = [q, kc, vc, out, None, None, None, None]
        payload = struct.pack("<9Iff5I", Hq, Hkv, t_new, 1, Hq * D, t_new, 0, 1 << 20, 1,
                              1 / 16.0, 0.0, 0, 1, 0, 0, Hq * D)
        d = g.launch(g.pipeline(T.ATTN_SRC, "sml_llm_attention_fwd_f16kv_d256"), tensors, payload, (1, t_new, Hkv))
        ms = g.bench(d, repeat=2, trials=3)
        row(f"prefill T={t_new} causal", ms, flops=4.0 * Hq * D * t_new * t_new / 2)
        for b in (kc, vc, q, out):
            g.free(b)


def bench_linear_attention() -> None:
    print("\n# Gated DeltaNet (Qwen3.8: Hk=16, Hv=48, dk=dv=128)")
    Hk, Hv, dk, dv = 16, 48, 128, 128
    ld = 2 * Hk * dk + Hv * dv
    for T_ in (1, 512):
        qkv = g.array(RNG.normal(size=(T_, ld)).astype(np.float32))
        a = g.array(RNG.normal(size=(T_, Hv)).astype(np.float32))
        b = g.array(RNG.normal(size=(T_, Hv)).astype(np.float32))
        st = g.array(RNG.normal(size=(Hv, dk, dv)).astype(np.float32) * 0.01)
        out = g.zeros(T_ * Hv * dv * 4)
        payload = struct.pack("<10IffII", T_, Hk, Hv, dv, ld, 0, Hk * dk, 2 * Hk * dk, Hv, Hv * dv,
                              1 / np.sqrt(dk), 1e-6, 0, 0)
        d = g.launch(g.pipeline(T.GDN_SRC, "sml_linattn_gated_delta_rule_fwd_f32"),
                     [qkv, a, b, g.array(np.zeros(Hv, np.float32)), g.array(np.zeros(Hv, np.float32)), st, out],
                     payload, (dv // 32, Hv, 1))
        ms = g.bench(d, repeat=20 if T_ == 1 else 3)
        row(f"gated_delta_rule T={T_}", ms, 2 * Hv * dk * dv * 4 + T_ * (ld + Hv * dv) * 4)
    C = ld
    for T_ in (1, 512):
        x = g.array(RNG.normal(size=(T_, C)).astype(np.float32))
        y = g.zeros(T_ * C * 4)
        st = g.array(RNG.normal(size=(3, C)).astype(np.float32))
        so = g.zeros(3 * C * 4)
        tpt = 1 if T_ == 1 else 32
        d = g.launch(g.pipeline(T.CONV_SRC, "sml_linattn_conv1d_fwd_f32"),
                     [x, g.array(RNG.normal(size=(C, 4)).astype(np.float32)), None, st, y, so if T_ > 1 else st],
                     struct.pack("<8I", T_, C, 4, 1 | 4 | 8, C, C, tpt, 0), (h.ceil_div(C, 128), h.ceil_div(T_, tpt), 1))
        ms = g.bench(d, repeat=20)
        row(f"conv1d+silu T={T_} C={C}", ms, 2 * T_ * C * 4)


def bench_small_ops() -> None:
    print("\n# fused small ops (decode, one token)")
    x = g.array(RNG.normal(size=(1, 5120)).astype(np.float32))
    r = g.array(RNG.normal(size=(1, 5120)).astype(np.float32))
    w = g.array(RNG.normal(size=5120).astype(np.float32))
    y, hb, yq, ys = g.zeros(5120 * 4), g.zeros(5120 * 4), g.zeros(5120), g.zeros(5120 // 32 * 8)
    d = g.launch(g.pipeline(T.NORM_SRC, "sml_llm_rmsnorm_fwd_f32_row"), [x, r, w, y, hb, None, (yq, h.DTYPE_INT8, 0), ys],
                 struct.pack("<2IfI4I", 1, 5120, 1e-6, 1 | 2 | 4 | 16 | 32, 5120, 5120, 5120, 5120), 1)
    row("add + rmsnorm(1+w) + int8 quantize, 5120", g.bench(d, repeat=100))
    logits = g.array(RNG.normal(size=(1, 248320)).astype(np.float32))
    ch = h.ceil_div(248320, 4096)
    wv, wi, tok = g.zeros(ch * 4), g.zeros(ch * 4), g.zeros(4)
    pa = struct.pack("<5IfffIIII", 1, 248320, 248320, 1, ch, 1.0, 1.0, 0.0, 0, 0, 0, 0)
    disp = [g.launch(g.pipeline(T.SAMPLE_SRC, "sml_llm_argmax_partial_fwd_f32"), [logits, wv, wi, tok], pa, (ch, 1, 1)),
            g.launch(g.pipeline(T.SAMPLE_SRC, "sml_llm_argmax_final_fwd_f32"), [logits, wv, wi, tok], pa, 1)]
    row("argmax over 248320 logits", g.bench(disp, repeat=50), 248320 * 4)
    K = 64
    cv, ci, tv, ti = g.zeros(ch * K * 4), g.zeros(ch * K * 4), g.zeros(K * 4), g.zeros(K * 4)
    pk = struct.pack("<5IfffIIII", 1, 248320, 248320, K, ch, 1.0, 1.0, 0.0, 0, 0, 0, 0)
    disp = [g.launch(g.pipeline(T.SAMPLE_SRC, "sml_llm_topk_partial_fwd_f32"), [logits, cv, ci], pk, (ch, 1, 1)),
            g.launch(g.pipeline(T.SAMPLE_SRC, "sml_llm_topk_merge_fwd_f32"), [logits, cv, ci, tv, ti], pk, 1)]
    row("top-64 over 248320 logits", g.bench(disp, repeat=20), 248320 * 4)


BENCHES = [bench_gemv, bench_gptoss, bench_gemm, bench_attention, bench_linear_attention, bench_small_ops]


def main(argv: list[str]) -> int:
    pattern = argv[1] if len(argv) > 1 else ""
    setup()
    for b in BENCHES:
        if pattern and pattern not in b.__name__:
            continue
        b()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
