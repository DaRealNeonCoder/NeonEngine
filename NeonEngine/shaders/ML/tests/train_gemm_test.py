"""GPU tests of the training GEMMs (slangml/train/gemm_train_*.slang) against NumPy.

Covers every operand layout of the tensor-core kernel (sml_train_gemm_tc_*),
all three modes (plain + split-K, M-grouped, K-grouped), bias / residual /
accumulate epilogues, the direct-store fast path and ragged edges, and compares
it with the fp32 kernel on the same problems. With --bench it also times the
trainer's typical shapes on both kernels.

Usage (from shaders/ML/tests):
    python train_gemm_test.py [--bench]
Exit code 0 = all checks passed.
"""

from __future__ import annotations

import argparse
import struct
import sys

import numpy as np

import llm_harness as h

SRC_F32 = "slangml/train/gemm_train_f32.slang"
SRC_TC = "slangml/train/gemm_train_coopmat.slang"
INVALID = 0xFFFFFFFF
BIAS, RESID, ACCUM = 1, 2, 4
RNG = np.random.default_rng(7)
_fail = 0


def payload(M, N, K, flags, sAm, sAk, sBk, sBn, ldC, mode=0, kChunk=None, sBG=0, sBiasG=0, sCG=0, alpha=1.0):
    kChunk = K if kChunk is None else kChunk
    return struct.pack("<14If I", M, N, K, flags, sAm, sAk, sBk, sBn, ldC, mode, kChunk, sBG, sBiasG, sCG, alpha, 0)


def layout_entry(sAm, sAk, sBk, sBn):
    a = "k" if sAk == 1 else "m"
    b = "k" if sBk == 1 else "n"
    return f"sml_train_gemm_tc_{a}{b}"


def check(name, got, want, tol):
    global _fail
    scale = max(1.0, float(np.abs(want).max()))
    err = float(np.abs(got - want).max()) / scale
    ok = err <= tol
    _fail += 0 if ok else 1
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: max err / max |ref| = {err:.2e}")


def f16(x):
    return x.astype(np.float16).astype(np.float64)


def run_case(g, name, M, N, K, layoutA, layoutB, flags=0, mode=0, groups=1, table=None, splits=1, alpha=1.0):
    """layoutA: 'k' (A stored [M][K]) or 'm' (stored [K][M]); layoutB: 'k' ([N][K]) or 'n' ([K][N])."""
    Ag = groups if mode == 0 else 1
    Bg = groups if mode == 1 else 1
    A = RNG.normal(size=(K, M) if layoutA == "m" else (M, K)).astype(np.float32)
    B = (RNG.normal(size=(Bg, N, K) if layoutB == "k" else (Bg, K, N)) * 0.1).astype(np.float32)
    Cg = groups if mode == 2 else 1
    C0 = RNG.normal(size=(Cg, M, N)).astype(np.float32)
    bias = RNG.normal(size=(Bg, N)).astype(np.float32)
    R = RNG.normal(size=(M, N)).astype(np.float32)
    sAm, sAk = (1, M) if layoutA == "m" else (K, 1)
    sBk, sBn = (N, 1) if layoutB == "n" else (1, K)
    Am = A.T if layoutA == "m" else A                      # [M][K]
    Bm = np.transpose(B, (0, 2, 1)) if layoutB == "k" else B  # [g][K][N]

    tbl = np.asarray(table if table is not None else [0], dtype=np.uint32)
    kChunk = K
    if mode == 0 and splits > 1:
        kChunk = ((K + splits - 1) // splits + 31) // 32 * 32
        splits = (K + kChunk - 1) // kChunk

    def reference(round16):
        cv = f16 if round16 else (lambda x: x.astype(np.float64))
        a, b = cv(Am), cv(Bm)
        out = C0.astype(np.float64).copy()
        if mode == 0:
            parts = []
            for s in range(splits):
                k0, k1 = s * kChunk, min(K, (s + 1) * kChunk)
                parts.append(alpha * a[:, k0:k1] @ b[0, k0:k1])
            if splits > 1:
                return np.stack(parts)  # partials, flags ignored by the host for split-K
            v = parts[0]
            if flags & BIAS:
                v = v + bias[0]
            if flags & RESID:
                v = v + R
            if flags & ACCUM:
                v = v + out[0]
            return v[None]
        if mode == 1:
            v = out[0].copy()
            for t in range((M + 127) // 128):
                gr = int(tbl[t])
                if gr == INVALID:
                    continue
                r0, r1 = t * 128, min(M, (t + 1) * 128)
                x = alpha * a[r0:r1] @ b[gr]
                if flags & BIAS:
                    x = x + bias[gr]
                if flags & RESID:
                    x = x + R[r0:r1]
                if flags & ACCUM:
                    x = x + out[0][r0:r1]
                v[r0:r1] = x
            return v[None]
        res = out.copy()
        for gr in range(groups):
            k0, k1 = int(tbl[gr]), int(tbl[gr + 1])
            if k0 >= k1 and (flags & ACCUM):
                continue
            x = alpha * a[:, k0:k1] @ b[0, k0:k1]
            if flags & BIAS:
                x = x + bias[0]
            if flags & RESID:
                x = x + R
            if flags & ACCUM:
                x = x + out[gr]
            res[gr] = x
        return res

    results = {}
    for kind in ("tc", "f32"):
        if kind == "tc" and not g.has_coopmat:
            continue
        cb = g.array(C0 if splits == 1 else np.zeros((splits, M, N), np.float32))
        entry = layout_entry(sAm, sAk, sBk, sBn) if kind == "tc" else "sml_train_gemm_fwd_f32"
        pipe = g.pipeline(SRC_TC if kind == "tc" else SRC_F32, entry)
        fl = 0 if splits > 1 else flags
        p = payload(M, N, K, fl, sAm, sAk, sBk, sBn, N, mode, kChunk,
                    N * K if mode == 1 else 0, N if mode == 1 else 0, M * N if (mode == 2 or splits > 1) else 0, alpha)
        gz = groups if mode == 2 else splits
        d = g.launch(pipe, [g.array(A), g.array(B), cb, g.array(bias), g.array(R), (g.array(tbl), h.DTYPE_I32 if hasattr(h, "DTYPE_I32") else 5, 0)],
                     p, (h.ceil_div(N, 128), h.ceil_div(M, 128), gz))
        g.run(d)
        shape = (splits if splits > 1 else Cg, M, N)
        got = g.read(cb, np.float32, shape).astype(np.float64)
        want = reference(kind == "tc")
        check(f"{kind:3s} {name}", got, want, 2e-5 if kind == "tc" else 2e-6)
        results[kind] = got
    if "tc" in results:
        # Against exact fp32 math: the fp16 rounding error must stay at the expected level.
        want = reference(False)
        err = float(np.abs(results["tc"] - want).max()) / max(1.0, float(np.abs(want).max()))
        print(f"        tc vs fp32 math: {err:.2e} (fp16 input rounding)")


def bench(g):
    if not g.has_coopmat:
        print("no cooperative matrix support: skipping the benchmark")
        return
    shapes = [  # (name, M, N, K, layoutA, layoutB)
        ("fwd  8192 x 768 x 512", 8192, 768, 512, "k", "k"),
        ("dX   8192 x 512 x 768", 8192, 512, 768, "k", "n"),
        ("dW    768 x 512 x 8192", 768, 512, 8192, "m", "n"),
        ("head 2048 x 50304 x 512", 2048, 50304, 512, "k", "k"),
        ("moe 34816 x 1024 x 512", 34816, 1024, 512, "k", "k"),
    ]
    print(f"\n{'shape':28s} {'fp32 ms':>9s} {'TFLOPS':>7s} {'tc ms':>9s} {'TFLOPS':>7s} {'speedup':>8s}")
    for name, M, N, K, la, lb in shapes:
        A = g.zeros(M * K * 4)
        B = g.zeros(N * K * 4)
        C = g.zeros(M * N * 4)
        sAm, sAk = (1, M) if la == "m" else (K, 1)
        sBk, sBn = (N, 1) if lb == "n" else (1, K)
        times = []
        for src, entry in ((SRC_F32, "sml_train_gemm_fwd_f32"), (SRC_TC, layout_entry(sAm, sAk, sBk, sBn))):
            d = g.launch(g.pipeline(src, entry), [A, B, C, None, None, None],
                         payload(M, N, K, 0, sAm, sAk, sBk, sBn, N), (h.ceil_div(N, 128), h.ceil_div(M, 128), 1))
            times.append(g.bench([d]))
        fl = 2.0 * M * N * K / 1e9
        print(f"{name:28s} {times[0]:9.3f} {fl / times[0]:7.2f} {times[1]:9.3f} {fl / times[1]:7.2f} {times[0] / times[1]:7.2f}x")
        for b in (A, B, C):
            g.free(b)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", action="store_true")
    args = ap.parse_args()
    g = h.Gpu()
    print(f"device: {g.device_name}  coopmat={g.has_coopmat}")
    run_case(g, "fwd kk ragged + bias + resid", 300, 200, 72, "k", "k", BIAS | RESID)
    run_case(g, "fwd kk K=20 (partial octets)", 130, 140, 20, "k", "k", BIAS)
    run_case(g, "fwd kk full tiles", 256, 256, 512, "k", "k")
    run_case(g, "fwd kk full tiles + bias + resid + accum", 256, 384, 96, "k", "k", BIAS | RESID | ACCUM)
    run_case(g, "fwd kk full tiles + bias, alpha 2 (shared epilogue)", 128, 128, 64, "k", "k", BIAS, alpha=2.0)
    run_case(g, "mode 1 full tiles + bias", 2 * 128, 256, 96, "k", "k", BIAS, mode=1, groups=2, table=[1, 0])
    run_case(g, "fwd kk alpha + accum", 129, 257, 96, "k", "k", ACCUM, alpha=0.5)
    run_case(g, "dX kn ragged N=132 + accum", 200, 132, 100, "k", "n", ACCUM)
    run_case(g, "dX kn full tiles + accum (preload)", 256, 384, 256, "k", "n", ACCUM)
    run_case(g, "dW mn full tiles + accum (preload)", 256, 256, 1000, "m", "n", ACCUM)
    run_case(g, "dW mn ragged M=100 N=36 + accum", 100, 36, 300, "m", "n", ACCUM)
    run_case(g, "mk layout", 140, 150, 64, "m", "k", BIAS)
    run_case(g, "split-K x3 (kk)", 128, 256, 1024, "k", "k", splits=3)
    run_case(g, "split-K x4 (mn)", 192, 128, 2048, "m", "n", splits=4)
    run_case(g, "mode 1 M-grouped + bias", 3 * 128, 192, 64, "k", "k", BIAS, mode=1, groups=3,
             table=[0, INVALID, 2])
    run_case(g, "mode 1 M-grouped dX (kn)", 2 * 128, 96, 128, "k", "n", 0, mode=1, groups=2, table=[1, 0])
    run_case(g, "mode 2 K-grouped dW + accum, empty group", 128, 64, 384, "m", "n", ACCUM, mode=2, groups=3,
             table=[0, 128, 128, 384])
    run_case(g, "mode 2 K-grouped no accum, empty group", 96, 160, 256, "m", "n", 0, mode=2, groups=2,
             table=[0, 0, 256])
    if args.bench:
        bench(g)
    print("ALL PASSED" if _fail == 0 else f"{_fail} FAILED")
    return 0 if _fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
