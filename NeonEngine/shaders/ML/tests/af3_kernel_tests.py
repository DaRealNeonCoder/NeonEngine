"""CPU-reference tests for AlphaFold 3 SlangML kernels.

Usage:
    python af3_kernel_tests.py            # run every test
    python af3_kernel_tests.py binned     # run tests whose name contains "binned"

Each test compiles one entry point, dispatches it on the GPU through af3_harness,
and compares the output with a NumPy reference that follows the same math as the
kernel header comment.  Failures print the largest absolute error.
"""

from __future__ import annotations

import struct
import sys
import tempfile
from pathlib import Path

import numpy as np

import af3_harness as h

RNG = np.random.default_rng(1234)
TOL = 1e-4

_results: list[tuple[str, bool, str]] = []
_gpu: h.Gpu | None = None
_tmp = Path(tempfile.mkdtemp(prefix="af3_spv_"))


def gpu() -> h.Gpu:
    global _gpu
    if _gpu is None:
        _gpu = h.Gpu()
        print(f"Device: {_gpu.device_name}")
    return _gpu


def spv(source: str, entry: str) -> bytes:
    return h.compile_entry(source, entry, _tmp)


def check(name: str, got: np.ndarray, want: np.ndarray, tol: float = TOL) -> None:
    err = float(np.max(np.abs(got.astype(np.float64) - want.astype(np.float64))))
    scale = max(float(np.max(np.abs(want))), 1e-30)  # tolerance is relative to the largest reference value
    ok = err <= tol * scale
    _results.append((name, ok, f"max abs err {err:.3e}"))


def record(name: str, ok: bool, detail: str) -> None:
    _results.append((name, ok, detail))


# ---------------------------------------------------------------------------
# Binned cross-entropy  [SI Eq. 9, 10, 12, 14]
# ---------------------------------------------------------------------------

def _bce_ref(logits: np.ndarray, targets: np.ndarray) -> np.ndarray:
    m = logits.max(axis=1, keepdims=True)
    lse = m + np.log(np.exp(logits - m).sum(axis=1, keepdims=True))
    return -(targets * (logits - lse)).sum(axis=1)


def _bce_grad_ref(logits: np.ndarray, targets: np.ndarray, upstream: np.ndarray) -> np.ndarray:
    m = logits.max(axis=1, keepdims=True)
    e = np.exp(logits - m)
    softmax = e / e.sum(axis=1, keepdims=True)
    t_sum = targets.sum(axis=1, keepdims=True)
    return upstream[:, None] * (softmax * t_sum - targets)


def test_binned_cross_entropy_fwd() -> None:
    rows, bins = 300, 64
    logits = RNG.normal(size=(rows, bins)).astype(np.float32) * 3.0
    targets = np.zeros((rows, bins), dtype=np.float32)
    targets[np.arange(rows), RNG.integers(0, bins, rows)] = 1.0
    payload = struct.pack("<4I", rows, bins, 0, 0)
    out = h.run_entry(
        gpu(),
        spv("slangml/loss/binned_cross_entropy_f32.slang", "sml_loss_binned_cross_entropy_fwd_f32"),
        "sml_loss_binned_cross_entropy_fwd_f32",
        payload,
        [
            h.Tensor(logits),
            h.Tensor(targets),
            h.Tensor(np.zeros(rows, dtype=np.float32), output=True),
        ],
        groups=h.ceil_div(rows, 64),
    )
    check("binned_cross_entropy_fwd", out[2], _bce_ref(logits.astype(np.float64), targets.astype(np.float64)))


def test_binned_cross_entropy_bwd() -> None:
    rows, bins = 300, 64
    logits = RNG.normal(size=(rows, bins)).astype(np.float32) * 3.0
    targets = RNG.random((rows, bins)).astype(np.float32)
    upstream = RNG.normal(size=rows).astype(np.float32)
    payload = struct.pack("<4I", rows, bins, 0, 0)
    out = h.run_entry(
        gpu(),
        spv("slangml/loss/binned_cross_entropy_f32.slang", "sml_loss_binned_cross_entropy_bwd_f32"),
        "sml_loss_binned_cross_entropy_bwd_f32",
        payload,
        [
            h.Tensor(logits),
            h.Tensor(targets),
            h.Tensor(upstream),
            h.Tensor(np.zeros((rows, bins), dtype=np.float32), output=True),
        ],
        groups=h.ceil_div(rows, 64),
    )
    want = _bce_grad_ref(logits.astype(np.float64), targets.astype(np.float64), upstream.astype(np.float64))
    check("binned_cross_entropy_bwd", out[3], want)


# ---------------------------------------------------------------------------
# Diffusion conditioning and preconditioning  [SI Alg 20, Alg 22]
# ---------------------------------------------------------------------------

def test_fourier_fwd() -> None:
    rows, dims, sigma = 17, 256, 16.0
    t = (RNG.random(rows).astype(np.float32) * 200.0 + 0.01).astype(np.float32)
    w = RNG.normal(size=dims).astype(np.float32)
    b = RNG.normal(size=dims).astype(np.float32)
    payload = struct.pack("<IIff", rows, dims, sigma, 0.0)
    out = h.run_entry(
        gpu(),
        spv("slangml/diffusion/conditioning_f32.slang", "sml_diffusion_fourier_fwd_f32"),
        "sml_diffusion_fourier_fwd_f32",
        payload,
        [
            h.Tensor(t),
            h.Tensor(w),
            h.Tensor(b),
            h.Tensor(np.zeros(rows * dims, dtype=np.float32), output=True),
        ],
        groups=h.ceil_div(rows * dims, 256),
    )
    x = 0.25 * np.log(t.astype(np.float64) / sigma)
    phase = x[:, None] * w.astype(np.float64)[None, :] + b.astype(np.float64)[None, :]
    want = np.cos(2.0 * np.pi * phase).reshape(-1)
    check("fourier_fwd", out[3], want)


def test_input_scale_fwd() -> None:
    atoms, sigma = 123, 16.0
    x = RNG.normal(size=atoms * 3).astype(np.float32) * 10.0
    t = (RNG.random(atoms).astype(np.float32) * 50.0).astype(np.float32)
    payload = struct.pack("<IIff", atoms, 0, sigma, 0.0)
    out = h.run_entry(
        gpu(),
        spv("slangml/diffusion/conditioning_f32.slang", "sml_diffusion_input_scale_fwd_f32"),
        "sml_diffusion_input_scale_fwd_f32",
        payload,
        [h.Tensor(x), h.Tensor(t), h.Tensor(np.zeros(atoms * 3, dtype=np.float32), output=True)],
        groups=h.ceil_div(atoms * 3, 256),
    )
    t_rep = np.repeat(t.astype(np.float64), 3)
    want = x.astype(np.float64) / np.sqrt(t_rep ** 2 + sigma ** 2)
    check("input_scale_fwd", out[2], want)


def test_output_combine_fwd() -> None:
    atoms, sigma = 123, 16.0
    x_noisy = RNG.normal(size=atoms * 3).astype(np.float32) * 10.0
    r_update = RNG.normal(size=atoms * 3).astype(np.float32)
    t = (RNG.random(atoms).astype(np.float32) * 50.0).astype(np.float32)
    payload = struct.pack("<IIff", atoms, 0, sigma, 0.0)
    out = h.run_entry(
        gpu(),
        spv("slangml/diffusion/conditioning_f32.slang", "sml_diffusion_output_combine_fwd_f32"),
        "sml_diffusion_output_combine_fwd_f32",
        payload,
        [
            h.Tensor(x_noisy),
            h.Tensor(r_update),
            h.Tensor(t),
            h.Tensor(np.zeros(atoms * 3, dtype=np.float32), output=True),
        ],
        groups=h.ceil_div(atoms * 3, 256),
    )
    t_rep = np.repeat(t.astype(np.float64), 3)
    c_skip = sigma ** 2 / (sigma ** 2 + t_rep ** 2)
    c_out = sigma * t_rep / np.sqrt(sigma ** 2 + t_rep ** 2)
    want = c_skip * x_noisy.astype(np.float64) + c_out * r_update.astype(np.float64)
    check("output_combine_fwd", out[3], want)


# ---------------------------------------------------------------------------
# Smooth LDDT  [SI Alg 27]
# ---------------------------------------------------------------------------

_THRESHOLDS = np.array([0.5, 1.0, 2.0, 4.0])


def _lddt_rows(x_pred: np.ndarray, x_gt: np.ndarray, nuc: np.ndarray):
    """rowNum and rowDen from the kernel header comment, in float64."""
    d_gt = np.linalg.norm(x_gt[:, None, :] - x_gt[None, :, :], axis=-1)
    d_pr = np.linalg.norm(x_pred[:, None, :] - x_pred[None, :, :], axis=-1)
    radius = np.where(nuc > 0.5, 30.0, 15.0)
    c = (d_gt < radius[:, None]).astype(np.float64)
    np.fill_diagonal(c, 0.0)
    delta = np.abs(d_gt - d_pr)
    sig = 1.0 / (1.0 + np.exp(-(_THRESHOLDS[None, None, :] - delta[..., None])))
    l_lm = 0.25 * sig.sum(axis=-1)
    return (c * l_lm).sum(axis=1), c.sum(axis=1)


def test_smooth_lddt_fwd() -> None:
    n = 90
    x_pred = RNG.normal(size=(n, 3)).astype(np.float32) * 8.0
    x_gt = RNG.normal(size=(n, 3)).astype(np.float32) * 8.0
    nuc = (RNG.random(n) < 0.3).astype(np.float32)
    payload = struct.pack("<4If3I", n, 0, 0, 0, 0.0, 0, 0, 0)
    out = h.run_entry(
        gpu(),
        spv("slangml/loss/smooth_lddt_f32.slang", "sml_loss_smooth_lddt_fwd_f32"),
        "sml_loss_smooth_lddt_fwd_f32",
        payload,
        [
            h.Tensor(x_pred.reshape(-1)),
            h.Tensor(x_gt.reshape(-1)),
            h.Tensor(nuc),
            h.Tensor(np.zeros(n, dtype=np.float32), output=True),
            h.Tensor(np.zeros(n, dtype=np.float32), output=True),
        ],
        groups=h.ceil_div(n, 64),
    )
    want_num, want_den = _lddt_rows(x_pred.astype(np.float64), x_gt.astype(np.float64), nuc.astype(np.float64))
    check("smooth_lddt_fwd rowNum", out[3], want_num)
    check("smooth_lddt_fwd rowDen", out[4], want_den)


def test_smooth_lddt_bwd() -> None:
    n = 24
    x_pred = RNG.normal(size=(n, 3)) * 6.0
    x_gt = RNG.normal(size=(n, 3)) * 6.0
    nuc = (RNG.random(n) < 0.3).astype(np.float64)
    _, den = _lddt_rows(x_pred, x_gt, nuc)
    total_den = den.sum()
    grad_scale = -1.0 / total_den  # upstream = 1; the GT-side weights do not depend on x_pred

    def loss(xp: np.ndarray) -> float:
        rn, _ = _lddt_rows(xp, x_gt, nuc)
        return 1.0 - rn.sum() / total_den

    eps = 1e-5
    want = np.zeros_like(x_pred)
    for k in range(n):
        for a in range(3):
            hi = x_pred.copy(); hi[k, a] += eps
            lo = x_pred.copy(); lo[k, a] -= eps
            want[k, a] = (loss(hi) - loss(lo)) / (2 * eps)

    payload = struct.pack("<4If3I", n, 0, 0, 0, grad_scale, 0, 0, 0)
    out = h.run_entry(
        gpu(),
        spv("slangml/loss/smooth_lddt_f32.slang", "sml_loss_smooth_lddt_bwd_f32"),
        "sml_loss_smooth_lddt_bwd_f32",
        payload,
        [
            h.Tensor(x_pred.astype(np.float32).reshape(-1)),
            h.Tensor(x_gt.astype(np.float32).reshape(-1)),
            h.Tensor(nuc.astype(np.float32)),
            h.Tensor(np.zeros(n * 3, dtype=np.float32), output=True),
        ],
        groups=h.ceil_div(n, 64),
    )
    check("smooth_lddt_bwd gradX", out[3].reshape(n, 3), want, tol=1e-3)


# ---------------------------------------------------------------------------
# Pair-biased attention  [SI Alg 24] and its backward
# ---------------------------------------------------------------------------

def _pb_logits(q, k, bias, scale):
    """logits[i,h,j] = scale * q_i.k_j + bias[i,j,h]."""
    return scale * np.einsum("ihd,jhd->ihj", q, k) + bias.transpose(0, 2, 1)


def _pb_fwd(q, k, v, bias, gate, scale):
    """out[i,h,:] = sigmoid(gate) * sum_j softmax_j(logits) v_j, in float64."""
    logits = _pb_logits(q, k, bias, scale)
    logits = logits - logits.max(axis=-1, keepdims=True)
    a = np.exp(logits)
    a = a / a.sum(axis=-1, keepdims=True)
    acc = np.einsum("ihj,jhd->ihd", a, v)
    return (1.0 / (1.0 + np.exp(-gate))) * acc


def _pb_loss(q, k, v, bias, gate, scale, d_out):
    return float(np.sum(_pb_fwd(q, k, v, bias, gate, scale) * d_out))


def _pb_inputs(n, hh, d):
    q = RNG.normal(size=(n, hh, d)).astype(np.float32)
    k = RNG.normal(size=(n, hh, d)).astype(np.float32)
    v = RNG.normal(size=(n, hh, d)).astype(np.float32)
    bias = RNG.normal(size=(n, n, hh)).astype(np.float32)
    gate = RNG.normal(size=(n, hh, d)).astype(np.float32)
    return q, k, v, bias, gate


def test_pair_bias_attention_fwd() -> None:
    n, hh, d = 13, 3, 8
    scale = 1.0 / np.sqrt(d)
    q, k, v, bias, gate = _pb_inputs(n, hh, d)
    payload = struct.pack("<3If", n, hh, d, scale)
    out = h.run_entry(
        gpu(),
        spv("slangml/attention/pair_bias_attention_f32.slang", "sml_attention_pair_bias_fwd_f32"),
        "sml_attention_pair_bias_fwd_f32",
        payload,
        [
            h.Tensor(q.reshape(-1)), h.Tensor(k.reshape(-1)), h.Tensor(v.reshape(-1)),
            h.Tensor(bias.reshape(-1)), h.Tensor(gate.reshape(-1)),
            h.Tensor(np.zeros(n * hh * d, dtype=np.float32), output=True),
        ],
        groups=h.ceil_div(n * hh, 64),
    )
    f64 = [x.astype(np.float64) for x in (q, k, v, bias, gate)]
    check("pair_bias_attention_fwd", out[5].reshape(n, hh, d), _pb_fwd(*f64, scale))


def test_pair_bias_attention_bwd() -> None:
    n, hh, d = 6, 2, 4
    scale = 1.0 / np.sqrt(d)
    q, k, v, bias, gate = _pb_inputs(n, hh, d)
    d_out = RNG.normal(size=(n, hh, d)).astype(np.float32)
    base = [x.astype(np.float64) for x in (q, k, v, bias, gate)]
    d_out64 = d_out.astype(np.float64)

    def numeric(index: int) -> np.ndarray:
        """Central-difference gradient of sum(out * dOut) with respect to base[index]."""
        grad = np.zeros(base[index].size)
        eps = 1e-6
        for e in range(grad.size):
            hi = [x.copy() for x in base]
            lo = [x.copy() for x in base]
            hi[index].reshape(-1)[e] += eps
            lo[index].reshape(-1)[e] -= eps
            grad[e] = (_pb_loss(*hi, scale, d_out64) - _pb_loss(*lo, scale, d_out64)) / (2 * eps)
        return grad.reshape(base[index].shape)

    payload = struct.pack("<3If", n, hh, d, scale)
    src = "slangml/attention/pair_bias_attention_bwd_f32.slang"
    flat = lambda a: a.astype(np.float32).reshape(-1)
    zeros = lambda size: np.zeros(size, dtype=np.float32)
    inputs = [flat(q), flat(k), flat(v), flat(bias)]

    stats_entry = "sml_attention_pair_bias_bwd_f32_stats"
    stats = h.run_entry(
        gpu(), spv(src, stats_entry), stats_entry, payload,
        [h.Tensor(x) for x in inputs]
        + [h.Tensor(flat(gate)), h.Tensor(flat(d_out))]
        + [
            h.Tensor(zeros(n * hh), output=True),
            h.Tensor(zeros(n * hh * d), output=True),
            h.Tensor(zeros(n * hh * d), output=True),
            h.Tensor(zeros(n * hh), output=True),
            h.Tensor(zeros(n * hh * d), output=True),
        ],
        groups=h.ceil_div(n * hh, 64),
    )
    lse, d_oacc, delta, d_gate = stats[6], stats[8], stats[9], stats[10]

    dqdb_entry = "sml_attention_pair_bias_bwd_f32_dq_dbias"
    dqdb = h.run_entry(
        gpu(), spv(src, dqdb_entry), dqdb_entry, payload,
        [h.Tensor(x) for x in inputs]
        + [
            h.Tensor(lse), h.Tensor(d_oacc), h.Tensor(delta),
            h.Tensor(zeros(n * hh * d), output=True),
            h.Tensor(zeros(n * n * hh), output=True),
        ],
        groups=h.ceil_div(n * hh, 64),
    )

    dkdv_entry = "sml_attention_pair_bias_bwd_f32_dkdv"
    dkdv = h.run_entry(
        gpu(), spv(src, dkdv_entry), dkdv_entry, payload,
        [h.Tensor(x) for x in inputs]
        + [
            h.Tensor(lse), h.Tensor(d_oacc), h.Tensor(delta),
            h.Tensor(zeros(n * hh * d), output=True),
            h.Tensor(zeros(n * hh * d), output=True),
        ],
        groups=h.ceil_div(n * hh, 64),
    )

    check("pair_bias bwd dGate", d_gate.reshape(n, hh, d), numeric(4), tol=1e-3)
    check("pair_bias bwd dQ", dqdb[7].reshape(n, hh, d), numeric(0), tol=1e-3)
    check("pair_bias bwd dBias", dqdb[8].reshape(n, n, hh), numeric(3), tol=1e-3)
    check("pair_bias bwd dK", dkdv[7].reshape(n, hh, d), numeric(1), tol=1e-3)
    check("pair_bias bwd dV", dkdv[8].reshape(n, hh, d), numeric(2), tol=1e-3)


# ---------------------------------------------------------------------------
# Triangular gated self-attention  [SI Alg 14, Alg 15] and its backward
# ---------------------------------------------------------------------------

def _tri_fwd(q, k, v, bias, gate, scale, ending):
    """Reference forward in float64.  q, k, v, gate: [N,N,H,C]; bias: [N,N,H]."""
    if ending:
        # logit[i,j,k,h] = scale q[i,j].k[k,j] + bias[k,i,h]
        logits = scale * np.einsum("ijhc,kjhc->ijkh", q, k) + bias.transpose(1, 0, 2)[:, None, :, :]
    else:
        # logit[i,j,k,h] = scale q[i,j].k[i,k] + bias[j,k,h]
        logits = scale * np.einsum("ijhc,ikhc->ijkh", q, k) + bias[None, :, :, :]
    logits = logits - logits.max(axis=2, keepdims=True)
    a = np.exp(logits)
    a = a / a.sum(axis=2, keepdims=True)
    if ending:
        acc = np.einsum("ijkh,kjhc->ijhc", a, v)
    else:
        acc = np.einsum("ijkh,ikhc->ijhc", a, v)
    return (1.0 / (1.0 + np.exp(-gate))) * acc


def _tri_loss(q, k, v, bias, gate, scale, ending, d_out):
    return float(np.sum(_tri_fwd(q, k, v, bias, gate, scale, ending) * d_out))


def _tri_bwd(ending: bool) -> None:
    n, hh, c = 5, 2, 4
    scale = 1.0 / np.sqrt(c)
    q = RNG.normal(size=(n, n, hh, c)).astype(np.float32)
    k = RNG.normal(size=(n, n, hh, c)).astype(np.float32)
    v = RNG.normal(size=(n, n, hh, c)).astype(np.float32)
    bias = RNG.normal(size=(n, n, hh)).astype(np.float32)
    gate = RNG.normal(size=(n, n, hh, c)).astype(np.float32)
    d_out = RNG.normal(size=(n, n, hh, c)).astype(np.float32)
    tag = "end" if ending else "start"
    src = "slangml/pairformer/triangle_attention_bwd_f32.slang"
    base = [x.astype(np.float64) for x in (q, k, v, bias, gate)]
    d_out64 = d_out.astype(np.float64)

    # Forward check on the GPU forward kernel, so the reference is known to match first.
    fwd_entry = f"sml_pairformer_triangle_attention_{tag}_fwd_f32"
    fwd_payload = struct.pack("<3If", n, hh, c, scale)
    flat = lambda a: a.astype(np.float32).reshape(-1)
    zeros = lambda size: np.zeros(size, dtype=np.float32)
    fwd = h.run_entry(
        gpu(),
        spv("slangml/pairformer/triangle_attention_f32.slang", fwd_entry),
        fwd_entry,
        fwd_payload,
        [h.Tensor(flat(q)), h.Tensor(flat(k)), h.Tensor(flat(v)), h.Tensor(flat(bias)),
         h.Tensor(flat(gate)), h.Tensor(zeros(n * n * hh * c), output=True)],
        groups=h.ceil_div(n * n * hh, 128),
    )
    check(f"triangle_attention_{tag}_fwd", fwd[5].reshape(n, n, hh, c), _tri_fwd(*base, scale, ending))

    def numeric(index: int) -> np.ndarray:
        grad = np.zeros(base[index].size)
        eps = 1e-6
        for e in range(grad.size):
            hi = [x.copy() for x in base]
            lo = [x.copy() for x in base]
            hi[index].reshape(-1)[e] += eps
            lo[index].reshape(-1)[e] -= eps
            grad[e] = (_tri_loss(*hi, scale, ending, d_out64) - _tri_loss(*lo, scale, ending, d_out64)) / (2 * eps)
        return grad.reshape(base[index].shape)

    payload = struct.pack("<3If", n, hh, c, scale)
    inputs = [flat(q), flat(k), flat(v), flat(bias)]
    e_stats = f"sml_pairformer_triangle_attention_{tag}_bwd_f32_stats"
    stats = h.run_entry(
        gpu(), spv(src, e_stats), e_stats, payload,
        [h.Tensor(x) for x in inputs]
        + [h.Tensor(flat(gate)), h.Tensor(flat(d_out)),
           h.Tensor(zeros(n * n * hh), output=True),
           h.Tensor(zeros(n * n * hh * c), output=True),
           h.Tensor(zeros(n * n * hh * c), output=True),
           h.Tensor(zeros(n * n * hh), output=True),
           h.Tensor(zeros(n * n * hh * c), output=True)],
        groups=h.ceil_div(n * n * hh, 128),
    )
    lse, d_oacc, delta, d_gate = stats[6], stats[8], stats[9], stats[10]

    e_dq = f"sml_pairformer_triangle_attention_{tag}_bwd_f32_dq"
    dq = h.run_entry(
        gpu(), spv(src, e_dq), e_dq, payload,
        [h.Tensor(x) for x in inputs]
        + [h.Tensor(lse), h.Tensor(d_oacc), h.Tensor(delta),
           h.Tensor(zeros(n * n * hh * c), output=True)],
        groups=h.ceil_div(n * n * hh, 128),
    )
    e_dbias = f"sml_pairformer_triangle_attention_{tag}_bwd_f32_dbias"
    dbias = h.run_entry(
        gpu(), spv(src, e_dbias), e_dbias, payload,
        [h.Tensor(x) for x in inputs]
        + [h.Tensor(lse), h.Tensor(d_oacc), h.Tensor(delta),
           h.Tensor(zeros(n * n * hh), output=True)],
        groups=h.ceil_div(n * n * hh, 128),
    )
    e_dkdv = f"sml_pairformer_triangle_attention_{tag}_bwd_f32_dkdv"
    dkdv = h.run_entry(
        gpu(), spv(src, e_dkdv), e_dkdv, payload,
        [h.Tensor(x) for x in inputs]
        + [h.Tensor(lse), h.Tensor(d_oacc), h.Tensor(delta),
           h.Tensor(zeros(n * n * hh * c), output=True),
           h.Tensor(zeros(n * n * hh * c), output=True)],
        groups=h.ceil_div(n * n * hh, 128),
    )

    check(f"triangle_{tag} bwd dGate", d_gate.reshape(n, n, hh, c), numeric(4), tol=1e-3)
    check(f"triangle_{tag} bwd dQ", dq[7].reshape(n, n, hh, c), numeric(0), tol=1e-3)
    check(f"triangle_{tag} bwd dBias", dbias[7].reshape(n, n, hh), numeric(3), tol=1e-3)
    check(f"triangle_{tag} bwd dK", dkdv[7].reshape(n, n, hh, c), numeric(1), tol=1e-3)
    check(f"triangle_{tag} bwd dV", dkdv[8].reshape(n, n, hh, c), numeric(2), tol=1e-3)


def test_triangle_attention_bwd_start() -> None:
    _tri_bwd(ending=False)


def test_triangle_attention_bwd_end() -> None:
    _tri_bwd(ending=True)


# ---------------------------------------------------------------------------
# Helper: central-difference gradient of fn(list_of_arrays) with respect to one input.
# ---------------------------------------------------------------------------

def _numeric_grad(fn, arrays, index, eps=1e-6):
    base = [x.astype(np.float64) for x in arrays]
    flat = base[index].reshape(-1)
    grad = np.zeros(flat.size)
    for e in range(flat.size):
        orig = flat[e]
        flat[e] = orig + eps
        hi = fn(base)
        flat[e] = orig - eps
        lo = fn(base)
        flat[e] = orig
        grad[e] = (hi - lo) / (2 * eps)
    return grad.reshape(base[index].shape)


# ---------------------------------------------------------------------------
# Sequence-local atom attention  [SI Alg 5, 6, 7] and its backward
# ---------------------------------------------------------------------------

def _alb_fwd(q, k, v, bias, gate, scale, n):
    """Reference forward in float64.  q, k, v, gate: [N,H,D]; bias: [N,128,H]."""
    t = np.arange(128)
    key = (32 * (np.arange(n) // 32))[:, None] - 48 + t[None, :]
    valid = (key >= 0) & (key < n)
    kc = np.clip(key, 0, n - 1)
    kk = k[kc]  # [N,128,H,D]
    vv = v[kc]
    logits = scale * np.einsum("lhd,lthd->lth", q, kk) + bias
    logits = np.where(valid[:, :, None], logits, -np.inf)
    m = logits.max(axis=1, keepdims=True)
    a = np.exp(logits - m)
    a = a / a.sum(axis=1, keepdims=True)
    acc = np.einsum("lth,lthd->lhd", a, vv)
    return (1.0 / (1.0 + np.exp(-gate))) * acc


def test_sequence_local_attention_bwd() -> None:
    n, hh, d = 40, 2, 4
    scale = 1.0 / np.sqrt(d)
    q = RNG.normal(size=(n, hh, d)).astype(np.float32)
    k = RNG.normal(size=(n, hh, d)).astype(np.float32)
    v = RNG.normal(size=(n, hh, d)).astype(np.float32)
    bias = RNG.normal(size=(n, 128, hh)).astype(np.float32)
    gate = RNG.normal(size=(n, hh, d)).astype(np.float32)
    d_out = RNG.normal(size=(n, hh, d)).astype(np.float32)
    flat = lambda x: x.astype(np.float32).reshape(-1)
    zeros = lambda size: np.zeros(size, dtype=np.float32)
    src = "slangml/atom/sequence_local_attention_bwd_f32.slang"
    payload = struct.pack("<3If", n, hh, d, scale)
    arrays = [q, k, v, bias, gate]

    fwd = h.run_entry(
        gpu(),
        spv("slangml/atom/sequence_local_attention_f32.slang", "sml_atom_sequence_local_attention_fwd_f32"),
        "sml_atom_sequence_local_attention_fwd_f32",
        payload,
        [h.Tensor(flat(x)) for x in arrays] + [h.Tensor(zeros(n * hh * d), output=True)],
        groups=h.ceil_div(n * hh, 64),
    )
    check("sequence_local_attention_fwd", fwd[5].reshape(n, hh, d),
          _alb_fwd(*[x.astype(np.float64) for x in arrays], scale, n))

    def loss(xs):
        return float(np.sum(_alb_fwd(*xs, scale, n) * d_out.astype(np.float64)))

    base_inputs = [flat(x) for x in arrays]
    e_stats = "sml_atom_sequence_local_attention_bwd_f32_stats"
    stats = h.run_entry(
        gpu(), spv(src, e_stats), e_stats, payload,
        [h.Tensor(x) for x in base_inputs]
        + [h.Tensor(flat(d_out)),
           h.Tensor(zeros(n * hh), output=True),
           h.Tensor(zeros(n * hh * d), output=True),
           h.Tensor(zeros(n * hh * d), output=True),
           h.Tensor(zeros(n * hh), output=True),
           h.Tensor(zeros(n * hh * d), output=True)],
        groups=h.ceil_div(n * hh, 64),
    )
    lse, d_oacc, delta, d_gate = stats[6], stats[8], stats[9], stats[10]

    e_dq = "sml_atom_sequence_local_attention_bwd_f32_dq"
    dq = h.run_entry(
        gpu(), spv(src, e_dq), e_dq, payload,
        [h.Tensor(x) for x in base_inputs[:4]]
        + [h.Tensor(lse), h.Tensor(d_oacc), h.Tensor(delta), h.Tensor(zeros(n * hh * d), output=True)],
        groups=h.ceil_div(n * hh, 64),
    )
    e_dbias = "sml_atom_sequence_local_attention_bwd_f32_dbias"
    dbias = h.run_entry(
        gpu(), spv(src, e_dbias), e_dbias, payload,
        [h.Tensor(x) for x in base_inputs[:4]]
        + [h.Tensor(lse), h.Tensor(d_oacc), h.Tensor(delta), h.Tensor(zeros(n * 128 * hh), output=True)],
        groups=h.ceil_div(n * 128 * hh, 128),
    )
    e_dkdv = "sml_atom_sequence_local_attention_bwd_f32_dkdv"
    dkdv = h.run_entry(
        gpu(), spv(src, e_dkdv), e_dkdv, payload,
        [h.Tensor(x) for x in base_inputs[:4]]
        + [h.Tensor(lse), h.Tensor(d_oacc), h.Tensor(delta),
           h.Tensor(zeros(n * hh * d), output=True),
           h.Tensor(zeros(n * hh * d), output=True)],
        groups=h.ceil_div(n * hh, 64),
    )

    check("sequence_local bwd dGate", d_gate.reshape(n, hh, d), _numeric_grad(loss, arrays, 4), tol=1e-3)
    check("sequence_local bwd dQ", dq[7].reshape(n, hh, d), _numeric_grad(loss, arrays, 0), tol=1e-3)
    check("sequence_local bwd dBias", dbias[7].reshape(n, 128, hh), _numeric_grad(loss, arrays, 3), tol=1e-3)
    check("sequence_local bwd dK", dkdv[7].reshape(n, hh, d), _numeric_grad(loss, arrays, 1), tol=1e-3)
    check("sequence_local bwd dV", dkdv[8].reshape(n, hh, d), _numeric_grad(loss, arrays, 2), tol=1e-3)


# ---------------------------------------------------------------------------
# Outer product mean  [SI Alg 9] and its backward
# ---------------------------------------------------------------------------

def _opm_fwd(a, b, w, bias, z_in, s):
    """zOut = zIn + bias + W o, o_ij = (1/S) sum_s a_si (x) b_sj.  Float64."""
    o = np.einsum("sic,sjd->ijcd", a, b).reshape(a.shape[1], a.shape[1], 1024) / s
    return z_in + bias[None, None, :] + np.einsum("ijp,zp->ijz", o, w)


def test_outer_product_mean_bwd() -> None:
    s_count, n = 3, 6
    a = RNG.normal(size=(s_count, n, 32)).astype(np.float32)
    b = RNG.normal(size=(s_count, n, 32)).astype(np.float32)
    w = (RNG.normal(size=(128, 1024)) / 32.0).astype(np.float32)
    bias = RNG.normal(size=128).astype(np.float32)
    z_in = RNG.normal(size=(n, n, 128)).astype(np.float32)
    d_z = RNG.normal(size=(n, n, 128)).astype(np.float32)
    flat = lambda x: x.astype(np.float32).reshape(-1)
    zeros = lambda size: np.zeros(size, dtype=np.float32)
    src = "slangml/msa/outer_product_mean_bwd_f32.slang"
    payload = struct.pack("<4I", s_count, n, 0, 0)

    fwd = h.run_entry(
        gpu(),
        spv("slangml/msa/outer_product_mean_f32.slang", "sml_msa_outer_product_mean_fwd_f32"),
        "sml_msa_outer_product_mean_fwd_f32",
        payload,
        [h.Tensor(flat(a)), h.Tensor(flat(b)), h.Tensor(flat(w)), h.Tensor(flat(bias)),
         h.Tensor(flat(z_in)), h.Tensor(zeros(n * n * 128), output=True)],
        groups=n * n,
    )
    f64 = [x.astype(np.float64) for x in (a, b, w, bias, z_in)]
    z_ref = _opm_fwd(*f64, s_count)
    check("outer_product_mean_fwd", fwd[5].reshape(n, n, 128), z_ref)

    # Analytic reference for the gradients, in float64.
    o_ref = (np.einsum("sic,sjd->ijcd", f64[0], f64[1]).reshape(n, n, 1024) / s_count)
    do_ref = np.einsum("ijz,zc->ijc", d_z.astype(np.float64), w.astype(np.float64))
    dw_ref = np.einsum("ijz,ijc->zc", d_z.astype(np.float64), o_ref)

    def loss(xs):
        return float(np.sum(_opm_fwd(xs[0], xs[1], f64[2], xs[3], f64[4], s_count) * d_z.astype(np.float64)))

    arrays = [a, b, bias]
    loss_ab = lambda xs: float(np.sum(_opm_fwd(xs[0], xs[1], f64[2], xs[2], f64[4], s_count) * d_z.astype(np.float64)))
    e_pairs = "sml_msa_outer_product_mean_bwd_f32_pairs"
    pairs = h.run_entry(
        gpu(), spv(src, e_pairs), e_pairs, payload,
        [h.Tensor(flat(a)), h.Tensor(flat(b)), h.Tensor(flat(w)), h.Tensor(flat(d_z)),
         h.Tensor(zeros(n * n * 1024), output=True),
         h.Tensor(zeros(n * n * 1024), output=True)],
        groups=n * n,
    )
    o_ws, do_ws = pairs[4], pairs[5]
    check("outer_product_mean bwd oWs", o_ws.reshape(n, n, 1024), o_ref)
    check("outer_product_mean bwd dO", do_ws.reshape(n, n, 1024), do_ref)

    e_dbias = "sml_msa_outer_product_mean_bwd_f32_dbias"
    dbias = h.run_entry(
        gpu(), spv(src, e_dbias), e_dbias, payload,
        [h.Tensor(flat(d_z)), h.Tensor(zeros(128), output=True)],
        groups=1,
    )
    check("outer_product_mean bwd dbias", dbias[1], d_z.astype(np.float64).sum(axis=(0, 1)), tol=1e-3)

    e_dw = "sml_msa_outer_product_mean_bwd_f32_dw"
    dw = h.run_entry(
        gpu(), spv(src, e_dw), e_dw, payload,
        [h.Tensor(flat(d_z)), h.Tensor(o_ws), h.Tensor(zeros(128 * 1024), output=True)],
        groups=h.ceil_div(128 * 1024, 256),
    )
    check("outer_product_mean bwd dW", dw[2].reshape(128, 1024), dw_ref, tol=1e-3)

    e_da = "sml_msa_outer_product_mean_bwd_f32_da"
    da = h.run_entry(
        gpu(), spv(src, e_da), e_da, payload,
        [h.Tensor(flat(b)), h.Tensor(do_ws), h.Tensor(zeros(s_count * n * 32), output=True)],
        groups=h.ceil_div(s_count * n * 32, 64),
    )
    e_db = "sml_msa_outer_product_mean_bwd_f32_db"
    db = h.run_entry(
        gpu(), spv(src, e_db), e_db, payload,
        [h.Tensor(flat(a)), h.Tensor(do_ws), h.Tensor(zeros(s_count * n * 32), output=True)],
        groups=h.ceil_div(s_count * n * 32, 64),
    )
    check("outer_product_mean bwd da", da[2].reshape(s_count, n, 32), _numeric_grad(loss_ab, [a, b, bias], 0), tol=1e-3)
    check("outer_product_mean bwd db", db[2].reshape(s_count, n, 32), _numeric_grad(loss_ab, [a, b, bias], 1), tol=1e-3)


# ---------------------------------------------------------------------------
# MSA pair-weighted averaging  [SI Alg 10] and its backward
# ---------------------------------------------------------------------------

def _msa_fwd(v, gate, bias, s_count, n, hh, c):
    """out[s,i,h,c] = sigmoid(gate) * sum_j softmax_j(bias[i,j,h]) v[s,j,h,c].  Float64."""
    v4 = v.reshape(s_count, n, hh, c)
    g4 = gate.reshape(s_count, n, hh, c)
    m = bias.max(axis=1, keepdims=True)
    a = np.exp(bias - m)
    a = a / a.sum(axis=1, keepdims=True)  # [i, j, h]
    acc = np.einsum("ijh,sjhc->sihc", a, v4)
    return ((1.0 / (1.0 + np.exp(-g4))) * acc).reshape(s_count, n, hh * c)


def test_msa_pair_weighted_avg_bwd() -> None:
    s_count, n, hh, c = 3, 5, 2, 4
    v = RNG.normal(size=(s_count, n, hh * c)).astype(np.float32)
    gate = RNG.normal(size=(s_count, n, hh * c)).astype(np.float32)
    bias = RNG.normal(size=(n, n, hh)).astype(np.float32)
    d_out = RNG.normal(size=(s_count, n, hh * c)).astype(np.float32)
    flat = lambda x: x.astype(np.float32).reshape(-1)
    zeros = lambda size: np.zeros(size, dtype=np.float32)
    src = "slangml/msa/msa_pair_weighted_avg_bwd_f32.slang"
    payload = struct.pack("<4I", s_count, n, hh, c)
    arrays = [v, gate, bias]

    fwd = h.run_entry(
        gpu(),
        spv("slangml/msa/msa_pair_weighted_avg_f32.slang", "sml_msa_pair_weighted_avg_fwd_f32"),
        "sml_msa_pair_weighted_avg_fwd_f32",
        struct.pack("<4I", s_count, n, hh, c),
        [h.Tensor(flat(v)), h.Tensor(flat(gate)), h.Tensor(flat(bias)),
         h.Tensor(zeros(s_count * n * hh * c), output=True)],
        groups=h.ceil_div(s_count * n * hh, 64),
    )
    check("msa_pair_weighted_avg_fwd", fwd[3].reshape(s_count, n, hh * c),
          _msa_fwd(*[x.astype(np.float64) for x in arrays], s_count, n, hh, c))

    def loss(xs):
        return float(np.sum(_msa_fwd(*xs, s_count, n, hh, c) * d_out.astype(np.float64)))

    base = [flat(x) for x in arrays]
    e_lse = "sml_msa_pair_weighted_avg_bwd_f32_lse"
    lse = h.run_entry(
        gpu(), spv(src, e_lse), e_lse, payload,
        [h.Tensor(flat(bias)), h.Tensor(zeros(n * hh), output=True)],
        groups=h.ceil_div(n * hh, 64),
    )[1]
    e_rows = "sml_msa_pair_weighted_avg_bwd_f32_rows"
    rows = h.run_entry(
        gpu(), spv(src, e_rows), e_rows, payload,
        [h.Tensor(base[2]), h.Tensor(base[0]), h.Tensor(base[1]), h.Tensor(lse),
         h.Tensor(flat(d_out)),
         h.Tensor(zeros(s_count * n * hh * c), output=True),
         h.Tensor(zeros(s_count * n * hh * c), output=True),
         h.Tensor(zeros(s_count * n * hh), output=True),
         h.Tensor(zeros(s_count * n * hh * c), output=True)],
        groups=h.ceil_div(s_count * n * hh, 64),
    )
    d_oacc, delta, d_gate = rows[6], rows[7], rows[8]
    e_dv = "sml_msa_pair_weighted_avg_bwd_f32_dv"
    dv = h.run_entry(
        gpu(), spv(src, e_dv), e_dv, payload,
        [h.Tensor(base[2]), h.Tensor(lse), h.Tensor(d_oacc),
         h.Tensor(zeros(s_count * n * hh * c), output=True)],
        groups=h.ceil_div(s_count * n * hh, 64),
    )
    e_dbias = "sml_msa_pair_weighted_avg_bwd_f32_dbias"
    dbias = h.run_entry(
        gpu(), spv(src, e_dbias), e_dbias, payload,
        [h.Tensor(base[2]), h.Tensor(lse), h.Tensor(base[0]), h.Tensor(d_oacc), h.Tensor(delta),
         h.Tensor(zeros(n * n * hh), output=True)],
        groups=h.ceil_div(n * n * hh, 64),
    )

    check("msa_pair_avg bwd dGate", d_gate.reshape(s_count, n, hh * c), _numeric_grad(loss, arrays, 1), tol=1e-3)
    check("msa_pair_avg bwd dV", dv[3].reshape(s_count, n, hh * c), _numeric_grad(loss, arrays, 0), tol=1e-3)
    check("msa_pair_avg bwd dBias", dbias[5].reshape(n, n, hh), _numeric_grad(loss, arrays, 2), tol=1e-3)


# ---------------------------------------------------------------------------
# AdaLN and adaLN-Zero  [SI Alg 25, Alg 26]
# ---------------------------------------------------------------------------

_LN_EPS = 1e-5


def _adaln_ref(a, sp, sh):
    mean = a.mean(axis=1, keepdims=True)
    var = ((a - mean) ** 2).mean(axis=1, keepdims=True)
    ahat = (a - mean) / np.sqrt(var + _LN_EPS)
    return (1.0 / (1.0 + np.exp(-sp))) * ahat + sh


def test_adaln_fwd_bwd() -> None:
    rows, cols = 4, 300  # cols > 256 exercises the strided lane loop
    a = (RNG.normal(size=(rows, cols)) * 3.0 + 1.0).astype(np.float32)
    sp = RNG.normal(size=(rows, cols)).astype(np.float32)
    sh = RNG.normal(size=(rows, cols)).astype(np.float32)
    dy = RNG.normal(size=(rows, cols)).astype(np.float32)
    zeros = lambda size: np.zeros(size, dtype=np.float32)
    src = "slangml/diffusion/adaln_f32.slang"
    payload = struct.pack("<IIff", rows, cols, _LN_EPS, 0.0)

    fwd = h.run_entry(
        gpu(), spv(src, "sml_diffusion_adaln_fwd_f32"), "sml_diffusion_adaln_fwd_f32", payload,
        [h.Tensor(a.reshape(-1)), h.Tensor(sp.reshape(-1)), h.Tensor(sh.reshape(-1)),
         h.Tensor(zeros(rows * cols), output=True)],
        groups=rows,
    )
    f64 = [v.astype(np.float64) for v in (a, sp, sh)]
    check("adaln_fwd", fwd[3].reshape(rows, cols), _adaln_ref(*f64))

    def loss(xs):
        return float(np.sum(_adaln_ref(xs[0], xs[1], f64[2]) * dy.astype(np.float64)))

    bwd = h.run_entry(
        gpu(), spv(src, "sml_diffusion_adaln_bwd_f32"), "sml_diffusion_adaln_bwd_f32", payload,
        [h.Tensor(a.reshape(-1)), h.Tensor(sp.reshape(-1)), h.Tensor(dy.reshape(-1)),
         h.Tensor(zeros(rows * cols), output=True), h.Tensor(zeros(rows * cols), output=True)],
        groups=rows,
    )
    check("adaln bwd da", bwd[3].reshape(rows, cols), _numeric_grad(loss, [a, sp], 0), tol=1e-3)
    check("adaln bwd dScalePre", bwd[4].reshape(rows, cols), _numeric_grad(loss, [a, sp], 1), tol=1e-3)


def test_adaln_zero_residual_fwd_bwd() -> None:
    numel = 1000
    x = RNG.normal(size=numel).astype(np.float32)
    gp = RNG.normal(size=numel).astype(np.float32)
    resid = RNG.normal(size=numel).astype(np.float32)
    dy = RNG.normal(size=numel).astype(np.float32)
    zeros = lambda size: np.zeros(size, dtype=np.float32)
    src = "slangml/diffusion/adaln_f32.slang"
    payload = struct.pack("<I3I", numel, 0, 0, 0)
    groups = h.ceil_div(numel, 256)

    fwd = h.run_entry(
        gpu(), spv(src, "sml_diffusion_adaln_zero_residual_fwd_f32"), "sml_diffusion_adaln_zero_residual_fwd_f32",
        payload, [h.Tensor(x), h.Tensor(gp), h.Tensor(resid), h.Tensor(zeros(numel), output=True)], groups=groups,
    )
    f64 = [v.astype(np.float64) for v in (x, gp, resid)]
    check("adaln_zero_residual_fwd", fwd[3], f64[2] + (1.0 / (1.0 + np.exp(-f64[1]))) * f64[0])

    def loss(xs):
        return float(np.sum((xs[2] + (1.0 / (1.0 + np.exp(-xs[1]))) * xs[0]) * dy.astype(np.float64)))

    bwd = h.run_entry(
        gpu(), spv(src, "sml_diffusion_adaln_zero_residual_bwd_f32"), "sml_diffusion_adaln_zero_residual_bwd_f32",
        payload, [h.Tensor(x), h.Tensor(gp), h.Tensor(dy), h.Tensor(zeros(numel), output=True),
                  h.Tensor(zeros(numel), output=True)], groups=groups,
    )
    arrays = [x, gp, resid]
    check("adaln_zero_residual bwd dx", bwd[3], _numeric_grad(loss, arrays, 0), tol=1e-3)
    check("adaln_zero_residual bwd dGatePre", bwd[4], _numeric_grad(loss, arrays, 1), tol=1e-3)


# ---------------------------------------------------------------------------
# Diffusion sampler  [SI Alg 18, Alg 19].  Philox draws are checked statistically
# or with noiseScale = 0, since NumPy cannot reproduce the Philox stream.
# ---------------------------------------------------------------------------

def test_sampler_init() -> None:
    elements = 4096
    c0 = 2.0
    src = "slangml/diffusion/sampler_f32.slang"
    entry = "sml_diffusion_sampler_init_fwd_f32"
    payload = struct.pack("<4I8f", elements, 0, 5, 0, c0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    out = h.run_entry(
        gpu(), spv(src, entry), entry, payload,
        [h.Tensor(np.zeros(elements, dtype=np.float32), output=True)],
        groups=h.ceil_div(elements, 256),
    )[0].astype(np.float64)
    record("sampler_init mean ~ 0", abs(out.mean()) < 0.15, f"mean {out.mean():.3f}")
    record("sampler_init std ~ c0", abs(out.std() - c0) < 0.15 * c0, f"std {out.std():.3f}, expect {c0}")


def test_sampler_step() -> None:
    elements = 777
    x = RNG.normal(size=elements).astype(np.float32)
    x_den = RNG.normal(size=elements).astype(np.float32)
    src = "slangml/diffusion/sampler_f32.slang"
    entry = "sml_diffusion_sampler_step_fwd_f32"
    zeros = lambda size: np.zeros(size, dtype=np.float32)
    for c_now in (1.5, 0.2):  # above and below gammaMin, so both gamma branches run
        c_prev, gamma0, gamma_min, step_scale = 2.0, 0.8, 0.5, 1.5
        payload = struct.pack("<4I8f", elements, 3, 1, 2, c_prev, c_now, gamma0, gamma_min, 0.0, step_scale, 0.0, 0.0)
        out = h.run_entry(
            gpu(), spv(src, entry), entry, payload,
            [h.Tensor(x), h.Tensor(x_den), h.Tensor(zeros(elements), output=True)],
            groups=h.ceil_div(elements, 256),
        )
        gamma = gamma0 if c_now > gamma_min else 0.0
        t_hat = c_prev * (gamma + 1.0)
        x64, den64 = x.astype(np.float64), x_den.astype(np.float64)
        want = x64 + step_scale * (c_now - t_hat) * (x64 - den64) / t_hat
        check(f"sampler_step c={c_now}", out[2], want)


def test_centre_augment() -> None:
    structures, atoms = 3, 50
    x = (RNG.normal(size=(structures, atoms, 3)) * 5.0 + 20.0).astype(np.float32)
    src = "slangml/diffusion/sampler_f32.slang"
    entry = "sml_diffusion_centre_augment_fwd_f32"
    payload = struct.pack("<4I4f", structures, atoms, 7, 1, 0.0, 0.0, 0.0, 0.0)
    out = h.run_entry(
        gpu(), spv(src, entry), entry, payload,
        [h.Tensor(x.reshape(-1)), h.Tensor(np.zeros(x.size, dtype=np.float32), output=True)],
        groups=structures,
    )[1].reshape(structures, atoms, 3).astype(np.float64)
    centred = x.astype(np.float64) - x.astype(np.float64).mean(axis=1, keepdims=True)
    centroid = float(np.abs(out.mean(axis=1)).max())
    record("centre_augment centroid", centroid < 1e-4, f"max |centroid| {centroid:.2e}")

    def pair_dist(p):
        return np.linalg.norm(p[:, :, None, :] - p[:, None, :, :], axis=-1)

    check("centre_augment pairwise distances", pair_dist(out), pair_dist(centred), tol=1e-4)


# ---------------------------------------------------------------------------
# Shared-axis dropout with residual  [SI Alg 8, 16, 17]
# ---------------------------------------------------------------------------

def test_dropout_residual() -> None:
    A, B, C = 6, 40, 40
    total = A * B * C
    p_drop = 0.25
    keep_scale = 1.0 / (1.0 - p_drop)
    y = RNG.uniform(1.0, 2.0, size=(A, B, C)).astype(np.float32)
    resid = RNG.normal(size=(A, B, C)).astype(np.float32)
    ones = np.ones(total, dtype=np.float32)
    src = "slangml/pairformer/dropout_residual_f32.slang"
    zeros = lambda size: np.zeros(size, dtype=np.float32)
    for shared in (0, 1):
        payload = struct.pack("<4IfIII", A, B, C, shared, p_drop, 11, 3, 0)
        out = h.run_entry(
            gpu(), spv(src, "sml_pairformer_dropout_residual_fwd_f32"), "sml_pairformer_dropout_residual_fwd_f32",
            payload, [h.Tensor(y.reshape(-1)), h.Tensor(resid.reshape(-1)), h.Tensor(zeros(total), output=True)],
            groups=h.ceil_div(total, 256),
        )[2].reshape(A, B, C).astype(np.float64)
        scale = (out - resid) / y
        near = np.minimum(np.abs(scale), np.abs(scale - keep_scale))
        record(f"dropout scale is 0 or 1/(1-p) shared={shared}", float(near.max()) < 1e-4,
               f"max distance {near.max():.2e}")
        # The mask must be constant along the shared axis.
        spread = np.abs(scale - scale[:1]).max() if shared == 0 else np.abs(scale - scale[:, :1]).max()
        record(f"dropout mask shared along axis {shared}", float(spread) < 1e-4, f"max spread {spread:.2e}")
        keep = float(np.mean(scale > 0.5))
        record(f"dropout keep fraction shared={shared}", 0.65 < keep < 0.85, f"keep {keep:.3f}, expect 0.75")
        dy = h.run_entry(
            gpu(), spv(src, "sml_pairformer_dropout_bwd_f32"), "sml_pairformer_dropout_bwd_f32", payload,
            [h.Tensor(ones), h.Tensor(zeros(total), output=True)],
            groups=h.ceil_div(total, 256),
        )[1].reshape(A, B, C).astype(np.float64)
        check(f"dropout bwd matches forward mask shared={shared}", dy, scale, tol=1e-4)


# ---------------------------------------------------------------------------
# Relative position encoding  [SI Alg 3] and distance-bin embedding  [SI Alg 4]
# ---------------------------------------------------------------------------

def _relpos_ref(res, tok, asym, sym, ent, w_rel, bonds, w_bond):
    n = res.shape[0]
    out = np.zeros((n, n, w_rel.shape[1]))
    for i in range(n):
        for j in range(n):
            same_chain = asym[i] == asym[j]
            same_res = res[i] == res[j]
            same_ent = ent[i] == ent[j]
            rel_pos = int(np.clip(res[i] - res[j] + 32, 0, 64)) if same_chain else 65
            rel_tok = int(np.clip(tok[i] - tok[j] + 32, 0, 64)) if (same_chain and same_res) else 65
            rel_chain = 5 if same_chain else int(np.clip(sym[i] - sym[j] + 2, 0, 4))
            v = w_rel[rel_pos] + w_rel[66 + rel_tok] + w_rel[133 + rel_chain]
            if same_ent:
                v = v + w_rel[132]
            out[i, j] = v + bonds[i, j] * w_bond
    return out


def test_relative_position_encoding() -> None:
    n, cz = 12, 8
    res = RNG.integers(0, 4, n).astype(np.int32)
    tok = RNG.integers(0, 6, n).astype(np.int32)
    asym = RNG.integers(0, 2, n).astype(np.int32)
    sym = RNG.integers(0, 3, n).astype(np.int32)
    ent = RNG.integers(0, 2, n).astype(np.int32)
    w_rel = RNG.normal(size=(139, cz)).astype(np.float32)
    bonds = (RNG.random((n, n)) < 0.2).astype(np.float32)
    w_bond = RNG.normal(size=cz).astype(np.float32)
    src = "slangml/features/pair_embedding_f32.slang"
    entry = "sml_features_relative_position_encoding_fwd_f32"
    total = n * n * cz
    out = h.run_entry(
        gpu(), spv(src, entry), entry, struct.pack("<4I", n, cz, 1, 0),
        [h.Tensor(res), h.Tensor(tok), h.Tensor(asym), h.Tensor(sym), h.Tensor(ent),
         h.Tensor(w_rel.reshape(-1)), h.Tensor(bonds.reshape(-1)), h.Tensor(w_bond),
         h.Tensor(np.zeros(total, dtype=np.float32), output=True)],
        groups=h.ceil_div(total, 256),
    )[8].reshape(n, n, cz)
    want = _relpos_ref(res, tok, asym, sym, ent, w_rel.astype(np.float64), bonds.astype(np.float64),
                       w_bond.astype(np.float64))
    check("relative_position_encoding", out, want)


def test_distance_bin_embed() -> None:
    n, cz, bins = 10, 6, 12
    d = (RNG.random((n, n)) * 30.0).astype(np.float32)
    v_bins = np.linspace(0.0, 30.0, bins).astype(np.float32)
    w = RNG.normal(size=(bins, cz)).astype(np.float32)
    src = "slangml/features/pair_embedding_f32.slang"
    entry = "sml_features_distance_bin_embed_fwd_f32"
    total = n * n * cz
    out = h.run_entry(
        gpu(), spv(src, entry), entry, struct.pack("<4I", n, cz, bins, 0),
        [h.Tensor(d.reshape(-1)), h.Tensor(v_bins), h.Tensor(w.reshape(-1)),
         h.Tensor(np.zeros(total, dtype=np.float32), output=True)],
        groups=h.ceil_div(total, 256),
    )[3].reshape(n, n, cz)
    idx = np.argmin(np.abs(d.astype(np.float64)[:, :, None] - v_bins.astype(np.float64)[None, None, :]), axis=-1)
    check("distance_bin_embed", out, w.astype(np.float64)[idx])


# ---------------------------------------------------------------------------
# pTM / ipTM row score and max  [SI Eq. 17, 18]
# ---------------------------------------------------------------------------

def _ptm_row_ref(pae, centers, in_d, chain, valid, d0, interface_only):
    n = pae.shape[0]
    scores = np.zeros(n)
    for i in range(n):
        if valid[i] < 0.5:
            scores[i] = -1.0
            continue
        total, count = 0.0, 0.0
        for j in range(n):
            if in_d[j] < 0.5:
                continue
            if interface_only and chain[j] == chain[i]:
                continue
            weight = pae[i, j] / (1.0 + (centers / d0) ** 2)
            total += weight.sum()
            count += 1.0
        scores[i] = total / count if count > 0 else 0.0
    return scores


def test_ptm_row_and_max() -> None:
    n, nb = 15, 64
    d0 = 1.24 * np.cbrt(max(n, 19) - 15) - 1.8
    centers = np.linspace(0.0, 31.5, nb).astype(np.float32)
    pae = RNG.random((n, n, nb)).astype(np.float32)
    pae = pae / pae.sum(axis=-1, keepdims=True)
    in_d = (RNG.random(n) < 0.8).astype(np.float32)
    in_d[0] = 1.0
    chain = RNG.integers(0, 3, n).astype(np.uint32)
    valid = (RNG.random(n) < 0.85).astype(np.float32)
    valid[1] = 1.0
    src = "slangml/confidence/ptm_f32.slang"
    for interface_only in (0, 1):
        payload = struct.pack("<4I4f", n, nb, interface_only, 0, d0, 0.0, 0.0, 0.0)
        row = h.run_entry(
            gpu(), spv(src, "sml_confidence_ptm_row_fwd_f32"), "sml_confidence_ptm_row_fwd_f32", payload,
            [h.Tensor(pae.reshape(-1)), h.Tensor(centers), h.Tensor(in_d), h.Tensor(chain),
             h.Tensor(valid), h.Tensor(np.zeros(n, dtype=np.float32), output=True)],
            groups=h.ceil_div(n, 64),
        )[5]
        want = _ptm_row_ref(pae.astype(np.float64), centers.astype(np.float64), in_d, chain, valid, d0,
                            interface_only)
        check(f"ptm_row interface_only={interface_only}", row, want)
        best = h.run_entry(
            gpu(), spv(src, "sml_confidence_ptm_max_fwd_f32"), "sml_confidence_ptm_max_fwd_f32",
            struct.pack("<4I", n, 0, 0, 0),
            [h.Tensor(row.astype(np.float32)), h.Tensor(np.zeros(1, dtype=np.float32), output=True)],
            groups=1,
        )[1]
        check(f"ptm_max interface_only={interface_only}", best, np.array([want.max()]))


# ---------------------------------------------------------------------------
# Diffusion losses  [SI 3.7.1, Alg 28]: weighted Kabsch, MSE, bond loss
# ---------------------------------------------------------------------------

def _kabsch_ref(p, g, w):
    """Weighted Kabsch alignment of g onto p, float64.  p, g: [A,3]; w: [A]."""
    mu_p = (w[:, None] * p).sum(axis=0) / w.sum()
    mu_g = (w[:, None] * g).sum(axis=0) / w.sum()
    m = ((p - mu_p) * w[:, None]).T @ (g - mu_g)
    u, s, vt = np.linalg.svd(m)
    rot = u @ vt
    if np.linalg.det(rot) < 0.0:
        rot = u @ np.diag([1.0, 1.0, -1.0]) @ vt
    return (rot @ (g - mu_g).T).T + mu_p


def _random_rotation() -> np.ndarray:
    q, r = np.linalg.qr(RNG.normal(size=(3, 3)))
    q = q * np.sign(np.diag(r))  # unique decomposition
    if np.linalg.det(q) < 0.0:
        q[:, 0] = -q[:, 0]
    return q


def test_weighted_kabsch() -> None:
    structures, atoms = 2, 50
    p = RNG.normal(size=(structures, atoms, 3)) * 4.0
    rot = np.stack([_random_rotation() for _ in range(structures)])
    g = np.einsum("sij,saj->sai", rot, p) + RNG.normal(size=p.shape) * 0.05  # rotated, noisy copy
    w = RNG.random((structures, atoms)) + 0.1
    src = "slangml/loss/diffusion_loss_f32.slang"
    entry = "sml_loss_weighted_rigid_align_fwd_f32"
    f32 = lambda x: x.astype(np.float32).reshape(-1)
    out = h.run_entry(
        gpu(), spv(src, entry), entry, struct.pack("<4I", structures, atoms, 0, 0),
        [h.Tensor(f32(p)), h.Tensor(f32(g)), h.Tensor(f32(w)),
         h.Tensor(np.zeros(p.size, dtype=np.float32), output=True)],
        groups=structures,
    )[3].reshape(structures, atoms, 3)
    want = np.stack([_kabsch_ref(p[s], g[s].astype(np.float64), w[s]) for s in range(structures)])
    check("weighted_rigid_align", out, want, tol=1e-3)
    residual = float(np.abs(out - p).max())
    record("weighted_rigid_align recovers the pose", residual < 0.5, f"max |aligned - pred| {residual:.3f}")


def _mse_ref(xp, al, ww, atoms):
    return (ww * ((xp - al) ** 2).sum(axis=-1)).sum(axis=-1) / (3.0 * atoms)


def test_diffusion_mse_and_bond() -> None:
    structures, atoms = 2, 30
    p = RNG.normal(size=(structures, atoms, 3)) * 4.0
    aligned = RNG.normal(size=(structures, atoms, 3)) * 4.0
    w = RNG.random((structures, atoms)) + 0.1
    grad_scale = np.array([0.7, -1.3])
    src = "slangml/loss/diffusion_loss_f32.slang"
    f32 = lambda x: x.astype(np.float32).reshape(-1)
    zeros = lambda size: np.zeros(size, dtype=np.float32)

    per = h.run_entry(
        gpu(), spv(src, "sml_loss_diffusion_mse_fwd_f32"), "sml_loss_diffusion_mse_fwd_f32",
        struct.pack("<4I", structures, atoms, 0, 0),
        [h.Tensor(f32(p)), h.Tensor(f32(aligned)), h.Tensor(f32(w)), h.Tensor(zeros(structures), output=True)],
        groups=structures,
    )[3]
    check("diffusion_mse_fwd", per, _mse_ref(p, aligned, w, atoms), tol=1e-4)

    def loss(xs):
        return float(np.sum(grad_scale * _mse_ref(xs[0], aligned, xs[1], atoms)))

    grad = h.run_entry(
        gpu(), spv(src, "sml_loss_diffusion_mse_bwd_f32"), "sml_loss_diffusion_mse_bwd_f32",
        struct.pack("<4I", structures, atoms, 0, 0),
        [h.Tensor(f32(p)), h.Tensor(f32(aligned)), h.Tensor(f32(w)), h.Tensor(f32(grad_scale)),
         h.Tensor(zeros(p.size), output=True)],
        groups=h.ceil_div(p.size, 256),
    )[4].reshape(structures, atoms, 3)
    check("diffusion_mse_bwd gradX", grad, _numeric_grad(loss, [p, w], 0), tol=1e-3)

    # Bond loss: perBond = |(x_l - x_m) - (g_l - g_m)|^2 over random distinct atom pairs.
    n_atoms, n_bonds = 20, 30
    xp = RNG.normal(size=(n_atoms, 3)) * 3.0
    xg = RNG.normal(size=(n_atoms, 3)) * 3.0
    pairs = np.zeros((n_bonds, 2), dtype=np.uint32)
    for b in range(n_bonds):
        pairs[b] = RNG.choice(n_atoms, size=2, replace=False)
    bond_scale = RNG.random(n_bonds) + 0.5

    def bond_per(p_pred, g_gt):
        r = (p_pred[pairs[:, 0]] - p_pred[pairs[:, 1]]) - (g_gt[pairs[:, 0]] - g_gt[pairs[:, 1]])
        return (r ** 2).sum(axis=-1)

    src_bond = "slangml/loss/diffusion_loss_f32.slang"
    fwd = h.run_entry(
        gpu(), spv(src_bond, "sml_loss_bond_fwd_f32"), "sml_loss_bond_fwd_f32", struct.pack("<4I", n_bonds, 0, 0, 0),
        [h.Tensor(f32(xp)), h.Tensor(f32(xg)), h.Tensor(pairs.reshape(-1)), h.Tensor(zeros(n_bonds), output=True)],
        groups=h.ceil_div(n_bonds, 256),
    )[3]
    check("bond_fwd", fwd, bond_per(xp, xg), tol=1e-4)

    contrib = h.run_entry(
        gpu(), spv(src_bond, "sml_loss_bond_bwd_f32"), "sml_loss_bond_bwd_f32", struct.pack("<4I", n_bonds, 0, 0, 0),
        [h.Tensor(f32(xp)), h.Tensor(f32(xg)), h.Tensor(pairs.reshape(-1)), h.Tensor(f32(bond_scale)),
         h.Tensor(zeros(n_bonds * 6), output=True)],
        groups=h.ceil_div(n_bonds, 256),
    )[4].reshape(n_bonds, 2, 3)
    grad_bond = np.zeros((n_atoms, 3))
    np.add.at(grad_bond, pairs[:, 0], contrib[:, 0].astype(np.float64))
    np.add.at(grad_bond, pairs[:, 1], contrib[:, 1].astype(np.float64))

    def bond_loss(xs):
        return float(np.sum(bond_scale * bond_per(xs[0], xs[1])))

    check("bond_bwd gradX (host scatter)", grad_bond, _numeric_grad(bond_loss, [xp, xg], 0), tol=1e-3)


# ---------------------------------------------------------------------------
# Regression for the reduce entry barrier (common/subgroup.slang).  reduce_stats
# runs a max then a sum, or a sum then a sum, in one workgroup, so a missing
# barrier shows up as a wrong logsumexp or variance.  Not a proof of absence:
# the race is timing-dependent.
# ---------------------------------------------------------------------------

def test_reduce_stats_back_to_back() -> None:
    rows, cols = 64, 600
    x = (RNG.normal(size=(rows, cols)) * 4.0 + 2.0).astype(np.float32)
    x64 = x.astype(np.float64)
    src = "slangml/reduce/reduce_stats_f32.slang"
    zeros = np.zeros(rows, dtype=np.float32)
    payload = struct.pack("<2IfI", rows, cols, 0.0, 0)

    lse_entry = "sml_reduce_logsumexp_fwd_f32_subgroup"
    lse = h.run_entry(
        gpu(), spv(src, lse_entry), lse_entry, payload,
        [h.Tensor(x.reshape(-1)), h.Tensor(zeros, output=True)],
        groups=rows,
    )[1]
    m = x64.max(axis=1, keepdims=True)
    check("reduce_stats logsumexp (max then sum)", lse, (m + np.log(np.exp(x64 - m).sum(axis=1, keepdims=True)))[:, 0])

    var_entry = "sml_reduce_var_fwd_f32_subgroup"
    var = h.run_entry(
        gpu(), spv(src, var_entry), var_entry, payload,
        [h.Tensor(x.reshape(-1)), h.Tensor(zeros, output=True)],
        groups=rows,
    )[1]
    check("reduce_stats variance (sum then sum)", var, x64.var(axis=1))


# ---------------------------------------------------------------------------
# Split dispatch (contract section 13): two dispatches over disjoint slices of
# one logical grid must match one full dispatch.  Ignoring dispatchBase makes
# the second slice overwrite the first, so this test catches that.
# ---------------------------------------------------------------------------

def test_split_dispatch_matches_single() -> None:
    zeros = lambda size: np.zeros(size, dtype=np.float32)

    numel = 1000
    x = RNG.normal(size=numel).astype(np.float32)
    gp = RNG.normal(size=numel).astype(np.float32)
    resid = RNG.normal(size=numel).astype(np.float32)
    src = "slangml/diffusion/adaln_f32.slang"
    entry = "sml_diffusion_adaln_zero_residual_fwd_f32"
    payload = struct.pack("<I3I", numel, 0, 0, 0)

    def run_adaln(base: int, groups: int) -> np.ndarray:
        return h.run_entry(
            gpu(), spv(src, entry), entry, payload,
            [h.Tensor(x), h.Tensor(gp), h.Tensor(resid), h.Tensor(zeros(numel), output=True)],
            groups=groups, dispatch_base=(base, 0, 0),
        )[3]

    full = run_adaln(0, h.ceil_div(numel, 256))
    head = run_adaln(0, 2)      # global elements 0..511
    tail = run_adaln(512, 2)    # global elements 512..999; the numel guard stops at 1000
    combined = head.copy()
    combined[512:] = tail[512:]
    check("split dispatch adaln_zero_residual", combined, full)

    A, B, C = 6, 40, 40
    total = A * B * C
    y = RNG.uniform(1.0, 2.0, size=(A, B, C)).astype(np.float32)
    resid_d = RNG.normal(size=(A, B, C)).astype(np.float32)
    src_d = "slangml/pairformer/dropout_residual_f32.slang"
    entry_d = "sml_pairformer_dropout_residual_fwd_f32"
    payload_d = struct.pack("<4IfIII", A, B, C, 0, 0.25, 11, 3, 0)

    def run_dropout(base: int, groups: int) -> np.ndarray:
        return h.run_entry(
            gpu(), spv(src_d, entry_d), entry_d, payload_d,
            [h.Tensor(y.reshape(-1)), h.Tensor(resid_d.reshape(-1)), h.Tensor(zeros(total), output=True)],
            groups=groups, dispatch_base=(base, 0, 0),
        )[2]

    full_d = run_dropout(0, h.ceil_div(total, 256))
    head_d = run_dropout(0, 16)     # global elements 0..4095
    tail_d = run_dropout(4096, 22)  # global elements 4096..9599
    combined_d = head_d.copy()
    combined_d[4096:] = tail_d[4096:]
    check("split dispatch dropout (mask index math)", combined_d, full_d)


TESTS = [
    test_binned_cross_entropy_fwd,
    test_binned_cross_entropy_bwd,
    test_fourier_fwd,
    test_input_scale_fwd,
    test_output_combine_fwd,
    test_smooth_lddt_fwd,
    test_smooth_lddt_bwd,
    test_pair_bias_attention_fwd,
    test_pair_bias_attention_bwd,
    test_triangle_attention_bwd_start,
    test_triangle_attention_bwd_end,
    test_sequence_local_attention_bwd,
    test_outer_product_mean_bwd,
    test_msa_pair_weighted_avg_bwd,
    test_adaln_fwd_bwd,
    test_adaln_zero_residual_fwd_bwd,
    test_sampler_init,
    test_sampler_step,
    test_centre_augment,
    test_dropout_residual,
    test_relative_position_encoding,
    test_distance_bin_embed,
    test_ptm_row_and_max,
    test_weighted_kabsch,
    test_diffusion_mse_and_bond,
    test_reduce_stats_back_to_back,
    test_split_dispatch_matches_single,
]


def main(argv: list[str]) -> int:
    pattern = argv[1] if len(argv) > 1 else ""
    for test in TESTS:
        if pattern and pattern not in test.__name__:
            continue
        try:
            test()
        except Exception as exc:  # report and continue with the other kernels
            record(test.__name__, False, f"exception: {exc!r}")
    failed = 0
    for name, ok, detail in _results:
        print(f"{'PASS' if ok else 'FAIL'}  {name:36s} {detail}")
        failed += 0 if ok else 1
    print(f"{len(_results) - failed}/{len(_results)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
