"""Trains Nielsen's network.py on the same data and hyperparameters as the
NeonEngine GPU digit recognizer, and reports training time and test accuracy.

Data: the official MNIST IDX files (60k train / 10k test), resized from 28x28
to 27x27 with the same bilinear mapping as digit_recognizer_train.cpp, pixels
scaled to [0, 1].

Usage: python train_nielsen.py [mnist_dir]
"""

import os

# network.py does many tiny matrix ops; multithreaded OpenBLAS makes them ~1.8x slower here.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import random
import sys
import time
from pathlib import Path

import numpy as np

import network

# Same values as TrainConfig in src/digit_recognizer_train.hpp.
SIZES = [729, 128, 10]
EPOCHS = 30
MINI_BATCH_SIZE = 10
ETA = 3.0
SEED = 20241

SRC = 28
DST = 27


def read_images(path):
    raw = Path(path).read_bytes()
    assert int.from_bytes(raw[0:4], "big") == 2051
    n = int.from_bytes(raw[4:8], "big")
    imgs = np.frombuffer(raw, dtype=np.uint8, offset=16).reshape(n, SRC, SRC).astype(np.float32)

    # Bilinear 28 -> 27, same as align_corners=False in the C++ loader.
    d = np.arange(DST, dtype=np.float32)
    src = np.clip((d + 0.5) * SRC / DST - 0.5, 0, SRC - 1).astype(np.float32)
    lo = np.floor(src).astype(np.int64)
    hi = np.minimum(lo + 1, SRC - 1)
    fr = (src - lo).astype(np.float32)

    a = imgs[:, lo][:, :, lo]
    b = imgs[:, lo][:, :, hi]
    c = imgs[:, hi][:, :, lo]
    e = imgs[:, hi][:, :, hi]
    top = a + (b - a) * fr[None, None, :]
    bot = c + (e - c) * fr[None, None, :]
    out = (top + (bot - top) * fr[None, :, None]) / np.float32(255.0)
    # float64 to match the weights: mixed float32/float64 np.dot skips the BLAS fast path.
    return out.reshape(n, DST * DST, 1).astype(np.float64)


def read_labels(path):
    raw = Path(path).read_bytes()
    assert int.from_bytes(raw[0:4], "big") == 2049
    return np.frombuffer(raw, dtype=np.uint8, offset=8).astype(np.int64)


def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e


def main():
    mnist = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[5] / "data" / "mnist"

    t0 = time.perf_counter()
    train_x = read_images(mnist / "train-images-idx3-ubyte")
    train_y = read_labels(mnist / "train-labels-idx1-ubyte")
    test_x = read_images(mnist / "t10k-images-idx3-ubyte")
    test_y = read_labels(mnist / "t10k-labels-idx1-ubyte")
    training_data = [(x, vectorized_result(y)) for x, y in zip(train_x, train_y)]
    test_data = [(x, int(y)) for x, y in zip(test_x, test_y)]
    load_s = time.perf_counter() - t0
    print(f"MNIST: {len(training_data)} train, {len(test_data)} test (27x27), loaded in {load_s:.2f} s")
    print(f"sizes={SIZES}  epochs={EPOCHS}  batch={MINI_BATCH_SIZE}  eta={ETA}  seed={SEED}", flush=True)

    random.seed(SEED)
    np.random.seed(SEED)

    t0 = time.perf_counter()
    net = network.Network(SIZES)
    net.SGD(training_data, EPOCHS, MINI_BATCH_SIZE, ETA)  # no per-epoch test eval, same as the GPU run
    train_s = time.perf_counter() - t0

    correct = net.evaluate(test_data)
    print(f"Python training: {EPOCHS} epochs in {train_s:.2f} s ({train_s / EPOCHS:.2f} s/epoch)")
    print(f"MNIST test accuracy: {correct}/{len(test_data)} = {correct / len(test_data):.4f}")


if __name__ == "__main__":
    main()
