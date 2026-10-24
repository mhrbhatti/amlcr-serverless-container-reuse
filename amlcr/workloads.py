"""Synthetic invocation streams built from the 13 catalog functions.

Kinds follow the workload design of FStartBench (similarity, arrival pattern) plus a
mixed stream used for training and a drift stream that changes its function mix midway.
"""
import numpy as np
from .catalog import CAT, N_FUNC

HISIM = [0, 1, 2, 3, 10]       # functions 1,2,3,4,11 of Table II
LOSIM = [0, 1, 4, 8, 12]       # functions 1,2,5,9,13
PEAKSET = [0, 1, 4, 5, 12]     # functions 1,2,5,6,13
FAM_A = [1, 2, 3, 4, 5, 6, 7, 9, 12]    # python family plus alpine node and go
FAM_B = [0, 8, 10, 11]                  # java, c++ and node web (rare before the drift)


def _exec_times(f, rng):
    return CAT.exec_median[f] * np.exp(rng.normal(0.0, 0.35, size=len(f)))


def _poisson_times(n, rate, rng):
    return np.cumsum(rng.exponential(1.0 / rate, size=n))


def _peak_times(n, hi, lo, period, rng):
    t, out = 0.0, []
    while len(out) < n:
        rate = hi if int(t // period) % 2 == 0 else lo
        t += rng.exponential(1.0 / rate)
        out.append(t)
    return np.array(out)


N_INST = 6      # application instances (distinct functions) per package set


def _instances(f, rng):
    """Each package set is shared by several distinct functions with a skewed popularity."""
    w = np.exp(rng.normal(0, 1.2, N_INST))
    v = rng.choice(N_INST, size=len(f), p=w / w.sum())
    return f + N_FUNC * v


def _pick(funcs, w, n, rng):
    funcs = np.asarray(funcs)
    w = np.asarray(w, float)
    return funcs[rng.choice(len(funcs), size=n, p=w / w.sum())]


def make_episode(kind, seed, n=600):
    rng = np.random.default_rng(seed)
    if kind == "mixed":
        w = np.exp(rng.normal(0, 1.0, N_FUNC))
        f = _pick(range(N_FUNC), w, n, rng)
        t = _poisson_times(n, rng.uniform(1.6, 2.8), rng)
    elif kind == "hisim":
        f = _pick(HISIM, np.ones(len(HISIM)), n, rng)
        t = _poisson_times(n, 2.2, rng)
    elif kind == "losim":
        f = _pick(LOSIM, np.ones(len(LOSIM)), n, rng)
        t = _poisson_times(n, 2.2, rng)
    elif kind == "peak":
        f = _pick(PEAKSET, np.ones(len(PEAKSET)), n, rng)
        t = _peak_times(n, 3.6, 0.9, 60.0, rng)
    elif kind == "uniform":
        f = _pick(PEAKSET, np.ones(len(PEAKSET)), n, rng)
        t = np.arange(1, n + 1) / 2.25 + rng.uniform(0, 0.05, n)
    elif kind == "familyA":          # training regime for the drift study
        wa = np.exp(rng.normal(0, 0.8, len(FAM_A)))
        fa = _pick(FAM_A, wa, n, rng)
        rare = rng.random(n) < 0.01
        fb = _pick(FAM_B, np.ones(len(FAM_B)), n, rng)
        f = np.where(rare, fb, fa)
        t = _poisson_times(n, rng.uniform(1.6, 2.8), rng)
    elif kind == "drift":            # first half family A, second half dominated by family B
        wa = np.exp(rng.normal(0, 0.8, len(FAM_A)))
        h = n // 2
        f1 = _pick(FAM_A, wa, h, rng)
        wb = np.exp(rng.normal(0, 0.5, len(FAM_B)))
        fb = _pick(FAM_B, wb, n - h, rng)
        fa2 = _pick(FAM_A, wa, n - h, rng)
        use_b = rng.random(n - h) < 0.8
        f = np.concatenate([f1, np.where(use_b, fb, fa2)])
        t = _poisson_times(n, 2.2, rng)
    elif kind == "train":
        k = rng.choice(["mixed", "hisim", "losim", "peak", "uniform"], p=[0.4, 0.15, 0.15, 0.2, 0.1])
        return make_episode(k, seed + 7919, n)
    else:
        raise ValueError(kind)
    e = _exec_times(f, rng)
    return {"kind": kind, "seed": seed, "t": t, "f": f.astype(int), "e": e,
            "inst": _instances(f.astype(int), rng)}
