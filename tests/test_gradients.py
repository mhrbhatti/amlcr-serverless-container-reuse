"""Finite difference check of the manual back propagation (float64)."""
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from amlcr.nn import TokenQNet, MLP


def check_token():
    rng = np.random.default_rng(1)
    net = TokenQNet(7, d=8, heads=2, seed=3, dtype=np.float64)
    for k in ("wh", "bh"):
        net.p[k] = rng.standard_normal(net.p[k].shape)      # make the head non trivial
    B, T = 3, 6
    X = rng.standard_normal((B, T, 7))
    mask = np.ones((B, T), bool)
    mask[0, 4:] = False
    mask[1, 5:] = False
    W = rng.standard_normal((B, T))

    def loss():
        q = net.forward(X, mask)
        return float((q * W).sum())

    net.forward(X, mask, cache=True)
    g = net.backward(W)
    worst = 0.0
    for k, v in net.p.items():
        flat = v.reshape(-1)
        idx = rng.choice(flat.size, size=min(6, flat.size), replace=False)
        for i in idx:
            old = flat[i]
            flat[i] = old + 1e-6
            lp = loss()
            flat[i] = old - 1e-6
            lm = loss()
            flat[i] = old
            num = (lp - lm) / 2e-6
            ana = g[k].reshape(-1)[i]
            err = abs(num - ana) / max(1e-7, abs(num) + abs(ana))
            worst = max(worst, err)
    print("TokenQNet worst relative error", worst)
    assert worst < 1e-4


def check_mlp():
    rng = np.random.default_rng(2)
    net = MLP([5, 7, 6, 2], seed=1, dtype=np.float64)
    X = rng.standard_normal((4, 5))
    W = rng.standard_normal((4, 2))
    net.forward(X, cache=True)
    g = net.backward(W)
    worst = 0.0
    for k, v in net.p.items():
        flat = v.reshape(-1)
        for i in rng.choice(flat.size, size=min(6, flat.size), replace=False):
            old = flat[i]
            flat[i] = old + 1e-6
            lp = float((net.forward(X) * W).sum())
            flat[i] = old - 1e-6
            lm = float((net.forward(X) * W).sum())
            flat[i] = old
            num = (lp - lm) / 2e-6
            ana = g[k].reshape(-1)[i]
            worst = max(worst, abs(num - ana) / max(1e-7, abs(num) + abs(ana)))
    print("MLP worst relative error", worst)
    assert worst < 1e-4


if __name__ == "__main__":
    check_token()
    check_mlp()
    print("gradient checks passed")
