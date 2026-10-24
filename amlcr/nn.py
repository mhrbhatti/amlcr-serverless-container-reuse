"""Small neural network toolkit written in plain numpy with manual back propagation.

TokenQNet : token wise Q network with one multi head self attention layer.
            Token 0 stands for the cold start action, tokens 1..K for idle containers.
MLP       : fully connected regressor with several output heads.
Adam      : optimiser shared by both.
"""
import numpy as np


class Adam:
    def __init__(self, params, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8, clip=5.0):
        self.p = params
        self.lr = lr
        self.b1, self.b2, self.eps, self.clip = b1, b2, eps, clip
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, grads, lr_scale=1.0):
        # global norm clipping
        tot = np.sqrt(sum(float((g * g).sum()) for g in grads.values()))
        sc = min(1.0, self.clip / (tot + 1e-12))
        self.t += 1
        for k, g in grads.items():
            g = g * sc
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * g * g
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            self.p[k] -= self.lr * lr_scale * mh / (np.sqrt(vh) + self.eps)


def _softmax(x):
    x = x - x.max(-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(-1, keepdims=True)


class TokenQNet:
    def __init__(self, f_in, d=32, heads=2, seed=0, dtype=np.float32):
        rng = np.random.default_rng(seed)
        self.f_in, self.d, self.h = f_in, d, heads
        self.dk = d // heads
        self.dtype = dtype

        def w(a, b, s=None):
            s = s if s is not None else np.sqrt(2.0 / a)
            return (rng.standard_normal((a, b)) * s).astype(dtype)

        self.p = {
            "W1": w(f_in, d), "b1": np.zeros(d, dtype),
            "Wq": w(d, d, 0.15), "Wk": w(d, d, 0.15), "Wv": w(d, d, 0.15), "Wo": w(d, d, 0.15),
            "W2": w(d, d), "b2": np.zeros(d, dtype),
            "wh": np.zeros((2, d), dtype),          # zero start: Q residual begins at exactly zero
            "bh": np.zeros(2, dtype),
        }

    def copy_from(self, other):
        for k in self.p:
            self.p[k][...] = other.p[k]

    # ------------------------------------------------------------------
    def forward(self, X, mask, cache=False):
        p, B, T = self.p, X.shape[0], X.shape[1]
        H, dk, D = self.h, self.dk, self.d
        Z1 = X @ p["W1"] + p["b1"]
        H0 = np.maximum(Z1, 0)
        Q = (H0 @ p["Wq"]).reshape(B, T, H, dk).transpose(0, 2, 1, 3)
        K = (H0 @ p["Wk"]).reshape(B, T, H, dk).transpose(0, 2, 1, 3)
        V = (H0 @ p["Wv"]).reshape(B, T, H, dk).transpose(0, 2, 1, 3)
        S = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(dk)
        S = np.where(mask[:, None, None, :], S, -1e9)
        A = _softmax(S)
        O = (A @ V).transpose(0, 2, 1, 3).reshape(B, T, D)
        H1 = H0 + O @ p["Wo"]
        Z2 = H1 @ p["W2"] + p["b2"]
        H2 = np.maximum(Z2, 0)
        typ = np.ones(T, int)
        typ[0] = 0
        q = np.einsum("btd,td->bt", H2, p["wh"][typ]) + p["bh"][typ]
        if cache:
            self._c = (X, mask, Z1, H0, Q, K, V, A, O, H1, Z2, H2, typ)
        return q

    def backward(self, dq):
        p = self.p
        X, mask, Z1, H0, Q, K, V, A, O, H1, Z2, H2, typ = self._c
        B, T, D = H0.shape
        H, dk = self.h, self.dk
        g = {}
        wh_t = p["wh"][typ]
        dH2 = dq[:, :, None] * wh_t[None]
        g["wh"] = np.zeros_like(p["wh"])
        g["bh"] = np.zeros_like(p["bh"])
        g["wh"][0] = np.einsum("b,bd->d", dq[:, 0], H2[:, 0])
        g["wh"][1] = np.einsum("bt,btd->d", dq[:, 1:], H2[:, 1:])
        g["bh"][0] = dq[:, 0].sum()
        g["bh"][1] = dq[:, 1:].sum()
        dZ2 = dH2 * (Z2 > 0)
        g["W2"] = H1.reshape(-1, D).T @ dZ2.reshape(-1, D)
        g["b2"] = dZ2.sum((0, 1))
        dH1 = dZ2 @ p["W2"].T
        dH0 = dH1.copy()
        g["Wo"] = O.reshape(-1, D).T @ dH1.reshape(-1, D)
        dO = dH1 @ p["Wo"].T
        dOh = dO.reshape(B, T, H, dk).transpose(0, 2, 1, 3)
        dA = dOh @ V.transpose(0, 1, 3, 2)
        dV = A.transpose(0, 1, 3, 2) @ dOh
        dS = A * (dA - (dA * A).sum(-1, keepdims=True))
        dS = dS / np.sqrt(dk)
        dQ = dS @ K
        dK = dS.transpose(0, 1, 3, 2) @ Q
        mer = lambda x: x.transpose(0, 2, 1, 3).reshape(B, T, D)
        dQ, dK, dV = mer(dQ), mer(dK), mer(dV)
        H0f = H0.reshape(-1, D).T
        g["Wq"] = H0f @ dQ.reshape(-1, D)
        g["Wk"] = H0f @ dK.reshape(-1, D)
        g["Wv"] = H0f @ dV.reshape(-1, D)
        dH0 += dQ @ p["Wq"].T + dK @ p["Wk"].T + dV @ p["Wv"].T
        dZ1 = dH0 * (Z1 > 0)
        g["W1"] = X.reshape(-1, X.shape[-1]).T @ dZ1.reshape(-1, D)
        g["b1"] = dZ1.sum((0, 1))
        return g


class MLP:
    def __init__(self, sizes, seed=0, dtype=np.float32):
        rng = np.random.default_rng(seed)
        self.n = len(sizes) - 1
        self.p = {}
        for i in range(self.n):
            a, b = sizes[i], sizes[i + 1]
            self.p["W%d" % i] = (rng.standard_normal((a, b)) * np.sqrt(2.0 / a)).astype(dtype)
            self.p["b%d" % i] = np.zeros(b, dtype)
        self.p["W%d" % (self.n - 1)] *= 0.1

    def forward(self, X, cache=False):
        hs, zs = [X], []
        h = X
        for i in range(self.n):
            z = h @ self.p["W%d" % i] + self.p["b%d" % i]
            zs.append(z)
            h = np.maximum(z, 0) if i < self.n - 1 else z
            hs.append(h)
        if cache:
            self._c = (hs, zs)
        return h

    def backward(self, dout):
        hs, zs = self._c
        g = {}
        d = dout
        for i in reversed(range(self.n)):
            g["W%d" % i] = hs[i].T @ d
            g["b%d" % i] = d.sum(0)
            if i > 0:
                d = (d @ self.p["W%d" % i].T) * (zs[i - 1] > 0)
        return g
