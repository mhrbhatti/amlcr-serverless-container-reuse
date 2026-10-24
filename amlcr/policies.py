"""Warm pool policies: baselines, the DQN scheduler (MLCR re implementation and the
proposed AMLCR variants), the retention agent and the drift detector."""
import numpy as np
from .catalog import CAT, N_FUNC
from .sim import K_MAX
from .nn import TokenQNet, MLP, Adam

F_TOK = 20
T_TOK = K_MAX + 1
OS_OH = np.eye(3, dtype=np.float32)[CAT.os]
LANG_OH = np.eye(5, dtype=np.float32)[CAT.lang]
TYPE_OH = np.eye(N_FUNC, dtype=np.float32)


# =============================================================== baselines
class BasePolicy:
    ttl = None

    def reset(self, sim): pass
    def select(self, sim, f): return -1
    def observe(self, sim, f, lat, cold, a): pass
    def retain(self, sim, c): return True
    def recheck(self, sim): return []
    def victim(self, sim): return int(np.argmin([c.last_used for c in sim.idle]))
    def stay_features(self, sim, c): return None
    def on_stay_end(self, sim, x, savings, memtime): pass
    def end_episode(self, sim): pass


class ColdOnly(BasePolicy):
    def retain(self, sim, c): return False


class ExactLRU(BasePolicy):
    """Reuse only a container that already holds the same function, evict least recently used."""
    def select(self, sim, f):
        best, bt = -1, -1.0
        for i, c in enumerate(sim.idle):
            if c.inst == sim.cur_inst and c.last_used > bt:
                best, bt = i, c.last_used
        return best


class FaasCache(BasePolicy):
    """Greedy dual size frequency caching of Fuerst and Sharma (exact match reuse)."""
    def reset(self, sim):
        self.clock = 0.0

    def _prio(self, sim, c):
        return self.clock + sim.freq_inst[c.inst] * CAT.cold[c.typ] / CAT.mem[c.typ]

    def select(self, sim, f):
        cand = [(c.last_used, i) for i, c in enumerate(sim.idle) if c.inst == sim.cur_inst]
        return max(cand)[1] if cand else -1

    def victim(self, sim):
        pr = [self._prio(sim, c) for c in sim.idle]
        i = int(np.argmin(pr))
        self.clock = pr[i]
        return i


class KeepAlive(BasePolicy):
    """Fixed ten minute keep alive, new containers are rejected when the pool is full."""
    ttl = 600.0

    def select(self, sim, f):
        for i, c in enumerate(sim.idle):
            if c.inst == sim.cur_inst:
                return i
        return -1

    def retain(self, sim, c):
        return sim.idle_mem + CAT.mem[c.typ] <= sim.cap + 1e-9 and len(sim.idle) < K_MAX


class Greedy(BasePolicy):
    """Multi level matching with the locally cheapest container (Greedy Match of the base paper)."""
    def select(self, sim, f):
        if not sim.idle:
            return -1
        types = np.array([c.typ for c in sim.idle])
        w = sim.warm[f, types]
        lv = CAT.level[f, types]
        w = np.where(lv >= 1, w, np.inf)
        i = int(np.argmin(w))
        return i if w[i] < CAT.cold[f] - 1e-9 else -1


# =============================================================== features
def build_tokens(sim, f, variant):
    X = np.zeros((T_TOK, F_TOK), np.float32)
    mask = np.zeros(T_TOK, bool)
    anchor = np.zeros(T_TOK, np.float32)
    X[0, 0:3] = OS_OH[f]
    X[0, 3:8] = LANG_OH[f]
    X[0, 8] = CAT.cold[f] / 10
    X[0, 9] = CAT.sz_total[f] / 600
    X[0, 10] = CAT.mem[f] / 1000
    X[0, 11] = CAT.sz_rt[f] / 450
    X[0, 12] = min(sim.rate[f], 3.0)
    X[0, 13] = sim.idle_mem / sim.cap
    X[0, 14] = len(sim.idle) / K_MAX
    X[0, 15] = sim.cap / 2000
    X[0, 16] = sim.rate.sum() / 4
    X[0, 17] = 1.0
    mask[0] = True
    anchor[0] = CAT.cold[f] / 10
    n = len(sim.idle)
    if n:
        ty = np.fromiter((c.typ for c in sim.idle), int, n)
        lv = CAT.level[f, ty]
        warm = sim.warm[f, ty]
        X[1:1 + n, 0:3] = OS_OH[ty]
        X[1:1 + n, 3:8] = LANG_OH[ty]
        X[1:1 + n, 8] = warm / 10
        X[1:1 + n, 9] = (CAT.cold[f] - warm) / 10
        X[1:1 + n, 10] = CAT.mem[ty] / 1000
        X[1:1 + n, 11] = np.minimum((sim.now - np.fromiter((c.t_idle for c in sim.idle), float, n)) / 60.0, 2.0)
        X[1:1 + n, 12] = lv == 1
        X[1:1 + n, 13] = lv == 2
        X[1:1 + n, 14] = lv == 3
        if variant == "delta":
            X[1:1 + n, 15] = CAT.delta_mb[f, ty] / 450
            X[1:1 + n, 16] = CAT.jac[f, ty]
        X[1:1 + n, 18] = np.minimum(np.fromiter((c.uses for c in sim.idle), float, n), 5) / 5
        X[1:1 + n, 19] = 1.0
        mask[1:1 + n] = lv >= 1
        anchor[1:1 + n] = warm / 10
    return X, mask, anchor


# =============================================================== retention agent (contribution 2)
class RetentionAgent:
    """Predicts, for a container that is about to stay idle, the latency it will save and the
    memory time (GB s) it will occupy. The keep value is saved seconds minus lam times GB s."""
    N_IN = 2 * N_FUNC + 6

    def __init__(self, seed=0, train=False, lam=0.2, lr=1e-3, budget_mb=None):
        self.net = MLP([self.N_IN, 64, 64, 2], seed=seed)
        self.opt = Adam(self.net.p, lr=lr)
        self.rng = np.random.default_rng(seed + 11)
        self.train = train
        self.lam = lam
        self.budget_mb = budget_mb
        self.eps = 0.0
        self.cap_buf = 60000
        self.bx = np.zeros((self.cap_buf, self.N_IN), np.float32)
        self.by = np.zeros((self.cap_buf, 2), np.float32)
        self.nb = 0
        self.ptr = 0
        self.since = 0
        self.ready = False
        self.fit_mult = 1
        self.type_n = np.zeros(N_FUNC, int)
        self.win_start = 0.0
        self.lam_trace = []

    # features of a container of type tc, with the pool as it is (optionally leaving one slot out)
    def feat(self, sim, tc, exclude=None):
        pool = [c.typ for j, c in enumerate(sim.idle) if j != exclude]
        return self._feat(sim, tc, pool)

    def _feat(self, sim, tc, pool):
        best = CAT.cold.copy()
        if pool:
            best = np.minimum(best, sim.warm[:, pool].min(1))
        gain_m = np.maximum(0.0, best - sim.warm[:, tc])
        gain_p = np.maximum(0.0, CAT.cold - sim.warm[:, tc])
        r = sim.rate
        x = np.zeros(self.N_IN, np.float32)
        x[:N_FUNC] = TYPE_OH[tc]
        x[N_FUNC:2 * N_FUNC] = np.minimum(r, 3.0)
        x[2 * N_FUNC + 0] = sim.idle_mem / sim.cap
        x[2 * N_FUNC + 1] = sim.cap / 2000
        x[2 * N_FUNC + 2] = len(pool) / K_MAX
        x[2 * N_FUNC + 3] = CAT.mem[tc] / 1000
        x[2 * N_FUNC + 4] = float((r * gain_m).sum()) / 3
        x[2 * N_FUNC + 5] = float((r * gain_p).sum()) / 3
        return x

    def feat_subset(self, sim, j, alive):
        """Features of idle container j when only the containers in `alive` are in the pool."""
        pool = [sim.idle[i].typ for i in alive if i != j]
        return self._feat(sim, sim.idle[j].typ, pool)

    def predict(self, X):
        o = self.net.forward(X)
        return o[:, 0] * 5.0, np.maximum(o[:, 1] * 20.0, 0.0)

    def value(self, X):
        s, m = self.predict(X)
        return s - self.lam * m

    def add(self, x, savings, memtime):
        self.type_n[int(np.argmax(x[:N_FUNC]))] += 1
        i = self.ptr
        self.bx[i] = x
        self.by[i] = (savings / 5.0, memtime / 20.0)
        self.ptr = (self.ptr + 1) % self.cap_buf
        self.nb = min(self.nb + 1, self.cap_buf)
        self.since += 1
        if self.nb >= 200:
            self.ready = True
        if self.train and self.nb >= 200 and self.since >= 8:
            self.since = 0
            for _ in range(2 * self.fit_mult):
                self.fit_batch()

    def fit_batch(self, bs=128):
        idx = self.rng.integers(0, self.nb, bs)
        X, Y = self.bx[idx], self.by[idx]
        out = self.net.forward(X, cache=True)
        d = (out - Y)
        g = self.net.backward(d / bs)
        self.opt.step(g)

    def tick(self, sim):
        """Dual ascent on the memory price when a budget is set."""
        if self.budget_mb is None:
            return
        if sim.now - self.win_start >= 20.0:
            dur = sim.now - self.win_start
            occ = sim.window_area / dur
            self.lam = float(np.clip(self.lam * np.exp(0.8 * (occ / self.budget_mb - 1.0)), 1e-3, 20.0))
            sim.window_area = 0.0
            self.win_start = sim.now
            self.lam_trace.append((sim.now, self.lam, occ))


# =============================================================== drift detector (contribution 3)
class DriftDetector:
    def __init__(self, recent=60, ref=240, thr=0.25, cooldown=250):
        self.recent, self.ref, self.thr, self.cool = recent, ref, thr, cooldown
        self.buf = []
        self.last = -10 ** 9
        self.k = 0
        self.events = []

    @staticmethod
    def js(p, q):
        m = 0.5 * (p + q)
        kl = lambda a, b: float((a * (np.log(a + 1e-12) - np.log(b + 1e-12))).sum())
        return 0.5 * kl(p, m) + 0.5 * kl(q, m)

    def update(self, f):
        self.buf.append(f)
        self.k += 1
        if len(self.buf) > self.recent + self.ref:
            self.buf.pop(0)
        if len(self.buf) < self.recent + self.ref or self.k - self.last < self.cool:
            return False
        a = np.bincount(self.buf[-self.recent:], minlength=N_FUNC).astype(float) + 0.5
        b = np.bincount(self.buf[:-self.recent], minlength=N_FUNC).astype(float) + 0.5
        if self.js(a / a.sum(), b / b.sum()) > self.thr:
            self.last = self.k
            self.events.append(self.k)
            return True
        return False


# =============================================================== DQN scheduler
class Replay:
    def __init__(self, cap):
        self.cap = cap
        self.X = np.zeros((cap, T_TOK, F_TOK), np.float32)
        self.M = np.zeros((cap, T_TOK), bool)
        self.A = np.zeros((cap, T_TOK), np.float32)
        self.a = np.zeros(cap, np.int64)
        self.r = np.zeros(cap, np.float32)
        self.X2 = np.zeros((cap, T_TOK, F_TOK), np.float32)
        self.M2 = np.zeros((cap, T_TOK), bool)
        self.A2 = np.zeros((cap, T_TOK), np.float32)
        self.d = np.zeros(cap, np.float32)
        self.n = 0
        self.ptr = 0

    def push(self, X, M, A, a, r, X2, M2, A2, d):
        i = self.ptr
        self.X[i], self.M[i], self.A[i], self.a[i], self.r[i] = X, M, A, a, r
        self.X2[i], self.M2[i], self.A2[i], self.d[i] = X2, M2, A2, d
        self.ptr = (self.ptr + 1) % self.cap
        self.n = min(self.n + 1, self.cap)


class DQNSched(BasePolicy):
    def __init__(self, variant="level", retention=None, train=False, seed=0, net=None,
                 eps=0.0, lr=5e-4, gamma=0.9, online=False, detector=None, buf=20000):
        self.variant = variant
        self.ret = retention
        self.train = train
        self.online = online
        self.learning = train or online
        self.rng = np.random.default_rng(seed + 101)
        self.net = net if net is not None else TokenQNet(F_TOK, d=32, heads=2, seed=seed)
        self.tgt = TokenQNet(F_TOK, d=32, heads=2, seed=seed)
        self.tgt.copy_from(self.net)
        self.opt = Adam(self.net.p, lr=lr)
        self.base_lr = lr
        self.gamma = gamma
        self.eps = eps
        self.base_eps = eps
        self.replay = Replay(buf) if self.learning else None
        self.pending = None
        self.steps = 0
        self.detector = detector
        self.boost = 0
        self.recent_only = 0
        self.ret_base_eps = 0.0
        self.sync_every = 400
        self.learn_every = 2
        self.decision_ms = []

    # --- interface to the simulator
    def reset(self, sim):
        self.pending = None
        self._last_check = 0.0
        self.sim = sim
        if self.ret is not None:
            self.ret.win_start = 0.0

    @property
    def system(self):
        return self.variant

    def select(self, sim, f):
        X, M, A = build_tokens(sim, f, self.variant)
        if self.pending is not None and self.learning:
            Xp, Mp, Ap, ap, rp = self.pending
            self.replay.push(Xp, Mp, Ap, ap, rp, X, M, A, 0.0)
            self._after_push()
        g = self.net.forward(X[None], M[None])[0]
        q = np.where(M, -A + g, -np.inf)
        eps = self.eps
        if self.learning and self.rng.random() < eps:
            a = int(self.rng.choice(np.flatnonzero(M)))
        else:
            a = int(np.argmax(q))
        if self.learning:
            self.pending = (X, M, A, a, 0.0)
        return a - 1

    def observe(self, sim, f, lat, cold, a):
        if self.pending is not None:
            Xp, Mp, Ap, ap, _ = self.pending
            self.pending = (Xp, Mp, Ap, ap, -lat / 10.0)
        if self.ret is not None:
            self.ret.tick(sim)
        if self.detector is not None and self.detector.update(f):
            self.boost = 150
            self.recent_only = 300
            self.drift_flag_at = sim.now
        if self.boost > 0:
            frac = self.boost / 150.0
            self.eps = self.base_eps + 0.15 * frac
            if self.ret is not None and self.ret.train:
                self.ret.eps = self.ret_base_eps + 0.30 * frac
                self.ret.fit_mult = 3
            self.boost -= 1
        elif self.detector is not None:
            self.eps = self.base_eps
            if self.ret is not None and self.ret.train:
                self.ret.eps = self.ret_base_eps
                self.ret.fit_mult = 1

    def end_episode(self, sim):
        if self.pending is not None and self.learning:
            Xp, Mp, Ap, ap, rp = self.pending
            self.replay.push(Xp, Mp, Ap, ap, rp, Xp, Mp, Ap, 1.0)
            self._after_push()
            self.pending = None

    # --- retention hooks
    def retain(self, sim, c):
        if self.ret is None:
            return True
        r = self.ret
        if r.train and r.rng.random() < r.eps:
            return r.rng.random() < 0.5
        if not r.ready:
            return True
        if r.train and r.type_n[c.typ] < 6:
            return True          # optimism for function types with too few observed stays (online mode only)
        x = r.feat(sim, c.typ)
        return bool(r.value(x[None])[0] > 0)

    def recheck(self, sim, period=5.0, max_drop=3):
        """Drop idle containers whose keep value has turned negative as arrival rates change."""
        r = self.ret
        if r is None or not r.ready or not sim.idle or sim.now - self._last_check < period:
            return []
        self._last_check = sim.now
        drop = []
        alive = list(range(len(sim.idle)))
        for _ in range(max_drop):
            if not alive:
                break
            X = np.stack([r.feat_subset(sim, j, alive) for j in alive])
            v = r.value(X)
            k = int(np.argmin(v))
            if v[k] > 0:
                break
            drop.append(alive.pop(k))
        return drop

    def victim(self, sim):
        if self.ret is None or not self.ret.ready:
            return super().victim(sim)
        r = self.ret
        X = np.stack([r.feat(sim, c.typ, exclude=j) for j, c in enumerate(sim.idle)])
        v = r.value(X) / CAT.mem[[c.typ for c in sim.idle]]
        return int(np.argmin(v))

    def stay_features(self, sim, c):
        if self.ret is None:
            return None
        return self.ret.feat(sim, c.typ)

    def on_stay_end(self, sim, x, savings, memtime):
        if self.ret is not None and x is not None:
            self.ret.add(x, savings, memtime)

    # --- learning
    def _after_push(self):
        self.steps += 1
        if self.replay.n >= 500 and self.steps % self.learn_every == 0:
            self.learn()
        if self.steps % self.sync_every == 0:
            self.tgt.copy_from(self.net)

    def learn(self, bs=64):
        rp = self.replay
        if self.recent_only > 0:
            self.recent_only -= 1
            lo = max(0, rp.n - 1500) if rp.n < rp.cap else 0
            if rp.n < rp.cap:
                idx = self.rng.integers(lo, rp.n, bs)
            else:
                idx = (rp.ptr - 1 - self.rng.integers(0, 1500, bs)) % rp.cap
        else:
            idx = self.rng.integers(0, rp.n, bs)
        X, M, A, a, r = rp.X[idx], rp.M[idx], rp.A[idx], rp.a[idx], rp.r[idx]
        X2, M2, A2, d = rp.X2[idx], rp.M2[idx], rp.A2[idx], rp.d[idx]
        gt = self.tgt.forward(X2, M2)
        qn = np.where(M2, -A2 + gt, -1e9).max(1)
        y = r + self.gamma * (1.0 - d) * qn
        g = self.net.forward(X, M, cache=True)
        ar = np.arange(bs)
        qsa = -A[ar, a] + g[ar, a]
        td = np.clip(qsa - y, -1.0, 1.0)
        dq = np.zeros_like(g)
        dq[ar, a] = td / bs
        grads = self.net.backward(dq)
        scale = 3.0 if self.boost > 0 else 1.0
        self.opt.step(grads, lr_scale=scale)

    def save(self, path):
        np.savez(path, **self.net.p)

    def load(self, path):
        z = np.load(path)
        for k in self.net.p:
            self.net.p[k][...] = z[k]
        self.tgt.copy_from(self.net)
