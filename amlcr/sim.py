"""Discrete event simulator for warm container reuse.

A request arrives, the policy picks an idle warm container or a cold start, the startup
latency is read from the catalog tables, the container is busy for startup plus execution
and is then handed back to the policy, which may keep it in the warm pool or drop it.
Only idle containers count against the warm pool capacity, as in the base paper.
"""
import heapq
import numpy as np
from .catalog import CAT, N_FUNC, CLEAN_OTHER, CLEAN_SAME

K_MAX = 24                 # maximum number of idle containers shown to a policy
RATE_HALF_LIFE = 45.0      # s, half life of the per function arrival rate estimate


class Container:
    __slots__ = ("typ", "inst", "t_idle", "last_used", "uses", "uid", "stay")

    def __init__(self, typ, uid, inst=None):
        self.typ = typ
        self.inst = typ if inst is None else inst
        self.t_idle = 0.0
        self.last_used = 0.0
        self.uses = 0
        self.uid = uid
        self.stay = None


class Sim:
    def __init__(self, episode, cap_mb, system="level"):
        self.ep = episode
        self.cap = float(cap_mb)
        self.system = system
        self.warm = CAT.warm_delta if system == "delta" else CAT.warm_level
        self.now = 0.0
        self.idle = []
        self.busy = []                      # heap of (finish, uid, container)
        self.idle_mem = 0.0
        self.area = 0.0                     # MB * s
        self.peak = 0.0
        self._t_last = 0.0
        self._uid = 0
        self.rate = np.zeros(N_FUNC)
        self.rate_t = 0.0
        self.freq = np.zeros(N_FUNC)
        self.freq_inst = np.zeros(N_FUNC * 8)
        self.cur_inst = 0
        self.window_area = 0.0
        self.log_t, self.log_lat, self.log_cold, self.log_f, self.log_mem = [], [], [], [], []
        self.evictions = 0

    # ----------------------------------------------------------- time keeping
    def _advance(self, t):
        dt = t - self._t_last
        if dt > 0:
            a = self.idle_mem * dt
            self.area += a
            self.window_area += a
            self._t_last = t
        self.now = t

    def decay_rates(self, t):
        dt = t - self.rate_t
        if dt > 0:
            self.rate *= np.exp(-dt * np.log(2) / RATE_HALF_LIFE)
            self.rate_t = t

    # ----------------------------------------------------------- pool operations
    def _add_idle(self, c):
        self.idle.append(c)
        self.idle_mem += CAT.mem[c.typ]
        self.peak = max(self.peak, self.idle_mem)

    def remove_idle(self, i, evicted=False, policy=None):
        c = self.idle.pop(i)
        self.idle_mem -= CAT.mem[c.typ]
        if self.idle_mem < 1e-9:
            self.idle_mem = 0.0
        if evicted:
            self.evictions += 1
            self._end_stay(c, 0.0, policy)
        return c

    def _end_stay(self, c, savings, policy):
        if c.stay is not None and policy is not None:
            x, t_keep, mem = c.stay
            memtime = mem * (self.now - t_keep) / 1000.0
            policy.on_stay_end(self, x, savings, memtime)
        c.stay = None

    def _release(self, c, policy):
        c.t_idle = self.now
        c.last_used = self.now
        mem = CAT.mem[c.typ]
        if mem > self.cap:
            return
        if not policy.retain(self, c):
            return
        c.stay = (policy.stay_features(self, c), self.now, mem)
        self._add_idle(c)
        while (self.idle_mem > self.cap + 1e-9 or len(self.idle) > K_MAX) and self.idle:
            i = policy.victim(self)
            self.remove_idle(i, evicted=True, policy=policy)

    def _expire(self, policy):
        ttl = getattr(policy, "ttl", None)
        if ttl is None:
            return
        i = 0
        while i < len(self.idle):
            if self.now - self.idle[i].t_idle > ttl:
                self.remove_idle(i, evicted=True, policy=policy)
            else:
                i += 1

    # ----------------------------------------------------------- main loop
    def run(self, policy):
        ep = self.ep
        policy.reset(self)
        n = len(ep["t"])
        for k in range(n):
            t, f = float(ep["t"][k]), int(ep["f"][k])
            # releases up to t
            while self.busy and self.busy[0][0] <= t:
                fin, _, c = heapq.heappop(self.busy)
                self._advance(fin)
                self._expire(policy)
                self._release(c, policy)
            self._advance(t)
            self._expire(policy)
            self.decay_rates(t)
            self.rate[f] += np.log(2) / RATE_HALF_LIFE
            self.freq[f] += 1
            inst = int(ep["inst"][k])
            self.cur_inst = inst
            self.freq_inst[inst] += 1
            for j in sorted(policy.recheck(self), reverse=True):
                self.remove_idle(j, evicted=True, policy=policy)
            a = policy.select(self, f)
            cold = True
            if a is not None and a >= 0:
                c = self.idle[a]
                lv = CAT.level[f, c.typ]
                lat = float(self.warm[f, c.typ])
                if lv == 0:
                    a = -1
            if a is not None and a >= 0:
                c = self.remove_idle(a)
                lat = float(self.warm[f, c.typ])
                if c.inst == inst:
                    lat -= CLEAN_OTHER - CLEAN_SAME
                cold = lat >= CAT.cold[f] - 1e-9
                self._end_stay(c, max(0.0, CAT.cold[f] - lat), policy)
                c.typ = f
                c.inst = inst
                c.uses += 1
            else:
                self._uid += 1
                c = Container(f, self._uid, inst)
                lat = float(CAT.cold[f])
            c.last_used = t
            fin = t + lat + float(ep["e"][k])
            self._uid += 1
            heapq.heappush(self.busy, (fin, self._uid, c))
            self.log_t.append(t); self.log_lat.append(lat); self.log_cold.append(cold)
            self.log_f.append(f); self.log_mem.append(self.idle_mem)
            policy.observe(self, f, lat, cold, a if (a is not None and a >= 0) else -1)
        # drain
        end = t
        while self.busy:
            fin, _, c = heapq.heappop(self.busy)
            end = max(end, fin)
        policy.end_episode(self)
        span = max(end, 1e-9)
        return {
            "lat": np.array(self.log_lat), "cold": np.array(self.log_cold), "f": np.array(self.log_f),
            "t": np.array(self.log_t), "mem_trace": np.array(self.log_mem),
            "mean_lat": float(np.mean(self.log_lat)), "total_lat": float(np.sum(self.log_lat)),
            "cold_starts": int(np.sum(self.log_cold)),
            "avg_mem": self.area / max(self.log_t[-1], 1e-9), "peak_mem": self.peak,
            "gb_s": self.area / 1000.0, "evictions": self.evictions,
        }


def reference_capacity(episode):
    """Peak memory of concurrently running containers when nothing is ever reused (Loose)."""
    t, f, e = episode["t"], episode["f"], episode["e"]
    ev = []
    for k in range(len(t)):
        fin = t[k] + CAT.cold[f[k]] + e[k]
        ev.append((t[k], CAT.mem[f[k]]))
        ev.append((fin, -CAT.mem[f[k]]))
    ev.sort()
    cur = peak = 0.0
    for _, m in ev:
        cur += m
        peak = max(peak, cur)
    return peak
