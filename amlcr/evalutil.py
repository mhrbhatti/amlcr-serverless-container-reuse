"""Evaluation helpers: build every compared method from saved weights and run it."""
import numpy as np
from . import policies as P
from .sim import Sim, reference_capacity
from .workloads import make_episode


def load_q(path):
    net = P.TokenQNet(P.F_TOK, d=32, heads=2, seed=0)
    z = np.load(path)
    for k in net.p:
        net.p[k][...] = z[k]
    return net


def load_ret(path, lam=0.2, budget_mb=None):
    r = P.RetentionAgent(seed=0, train=False, lam=lam, budget_mb=budget_mb)
    z = np.load(path)
    for k in r.net.p:
        r.net.p[k][...] = z[k]
    r.ready = True
    return r


def make_policy(name, cfg, lam=0.2, budget_mb=None, online=False, detect=False, seed=0, ttl=None):
    """cfg holds file paths of the trained weights."""
    if name in ("LRU", "FaasCache", "KeepAlive", "GreedyMatch", "GreedyDelta", "ColdOnly"):
        return {"LRU": P.ExactLRU, "FaasCache": P.FaasCache, "KeepAlive": P.KeepAlive,
                "GreedyMatch": P.Greedy, "GreedyDelta": P.Greedy, "ColdOnly": P.ColdOnly}[name](), \
               ("delta" if name == "GreedyDelta" else "level")
    if name == "MLCR_TTL":
        pol = P.DQNSched("level", net=load_q(cfg["mlcr"]), seed=seed)
        pol.ttl = ttl
        return pol, "level"
    if name == "GreedyMatch_TTL":
        pol = P.Greedy()
        pol.ttl = ttl
        return pol, "level"
    if name == "MLCR":
        return P.DQNSched("level", net=load_q(cfg["mlcr"]), seed=seed), "level"
    if name == "AMLCR_C1":
        return P.DQNSched("delta", net=load_q(cfg["c1"]), seed=seed), "delta"
    if name == "AMLCR_C12":
        ret = load_ret(cfg["c12_ret"], lam=lam, budget_mb=budget_mb)
        ret.train = online
        ret.eps = 0.08 if online else 0.0
        det = P.DriftDetector() if detect else None
        pol = P.DQNSched("delta", retention=ret, net=load_q(cfg["c12_q"]), seed=seed,
                         online=online, detector=det, eps=0.02 if online else 0.0, lr=2e-4)
        pol.ret_base_eps = ret.eps
        return pol, "delta"
    raise ValueError(name)


def run_one(name, cfg, ep, cap_frac, **kw):
    pol, system = make_policy(name, cfg, **kw)
    ref = reference_capacity(ep)
    sim = Sim(ep, cap_frac * ref, system=system)
    res = sim.run(pol)
    res["cap"] = cap_frac * ref
    res["ref"] = ref
    res["policy"] = pol
    return res
