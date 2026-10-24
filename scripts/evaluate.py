"""Run every experiment on unseen test seeds and write raw csv files to results/.

python scripts/evaluate.py main workloads pareto drift overhead
"""
import json, os, sys, time
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from amlcr.evalutil import run_one, make_policy
from amlcr.workloads import make_episode
from amlcr.sim import Sim, reference_capacity
from amlcr import policies as P

ROOT = os.path.join(os.path.dirname(__file__), "..")
CFG = json.load(open(os.path.join(ROOT, "config.json")))
LAM = json.load(open(os.path.join(ROOT, "results", "lambda_selection.json")))["lambda"] \
    if os.path.exists(os.path.join(ROOT, "results", "lambda_selection.json")) else 0.45
CAPS = {"Tight": 0.2, "Moderate": 0.5, "Loose": 1.0}
TEST = [500000 + i for i in range(40)]
RES = os.path.join(ROOT, "results")


def row(exp, kind, method, cap, seed, r, **extra):
    d = dict(exp=exp, kind=kind, method=method, cap=cap, seed=seed, mean_lat=r["mean_lat"],
             p95_lat=float(np.percentile(r["lat"], 95)), cold=r["cold_starts"], avg_mem=r["avg_mem"],
             peak_mem=r["peak_mem"], gb_s=r["gb_s"], ref=r["ref"], evictions=r["evictions"])
    d.update(extra)
    return d


def exp_main():
    methods = [("LRU", "LRU", {}), ("FaasCache", "FaasCache", {}), ("KeepAlive", "KeepAlive", {}),
               ("GreedyMatch", "GreedyMatch", {}), ("GreedyDelta", "GreedyDelta", {}),
               ("MLCR", "MLCR", {}), ("AMLCR_C1", "AMLCR_C1", {}),
               ("AMLCR_L", "AMLCR_C12", {"lam": 0.1}), ("AMLCR", "AMLCR_C12", {"lam": LAM})]
    rows = []
    for s in TEST:
        ep = make_episode("mixed", s)
        for cn, fr in CAPS.items():
            for name, m, kw in methods:
                r = run_one(m, CFG, ep, fr, **kw)
                rows.append(row("main", "mixed", name, cn, s, r))
    pd.DataFrame(rows).to_csv(os.path.join(RES, "raw_main.csv"), index=False)
    print("main done", len(rows))


def exp_workloads():
    methods = [("LRU", "LRU", {}), ("FaasCache", "FaasCache", {}), ("KeepAlive", "KeepAlive", {}),
               ("GreedyMatch", "GreedyMatch", {}), ("MLCR", "MLCR", {}),
               ("AMLCR_L", "AMLCR_C12", {"lam": 0.1}), ("AMLCR", "AMLCR_C12", {"lam": LAM})]
    rows = []
    for kind in ["hisim", "losim", "peak", "uniform"]:
        for s in TEST[:30]:
            ep = make_episode(kind, s)
            for cn, fr in CAPS.items():
                for name, m, kw in methods:
                    r = run_one(m, CFG, ep, fr, **kw)
                    rows.append(row("workloads", kind, name, cn, s, r))
    pd.DataFrame(rows).to_csv(os.path.join(RES, "raw_workloads.csv"), index=False)
    print("workloads done", len(rows))


def exp_pareto():
    rows = []
    for s in TEST[:25]:
        ep = make_episode("mixed", s)
        ref = reference_capacity(ep)
        for cn, fr in CAPS.items():
            for lam in [0.0, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5]:
                r = run_one("AMLCR_C12", CFG, ep, fr, lam=lam)
                rows.append(row("pareto", "mixed", "AMLCR_lambda", cn, s, r, lam=lam))
            for ttl in [5, 10, 20, 40, 80, 160]:
                for m in ("MLCR_TTL", "GreedyMatch_TTL"):
                    r = run_one(m, CFG, ep, fr, ttl=ttl)
                    rows.append(row("pareto", "mixed", m, cn, s, r, lam=ttl))
            for phi in [0.10, 0.20, 0.30, 0.45]:
                r = run_one("AMLCR_C12", CFG, ep, fr, lam=0.3, budget_mb=phi * ref)
                lt = r["policy"].ret.lam_trace
                rows.append(row("pareto", "mixed", "AMLCR_budget", cn, s, r, lam=phi,
                                lam_final=float(lt[-1][1]) if lt else np.nan))
    pd.DataFrame(rows).to_csv(os.path.join(RES, "raw_pareto.csv"), index=False)
    print("pareto done", len(rows))


def exp_poolsweep():
    """Pool size sweep as in the base paper: 25, 50, 75 and 100 percent of the reference capacity."""
    methods = [("LRU", "LRU", {}), ("FaasCache", "FaasCache", {}), ("KeepAlive", "KeepAlive", {}),
               ("GreedyMatch", "GreedyMatch", {}), ("MLCR", "MLCR", {}), ("AMLCR_C1", "AMLCR_C1", {}),
               ("AMLCR_L", "AMLCR_C12", {"lam": 0.1}), ("AMLCR", "AMLCR_C12", {"lam": LAM})]
    rows = []
    for s in TEST[:20]:
        ep = make_episode("mixed", s)
        for fr in (0.25, 0.5, 0.75, 1.0):
            for name, m, kw in methods:
                r = run_one(m, CFG, ep, fr, **kw)
                rows.append(row("poolsweep", "mixed", name, fr, s, r))
    pd.DataFrame(rows).to_csv(os.path.join(RES, "raw_poolsweep.csv"), index=False)
    print("poolsweep done", len(rows))


def exp_drift():
    cfgA = dict(CFG)
    cfgA["mlcr"] = CFG["mlcr_A"]
    cfgA["c12_q"] = CFG["c12A_q"]
    cfgA["c12_ret"] = CFG["c12A_ret"]
    variants = [("MLCR (frozen)", "MLCR", {}),
                ("AMLCR frozen", "AMLCR_C12", {"lam": LAM}),
                ("AMLCR online", "AMLCR_C12", {"lam": LAM, "online": True}),
                ("AMLCR online+drift", "AMLCR_C12", {"lam": LAM, "online": True, "detect": True})]
    rows, traces = [], {}
    for s in TEST[:30]:
        ep = make_episode("drift", s)
        for cn in ("Tight", "Moderate"):
            fr = CAPS[cn]
            for label, m, kw in variants:
                r = run_one(m, cfgA, ep, fr, **kw)
                lat = r["lat"]
                n = len(lat)
                h = n // 2
                pol = r["policy"]
                ev = list(pol.detector.events) if getattr(pol, "detector", None) is not None else []
                rows.append(row("drift", "drift", label, cn, s, r,
                                pre=float(lat[:h].mean()), post=float(lat[h:].mean()),
                                post_first=float(lat[h:h + 100].mean()), post_last=float(lat[-150:].mean()),
                                n_false=int(sum(1 for e in ev if e < h)),
                                delay=(min([e for e in ev if e >= h]) - h if any(e >= h for e in ev) else -1)))
                traces.setdefault((label, cn), []).append(lat)
    pd.DataFrame(rows).to_csv(os.path.join(RES, "raw_drift.csv"), index=False)
    np.savez(os.path.join(RES, "drift_traces.npz"),
             **{("%s|%s" % k): np.array(v) for k, v in traces.items()})
    print("drift done", len(rows))


def exp_overhead():
    rows = []
    # decision time of one scheduling step, token network versus a flat network of the base paper style
    rng = np.random.default_rng(0)
    for K in [8, 16, 24, 48, 96, 192]:
        net = P.TokenQNet(P.F_TOK, seed=0)
        X = rng.standard_normal((1, K + 1, P.F_TOK)).astype(np.float32)
        M = np.ones((1, K + 1), bool)
        for _ in range(20):
            net.forward(X, M)
        t0 = time.perf_counter()
        for _ in range(300):
            net.forward(X, M)
        tok = (time.perf_counter() - t0) / 300 * 1000
        # flat network: embed the concatenated state, 512 wide, then one output per action
        din = (K + 1) * P.F_TOK
        W1 = rng.standard_normal((din, 512)).astype(np.float32) * 0.01
        W2 = rng.standard_normal((512, K + 1)).astype(np.float32) * 0.01
        x = X.reshape(1, -1)
        for _ in range(20):
            np.maximum(x @ W1, 0) @ W2
        t0 = time.perf_counter()
        for _ in range(300):
            np.maximum(x @ W1, 0) @ W2
        flat = (time.perf_counter() - t0) / 300 * 1000
        rows.append(dict(K=K, token_ms=tok, flat_ms=flat,
                         token_params=int(sum(v.size for v in net.p.values())), flat_params=int(W1.size + W2.size)))
    pd.DataFrame(rows).to_csv(os.path.join(RES, "raw_overhead_k.csv"), index=False)
    # end to end decision time inside the simulator
    out = []
    for m, kw in [("GreedyMatch", {}), ("MLCR", {}), ("AMLCR_C12", {"lam": LAM}),
                  ("AMLCR_C12", {"lam": LAM, "online": True, "detect": True})]:
        ts = []
        for s in TEST[:10]:
            ep = make_episode("mixed", s)
            t0 = time.perf_counter()
            r = run_one(m, CFG, ep, 0.5, **kw)
            ts.append((time.perf_counter() - t0) / len(ep["t"]) * 1000)
        tag = m + ("+online" if kw.get("online") else "")
        out.append(dict(method=tag, ms_per_invocation=float(np.mean(ts)), sd=float(np.std(ts))))
    pd.DataFrame(out).to_csv(os.path.join(RES, "raw_overhead_e2e.csv"), index=False)
    print("overhead done")


if __name__ == "__main__":
    todo = sys.argv[1:] or ["main", "workloads", "pareto", "drift", "overhead"]
    for t in todo:
        {"main": exp_main, "workloads": exp_workloads, "pareto": exp_pareto,
         "drift": exp_drift, "overhead": exp_overhead, "poolsweep": exp_poolsweep}[t]()
