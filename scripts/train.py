"""Offline training of the DQN schedulers.

python scripts/train.py --variant level --retention 0 --kind train --episodes 150 --out results/mlcr
python scripts/train.py --variant delta --retention 1 --kind train --episodes 150 --out results/amlcr
"""
import argparse, os, sys, time, json, pickle
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from amlcr.workloads import make_episode
from amlcr.sim import Sim, reference_capacity
from amlcr import policies as P

FRACS = [0.2, 0.35, 0.5, 0.75, 1.0]


def evaluate(pol, variant, kind, seeds, fracs=(0.2, 0.5, 1.0)):
    out = []
    for s in seeds:
        ep = make_episode(kind, s)
        ref = reference_capacity(ep)
        for fr in fracs:
            sim = Sim(ep, fr * ref, system=variant)
            r = sim.run(pol)
            out.append((r["mean_lat"], r["avg_mem"] / (fr * ref)))
    a = np.array(out)
    return a[:, 0].mean(), a[:, 1].mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="level")
    ap.add_argument("--retention", type=int, default=0)
    ap.add_argument("--kind", default="train")
    ap.add_argument("--episodes", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lr", type=float, default=5e-4)
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    ret = P.RetentionAgent(seed=a.seed, train=True) if a.retention else None
    pol = P.DQNSched(a.variant, retention=ret, train=True, seed=a.seed, eps=0.3, lr=a.lr)
    val_kind = "mixed" if a.kind == "train" else a.kind
    val_seeds = [900000 + i for i in range(5)]
    log, best, best_state = [], 1e9, None
    t0 = time.time()
    for k in range(a.episodes):
        frac_done = k / max(1, a.episodes - 1)
        pol.eps = max(0.04, 0.35 * (1 - frac_done / 0.7))
        if ret is not None:
            ret.eps = max(0.1, 0.5 * (1 - frac_done / 0.6))
            ret.lam = float(np.exp(rng.uniform(np.log(0.02), np.log(2.0))))
        ep = make_episode(a.kind, 100000 + a.seed * 100000 + k)
        ref = reference_capacity(ep)
        fr = FRACS[rng.integers(len(FRACS))]
        sim = Sim(ep, fr * ref, system=a.variant)
        r = sim.run(pol)
        log.append((k, fr, r["mean_lat"], r["avg_mem"] / (fr * ref), pol.eps, ret.lam if ret else 0.0))
        if (k + 1) % 25 == 0 or k == a.episodes - 1:
            # validation with exploration off and a fixed price
            pol_eval = P.DQNSched(a.variant, retention=None, train=False, net=pol.net)
            if ret is not None:
                ev = P.RetentionAgent(seed=0, train=False, lam=0.2)
                ev.net = ret.net
                ev.ready = True
                pol_eval.ret = ev
            vl, vm = evaluate(pol_eval, a.variant, val_kind, val_seeds)
            print("ep %d  train lat %.3f  val lat %.3f  val mem frac %.3f  eps %.2f  %.0fs" %
                  (k + 1, np.mean([x[2] for x in log[-25:]]), vl, vm, pol.eps, time.time() - t0), flush=True)
            if vl < best and frac_done > 0.5:
                best = vl
                best_state = ({kk: v.copy() for kk, v in pol.net.p.items()},
                              {kk: v.copy() for kk, v in ret.net.p.items()} if ret else None, ret.nb if ret else 0)
    if best_state is None:
        best_state = ({kk: v.copy() for kk, v in pol.net.p.items()},
                      {kk: v.copy() for kk, v in ret.net.p.items()} if ret else None, ret.nb if ret else 0)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    np.savez(a.out + "_q.npz", **best_state[0])
    if ret is not None:
        np.savez(a.out + "_ret.npz", **best_state[1])
    np.savetxt(a.out + "_trainlog.csv", np.array(log), delimiter=",",
               header="episode,cap_frac,mean_latency,mem_frac,eps,lambda", comments="")
    print("saved", a.out, "best val", best)


if __name__ == "__main__":
    main()
