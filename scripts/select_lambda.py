"""Choose the memory price lambda on validation seeds only (never on test seeds).

Rule: the largest lambda whose mean validation latency, averaged over the three pool sizes,
stays within 3 percent of the latency obtained at lambda = 0.1.
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from amlcr.evalutil import run_one
from amlcr.workloads import make_episode

CFG = json.load(open(os.path.join(os.path.dirname(__file__), "..", "config.json")))
VAL = [900000 + i for i in range(12)]
GRID = [0.1, 0.2, 0.3, 0.45, 0.6, 0.8, 1.0, 1.3]


def main():
    res = {}
    for lam in GRID:
        L, M = [], []
        for s in VAL:
            ep = make_episode("mixed", s)
            for fr in (0.2, 0.5, 1.0):
                r = run_one("AMLCR_C12", CFG, ep, fr, lam=lam)
                L.append(r["mean_lat"]); M.append(r["avg_mem"] / r["ref"])
        res[lam] = (float(np.mean(L)), float(np.mean(M)))
        print("lam %.2f  val latency %.3f  mem frac %.3f" % (lam, *res[lam]), flush=True)
    base = res[0.1][0]
    ok = [l for l in GRID if res[l][0] <= 1.03 * base]
    lam_sel = max(ok)
    json.dump({"lambda": lam_sel, "grid": {str(k): v for k, v in res.items()}},
              open(os.path.join(os.path.dirname(__file__), "..", "results", "lambda_selection.json"), "w"), indent=1)
    print("selected lambda", lam_sel)


if __name__ == "__main__":
    main()
