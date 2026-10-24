"""Sensitivity of the drift detector to its threshold: false alarms before the shift, detection rate and delay.
Uses only the function id stream, so no policy is needed."""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from amlcr.workloads import make_episode
from amlcr import policies as P

ROOT = os.path.join(os.path.dirname(__file__), "..")
SEEDS = [500000 + i for i in range(30)]
rows = []
for thr in [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
    for kind in ("drift", "mixed", "peak"):
        for s in SEEDS:
            ep = make_episode(kind, s)
            det = P.DriftDetector(thr=thr)
            for f in ep["f"]:
                det.update(int(f))
            ev = det.events
            h = len(ep["f"]) // 2
            if kind == "drift":
                after = [e for e in ev if e >= h]
                rows.append(dict(thr=thr, kind=kind, seed=s, false_alarms=sum(1 for e in ev if e < h),
                                 detected=int(bool(after)), delay=(min(after) - h) if after else np.nan))
            else:
                rows.append(dict(thr=thr, kind=kind, seed=s, false_alarms=len(ev), detected=np.nan, delay=np.nan))
df = pd.DataFrame(rows)
df.to_csv(os.path.join(ROOT, "results", "raw_detector.csv"), index=False)
g = df.groupby(["thr", "kind"]).agg(false_alarms=("false_alarms", "mean"), detected=("detected", "mean"), delay=("delay", "mean"))
print(g.round(2).to_string())
