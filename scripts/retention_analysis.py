"""Which containers does the retention agent keep? Counts keep decisions per function type on the test seeds
(mixed stream, Loose pool) for the latency price 0.1 and the selected memory price."""
import json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from amlcr.evalutil import make_policy
from amlcr.workloads import make_episode
from amlcr.sim import Sim, reference_capacity
from amlcr.catalog import CAT, N_FUNC

ROOT = os.path.join(os.path.dirname(__file__), "..")
CFG = json.load(open(os.path.join(ROOT, "config.json")))
LAM = json.load(open(os.path.join(ROOT, "results", "lambda_selection.json")))["lambda"]
rows = []
for lam, tag in ((0.1, "AMLCR_L"), (LAM, "AMLCR_M")):
    kept = np.zeros(N_FUNC); total = np.zeros(N_FUNC); dropped = np.zeros(N_FUNC)
    gbs = np.zeros(N_FUNC); stays = np.zeros(N_FUNC); sav = np.zeros(N_FUNC)
    for s in [500000 + i for i in range(40)]:
        ep = make_episode("mixed", s)
        pol, system = make_policy("AMLCR_C12", CFG, lam=lam)
        orig = pol.retain
        def retain(sim, c, orig=orig):
            d = orig(sim, c)
            total[c.typ] += 1
            kept[c.typ] += bool(d)
            return d
        pol.retain = retain
        orig_end = pol.on_stay_end
        def on_end(sim, x, savings, memtime, orig_end=orig_end):
            t = int(np.argmax(x[:N_FUNC]))
            gbs[t] += memtime; stays[t] += 1; sav[t] += savings
            return orig_end(sim, x, savings, memtime)
        pol.on_stay_end = on_end
        orig_re = pol.recheck
        def recheck(sim, orig_re=orig_re, **k):
            drop = orig_re(sim, **k)
            for j in drop:
                dropped[sim.idle[j].typ] += 1
            return drop
        pol.recheck = recheck
        ref = reference_capacity(ep)
        Sim(ep, 1.0 * ref, system=system).run(pol)
    for f in range(N_FUNC):
        rows.append(dict(method=tag, func=f + 1, mem_mb=float(CAT.mem[f]), cold_s=float(CAT.cold[f]), finished=total[f],
                         keep_rate=kept[f] / max(total[f], 1), recheck_drop_rate=dropped[f] / max(kept[f], 1),
                         mean_stay_s=gbs[f] / max(stays[f], 1) / (CAT.mem[f] / 1000.0), saved_s_per_stay=sav[f] / max(stays[f], 1), gb_s_total=gbs[f]))
df = pd.DataFrame(rows)
df.to_csv(os.path.join(ROOT, "results", "retention_by_type.csv"), index=False)
print(df.round(3).to_string())
