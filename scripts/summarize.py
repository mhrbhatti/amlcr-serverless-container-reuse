"""Turn the raw csv files into results/summary.json (all numbers quoted in the paper)."""
import json, os, sys
import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.join(os.path.dirname(__file__), "..")
RES = os.path.join(ROOT, "results")
ORDER = ["LRU", "FaasCache", "KeepAlive", "GreedyMatch", "GreedyDelta", "MLCR", "AMLCR_C1", "AMLCR_L", "AMLCR"]


def ci(x):
    x = np.asarray(x, float)
    if len(x) < 2:
        return 0.0
    return float(stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)))


def agg(df, by, cols=("mean_lat", "p95_lat", "cold", "avg_mem", "gb_s")):
    out = {}
    for key, g in df.groupby(by):
        k = key if isinstance(key, str) else "|".join(map(str, key))
        out[k] = {c: [float(g[c].mean()), ci(g[c])] for c in cols}
        out[k]["n"] = int(len(g))
    return out


def paired(df, a, b, col="mean_lat", cap=None):
    d = df if cap is None else df[df.cap == cap]
    x = d[d.method == a].sort_values("seed")[col].values
    y = d[d.method == b].sort_values("seed")[col].values
    red = (y - x) / y * 100.0
    try:
        p = float(stats.wilcoxon(x, y).pvalue)
    except ValueError:
        p = 1.0
    return {"mean_red_pct": float(red.mean()), "ci": ci(red), "p": p, "wins": int((x < y).sum()), "n": int(len(x))}


def main():
    S = {}
    m = pd.read_csv(os.path.join(RES, "raw_main.csv"))
    S["main"] = agg(m, ["cap", "method"])
    S["main_pairs"] = {}
    for cap in ["Tight", "Moderate", "Loose"]:
        for base in ["LRU", "FaasCache", "KeepAlive", "GreedyMatch", "MLCR"]:
            S["main_pairs"]["%s|%s" % (cap, base)] = paired(m, "AMLCR", base, cap=cap)
        S["main_pairs"]["%s|c1_vs_mlcr" % cap] = paired(m, "AMLCR_C1", "MLCR", cap=cap)
        S["main_pairs"]["%s|greedydelta_vs_greedymatch" % cap] = paired(m, "GreedyDelta", "GreedyMatch", cap=cap)
        S["main_pairs"]["%s|amlcr_mem_vs_mlcr" % cap] = paired(m, "AMLCR", "MLCR", col="avg_mem", cap=cap)
        S["main_pairs"]["%s|amlcr_mem_vs_greedy" % cap] = paired(m, "AMLCR", "GreedyMatch", col="avg_mem", cap=cap)
        S["main_pairs"]["%s|amlcr_vs_c1" % cap] = paired(m, "AMLCR", "AMLCR_C1", cap=cap)
        for who in ("AMLCR_L", "AMLCR_C1"):
            for base in ["LRU", "FaasCache", "KeepAlive", "GreedyMatch", "GreedyDelta", "MLCR"]:
                S["main_pairs"]["%s|%s|%s" % (cap, who, base)] = paired(m, who, base, cap=cap)
            S["main_pairs"]["%s|%s|mem_vs_mlcr" % (cap, who)] = paired(m, who, "MLCR", col="avg_mem", cap=cap)
        S["main_pairs"]["%s|AMLCR_L|vs_c1" % cap] = paired(m, "AMLCR_L", "AMLCR_C1", cap=cap)
        S["main_pairs"]["%s|AMLCR_L|mem_vs_c1" % cap] = paired(m, "AMLCR_L", "AMLCR_C1", col="avg_mem", cap=cap)
        S["main_pairs"]["%s|amlcr_mem_vs_c1" % cap] = paired(m, "AMLCR", "AMLCR_C1", col="avg_mem", cap=cap)
    S["main_all_caps"] = {}
    for base in ["LRU", "FaasCache", "KeepAlive", "GreedyMatch", "MLCR"]:
        x = m[m.method == "AMLCR_L"].sort_values(["cap", "seed"])["mean_lat"].values
        y = m[m.method == base].sort_values(["cap", "seed"])["mean_lat"].values
        red = (y - x) / y * 100
        S["main_all_caps"][base] = {"mean_red_pct": float(red.mean()), "max_red_pct": float(red.max())}
    # headline range as in the base paper: reduction of total startup latency per pool size
    w = pd.read_csv(os.path.join(RES, "raw_workloads.csv"))
    S["workloads"] = agg(w, ["kind", "cap", "method"])
    S["workloads_pairs"] = {}
    for kind in w.kind.unique():
        for cap in ["Tight", "Moderate", "Loose"]:
            d = w[(w.kind == kind)]
            for base in ["GreedyMatch", "MLCR", "LRU", "FaasCache", "KeepAlive"]:
                S["workloads_pairs"]["%s|%s|%s" % (kind, cap, base)] = paired(d, "AMLCR_L", base, cap=cap)
                S["workloads_pairs"]["%s|%s|M|%s" % (kind, cap, base)] = paired(d, "AMLCR", base, cap=cap)
    p = pd.read_csv(os.path.join(RES, "raw_pareto.csv"))
    S["pareto"] = agg(p, ["cap", "method", "lam"])
    if os.path.exists(os.path.join(RES, "raw_drift.csv")):
        d = pd.read_csv(os.path.join(RES, "raw_drift.csv"))
        S["drift"] = agg(d, ["cap", "method"], cols=("mean_lat", "pre", "post", "post_first", "post_last", "cold", "avg_mem"))
        for cap in ["Tight", "Moderate"]:
            for base in ["MLCR (frozen)", "AMLCR frozen", "AMLCR online"]:
                S["drift"]["%s|full_vs_%s" % (cap, base)] = {
                    "post": paired(d.assign(mean_lat=d.post), "AMLCR online+drift", base, cap=cap),
                    "post_first": paired(d.assign(mean_lat=d.post_first), "AMLCR online+drift", base, cap=cap)}
        f = d[d.method == "AMLCR online+drift"]
        S["drift"]["detector"] = {"false_alarms_mean": float(f.n_false.mean()),
                                  "detected_frac": float((f.delay >= 0).mean()),
                                  "delay_mean": float(f[f.delay >= 0].delay.mean()) if (f.delay >= 0).any() else -1}
    for name in ("raw_overhead_k.csv", "raw_overhead_e2e.csv"):
        if os.path.exists(os.path.join(RES, name)):
            S[name[4:-4]] = pd.read_csv(os.path.join(RES, name)).to_dict("records")
    json.dump(S, open(os.path.join(RES, "summary.json"), "w"), indent=1)
    print("summary written")


if __name__ == "__main__":
    main()
