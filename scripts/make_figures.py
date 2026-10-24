"""Build every figure from the csv files in results/. Run after evaluate.py."""
import json, os, re, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from amlcr.catalog import CAT

ROOT = os.path.join(os.path.dirname(__file__), "..")
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({
    "font.family": ["Carlito", "DejaVu Sans"], "font.size": 7.5, "axes.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "axes.labelsize": 7.5, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "legend.fontsize": 6.8, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "axes.titlesize": 7.8, "axes.titleweight": "bold",
    "figure.dpi": 100, "savefig.dpi": 300, "axes.grid": True, "grid.color": "#e3e6ea", "grid.linewidth": 0.5,
    "axes.axisbelow": True,
})
COL = {"LRU": "#b3bbc4", "FaasCache": "#8e99a5", "KeepAlive": "#6b7886", "GreedyMatch": "#6aa0cf",
       "GreedyDelta": "#2f6396", "MLCR": "#1f2f44", "AMLCR_C1": "#e8b04a", "AMLCR_L": "#e07b39", "AMLCR": "#9c3d08"}
NAME = {"LRU": "LRU", "FaasCache": "FaasCache", "KeepAlive": "KeepAlive", "GreedyMatch": "Greedy Match",
        "GreedyDelta": "Greedy Delta", "MLCR": "MLCR", "AMLCR_C1": "AMLCR C1", "AMLCR_L": "AMLCR L",
        "AMLCR": "AMLCR M"}
ORDER = ["LRU", "FaasCache", "KeepAlive", "GreedyMatch", "GreedyDelta", "MLCR", "AMLCR_C1", "AMLCR_L", "AMLCR"]
CAPS = ["Tight", "Moderate", "Loose"]
W1, W2 = 3.45, 7.1


def ci(x):
    x = np.asarray(x, float)
    return stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0


def save(fig, name):
    p = os.path.join(FIG, name)
    fig.savefig(p, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print("saved", p)


def panel_tag(ax, t):
    ax.text(-0.02, 1.07, t, transform=ax.transAxes, fontsize=8.5, fontweight="bold", va="bottom", ha="right")


# ---------------------------------------------------------------- Fig: latency tables
def fig_heatmap():
    fig, axs = plt.subplots(1, 2, figsize=(W2, 2.95), gridspec_kw={"wspace": 0.12})
    rl = CAT.warm_level / CAT.cold[:, None]
    rd = CAT.warm_delta / CAT.cold[:, None]
    im = None
    for ax, M, title in zip(axs, (rl, rd), ("(a) Level granular repack", "(b) Package granular repack")):
        im = ax.imshow(M, cmap="YlGnBu_r", vmin=0, vmax=1, aspect="equal")
        ax.set_xticks(range(13)); ax.set_yticks(range(13))
        ax.set_xticklabels(["F%d" % (i + 1) for i in range(13)], fontsize=6)
        ax.set_yticklabels(["F%d" % (i + 1) for i in range(13)], fontsize=6)
        ax.grid(False)
        ax.set_xlabel("Container type held in the pool")
        for i in range(13):
            for j in range(13):
                v = M[i, j]
                ax.text(j, i, "%d" % round(100 * v), ha="center", va="center", fontsize=4.6,
                        color="white" if v < 0.45 else "#222")
        ax.set_title(title, loc="left")
        for sp in ax.spines.values():
            sp.set_visible(False)
    axs[0].set_ylabel("Requested function")
    axs[1].set_yticklabels([])
    # outline the cells that gain from the package granular repack
    for i in range(13):
        for j in range(13):
            if rl[i, j] - rd[i, j] > 0.02:
                axs[1].add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec="#c4620a", lw=1.2))
    cb = fig.colorbar(im, ax=axs, fraction=0.018, pad=0.015)
    cb.set_label("Warm start latency as % of cold start (cell labels)", fontsize=6.8)
    cb.ax.tick_params(labelsize=6)
    save(fig, "fig_heatmap.png")


# ---------------------------------------------------------------- Fig: training
def parse_val(path):
    pts = []
    for line in open(path):
        m = re.match(r"ep (\d+)\s+train lat ([\d.]+)\s+val lat ([\d.]+)", line)
        if m:
            pts.append((int(m.group(1)), float(m.group(3))))
    return np.array(pts)


def fig_training():
    from amlcr.runner import run_baseline
    from amlcr.workloads import make_episode
    vals = {}
    for nm in ("GreedyMatch", "GreedyDelta"):
        L = []
        for s in [900000 + i for i in range(5)]:
            ep = make_episode("mixed", s)
            for fr in (0.2, 0.5, 1.0):
                L.append(run_baseline(nm, ep, fr)["mean_lat"])
        vals[nm] = np.mean(L)
    fig, ax = plt.subplots(figsize=(W1, 2.35))
    for key, lab, c in (("mlcr", "MLCR", COL["MLCR"]), ("c1", "AMLCR C1", COL["AMLCR_C1"]), ("c12", "AMLCR", COL["AMLCR"])):
        p = os.path.join(RES, "models", key + ".log")
        if os.path.exists(p):
            a = parse_val(p)
            ax.plot(a[:, 0], a[:, 1], "-o", ms=2.8, lw=1.2, color=c, label=lab)
    ax.axhline(vals["GreedyMatch"], color=COL["GreedyMatch"], ls="--", lw=1)
    ax.axhline(vals["GreedyDelta"], color=COL["GreedyDelta"], ls="--", lw=1)
    ax.text(201, vals["GreedyMatch"], " Greedy Match", fontsize=6, va="bottom", ha="right", color=COL["GreedyMatch"])
    ax.text(201, vals["GreedyDelta"], " Greedy Delta", fontsize=6, va="top", ha="right", color=COL["GreedyDelta"])
    ax.set_xlabel("Training episodes")
    ax.set_ylabel("Validation startup latency (s)")
    ax.legend(frameon=False, loc="upper right", ncol=3, bbox_to_anchor=(1.0, 1.16), columnspacing=1.0, handlelength=1.4)
    save(fig, "fig_training.png")


# ---------------------------------------------------------------- Fig: main comparison
def fig_main():
    m = pd.read_csv(os.path.join(RES, "raw_main.csv"))
    fig, axs = plt.subplots(1, 3, figsize=(W2, 2.55), gridspec_kw={"wspace": 0.28})
    specs = [("mean_lat", "Mean startup latency (s)", 1.0), ("cold", "Cold starts per 600 requests", 1.0),
             ("avg_mem", "Average idle pool memory (GB)", 1000.0)]
    bw = 0.088
    for ax, (col, ylab, div), tag in zip(axs, specs, "abc"):
        for k, name in enumerate(ORDER):
            mu, er = [], []
            for cap in CAPS:
                v = m[(m.method == name) & (m.cap == cap)][col].values / div
                mu.append(v.mean()); er.append(ci(v))
            x = np.arange(3) + (k - 4) * bw
            ax.bar(x, mu, bw * 0.92, yerr=er, color=COL[name], error_kw=dict(lw=0.6, capsize=1.0, capthick=0.6),
                   label=NAME[name], linewidth=0)
        ax.set_xticks(range(3)); ax.set_xticklabels(CAPS)
        ax.set_ylabel(ylab)
        ax.grid(axis="x", visible=False)
        panel_tag(ax, "(%s)" % tag)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=9, frameon=False, bbox_to_anchor=(0.5, 1.07), columnspacing=1.2,
               handlelength=1.1, handletextpad=0.4)
    save(fig, "fig_main.png")


# ---------------------------------------------------------------- Fig: workloads and Pareto
def fig_pareto():
    w = pd.read_csv(os.path.join(RES, "raw_workloads.csv"))
    p = pd.read_csv(os.path.join(RES, "raw_pareto.csv"))
    m = pd.read_csv(os.path.join(RES, "raw_main.csv"))
    fig, axs = plt.subplots(1, 3, figsize=(W2, 2.5), gridspec_kw={"wspace": 0.3, "width_ratios": [1.15, 1, 1]})
    ax = axs[0]
    kinds = [("hisim", "HI Sim"), ("losim", "LO Sim"), ("peak", "Peak"), ("uniform", "Uniform")]
    meths = ["LRU", "FaasCache", "KeepAlive", "GreedyMatch", "MLCR", "AMLCR_L", "AMLCR"]
    bw = 0.12
    for k, name in enumerate(meths):
        mu, er = [], []
        for kd, _ in kinds:
            v = w[(w.kind == kd) & (w.cap == "Moderate") & (w.method == name)].mean_lat.values
            mu.append(v.mean()); er.append(ci(v))
        ax.bar(np.arange(4) + (k - 3) * bw, mu, bw * 0.92, yerr=er, color=COL[name], linewidth=0,
               error_kw=dict(lw=0.6, capsize=1.0, capthick=0.6), label=NAME[name])
    ax.set_xticks(range(4)); ax.set_xticklabels([n for _, n in kinds])
    ax.set_ylabel("Mean startup latency (s)")
    ax.grid(axis="x", visible=False)
    ax.set_ylim(0, 4.9)
    ax.legend(frameon=False, ncol=3, loc="upper left", columnspacing=0.7, handlelength=0.9, fontsize=5.6,
              labelspacing=0.25, borderaxespad=0.1)
    panel_tag(ax, "(a)")
    for ax, cap, tag in ((axs[1], "Moderate", "(b)"), (axs[2], "Loose", "(c)")):
        for meth, c, lab, mk in (("GreedyMatch_TTL", COL["GreedyMatch"], "Greedy Match + TTL", "s"),
                                 ("MLCR_TTL", COL["MLCR"], "MLCR + TTL", "D"),
                                 ("AMLCR_lambda", COL["AMLCR"], "AMLCR price sweep", "o")):
            d = p[(p.method == meth) & (p.cap == cap)]
            xs, ys = [], []
            for lam, g in d.groupby("lam"):
                xs.append(g.avg_mem.mean() / 1000); ys.append(g.mean_lat.mean())
            o = np.argsort(xs)
            ax.plot(np.array(xs)[o], np.array(ys)[o], "-" + mk, color=c, ms=3, lw=1.1, label=lab)
        d = p[(p.method == "AMLCR_budget") & (p.cap == cap)]
        xs = [g.avg_mem.mean() / 1000 for _, g in d.groupby("lam")]
        ys = [g.mean_lat.mean() for _, g in d.groupby("lam")]
        ax.plot(xs, ys, "^", color="#7a2e00", ms=3.8, label="AMLCR budget control")
        for name in ("LRU", "FaasCache", "KeepAlive"):
            g = m[(m.method == name) & (m.cap == cap)]
            ax.plot(g.avg_mem.mean() / 1000, g.mean_lat.mean(), "x", color=COL[name], ms=4, mew=1.2,
                    label=NAME[name] if cap == "Moderate" else None)
        ax.set_xlabel("Average idle pool memory (GB)")
        ax.set_ylabel("Mean startup latency (s)")
        ax.set_title(cap + " pool", loc="right", fontsize=7)
        panel_tag(ax, tag)
    h, l = axs[1].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, loc="lower center", ncol=7, fontsize=6.0, handlelength=1.4,
               columnspacing=1.0, bbox_to_anchor=(0.5, -0.16))
    save(fig, "fig_pareto.png")


# ---------------------------------------------------------------- Fig: drift
def fig_drift():
    z = np.load(os.path.join(RES, "drift_traces.npz"))
    fig, ax = plt.subplots(figsize=(W1, 2.45))
    cols = {"MLCR (frozen)": COL["MLCR"], "AMLCR frozen": "#e0a45a", "AMLCR online": "#7c9fc4",
            "AMLCR online+drift": COL["AMLCR"]}
    labs = {"MLCR (frozen)": "MLCR frozen", "AMLCR frozen": "AMLCR frozen", "AMLCR online": "AMLCR online",
            "AMLCR online+drift": "AMLCR online and drift response"}
    win = 40
    for key in cols:
        a = z[key + "|Moderate"].mean(0)
        sm = np.convolve(a, np.ones(win) / win, mode="valid")
        ax.plot(np.arange(win - 1, len(a)) + 1, sm, color=cols[key], lw=1.3, label=labs[key])
    ax.axvline(300, color="#444", ls=":", lw=1)
    ax.set_ylim(0.5, 6.4)
    ax.text(304, 0.62, "mix shift", fontsize=6.5, va="bottom")
    ax.set_xlabel("Invocation index")
    ax.set_ylabel("Startup latency, 40 request mean (s)")
    ax.legend(frameon=False, loc="upper left", ncol=1, fontsize=6.0, handlelength=1.4, labelspacing=0.3)
    save(fig, "fig_drift.png")


# ---------------------------------------------------------------- Fig: budget control
def fig_budget():
    sys.path.insert(0, ROOT)
    from amlcr.evalutil import run_one
    from amlcr.workloads import make_episode
    from amlcr.sim import reference_capacity
    cfg = json.load(open(os.path.join(ROOT, "config.json")))
    lam0 = json.load(open(os.path.join(RES, "lambda_selection.json")))["lambda"]
    ep = make_episode("mixed", 500003)
    ref = reference_capacity(ep)
    fig, ax = plt.subplots(figsize=(W1, 2.35))
    ax2 = ax.twinx()
    ax2.spines["right"].set_visible(True)
    ax2.grid(False)
    for phi, c in ((0.10, "#7a2e00"), (0.25, "#d9822b")):
        r = run_one("AMLCR_C12", cfg, ep, 1.0, lam=lam0, budget_mb=phi * ref)
        tr = np.array(r["policy"].ret.lam_trace)
        ax.plot(tr[:, 0], tr[:, 2] / ref, color=c, lw=1.2, label="occupancy, budget %.2f" % phi)
        ax.axhline(phi, color=c, lw=0.8, ls="--")
        ax2.plot(tr[:, 0], tr[:, 1], color=c, lw=0.9, ls=":", label="price, budget %.2f" % phi)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Idle memory / reference (solid)")
    ax2.set_ylabel("Memory price lambda (dotted)")
    ax.legend(frameon=False, loc="lower center", ncol=2, fontsize=6.2, bbox_to_anchor=(0.5, 1.0))
    save(fig, "fig_budget.png")


# ---------------------------------------------------------------- Fig: pool size sweep
def fig_poolsweep():
    d = pd.read_csv(os.path.join(RES, "raw_poolsweep.csv"))
    fig, axs = plt.subplots(1, 2, figsize=(W1 * 2 * 0.9, 2.35), gridspec_kw={"wspace": 0.28})
    meths = ["LRU", "FaasCache", "KeepAlive", "GreedyMatch", "MLCR", "AMLCR_C1", "AMLCR_L", "AMLCR"]
    mk = {"LRU": "o", "FaasCache": "s", "KeepAlive": "^", "GreedyMatch": "v", "MLCR": "D", "AMLCR_C1": "P",
          "AMLCR_L": "o", "AMLCR": "X"}
    for ax, col, lab in ((axs[0], "mean_lat", "Mean startup latency (s)"), (axs[1], "avg_mem", "Average idle pool memory (GB)")):
        for m in meths:
            g = d[d.method == m]
            xs = sorted(g.cap.unique())
            mu = [g[g.cap == x][col].mean() / (1000 if col == "avg_mem" else 1) for x in xs]
            er = [ci(g[g.cap == x][col]) / (1000 if col == "avg_mem" else 1) for x in xs]
            ax.errorbar([x * 100 for x in xs], mu, yerr=er, color=COL[m], marker=mk[m], ms=3.2, lw=1.1, capsize=1.2,
                        elinewidth=0.6, label=NAME[m])
        ax.set_xlabel("Pool size (% of reference capacity)")
        ax.set_ylabel(lab)
        ax.set_xticks([25, 50, 75, 100])
    panel_tag(axs[0], "(a)"); panel_tag(axs[1], "(b)")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, loc="lower center", ncol=8, fontsize=6.2, handlelength=1.4, columnspacing=0.9,
               bbox_to_anchor=(0.5, -0.2))
    save(fig, "fig_poolsweep.png")


# ---------------------------------------------------------------- Fig: what the retention agent trims
def fig_retention():
    d = pd.read_csv(os.path.join(RES, "retention_by_type.csv"))
    fig, ax = plt.subplots(figsize=(W1, 2.3))
    x = np.arange(13)
    for k, (m, c) in enumerate((("AMLCR_L", COL["AMLCR_L"]), ("AMLCR_M", COL["AMLCR"]))):
        g = d[d.method == m].sort_values("func")
        ax.bar(x + (k - 0.5) * 0.38, g.gb_s_total.values / 1000.0, 0.36, color=c, linewidth=0,
               label="AMLCR L" if m == "AMLCR_L" else "AMLCR M")
    ax.set_xticks(x)
    ax.set_xticklabels(["F%d" % (i + 1) for i in range(13)], fontsize=6.3)
    ax.set_ylabel("Idle memory time (10$^3$ GB s)")
    ax.set_xlabel("Function type of the container")
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, loc="upper left")
    save(fig, "fig_retention.png")


if __name__ == "__main__":
    todo = sys.argv[1:] or ["heatmap", "training", "main", "pareto", "drift", "budget", "poolsweep", "retention"]
    fns = {"heatmap": fig_heatmap, "training": fig_training, "main": fig_main, "pareto": fig_pareto,
           "drift": fig_drift, "budget": fig_budget, "poolsweep": fig_poolsweep, "retention": fig_retention}
    for t in todo:
        fns[t]()
