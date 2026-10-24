"""Architecture figure with dotted boxes around the three novel components."""
import os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

OUT = sys.argv[1] if len(sys.argv) > 1 else "figures/fig_architecture.png"
plt.rcParams.update({"font.family": ["Carlito", "DejaVu Sans"], "font.size": 7.5})

C1, C2, C3 = "#1f5fa8", "#c4620a", "#2c7a4b"
GREY_F, GREY_E = "#eef1f5", "#59636e"
INK = "#2b2f36"

fig, ax = plt.subplots(figsize=(7.2, 3.55))
ax.set_xlim(-1, 101.5)
ax.set_ylim(-2.5, 46)
ax.axis("off")


def box(x, y, w, h, text, fc=GREY_F, ec=GREY_E, fs=6.7):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2,rounding_size=0.9", fc=fc, ec=ec, lw=0.9))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, linespacing=1.25, color=INK)


def dotted(x, y, w, h, color, tag):
    ax.add_patch(Rectangle((x, y), w, h, fill=False, ec=color, lw=1.5, ls=(0, (2.4, 2.0))))
    ax.text(x + 0.8, y + h - 0.6, tag, ha="left", va="top", fontsize=7.5, color="white", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.22", fc=color, ec=color))


def arrow(x1, y1, x2, y2, color=INK, ls="-", style="-|>"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=0.95, ls=ls, shrinkA=0, shrinkB=0,
                                mutation_scale=8))


def label(x, y, s, ha="center", fs=6.8):
    ax.text(x, y, s, fontsize=fs, ha=ha, va="center", color=INK, linespacing=1.15)


ym, hm = 24, 10
yc = ym + hm / 2
# ---------------------------------------------------------------- main pipeline
box(1, ym, 14, hm, "Invocation\nstream\n(function, time)")
box(20.5, ym, 18, hm, "State builder\nfunction token and\nidle container tokens\nwith delta features", fc="#e3edf8", ec=C1)
box(43.5, ym, 17, hm, "Token attention\nQ network with\nvalidity mask")
box(65.5, ym, 14.5, hm, "Container cleaner\nand package\ndelta repack", fc="#e3edf8", ec=C1)
box(85, ym, 14, hm, "Warm pool\nidle containers\nmemory cap")
arrow(15.4, yc, 20.1, yc)
arrow(38.9, yc, 43.1, yc)
arrow(60.9, yc, 65.1, yc)
arrow(80.4, yc, 84.6, yc)
label(52.0, 37.0, "action: reuse slot\nor cold start")

# ---------------------------------------------------------------- pool state returns to the state builder
ax.plot([92, 92], [ym + hm + 0.3, 40], color="#7b8794", lw=0.9, ls=(0, (3, 2)))
ax.plot([92, 29.5], [40, 40], color="#7b8794", lw=0.9, ls=(0, (3, 2)))
arrow(29.5, 40, 29.5, ym + hm + 0.3, color="#7b8794", ls=(0, (3, 2)))
label(61, 42.3, "pool state: idle containers, occupancy, arrival rates")

# ---------------------------------------------------------------- bottom row
yb, hb = 3, 10
box(1, yb, 16, hb, "Drift detector\nJS divergence of\nrecent function mix", fc="#e1f1e8", ec=C3)
box(21, yb, 17, hb, "Online adaptation\nexploration, step\nsize, retention probes", fc="#e1f1e8", ec=C3)
box(43.5, yb, 17, hb, "Replay buffer and\nlearner (Huber loss,\ntarget network)")
box(65.5, yb, 14.5, hb, "Memory price\nlambda by dual\nascent on budget", fc="#fbeadb", ec=C2)
box(85, yb, 14, hb, "Retention agent\nsaving and\nmemory time heads", fc="#fbeadb", ec=C2)
arrow(8.5, ym - 0.3, 8.5, yb + hb + 0.3)
label(12.8, 18.7, "function ids", fs=6.8)
arrow(17.4, yb + hb / 2, 20.6, yb + hb / 2)
arrow(38.4, yb + hb / 2, 43.1, yb + hb / 2)
arrow(47.0, yb + hb + 0.3, 47.0, ym - 0.3)
label(51.2, 18.7, "weights", fs=6.8)
# reward feedback
ax.plot([72.7, 72.7], [ym - 0.3, 19.2], color=INK, lw=0.95)
ax.plot([72.7, 56.5], [19.2, 19.2], color=INK, lw=0.95)
arrow(56.5, 19.2, 56.5, yb + hb + 0.3)
label(63.0, 20.7, "startup latency reward", fs=6.8)
arrow(80.4, yb + hb / 2, 84.6, yb + hb / 2)
label(82.5, yb + hb / 2 + 1.8, "price", fs=6.4)
arrow(92, yb + hb + 0.3, 92, ym - 0.3)
label(84.2, 18.7, "keep or evict,\nvictim ranking", fs=6.8)
# adaptation also acts on the policy network (drawn as a short note arrow)
# ---------------------------------------------------------------- dotted novelty boxes
dotted(19.0, ym - 1.5, 21.2, hm + 3.0, C1, "C1")
dotted(64.0, ym - 1.5, 17.6, hm + 3.0, C1, "C1")
dotted(-0.4, yb - 1.5, 39.8, hb + 3.0, C3, "C3")
dotted(64.0, yb - 1.5, 36.0, hb + 3.0, C2, "C2")

# ---------------------------------------------------------------- legend
ly = -2.0
for x0, col, txt in [(1, C1, "C1  package granular matching and repack"),
                     (37, C2, "C2  retention agent with memory price"),
                     (71, C3, "C3  drift aware online adaptation")]:
    ax.add_patch(Rectangle((x0, ly - 0.6), 3.0, 1.5, fill=False, ec=col, lw=1.4, ls=(0, (2.4, 2.0))))
    ax.text(x0 + 4.0, ly + 0.15, txt, fontsize=6.8, va="center", color=INK)

os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
fig.savefig(OUT, dpi=300, bbox_inches="tight", pad_inches=0.04)
print("saved", OUT)
