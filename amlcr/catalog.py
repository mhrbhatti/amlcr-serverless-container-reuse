"""Function catalog and startup latency model.

The 13 functions follow Table II of the FStartBench paper (Zhou et al., IPDPS 2024).
Package sizes for functions 2 to 5 (Debian Python Flask, Flask+Numpy, Flask+Numpy+Pandas,
Debian Java SpringBoot, Alpine Go Gin) are anchored to the image sizes printed in Fig. 1 of
that paper (124, 202, 246, 313 and 15 MB). The remaining sizes are estimates made by the
authors from typical public image footprints and are plain parameters that can be replaced
by measured values.
"""
import numpy as np

# ---------------------------------------------------------------- package sizes (MB)
PKG = {
    # OS level
    "alpine": 8, "debian": 75, "centos": 200,
    # language level
    "python": 35, "java_alpine": 120, "java_debian": 150, "node": 45, "go": 4, "cpp": 60,
    # runtime level
    "flask": 14, "numpy": 78, "pandas": 44, "matplotlib": 38, "tensorflow": 420,
    "springboot": 88, "express": 12, "gin": 3, "commlib": 10, "imglib": 25, "weblib": 6,
}
OS_ID = {"alpine": 0, "debian": 1, "centos": 2}
LANG_ID = {"python": 0, "java": 1, "node": 2, "go": 3, "cpp": 4}

# name, os, language package, runtime packages, language family, median exec time (s), extra init (s)
_FUNCS = [
    ("F1 hello java",     "alpine", "java_alpine", ["springboot"],                         "java",   0.06, 0.0),
    ("F2 hello node",     "alpine", "node",        ["express"],                            "node",   0.05, 0.0),
    ("F3 hello go",       "alpine", "go",          ["gin"],                                "go",     0.04, 0.0),
    ("F4 hello python",   "alpine", "python",      ["flask"],                              "python", 0.05, 0.0),
    ("F5 hello python",   "debian", "python",      ["flask"],                              "python", 0.05, 0.0),
    ("F6 analytics np",   "debian", "python",      ["flask", "numpy"],                     "python", 0.60, 0.0),
    ("F7 analytics pd",   "debian", "python",      ["flask", "numpy", "pandas"],           "python", 0.90, 0.0),
    ("F8 analytics plot", "debian", "python",      ["flask", "numpy", "pandas", "matplotlib"], "python", 1.40, 0.0),
    ("F9 comm cpp",       "centos", "cpp",         ["commlib"],                            "cpp",    0.30, 0.0),
    ("F10 alu python",    "debian", "python",      ["flask"],                              "python", 0.80, 0.0),
    ("F11 web node",      "alpine", "node",        ["express", "weblib"],                  "node",   0.10, 0.0),
    ("F12 image java",    "alpine", "java_alpine", ["springboot", "imglib"],               "java",   1.00, 0.0),
    ("F13 ml python",     "debian", "python",      ["flask", "tensorflow"],                "python", 1.50, 1.5),
]
N_FUNC = len(_FUNCS)

# short display labels used in plots
LABELS = ["F%d" % (i + 1) for i in range(N_FUNC)]

# ---------------------------------------------------------------- latency model constants
SANDBOX_CREATE = 0.55      # s, create and launch the sandbox
PULL_BW = 60.0             # MB/s effective code pull bandwidth
PULL_FIXED = 0.08          # s per package request
INSTALL_PER_MB = 0.003     # s per MB for extraction and installation
CLEAN_OTHER = 0.12         # s, volume unmount and mount when a container serves another function
CLEAN_SAME = 0.02          # s, container reused by the same function
LANG_INIT = {"python": 0.25, "java": 1.40, "node": 0.20, "go": 0.05, "cpp": 0.05}
RUNTIME_INIT_PER_MB = 0.0012  # s per MB of runtime packages loaded at function initialisation


class Catalog:
    def __init__(self):
        n = N_FUNC
        self.n = n
        self.names = [f[0] for f in _FUNCS]
        self.os = np.array([OS_ID[f[1]] for f in _FUNCS])
        self.lang = np.array([LANG_ID[f[4]] for f in _FUNCS])
        self.exec_median = np.array([f[5] for f in _FUNCS])
        self.extra_init = np.array([f[6] for f in _FUNCS])
        self.pk_os = [{f[1]} for f in _FUNCS]
        self.pk_lang = [{f[2]} for f in _FUNCS]
        self.pk_rt = [set(f[3]) for f in _FUNCS]
        self.sz_os = np.array([sum(PKG[p] for p in s) for s in self.pk_os], float)
        self.sz_lang = np.array([sum(PKG[p] for p in s) for s in self.pk_lang], float)
        self.sz_rt = np.array([sum(PKG[p] for p in s) for s in self.pk_rt], float)
        self.sz_total = self.sz_os + self.sz_lang + self.sz_rt
        langname = [f[4] for f in _FUNCS]
        self.resident = np.array([{"python": 60, "java": 220, "node": 50, "go": 20, "cpp": 25}[l] for l in langname], float)
        self.resident += np.where(self.extra_init > 0, 380.0, 0.0)
        self.mem = self.sz_total + self.resident          # MB held while the container is idle
        self.lang_init = np.array([LANG_INIT[l] for l in langname])
        self.init = self.lang_init + RUNTIME_INIT_PER_MB * self.sz_rt + self.extra_init

        def pull(size, count):
            return size / PULL_BW + PULL_FIXED * count + INSTALL_PER_MB * size

        self._pull = pull
        # cold start latency
        self.cold = np.zeros(n)
        for f in range(n):
            cnt = 1 + 1 + len(self.pk_rt[f])
            self.cold[f] = SANDBOX_CREATE + pull(self.sz_total[f], cnt) + self.init[f]
        # pair tables indexed [function, container type]
        self.level = np.zeros((n, n), int)      # 0 none, 1 OS, 2 OS+language, 3 full
        self.jac = np.zeros((n, n))
        self.delta_mb = np.zeros((n, n))        # MB that must be pulled when package volumes are shared
        self.level_mb = np.zeros((n, n))        # MB pulled when the whole unmatched level is reinstalled
        self.warm_level = np.zeros((n, n))      # warm cost of the baseline level granular cleaner
        self.warm_delta = np.zeros((n, n))      # warm cost of the package granular cleaner
        for f in range(n):
            for c in range(n):
                lv = 0
                if self.os[f] == self.os[c] and self.pk_os[f] == self.pk_os[c]:
                    lv = 1
                    if self.pk_lang[f] == self.pk_lang[c]:
                        lv = 2
                        if self.pk_rt[f] == self.pk_rt[c]:
                            lv = 3
                self.level[f, c] = lv
                pf = self.pk_os[f] | self.pk_lang[f] | self.pk_rt[f]
                pc = self.pk_os[c] | self.pk_lang[c] | self.pk_rt[c]
                self.jac[f, c] = len(pf & pc) / len(pf | pc)
                clean = CLEAN_OTHER   # same instance bonus is applied in the simulator
                # level granular: matched levels are kept, unmatched levels are reinstalled in full
                mb, cnt = 0.0, 0
                if lv < 2:
                    mb += self.sz_lang[f]; cnt += 1
                if lv < 3:
                    mb += self.sz_rt[f]; cnt += len(self.pk_rt[f])
                self.level_mb[f, c] = mb
                if lv == 0:
                    self.warm_level[f, c] = self.cold[f]
                else:
                    self.warm_level[f, c] = clean + (pull(mb, cnt) if cnt else 0.0) + self.init[f]
                # package granular: every package already present in the container is kept
                miss = [p for p in (list(pf)) if p not in pc]
                dmb = float(sum(PKG[p] for p in miss))
                self.delta_mb[f, c] = dmb
                if lv == 0:
                    # different OS image: only package volumes in common could be reused,
                    # but the root file system must be rebuilt, so we keep it a cold start
                    self.warm_delta[f, c] = self.cold[f]
                else:
                    self.warm_delta[f, c] = clean + (pull(dmb, len(miss)) if miss else 0.0) + self.init[f]
        # a reuse can never be worse than a cold start
        self.warm_level = np.minimum(self.warm_level, self.cold[:, None])
        self.warm_delta = np.minimum(self.warm_delta, self.cold[:, None])


CAT = Catalog()

if __name__ == "__main__":
    np.set_printoptions(precision=2, suppress=True, linewidth=160)
    c = CAT
    print("cold", c.cold)
    print("mem", c.mem)
    print("warm level\n", c.warm_level)
    print("warm delta\n", c.warm_delta)
