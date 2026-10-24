"""Helpers that wire episodes, capacities and policies together."""
import numpy as np
from .sim import Sim, reference_capacity
from .workloads import make_episode
from . import policies as P

POOL_FRACS = {"Tight": 0.2, "Moderate": 0.5, "Loose": 1.0}


def system_of(name):
    return "delta" if name in ("GreedyDelta",) or name.startswith("AMLCR") else "level"


def run_baseline(name, ep, cap_frac):
    cap = cap_frac * reference_capacity(ep)
    pol = {"ColdOnly": P.ColdOnly, "LRU": P.ExactLRU, "FaasCache": P.FaasCache,
           "KeepAlive": P.KeepAlive, "GreedyMatch": P.Greedy, "GreedyDelta": P.Greedy}[name]()
    sim = Sim(ep, cap, system="delta" if name == "GreedyDelta" else "level")
    res = sim.run(pol)
    res["cap"] = cap
    return res


def run_policy(policy, ep, cap_frac, system):
    cap = cap_frac * reference_capacity(ep)
    sim = Sim(ep, cap, system=system)
    res = sim.run(policy)
    res["cap"] = cap
    return res
