# AMLCR: Adaptive Multi-Level Container Reuse (simulation code)

Code, trained weights, raw results and figure scripts for the manuscript
"Adaptive Multi-Level Container Reuse for Cold-Start Latency and Memory Optimization in Serverless Computing".

Zenodo DOI: to be added by the authors after deposition.

## What is in this folder

| Path | Content |
|---|---|
| `amlcr/catalog.py` | The 13 functions of FStartBench (Table II of Zhou et al., IPDPS 2024), package sizes and the startup latency model |
| `amlcr/sim.py` | Discrete event simulator with idle-only pool accounting |
| `amlcr/workloads.py` | Mixed, HI Sim, LO Sim, Peak, Uniform, family A and drift streams |
| `amlcr/nn.py` | NumPy neural networks with manual backpropagation and Adam (token attention Q network, MLP) |
| `amlcr/policies.py` | Baselines (LRU, FaasCache, KeepAlive, Greedy Match, Greedy Delta), MLCR, AMLCR, retention agent, drift detector |
| `amlcr/evalutil.py`, `amlcr/runner.py` | Policy factory and run helpers |
| `scripts/train.py` | Training of one model |
| `scripts/select_lambda.py` | Memory price selection on validation seeds |
| `scripts/evaluate.py` | All experiments (main, workloads, pareto, drift, overhead, poolsweep) |
| `scripts/detector_sensitivity.py` | Drift detector threshold study |
| `scripts/retention_analysis.py` | Per function type keep and evict statistics of the retention agent |
| `scripts/summarize.py` | Confidence intervals, paired reductions, Wilcoxon tests, written to `results/summary.json` |
| `scripts/make_figures.py`, `scripts/fig_architecture.py` | All figures |
| `tests/test_gradients.py` | Finite difference check of both networks |
| `results/` | Raw csv files, `summary.json`, trained weights (`results/models`), training logs |
| `figures/` | Figures used in the paper (300 dpi PNG) |

## Reproduce

```
pip install -r requirements.txt
bash run_all.sh
```

Everything is NumPy; no GPU is needed. All random seeds are fixed: training seeds start at 100000, validation seeds
are 900000 to 900011, test seeds start at 500000. To regenerate tables and figures from the shipped result files without retraining, run
`python scripts/summarize.py && python scripts/make_figures.py` with `PYTHONPATH=.`.

## Important caveats

* This is a simulator study. Package sizes of the 5 reference images are taken from Fig. 1 of the base paper; all other sizes and every latency constant in `amlcr/catalog.py` are estimates and can be replaced by measured values.
* MLCR is a reimplementation with a greedy anchored token network, not the original authors' code.
* The drift study was run three times on the same 30 test seeds while the online adaptation was being designed (no exploration in the retention agent, then exploration, then exploration plus optimism for unseen function types). Only the last version is shipped. The intermediate csv of the second version is kept as `results/raw_drift_v1_explore_only.csv`.


## Citation

Please cite the manuscript above and the base paper: A. C. Zhou et al., "Tackling Cold Start in Serverless Computing with Multi-Level Container Reuse," IPDPS 2024, doi 10.1109/IPDPS57955.2024.00017.
