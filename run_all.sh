#!/usr/bin/env bash
# Reproduces every number and figure of the paper on one CPU core. Runs in roughly three to four hours.
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH=.
mkdir -p results/models figures

python tests/test_gradients.py                                   # finite difference check of both networks

# 1. training (200 episodes each, seed 0)
python scripts/train.py --variant level --retention 0 --kind train   --episodes 200 --out results/models/mlcr   | tee results/models/mlcr.log
python scripts/train.py --variant delta --retention 0 --kind train   --episodes 200 --out results/models/c1     | tee results/models/c1.log
python scripts/train.py --variant delta --retention 1 --kind train   --episodes 200 --out results/models/c12    | tee results/models/c12.log
python scripts/train.py --variant level --retention 0 --kind familyA --episodes 200 --out results/models/mlcrA  | tee results/models/mlcrA.log
python scripts/train.py --variant delta --retention 1 --kind familyA --episodes 200 --out results/models/c12A   | tee results/models/c12A.log

# 2. memory price chosen on validation seeds only
python scripts/select_lambda.py

# 3. experiments on the unseen test seeds (500000 and up)
python scripts/evaluate.py main workloads pareto drift overhead poolsweep
python scripts/detector_sensitivity.py
python scripts/retention_analysis.py

# 4. tables and figures
python scripts/summarize.py
python scripts/make_figures.py
python scripts/fig_architecture.py figures/fig_architecture.png
