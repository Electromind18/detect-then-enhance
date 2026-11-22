#!/usr/bin/env bash
# Paper figures. Fig. 3 from stored runs (CPU). Fig. 2 and Fig. 5 re-download the images (internet needed).
set -euo pipefail; cd "$(dirname "$0")/../src"
python make_figures.py --data ../results/phase1 --model_tag Qwen2.5-VL-3B-Instruct --out ../figures/graphs
python make_fig_examples.py --out ../figures/fig_examples
python make_fig_real.py --rrshid ../data/rrshid_raw --codrone ../data/codrone_raw --out ../figures/fig_real
