#!/usr/bin/env bash
# Main study: 700 UC Merced images x 3 questions, severities 1-3, both VLMs (27,300 runs per VLM, ~1 h each on 2x T4).
set -euo pipefail; cd "$(dirname "$0")/../src"
python prepare_ucm_ml.py --out ../data/ucm_ml
python make_versions.py --csv ../data/ucm_ml/ucm_ml.csv --out ../data/phase1 --n_images 700 --q_per_img 3 --graded
for M in Qwen/Qwen2.5-VL-3B-Instruct Qwen/Qwen2-VL-2B-Instruct; do bash ../scripts/run_vlm_2gpu.sh ../data/phase1 $M; done
for T in Qwen2.5-VL-3B-Instruct Qwen2-VL-2B-Instruct; do python analyze_pilot.py --data ../data/phase1 --model_tag $T; done
