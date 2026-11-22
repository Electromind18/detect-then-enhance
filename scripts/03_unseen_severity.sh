#!/usr/bin/env bash
# Unseen severity: 200 UC Merced images NOT used in the main study (skip 700), level 4 only.
set -euo pipefail; cd "$(dirname "$0")/../src"
python make_versions.py --csv ../data/ucm_ml/ucm_ml.csv --out ../data/unseen_v --n_images 200 --skip 700 --q_per_img 3 --fixed_level 4
for M in Qwen/Qwen2.5-VL-3B-Instruct Qwen/Qwen2-VL-2B-Instruct; do bash ../scripts/run_vlm_2gpu.sh ../data/unseen_v $M; done
for T in Qwen2.5-VL-3B-Instruct Qwen2-VL-2B-Instruct; do python transfer_eval.py --src ../data/phase1 --tgt ../data/unseen_v --model_tag $T; done
