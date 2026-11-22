#!/usr/bin/env bash
# Score one prepared folder with one VLM on two GPUs (one shard per GPU), as on Kaggle (2x T4).
# usage: bash scripts/run_vlm_2gpu.sh <data_dir> <hf_model_id>
set -euo pipefail
D=$1; M=${2:-Qwen/Qwen2.5-VL-3B-Instruct}; T=$(basename "$M")
cd "$(dirname "$0")/../src"
CUDA_VISIBLE_DEVICES=0 python run_vlm.py --data "$D" --model "$M" --bs 16 --shard 0 --nshards 2 > "log_${T}_$(basename "$D")_0.txt" 2>&1 &
CUDA_VISIBLE_DEVICES=1 python run_vlm.py --data "$D" --model "$M" --bs 16 --shard 1 --nshards 2 > "log_${T}_$(basename "$D")_1.txt" 2>&1 &
wait
tail -n 1 log_${T}_$(basename "$D")_*.txt
