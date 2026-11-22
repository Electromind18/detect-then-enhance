#!/usr/bin/env bash
# AID transfer: 600 scene + 400 multi-label images (1,400 questions), severities 1-3. Needs a Hugging Face token (HF_TOKEN).
set -euo pipefail; cd "$(dirname "$0")/../src"
python prepare_aid.py --out ../data/aid --per_class 20 --n_ml 400
python make_versions.py --csv ../data/aid/aid.csv --out ../data/aid_v --n_images 5000 --q_per_img 3 --graded
for M in Qwen/Qwen2.5-VL-3B-Instruct Qwen/Qwen2-VL-2B-Instruct; do bash ../scripts/run_vlm_2gpu.sh ../data/aid_v $M; done
for T in Qwen2.5-VL-3B-Instruct Qwen2-VL-2B-Instruct; do python transfer_eval.py --src ../data/phase1 --tgt ../data/aid_v --model_tag $T; done
python transfer_eval.py --src ../data/phase1 --tgt ../data/aid_v --model_tag Qwen2-VL-2B-Instruct --table_from Qwen2.5-VL-3B-Instruct
