#!/usr/bin/env bash
# Real night: CODrone (Ye et al., 2025) test split, 150 night + 100 day images (7.2 GB download; only test is unpacked).
set -euo pipefail; cd "$(dirname "$0")/../src"
gdown 1FQ6mUaOr_kATDaH7N2bObD5SRRkV7qJy -O /tmp/codrone.zip
mkdir -p ../data/codrone_raw && unzip -q -o /tmp/codrone.zip "CODrone/test/*" -d ../data/codrone_raw
python prepare_codrone.py --root ../data/codrone_raw --out ../data/codrone_v --n_night 150 --n_day 100
for M in Qwen/Qwen2.5-VL-3B-Instruct Qwen/Qwen2-VL-2B-Instruct; do bash ../scripts/run_vlm_2gpu.sh ../data/codrone_v $M; done
for T in Qwen2.5-VL-3B-Instruct Qwen2-VL-2B-Instruct; do python transfer_eval.py --src ../data/phase1 --tgt ../data/codrone_v --model_tag $T; done
