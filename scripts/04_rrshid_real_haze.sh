#!/usr/bin/env bash
# Real haze: RRSHID (Zhu et al., 2025), 50 test pairs per density. Download link from the RRSHID authors' GitHub page.
set -euo pipefail; cd "$(dirname "$0")/../src"
mkdir -p ../data/rrshid_raw
gdown 1uBwHM8tyw69xafFHd01vMERs3TadUmxT -O ../data/rrshid.zip && unzip -q -o ../data/rrshid.zip -d ../data/rrshid_raw
python prepare_rrshid.py --root ../data/rrshid_raw --out ../data/rrshid_v --per_density 50
for M in Qwen/Qwen2.5-VL-3B-Instruct Qwen/Qwen2-VL-2B-Instruct; do bash ../scripts/run_vlm_2gpu.sh ../data/rrshid_v $M; done
for K in 10 30; do python fewshot_rrshid.py --root ../data/rrshid_raw --k $K --out ../data/rrshid_v/fewshot_k$K.jsonl; done
for T in Qwen2.5-VL-3B-Instruct Qwen2-VL-2B-Instruct; do
  python transfer_eval.py --src ../data/phase1 --tgt ../data/rrshid_v --model_tag $T --pseudo 0.8                     # zero-shot
  python transfer_eval.py --src ../data/phase1 --tgt ../data/rrshid_v --model_tag $T --pseudo 0.8 \
         --extra_det ../data/rrshid_v/fewshot_k10.jsonl                                                                # few-shot
done
