#!/usr/bin/env bash
# Recompute every table of the paper from the stored runs in results/ (CPU only, a few minutes). No GPU, no images.
set -euo pipefail; cd "$(dirname "$0")/../src"; R=../results; mkdir -p ../recomputed
python main_table_ci.py --data $R/phase1 > ../recomputed/main_table_image_level_ci.txt   # Table III
for T in Qwen2.5-VL-3B-Instruct Qwen2-VL-2B-Instruct; do
  python analyze_pilot.py --data $R/phase1 --model_tag $T                         > ../recomputed/main_$T.txt
  for S in aid_v unseen_v codrone_v; do python transfer_eval.py --src $R/phase1 --tgt $R/$S --model_tag $T > ../recomputed/${S}_$T.txt; done
  python transfer_eval.py --src $R/phase1 --tgt $R/rrshid_v --model_tag $T --pseudo 0.8 > ../recomputed/rrshid_zeroshot_$T.txt
  python transfer_eval.py --src $R/phase1 --tgt $R/rrshid_v --model_tag $T --pseudo 0.8 --extra_det $R/rrshid_v/fewshot_k10.jsonl > ../recomputed/rrshid_fewshot_k10_$T.txt
done
python transfer_eval.py --src $R/phase1 --tgt $R/aid_v --model_tag Qwen2-VL-2B-Instruct --table_from Qwen2.5-VL-3B-Instruct > ../recomputed/aid_v_crossVLM_table.txt
python export_router.py --phase1 $R/phase1 --fewshot $R/rrshid_v/fewshot_k10.jsonl --out ../models
echo "reports written to recomputed/ ; router written to models/"
