# Reproducibility

## 1. Software and hardware of the original runs

| Item | Value |
|---|---|
| Platform | Kaggle notebooks, 2 x NVIDIA Tesla T4 (16 GB), 2-vCPU Intel Xeon 2.1 GHz |
| Python | 3.13 |
| PyTorch | 2.11.0 + CUDA 13.0 |
| Transformers | 4.57.6 (main study, AID, unseen severity, RRSHID); 5.20.2 (CODrone only, platform default at the time) |
| Precision | bfloat16, SDPA attention |
| Models | `Qwen/Qwen2.5-VL-3B-Instruct`, `Qwen/Qwen2-VL-2B-Instruct` (Hugging Face, default revision, Nov 2026) |
| Analysis | numpy, pandas, scikit-learn, OpenCV (CPU) |

Each `results/*/run_info_*.json` records the exact versions of that run. The original Kaggle logs are in
`results/kaggle_logs/`.

## 2. Seeds and determinism

| What | Seed / rule | Where |
|---|---|---|
| Question and image sampling | `random.Random(0)` | `make_versions.py --seed 0`, `prepare_*.py --seed 0` |
| Degraded image versions | `numpy.default_rng(crc32("0|<image_key>|<condition>"))`; severity drawn from the same generator | `make_versions.py` |
| Gradient-boosted detector, per-sample gate | `random_state=0` | `analyze_pilot.py`, `transfer_eval.py`, `export_router.py` |
| Bootstrap | `numpy.default_rng(0)`, 2,000 resamples | `analyze_pilot.py`, `transfer_eval.py` |
| Cross-validation | `GroupKFold(5)` by image (no randomness) | `analyze_pilot.py` |
| VLM scoring | no sampling: one forward pass, first-token logits | `run_vlm.py` |

Every image version can therefore be regenerated bit for bit from the source images, and the VLM answers are fixed
for fixed software versions and hardware.

## 3. Inference settings

See `configs/experiments.yaml`. In short: batch 16, at most 512 x 512 input pixels (min 64 x 28 x 28), left padding,
prompt suffix "Answer with Yes or No." or "Answer with only the letter of the correct option.", P(yes) from the
log-sum-exp of {Yes, yes} against {No, no}, A-D by softmax over the letter tokens.

## 4. Three levels of reproduction

1. **Tables from stored runs (CPU, minutes):** `bash scripts/06_tables_from_saved_runs.sh`. Writes all reports to
   `recomputed/` and refits the router into `models/`.
2. **Figures:** `bash scripts/07_figures.sh` (Fig. 2 and Fig. 5 download the source images).
3. **Full re-run (GPU):** `scripts/01_main_study.sh` to `scripts/05_codrone_real_night.sh`, in that order
   (01 must run first: every transfer study uses the main-study runs to fit the router).
   Approximate time on 2 x T4: main study about 2.3 and 1.5 h; AID 5 and 4 h; unseen 0.7 and 0.5 h; RRSHID 2 and 1.6 h;
   CODrone 1.4 and 1.1 h (Qwen2.5-VL-3B and Qwen2-VL-2B).

## 5. Known differences when recomputing

The paper reports the numbers computed on Kaggle. Recomputing the transfer tables with scikit-learn 1.9.1 changed
detector accuracies by at most about one point (for example 74.5% to 73.4% on RRSHID zero-shot) and router accuracies by
at most 0.2 points, because histogram binning differs between versions. All other numbers are identical.

## 6. Router models

`models/` holds the router used in the paper's transfer experiments, refitted from `results/phase1`
(`src/export_router.py`):

| File | Content |
|---|---|
| `detector_gbdt.joblib` | 11-statistic detector (clean / haze / lowlight) |
| `detector_gbdt_rrshid_k10.joblib` | same, plus 10 labelled RRSHID training pairs per density |
| `thresholds.json` | two-threshold detector |
| `lookup_tables.json` | best operator per condition, per VLM, with the per-condition accuracies |
| `metadata.json` | library versions and training sizes |

Load the `.joblib` files with the same scikit-learn version as in `metadata.json`, or refit them with
`export_router.py` (seconds on a CPU). `src/route.py` applies the router to any image.
