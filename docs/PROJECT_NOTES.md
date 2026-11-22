# Project notes (full context for future revisions)

This file lets authors continue the project from this
repository alone: revise the paper, answer reviewers, or rerun experiments. Read it together with
`docs/REPRODUCIBILITY.md` and `configs/experiments.yaml`. The LaTeX manuscript is kept by the authors (Overleaf),
not in this repository, so the two cannot drift apart; the paper's numbers are reproduced by `scripts/06`.

## How to resume with an AI assistant
Upload this repository (or point the assistant to it) and say: "Read docs/PROJECT_NOTES.md first. This is my
JSTARS paper. The task is: <reviewer comments / revision / new experiment>."

## Writing rules for this paper (from the authors)
- Deliver the manuscript as one Overleaf-style folder: main.tex, IEEEtran.cls, references.bib, figures.
- Authors in this order: Syed Muhammad Hamza Tauqeer (first), Muhammad Talha, Huma Ghafoor*, Seung Yeob Nam*
  (*corresponding). Affiliations and biographies as in the accepted FireNofire paper.
- Every citation must be a real, findable paper, and every claim attributed to it must be checkable in that paper.
- Plain, measured writing in the style of the authors' earlier accepted IEEE paper ("Comparative Study of Deep Neural
  Networks for Real-Time Fire Detection Using Mixed-Precision Inference"); no claims beyond the data.
- Related work should rely on closely relevant papers from 2024 onwards; older references only for datasets and methods.

## Kaggle practicalities
- Long runs: Save Version -> Save & Run All (Commit); interactive sessions time out.
- Pin `pip install "transformers>=4.51,<5"` in the first cell (the CODrone run missed this and used 5.20).
- Hugging Face downloads need an `HF_TOKEN` secret (Add-ons -> Secrets), or AID downloads hit HTTP 429.
- Two GPUs: one shard per GPU (`scripts/run_vlm_2gpu.sh`).

## Project history and results (chronological)
### 1. Goal
- IEEE JSTARS special issue "Towards Semantic and Reasoning-Enabled Remote Sensing: VLMs". Deadline **Nov 30, 2026**.
- Compute: Kaggle 2×T4. Long runs use Save Version → Save & Run All (Commit).
- Authors: **Syed Muhammad Hamza Tauqeer (1st)**, Muhammad Talha, Huma Ghafoor*, Seung Yeob Nam* (*corresponding).

### 2. Writing rules (strict)
- Deliver ONE Overleaf-style zip (main.tex, IEEEtran.cls, references.bib, figures), mirroring our accepted paper
  "Comparative Study of Deep Neural Networks for Real-Time Fire Detection Using Mixed-Precision Inference" (FireNofire_v4 zip).
- **Every citation must be a real, findable paper, and every claim must be verifiable from it. No invented references.**
- Write in the plain style of the accepted paper; it must not read as LLM-written.

### 3. Method (what the paper does)
- Frozen VLMs (no fine-tuning): Qwen2.5-VL-3B-Instruct, Qwen2-VL-2B-Instruct. Generation-free scoring from first-token logits
  (yes/no: sigmoid of the Yes−No logit difference; MCQ: softmax over A–D).
- Synthetic degradations: haze (atmospheric scattering model), low light (gamma + scale + noise), severities 1–3, plus level 4 = unseen
  (denser tinted haze; darker, noisier low light).
- Enhancers: CLAHE, adaptive gamma, DCP dehazing.
- **Router**: detect the condition from 11 cheap image statistics (GBDT, or a simple threshold rule), then apply the best enhancer for that
  condition from a lookup table. Baselines: never enhance, always-X, per-sample gate, oracle.
- Evaluation: 5-fold GroupKFold by image, paired bootstrap at image level, help/hurt rates, blank-image control.

### 4. Results so far (UC Merced multi-label: scene MCQ + presence, 700 images)
- **Qwen2.5-3B**: never 79.6 → router 83.4 (+3.8 [+3.0, +4.6]). Router vs always-gamma +0.78 [+0.30, +1.25] (image-level bootstrap).
  Detector accuracy 94.0%. Strong haze 76.0 → 86.6; strong low light 57.4 → 66.5. CLAHE hurts clean images (86.8 → 83.7).
- **Qwen2-2B**: never 85.2 → router 86.2 (+1.0 [+0.4, +1.6]); vs gamma +0.57 [+0.10, +1.03]. Caveat: in mild low light the router is
  below never (86.1 vs 87.6).
- Lookup tables: Qwen2.5 {clean: gamma, haze: dehaze, low: gamma}; Qwen2 {clean: none, haze: dehaze, low: gamma}.
  The tables transfer across VLMs with almost no loss.
- **The threshold router is as good as the GBDT router** (83.4 vs 83.3), so the gain comes from routing by degradation type, not from the classifier.
- RSVQA-LR was rejected: the clean image is no better than a grey image (67.0 vs 67.3), so the model ignores the image there.

### 5. Reviewer-style feedback we accepted
- Narrow the novelty claim. Say "without VLM fine-tuning", not "training-free".
- Proposed new title: "Detect, Then Enhance: Lightweight Degradation-Aware Enhancement Routing for Frozen Remote-Sensing VLMs".
- Cite closely related work (all verified): Cai et al. 2025 (arXiv 2506.05450); DA-CLIP (Luo et al., ICLR 2024);
  TDRE (Wan et al., ISPRS JPRS 242, 2026); LAD-Enhancer (ISPRS Annals XI-3-2026-307); YOLO-DER (MDPI Electronics 14(24):4851);
  JSTARS 2024 "Plug-and-Play Robust Aerial Object Detection Under Hazy Conditions".
- Fix the VQ-TTT description (input kernel + vision-encoder LoRA). Fix-Before-Search: use the PMLR author order.
- Soften the RSVQA-LR claim to the tested subset and model.

### 6. Plan at that time (all done since)
- **Phase 2 (Kaggle notebook rsvlm-gate-phase2, committed Nov 20)**: AID (600 scene MCQ + 400 multi-label images = 1,400 questions) and
  the unseen level-4 severity set, both VLMs, transfer_eval (the router is trained on UC Merced and applied unchanged).
  Output: rsvlm-gate_phase2.zip with transfer_*.md reports.
- TODO: real-haze test (RRSHID, arXiv 2503.17966; RICE1 as backup; metric = agreement with the answer on the clear image) and
  real-night test (CODrone night subset, arXiv 2504.20032; HazyDet as backup).
- TODO: update the paper with Phase 1B, threshold router, image-level bootstrap, Phase 2, real data, new references, new title;
  regenerate figures (make_figures.py, make_fig_samples.py); GitHub link.

### 7. Files
- Code: rsvqa-gate-v6.zip (common.py, make_versions.py, run_vlm.py, analyze_pilot.py, prepare_ucm_ml.py, prepare_aid.py,
  transfer_eval.py, make_figures.py, make_pipeline.py, make_fig_samples.py).
- Paper draft: JSTARS_DetectThenEnhance_v1.zip (Phase 1A numbers only; red \draftnote markers show what to update).
- Kaggle results: rsvlm-gate-phase1 / phase1b / phase2 output zips.

### 8. Phase 2 results (done Nov 21; router + table fit on UC Merced, applied unchanged; image-level 95% CI)
- AID, Qwen2.5: never 78.8 → router 81.3 (+2.50 [+1.61,+3.47]); always-gamma 81.4 (router vs gamma −0.12, n.s.);
  vs dehaze +1.98 [+1.13,+2.84]; oracle with target table 81.8. Detector 92.7% (thresholds 83.3%). Strong haze 77.7 → 86.4.
- AID, Qwen2: never 82.4 → router 83.3 (+0.90 [+0.14,+1.66]); ties gamma (83.0) and dehaze (83.1).
- Unseen level 4 (UCM held-out 200 images), Qwen2.5: never 67.4 → router 76.3 (+8.94 [+7.00,+10.78]);
  beats always-gamma +2.67 [+1.56,+3.78] and always-dehaze +2.28 [+1.06,+3.44]. Haze-4 63.2 → 85.3. Detector 97.0%.
- Unseen, Qwen2: never 75.3 → router 78.8 (+3.56 [+2.00,+5.00]); vs gamma +0.94 [+0.00,+1.78]; vs dehaze +0.61 (n.s.).
- Cross-VLM table (Qwen2.5 table on Qwen2, AID): 83.3, same as own table.
- Threshold router ≈ GBDT everywhere (within 0.3 pt, except Qwen2 unseen −0.6).
- Message: router is never significantly worse than the best fixed enhancer (which changes with model/dataset), and is clearly
  better when degradation is severe or unseen. Gains are small for the stronger-baseline Qwen2.

### 9. Paper v2 (Nov 21): JSTARS_DetectThenEnhance_v2.zip
- New title ("Lightweight ... for Frozen RS-VLMs"), Phase 1B for both VLMs, CV threshold router (91.1% det., 83.3 / 86.2),
  cross-VLM tables, transfer section (AID + unseen level 4) with Table V and Fig. 4, new refs (Cai 2025, DA-CLIP, Wu JSTARS 2024,
  AID, RRSHID, CODrone), Fix-Before-Search author order fixed (key zeng2026fixbeforesearch).
- Red \draftnote placeholders left: real-data subsection (RRSHID, CODrone), abstract/intro/conclusion sentences, CODrone counts
  in Table I, GitHub link, CPU re-timing.
- Real-data runs: rsvlm-gate-real (RRSHID, code v7, committed Nov 21); rsvlm-gate-night (CODrone, code v8, not yet run).

### 10. Real haze (RRSHID, 150 test pairs; reference = VLM answer on haze-free twin, P>0.8; kept 184 / 304 of 750 q)
- Zero-shot (detector fit on synthetic UCM only): detector 74.5% / 77.1%, misses ~1/3 of real haze (called clean).
  Q2.5: never 85.9, always-dehaze 91.8, router 86.1 (+0.3 n.s.). Q2: never 84.0, dehaze 89.5, router 88.8 (+4.8 [2.9,6.9]).
  On hazy inputs: dehazing 71.7->84.2 (Q2.5), 68.1->79.9 (Q2). Threshold detector ~47-48% on real haze.
- Few-shot detector recalibration (code v10, fewshot_rrshid.py + transfer_eval --extra_det): k=10 train pairs per density
  (30 pairs) -> detector 98.4% / 99.2%; k=30 gives the same. Hazy-input accuracy = oracle (84.2 / 79.9).
  Q2.5 router 89.1 (+3.3 [-1.1,7.6]); remaining gap to always-dehaze (-2.7) is all on clean images (table: clean->gamma,
  and the reference is defined by the clean un-enhanced answer, so any change on clean images counts against it).
  Q2 router 90.0 = oracle (+5.9 [3.3,8.4]), vs dehaze +0.49 [0.00,1.12]. Thresholds stay ~50%: two stats are not enough for real haze.

### 11. Real night (CODrone test split; 150 night + 100 day by file-name tags; 1,000 presence q from box labels; 12 classes)
- GBDT detector 41.2% (588/600 night inputs -> clean; real night noise MAD ~0.02 vs ~0.05 synthetic); thresholds 98.4%.
- Q2.5: never 70.0 (night 72.2), every operator lowers it; router -0.8. Q2: CLAHE +1.7, gamma +1.5, threshold router +1.2
  (= oracle with source table), GBDT router 0.0. Blank = always "no" (52.8%). CODrone runs used Transformers 5.20 (others 4.57).

### 12. Paper v4 (Nov 22): JSTARS_DetectThenEnhance_v4.zip, 13 pages
- Real-data section + Table VI + Fig. 5 placeholder (graphs/real_examples.pdf from make_fig_real.py on Kaggle).
- New verified 2024-2026 refs: MLLM-IC (ICCV 2025), R-Bench, VLM-RobustBench, Weng et al. GRSM, SenseBench,
  Kumar & Chadha 2025, Aryashad et al. 2025. Old pei2018/RESIDE/Hendrycks no longer cited.
- CPU timing re-measured: 15 ms (256 px) / 28 ms (512 px). Draft notes all removed (\draftfalse).
- Code repo: detect-then-enhance-repo.zip; cited as https://github.com/Electromind18/detect-then-enhance.

### 13. Status at submission (Nov 2026)
- All experiments finished; Table VI holds the
  real-data results and Fig. 5 the real-image examples (`src/make_fig_real.py`).
- Honest weak points a reviewer may raise: modest methodological novelty (degradation-aware selection exists; the
  contribution is the measured evidence on frozen RS VLMs and the transfer tests); small 2-3B VLMs; RRSHID metric is
  agreement with the model's own haze-free answer; small gains for the stronger Qwen2-VL-2B; GBDT detector fails on real
  night images (threshold detector works); CODrone runs used Transformers 5.20.
- If a reviewer asks for a larger VLM: `run_vlm.py --model <hf id>` works for Qwen2/2.5-VL-family models; 7B needs
  4-bit loading on a T4 (not implemented yet).

### 14. Pre-submission review fixes (Nov 22)
- Every confidence interval now resamples whole images (main table recomputed with `src/main_table_ci.py`).
- Cross-VLM table under the same CV: 83.3% (Qwen2.5 with Qwen2 table, own 83.3%) and 86.0% (Qwen2 with Qwen2.5 table,
  own 86.2%); the paper says "loss of at most 0.2 points", not "without loss".
- CV threshold router: 83.4% / 86.2% (GBDT router 83.3% / 86.2%).
- Evidence-drift section uses association, not causal, language.
- Input sizes after the Qwen processor (multiples of 28, area <= 512x512): UC Merced 252x252, AID/RRSHID 504x504,
  CODrone 672x364. Reported run times are batched throughput (16 per batch), not single-image latency.
- Added a limitations paragraph: RRSHID references are model answers, not human labels; T4/one CPU only; no memory,
  energy, or edge-device measurements.
- Lessons from the authors' earlier JSTARS reviews (FireNofire paper): reviewers asked for a public repo with configs,
  seeds and versions; 2024+ literature; visual samples; cross-dataset tests; hardware dependency and edge devices;
  explanations of *why* results differ; concise writing. Expect similar questions here.

### 15. Blind human labels for RRSHID (Nov 22)
- `src/make_label_sheet.py` built a blind labelling page (50 RRSHID haze-free images, seed 0, 16/16/18 per density,
  no model answers shown); one author labelled it. Labels: `results/rrshid_v/human_labels/rrshid_human_labels.csv`
  (102 yes, 66 no, 82 unsure). Scored with `src/human_eval.py` -> `human_eval_report.md` (paper Table VII).
- 168 questions / 48 images: haze-free 69.0 / 73.8; hazy never 44.6 / 49.4; always dehaze 57.7 / 58.3;
  router few-shot 57.7 / 58.3 (= always dehaze on these inputs); zero-shot router 54.2 / 54.8.
  VLM confident haze-free answers agree with human on 83.3% / 87.7% (validates the agreement metric partly).
