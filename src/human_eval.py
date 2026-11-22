"""Score the RRSHID runs against blind human labels (from make_label_sheet.py).

For every labelled question (Unsure answers dropped), the reference is the human answer on the haze-free image.
Reports, for both VLMs and the same fixed question set: accuracy on the haze-free image, accuracy on the real hazy
image with each fixed operator and with the router (zero-shot, few-shot, thresholds), and how often the old
reference (the VLM's own answer on the haze-free image) agrees with the human label.

  python human_eval.py --labels rrshid_human_labels.csv --data ../results/rrshid_v --models ../models
"""
import argparse, json, os
import joblib, numpy as np, pandas as pd
from analyze_pilot import load
from transfer_eval import img_boot

MODELS = ["Qwen2.5-VL-3B-Instruct", "Qwen2-VL-2B-Instruct"]
ap = argparse.ArgumentParser()
ap.add_argument("--labels", required=True)
ap.add_argument("--data", default="../results/rrshid_v")
ap.add_argument("--models", default="../models")
ap.add_argument("--out", default=None)
a = ap.parse_args()

lab = pd.read_csv(a.labels)
lab = lab[lab.human_answer.isin(["yes", "no"])]
print(f"labelled questions used: {len(lab)} on {lab.img_key.nunique()} images "
      f"({(lab.human_answer == 'yes').mean() * 100:.1f}% yes); unsure/blank dropped")
ref = lab.set_index("qid").human_answer

thr = json.load(open(os.path.join(a.models, "thresholds.json")))
tables = json.load(open(os.path.join(a.models, "lookup_tables.json")))
det = {k: joblib.load(os.path.join(a.models, f"detector_{k}.joblib")) for k in ["gbdt", "gbdt_rrshid_k10"]}
L = []
P = lambda *x: (print(*x), L.append(" ".join(map(str, x))))
P(f"# RRSHID against blind human labels: {len(lab)} questions, {lab.img_key.nunique()} images")
for m in MODELS:
    d, _ = load(a.data, m)
    d = d[d.qid.isin(ref.index)].copy()
    d["correct"] = ((d.p_yes > 0.5) == (d.qid.map(ref) == "yes")).astype(int)
    hz = d[d.cond == "haze"]
    inp = hz[hz.action == "none"].drop_duplicates("qid")
    pred = {"zero-shot GBDT": det["gbdt"]["model"].predict(inp[det["gbdt"]["features"]]),
            "few-shot GBDT (30 pairs)": det["gbdt_rrshid_k10"]["model"].predict(inp[det["gbdt_rrshid_k10"]["features"]]),
            "thresholds": np.where(inp.in_lum_mean < thr["lum_mean_below_is_lowlight"], "lowlight",
                                   np.where(inp.in_haze_dark > thr["haze_dark_above_is_haze"], "haze", "clean"))}
    never = hz[hz.action == "none"]
    pol = {"haze-free image, no enhancement (reference input)": d[(d.cond == "clean") & (d.action == "none")].assign(cond="haze"),
           "hazy, never enhance": never}
    for act in ["clahe", "gamma", "dehaze"]:
        pol[f"hazy, always {act}"] = hz[hz.action == act]
    for name, pc in pred.items():
        act = dict(zip(inp.qid, [tables[m]["table"][c] for c in pc]))
        s = hz[hz.action == hz.qid.map(act)]
        pol[f"hazy, router {name}"] = s
        P(f"{m} | {name}: {np.mean(pc == 'haze') * 100:.1f}% of hazy inputs detected as haze")
    P(f"\n## {m}")
    P("| input / policy | accuracy vs human (%) | gain vs never [95% CI, image-level] |")
    P("|---|---|---|")
    for k, v in pol.items():
        g = "--" if k == "hazy, never enhance" else "{:+.1f} [{:+.1f}, {:+.1f}]".format(*img_boot(v, never))
        P(f"| {k} | {v.correct.mean() * 100:.1f} | {g} |")
    clean = d[(d.cond == "clean") & (d.action == "none")]
    conf = clean[(clean.p_yes > 0.8) | (clean.p_yes < 0.2)]
    P(f"old reference check: VLM answer on haze-free image agrees with human on {clean.correct.mean() * 100:.1f}% "
      f"of questions ({conf.correct.mean() * 100:.1f}% of the {len(conf)} confident ones kept in the paper)")
if a.out:
    open(a.out, "w").write("\n".join(L) + "\n")
