"""Fit and save the router exactly as used in the transfer experiments (see transfer_eval.py):
  models/detector_gbdt.joblib            11-statistic HistGradientBoosting detector, fit on the UC Merced main study
  models/detector_gbdt_rrshid_k10.joblib same, plus 10 labelled RRSHID training pairs per haze density (few-shot)
  models/thresholds.json                 two-threshold detector (mean luminance, mean dark channel)
  models/lookup_tables.json              best operator per condition, per VLM
  models/metadata.json                   library versions and seeds

  python export_router.py --phase1 ../results/phase1 --fewshot ../results/rrshid_v/fewshot_k10.jsonl --out ../models
"""
import argparse, json, os, platform
import joblib, numpy as np, pandas as pd, sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from analyze_pilot import load
from common import ACTIONS
from transfer_eval import inputs, fit_threshold

MODELS = ["Qwen2.5-VL-3B-Instruct", "Qwen2-VL-2B-Instruct"]

ap = argparse.ArgumentParser()
ap.add_argument("--phase1", default="../results/phase1")
ap.add_argument("--fewshot", default="../results/rrshid_v/fewshot_k10.jsonl")
ap.add_argument("--out", default="../models")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

df, _ = load(a.phase1, MODELS[0])
si = inputs(df)
feats = [c for c in si.columns if c.startswith("in_")]
clf = HistGradientBoostingClassifier(max_iter=200, random_state=0).fit(si[feats], si.cond)
joblib.dump(dict(model=clf, features=feats), os.path.join(a.out, "detector_gbdt.joblib"))

ex = pd.read_json(a.fewshot, lines=True)
rep = max(1, len(si) // (5 * len(ex)))
si2 = pd.concat([si] + [ex[feats + ["cond"]]] * rep, ignore_index=True)
clf2 = HistGradientBoostingClassifier(max_iter=200, random_state=0).fit(si2[feats], si2.cond)
joblib.dump(dict(model=clf2, features=feats), os.path.join(a.out, "detector_gbdt_rrshid_k10.joblib"))

l, h = fit_threshold(si)
json.dump(dict(lum_mean_below_is_lowlight=float(l), haze_dark_above_is_haze=float(h)),
          open(os.path.join(a.out, "thresholds.json"), "w"), indent=2)

tables = {}
for m in MODELS:
    d, _ = load(a.phase1, m)
    acc = d.groupby(["cond", "action"]).correct.mean().unstack().reindex(columns=ACTIONS)
    tables[m] = dict(table=acc.idxmax(axis=1).to_dict(), accuracy=acc.round(4).to_dict(orient="index"))
json.dump(tables, open(os.path.join(a.out, "lookup_tables.json"), "w"), indent=2)

json.dump(dict(sklearn=sklearn.__version__, numpy=np.__version__, pandas=pd.__version__, python=platform.python_version(),
               detector="HistGradientBoostingClassifier(max_iter=200, random_state=0)",
               training_rows=len(si), fewshot_rows=len(ex), fewshot_repeat=rep, features=feats),
          open(os.path.join(a.out, "metadata.json"), "w"), indent=2)
print("saved router to", a.out, "| tables:", {m: t["table"] for m, t in tables.items()}, "| thresholds:", round(l, 3), round(h, 3))
