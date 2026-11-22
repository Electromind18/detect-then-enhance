"""Main-study table (paper Table III) with image-level paired bootstrap CIs, the CV threshold router,
and the cross-VLM lookup table under the same cross-validation.

  python main_table_ci.py --data ../results/phase1
"""
import argparse
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from analyze_pilot import load, cv_router, cv_gate
from common import ACTIONS
from transfer_eval import img_boot, inputs, fit_threshold

MODELS = ["Qwen2.5-VL-3B-Instruct", "Qwen2-VL-2B-Instruct"]


def routed(df, pick):
    """pick(train_inputs, train_df, test_inputs) -> list of actions for the test inputs; 5-fold CV by image."""
    ti, acts = inputs(df), {}
    for tr, te in GroupKFold(5).split(ti, groups=ti.img_key):
        x = ti.iloc[te]
        for q, c, a in zip(x.qid, x.cond, pick(ti.iloc[tr], df[df.img_key.isin(set(ti.img_key.iloc[tr]))], x)):
            acts[(q, c)] = a
    t = ti[["qid", "cond"]].assign(act=[acts[(q, c)] for q, c in zip(ti.qid, ti.cond)])
    s = df.merge(t, on=["qid", "cond"])
    return s[s.action == s.act]


def table(d):
    return d.groupby(["cond", "action"]).correct.mean().unstack().idxmax(axis=1).to_dict()


def threshold_pick(tri, trd, x):
    l, h = fit_threshold(tri)
    t = table(trd)
    return [t[c] for c in np.where(x.in_lum_mean < l, "lowlight", np.where(x.in_haze_dark > h, "haze", "clean"))]


def other_table_pick(other):
    def f(tri, trd, x):
        feats = [c for c in tri.columns if c.startswith("in_")]
        clf = HistGradientBoostingClassifier(max_iter=200, random_state=0).fit(tri[feats], tri.cond)
        t = table(other[other.img_key.isin(set(tri.img_key))])
        return [t[c] for c in clf.predict(x[feats])]
    return f


ap = argparse.ArgumentParser()
ap.add_argument("--data", default="../results/phase1")
a = ap.parse_args()
D = {m: load(a.data, m)[0] for m in MODELS}
for m in MODELS:
    df = D[m]; nev = df[df.action == "none"]
    pol = {f"always-{x}": df[df.action == x] for x in ACTIONS[1:]}
    g = cv_gate(df, 0.02); pol["per-sample gate"] = g[0] if isinstance(g, tuple) else g
    pol["router (GBDT)"] = cv_router(df)[0]
    pol["router (thresholds)"] = routed(df, threshold_pick)
    pol["router + question type"] = cv_router(df, use_q=True)[0]
    best = table(df); o = df.assign(act=df.cond.map(best)); pol["condition oracle"] = o[o.action == o.act]
    pol["router, other VLM's table"] = routed(df, other_table_pick(D[[n for n in MODELS if n != m][0]]))
    print(f"\n{m}")
    for k, v in pol.items():
        d, lo, hi = img_boot(v, nev)
        byc = v.groupby("cond").correct.mean().mul(100).round(1).to_dict()
        print(f"  {k:26s} {byc}  overall {v.correct.mean()*100:5.1f}  gain {d:+.2f} [{lo:+.2f}, {hi:+.2f}]")
    print("  router vs always-gamma:", [round(x, 2) for x in img_boot(pol["router (GBDT)"], pol["always-gamma"])])
