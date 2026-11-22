"""Apply the released router to one image (CPU only).

  python route.py --image my_scene.png --vlm Qwen2.5-VL-3B-Instruct [--detector gbdt|gbdt_rrshid_k10|thresholds] --save out.png

Prints the image statistics, the detected condition, and the operator; writes the enhanced image if --save is given.
The enhanced image is what the frozen VLM should receive. Images are processed at their given size; the paper resized
inputs to at most 512 px (768 px for CODrone) before routing.
"""
import argparse, json, os
import joblib, numpy as np, pandas as pd
from PIL import Image
from common import enhance, image_stats

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--image", required=True)
ap.add_argument("--vlm", default="Qwen2.5-VL-3B-Instruct", choices=["Qwen2.5-VL-3B-Instruct", "Qwen2-VL-2B-Instruct"])
ap.add_argument("--detector", default="gbdt", choices=["gbdt", "gbdt_rrshid_k10", "thresholds"])
ap.add_argument("--models", default=os.path.join(HERE, "..", "models"))
ap.add_argument("--save", default=None)
a = ap.parse_args()

img = np.array(Image.open(a.image).convert("RGB"))
st = image_stats(img)
if a.detector == "thresholds":
    t = json.load(open(os.path.join(a.models, "thresholds.json")))
    cond = ("lowlight" if st["lum_mean"] < t["lum_mean_below_is_lowlight"]
            else "haze" if st["haze_dark"] > t["haze_dark_above_is_haze"] else "clean")
else:
    d = joblib.load(os.path.join(a.models, f"detector_{a.detector}.joblib"))
    x = pd.DataFrame([{f"in_{k}": v for k, v in st.items()}])[d["features"]]
    cond = d["model"].predict(x)[0]
op = json.load(open(os.path.join(a.models, "lookup_tables.json")))[a.vlm]["table"][cond]
print(json.dumps(dict(statistics={k: round(float(v), 4) for k, v in st.items()}, condition=cond, operator=op), indent=2))
if a.save:
    Image.fromarray(enhance(img, op)).save(a.save)
    print("saved", a.save)
