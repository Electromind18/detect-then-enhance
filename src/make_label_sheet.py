"""Blind human-labelling sheet for the RRSHID haze-free images.

Picks n of the 150 RRSHID test pairs used in the paper (stratified by haze density, seed 0), and writes ONE
self-contained HTML page with the haze-free images and the same five presence questions the VLMs were asked.
No model answers are shown. The labeller clicks Yes / No / Unsure and downloads a CSV at the end.

  python make_label_sheet.py --rrshid /kaggle/working/rrshid_raw --features ../results/rrshid_v/features.jsonl \
         --n 50 --out rrshid_label_sheet.html
"""
import argparse, base64, glob, io, json, random
import pandas as pd
from PIL import Image

QUESTIONS = ["Is there buildings in this image?", "Is there trees in this image?", "Is there water in this image?",
             "Is there a road in this image?", "Is there farmland in this image?"]
DENS = {1: "thin", 2: "moderate", 3: "thick"}

ap = argparse.ArgumentParser()
ap.add_argument("--rrshid", required=True)
ap.add_argument("--features", default="../results/rrshid_v/features.jsonl")
ap.add_argument("--n", type=int, default=50)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default="rrshid_label_sheet.html")
a = ap.parse_args()

f = pd.read_json(a.features, lines=True)
h = f[(f.cond == "haze") & (f.action == "none")][["img_key", "source", "level"]].drop_duplicates("img_key")
rng = random.Random(a.seed)
per = {1: a.n // 3, 2: a.n // 3, 3: a.n - 2 * (a.n // 3)}
pick = []
for lv, k in per.items():
    keys = sorted(h[h.level == lv].img_key)
    pick += [(x, lv) for x in rng.sample(keys, min(k, len(keys)))]
rng.shuffle(pick)                                   # mixed order, so density is not obvious from position
src = h.set_index("img_key").source.astype(str)

items = []
for key, lv in pick:
    stem = src[key]
    hits = [p for p in glob.glob(f"{a.rrshid}/**/{DENS[lv]}_fog/*/*/{stem}.png", recursive=True) if "/hazy/" not in p]
    hits = sorted(hits, key=lambda p: "/test/" not in p)
    if not hits:
        raise SystemExit(f"haze-free image not found for {key} (stem {stem}, {DENS[lv]})")
    im = Image.open(hits[0]).convert("RGB").resize((512, 512), Image.BICUBIC)   # as in prepare_rrshid.py
    buf = io.BytesIO(); im.save(buf, "JPEG", quality=88)
    items.append(dict(key=key, img="data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()))
print(f"{len(items)} images ({per}) -> {a.out}")

html = """<!DOCTYPE html><html><head><meta charset="utf-8"><title>RRSHID blind labelling</title>
<style>body{font-family:Arial,sans-serif;max-width:980px;margin:20px auto;color:#222}
.card{display:flex;gap:18px;border:1px solid #ccc;border-radius:6px;padding:12px;margin:14px 0}
.card img{width:430px;height:430px;object-fit:contain;cursor:zoom-in}.qs{flex:1}
.q{margin:9px 0;padding:6px;border-radius:4px}.q.done{background:#eef7ee}.q span{display:block;font-weight:bold;margin-bottom:4px}
label{margin-right:14px}#bar{position:sticky;top:0;background:#fff;padding:8px 0;border-bottom:2px solid #444;z-index:2}
button{font-size:15px;padding:6px 14px}.big{position:fixed;inset:0;background:#000d;display:none;align-items:center;justify-content:center}
.big img{max-width:96vw;max-height:96vh}</style></head><body>
<div id="bar"><b>RRSHID blind labelling</b> &nbsp; Answered: <span id="cnt">0</span> / <span id="tot"></span>
 &nbsp; <button onclick="dl()">Download labels (CSV)</button></div>
<p><b>Instructions.</b> Each image is a haze-free satellite image. Answer each question from what you can see.
Use <i>Unsure</i> only if you really cannot decide. "Buildings", "trees", and "water" mean any visible amount;
"farmland" means cultivated fields; "a road" means any paved or unpaved road. Click an image to enlarge it.
Your answers are kept in this browser while the page stays open; download the CSV when you finish.</p>
<div id="list"></div><div class="big" id="big" onclick="this.style.display='none'"><img id="bigimg"></div>
<script>
const ITEMS=__ITEMS__, QS=__QS__;
const list=document.getElementById('list'); document.getElementById('tot').textContent=ITEMS.length*QS.length;
ITEMS.forEach((it,i)=>{const c=document.createElement('div');c.className='card';
 let h=`<div><b>Image ${i+1} / ${ITEMS.length}</b><br><img src="${it.img}" onclick="zoom(this.src)"></div><div class="qs">`;
 QS.forEach((q,j)=>{const n=it.key+'_q'+j;
  h+=`<div class="q" id="d_${n}"><span>${q}</span>`+['yes','no','unsure'].map(v=>
  `<label><input type="radio" name="${n}" value="${v}" onchange="upd('${n}')"> ${v[0].toUpperCase()+v.slice(1)}</label>`).join('')+`</div>`;});
 c.innerHTML=h+'</div>';list.appendChild(c);});
function zoom(s){document.getElementById('bigimg').src=s;document.getElementById('big').style.display='flex';}
function upd(n){document.getElementById('d_'+n).classList.add('done');
 document.getElementById('cnt').textContent=document.querySelectorAll('input:checked').length;}
function dl(){let rows=['qid,img_key,question,human_answer'];
 ITEMS.forEach(it=>QS.forEach((q,j)=>{const n=it.key+'_q'+j;const s=document.querySelector(`input[name="${n}"]:checked`);
  rows.push([n,it.key,'"'+q+'"',s?s.value:''].join(','));}));
 const miss=rows.filter(r=>r.endsWith(',')).length;
 if(miss&&!confirm(miss+' questions are still unanswered. Download anyway?'))return;
 const b=new Blob([rows.join('\\n')],{type:'text/csv'});const u=URL.createObjectURL(b);
 const a=document.createElement('a');a.href=u;a.download='rrshid_human_labels.csv';a.click();}
</script></body></html>"""
html = html.replace("__ITEMS__", json.dumps(items)).replace("__QS__", json.dumps(QUESTIONS))
open(a.out, "w").write(html)
