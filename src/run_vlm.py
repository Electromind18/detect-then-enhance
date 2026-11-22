"""Score every (question x image version) with a frozen VLM. ONE forward pass per run, no generation.
yes/no questions: P(yes) vs P(no) at the first answer token.
multiple choice (answer a-d): softmax over letter tokens A-D.
Resumable. Multi-GPU: run one process per GPU with --shard i --nshards N.

  python run_vlm.py --data pilot --model Qwen/Qwen2.5-VL-3B-Instruct --bs 16
"""
import argparse, glob, json, os, time, platform
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForImageTextToText
from common import CONDITIONS, ACTIONS

LETTERS = ["A", "B", "C", "D"]


def is_mcq(q):
    return q["answer"] in {"a", "b", "c", "d"}


def prompt(q):
    if is_mcq(q):
        return q["question"] + "\nAnswer with only the letter of the correct option."
    return q["question"] + " Answer with Yes or No."


def jobs_for(data):
    qs = [json.loads(l) for l in open(os.path.join(data, "questions.jsonl"))]
    jobs = []
    for q in qs:
        for c in CONDITIONS:
            for a in ACTIONS:
                fn = f"{q['img_key']}_{c}_{a}.png"
                if os.path.exists(os.path.join(data, "images", fn)):  # real-data sets have no low-light version
                    jobs.append((q, c, a, fn))
        jobs.append((q, "blank", "none", f"{q['img_key']}_blank.png"))
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="pilot")
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--max_pixels", type=int, default=512 * 512)
    ap.add_argument("--dtype", default="auto", choices=["auto", "fp16", "bf16", "fp32"])
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="run only the first N pending jobs (speed test)")
    a = ap.parse_args()

    tag = a.model.split("/")[-1]
    out_path = os.path.join(a.data, f"results_{tag}_s{a.shard}.jsonl")
    done = set()
    for p in glob.glob(os.path.join(a.data, f"results_{tag}*.jsonl")):
        done |= {json.loads(l)["key"] for l in open(p)}
    jobs = [j for i, j in enumerate(jobs_for(a.data)) if i % a.nshards == a.shard]
    jobs = [j for j in jobs if f"{j[0]['qid']}|{j[1]}|{j[2]}" not in done]
    if a.limit:
        jobs = jobs[:a.limit]
    print(f"shard {a.shard}/{a.nshards}: {len(jobs)} to run")
    if not jobs:
        return

    if a.dtype == "auto":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[a.dtype]
    proc = AutoProcessor.from_pretrained(a.model, min_pixels=64 * 28 * 28, max_pixels=a.max_pixels)
    proc.tokenizer.padding_side = "left"
    model = AutoModelForImageTextToText.from_pretrained(a.model, torch_dtype=dtype, device_map="cuda",
                                                        attn_implementation="sdpa").eval()
    tok = proc.tokenizer
    first = lambda w: tok.encode(w, add_special_tokens=False)[0]
    yes_ids = sorted({first(w) for w in ["Yes", "yes"]})
    no_ids = sorted({first(w) for w in ["No", "no"]})
    letter_ids = torch.tensor([first(L) for L in LETTERS], device="cuda")

    json.dump(dict(model=a.model, dtype=str(dtype), max_pixels=a.max_pixels,
                   gpu=torch.cuda.get_device_name(0), n_gpus_used=a.nshards, torch=torch.__version__,
                   python=platform.python_version(),
                   transformers=__import__("transformers").__version__),
              open(os.path.join(a.data, f"run_info_{tag}.json"), "w"), indent=1)

    t0, n = time.time(), 0
    with open(out_path, "a") as fout:
        for i in tqdm(range(0, len(jobs), a.bs), desc=f"gpu{a.shard}"):
            batch = jobs[i:i + a.bs]
            texts, imgs = [], []
            for q, c, act, fn in batch:
                msg = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt(q)}]}]
                texts.append(proc.apply_chat_template(msg, tokenize=False, add_generation_prompt=True))
                imgs.append(Image.open(os.path.join(a.data, "images", fn)).convert("RGB"))
            inp = proc(text=texts, images=imgs, return_tensors="pt", padding=True).to("cuda")
            with torch.inference_mode():
                logits = model(**inp).logits[:, -1, :].float()
            if torch.isnan(logits).any():
                raise RuntimeError("NaN logits: rerun with --dtype fp32")
            lp = torch.log_softmax(logits, -1)
            ly, ln = torch.logsumexp(lp[:, yes_ids], -1), torch.logsumexp(lp[:, no_ids], -1)
            p_yes = torch.sigmoid(ly - ln).cpu().tolist()
            yn_mass = (ly.exp() + ln.exp()).cpu().tolist()
            lett = lp[:, letter_ids]
            p_lett = torch.softmax(lett, -1).cpu().tolist()
            l_mass = lett.exp().sum(-1).cpu().tolist()
            for k, (q, c, act, fn) in enumerate(batch):
                if is_mcq(q):
                    gi = "abcd".index(q["answer"])
                    pc = p_lett[k][gi]
                    pred = LETTERS[max(range(4), key=lambda j: p_lett[k][j])].lower()
                    rec = dict(p_yes=None, yn_mass=l_mass[k], p_options=p_lett[k], pred=pred,
                               p_correct=pc, correct=int(pred == q["answer"]))
                else:
                    gt = q["answer"] == "yes"
                    py = p_yes[k]
                    rec = dict(p_yes=py, yn_mass=yn_mass[k], pred="yes" if py > 0.5 else "no",
                               p_correct=py if gt else 1 - py, correct=int((py > 0.5) == gt))
                fout.write(json.dumps(dict(key=f"{q['qid']}|{c}|{act}", qid=q["qid"], img_key=q["img_key"],
                                           cond=c, action=act, qtype=q["qtype"], qcat=q["qcat"],
                                           answer=q["answer"], **rec)) + "\n")
            fout.flush()
            n += len(batch)
    print(f"shard {a.shard}: {n} runs in {time.time() - t0:.0f}s ({(time.time() - t0) / n * 1000:.0f} ms/run)")


if __name__ == "__main__":
    main()
