"""Shared image ops: degradations, enhancers, gate features, question categories."""
import re
import numpy as np
import cv2

CONDITIONS = ["clean", "haze", "lowlight"]
ACTIONS = ["none", "clahe", "gamma", "dehaze"]


# ---------------- degradations (input/output: uint8 RGB) ----------------
def add_haze(img, rng, t_range=(0.35, 0.6), A_range=(0.8, 0.95), tint=False):
    """Atmospheric scattering model I = J*t + A*(1-t), smooth spatially varying t."""
    J = img.astype(np.float32) / 255.0
    h, w = J.shape[:2]
    t0 = rng.uniform(*t_range)
    gx = np.linspace(-1, 1, w)[None, :] * rng.uniform(-0.1, 0.1)
    gy = np.linspace(-1, 1, h)[:, None] * rng.uniform(-0.1, 0.1)
    t = np.clip(t0 + gx + gy, 0.15, 0.95)[..., None]
    A = rng.uniform(*A_range)
    if tint:  # per-channel airlight (bluish/yellowish cast) -- not seen during router fitting
        A = np.clip(A + rng.uniform(-0.08, 0.08, 3), 0.6, 1.0)[None, None, :]
    I = J * t + A * (1 - t)
    return (np.clip(I, 0, 1) * 255).round().astype(np.uint8)


def add_lowlight(img, rng, gamma_range=(1.8, 2.6), scale_range=(0.35, 0.55), noise_sigma=0.01):
    J = img.astype(np.float32) / 255.0
    I = rng.uniform(*scale_range) * J ** rng.uniform(*gamma_range)
    I = I + rng.normal(0, noise_sigma, I.shape)  # mild sensor noise
    return (np.clip(I, 0, 1) * 255).round().astype(np.uint8)


HAZE_LEVELS = {1: (0.60, 0.75), 2: (0.40, 0.55), 3: (0.20, 0.35),            # transmission t
               4: (0.08, 0.18)}                       # 4 = UNSEEN: denser + tinted airlight
LOW_LEVELS = {1: ((1.4, 1.8), (0.55, 0.70), 0.008),                          # gamma, scale, noise
              2: ((1.8, 2.4), (0.35, 0.50), 0.012),
              3: ((2.4, 3.0), (0.20, 0.30), 0.018),
              4: ((2.6, 3.4), (0.12, 0.22), 0.030)}           # 4 = UNSEEN: darker + noisier


def degrade(img, cond, rng, level=None):
    """level None = pilot settings; 1/2/3 = mild/medium/strong."""
    if cond == "clean":
        return img.copy()
    if cond == "haze":
        return add_haze(img, rng) if level is None else add_haze(img, rng, t_range=HAZE_LEVELS[level],
                                                                  tint=(level == 4))
    if cond == "lowlight":
        if level is None:
            return add_lowlight(img, rng)
        g, s, n = LOW_LEVELS[level]
        return add_lowlight(img, rng, gamma_range=g, scale_range=s, noise_sigma=n)
    raise ValueError(cond)


# ---------------- enhancers ----------------
def enh_clahe(img, clip=2.0, tile=8):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile)).apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def enh_gamma(img, target=0.5):
    """Adaptive gamma: maps mean luminance to `target`."""
    x = img.astype(np.float32) / 255.0
    m = float(np.clip(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).mean() / 255.0, 1e-3, 0.999))
    g = float(np.clip(np.log(target) / np.log(m), 0.3, 3.0))
    return (np.clip(x ** g, 0, 1) * 255).round().astype(np.uint8)


def _guided_filter(I, p, r, eps):
    box = lambda x: cv2.boxFilter(x, -1, (r, r))
    mI, mp = box(I), box(p)
    a = (box(I * p) - mI * mp) / (box(I * I) - mI * mI + eps)
    b = mp - a * mI
    return box(a) * I + box(b)


def enh_dehaze(img, patch=15, omega=0.95, t0=0.1, work=256):
    """Dark channel prior; transmission estimated on downscaled image, refined by guided filter."""
    I = img.astype(np.float32) / 255.0
    h, w = I.shape[:2]
    s = min(1.0, work / max(h, w))
    Is = cv2.resize(I, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else I
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (patch, patch))
    dark = cv2.erode(Is.min(axis=2), k)
    n = max(1, dark.size // 1000)
    idx = np.argsort(dark.ravel())[-n:]
    A = Is.reshape(-1, 3)[idx].max(axis=0)
    A = np.maximum(A, 0.1)
    t = 1 - omega * cv2.erode((Is / A).min(axis=2), k)
    gray = cv2.cvtColor((Is * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
    t = _guided_filter(gray, t.astype(np.float32), r=max(8, patch * 2), eps=1e-3)
    if s < 1:
        t = cv2.resize(t, (w, h), interpolation=cv2.INTER_LINEAR)
    t = np.clip(t, t0, 1)[..., None]
    J = (I - A) / t + A
    return (np.clip(J, 0, 1) * 255).round().astype(np.uint8)


def enhance(img, action):
    return {"none": lambda x: x.copy(), "clahe": enh_clahe,
            "gamma": enh_gamma, "dehaze": enh_dehaze}[action](img)


# ---------------- gate features (cheap, no VLM) ----------------
def image_stats(img):
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
    hist = np.histogram(g, bins=64, range=(0, 1))[0].astype(np.float64)
    p = hist / hist.sum()
    p = p[p > 0]
    lap = cv2.Laplacian(g, cv2.CV_32F)
    dark = cv2.erode(img.min(axis=2), np.ones((15, 15), np.uint8)).astype(np.float32) / 255
    return {
        "lum_mean": float(g.mean()), "lum_std": float(g.std()),
        "rms_contrast": float(g.std() / (g.mean() + 1e-6)),
        "entropy": float(-(p * np.log2(p)).sum()),
        "haze_dark": float(dark.mean()),
        "sat_mean": float(hsv[..., 1].mean() / 255), "sat_std": float(hsv[..., 1].std() / 255),
        "lap_var": float(lap.var()),
        "noise_mad": float(np.median(np.abs(lap)) / 0.6745),
        "frac_dark": float((g < 0.1).mean()), "frac_bright": float((g > 0.9).mean()),
    }


def _hf_energy(g):
    return float(np.abs(cv2.Laplacian(g, cv2.CV_32F)).mean())


def drift_stats(src, out):
    """How much an enhancement changes evidence: colour (dE, hue) and texture (HF ratio)."""
    la = cv2.cvtColor(src, cv2.COLOR_RGB2LAB).astype(np.float32)
    lb = cv2.cvtColor(out, cv2.COLOR_RGB2LAB).astype(np.float32)
    dE = float(np.sqrt(((la - lb) ** 2).sum(-1)).mean())
    ha = cv2.cvtColor(src, cv2.COLOR_RGB2HSV)[..., 0].astype(np.float32) * 2
    hb = cv2.cvtColor(out, cv2.COLOR_RGB2HSV)[..., 0].astype(np.float32) * 2
    sat = cv2.cvtColor(src, cv2.COLOR_RGB2HSV)[..., 1] > 30  # hue only meaningful where saturated
    dh = np.abs(ha - hb)
    dh = np.minimum(dh, 360 - dh)
    hue = float(dh[sat].mean()) if sat.any() else 0.0
    ga = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
    gb = cv2.cvtColor(out, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
    return {"dE": dE, "hue_drift": hue, "hf_ratio": _hf_energy(gb) / (_hf_energy(ga) + 1e-6)}


# ---------------- question categories (rule-based) ----------------
_RULES = [
    ("colour", r"\b(colou?r|red|green|blue|white|black|brown|yellow|grey|gray|dark|bright)\b"),
    ("count", r"\b(how many|number of|count)\b"),
    ("area", r"\b(area|size|m2|square|larger|smaller|bigger)\b"),
    ("texture", r"\b(texture|damage|damaged|debris|rubble|road condition|crack)\b"),
    ("scene", r"\b(rural|urban|scene|area type|land ?use)\b"),
    ("compare", r"\b(more|less|than|equal)\b"),
    ("presence", r"\b(is there|are there|present|exist)\b"),
]


def question_category(q):
    q = q.lower()
    for name, pat in _RULES:
        if re.search(pat, q):
            return name
    return "other"
