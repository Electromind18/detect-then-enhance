# Transfer report: ../results/phase1 -> ../results/aid_v | VLM Qwen2.5-VL-3B-Instruct
target: questions=1400 images=1000 | blank-image acc 43.5% | levels [np.int64(0), np.int64(1), np.int64(2), np.int64(3)]
lookup table (source): {'clean': 'gamma', 'haze': 'dehaze', 'lowlight': 'gamma'} | target-best: {'clean': 'clahe', 'haze': 'dehaze', 'lowlight': 'gamma'}
thresholds (source): lum<0.318 -> lowlight, dark>0.371 -> haze
detector accuracy on target: GBDT 92.5% | thresholds 83.3%
GBDT confusion (rows true, cols predicted):
| cond     |   clean |   haze |   lowlight |
|:---------|--------:|-------:|-----------:|
| clean    |    1249 |     72 |         79 |
| haze     |     140 |   1259 |          1 |
| lowlight |      22 |      0 |       1378 |

## Accuracy % on target (image-level 95% CI)
| policy                                |   clean |   haze |   lowlight |   overall | gain_vs_never       |
|:--------------------------------------|--------:|-------:|-----------:|----------:|:--------------------|
| never-enhance                         |    84.4 |   81.1 |       70.9 |      78.8 | nan                 |
| always-clahe                          |    84.7 |   82.8 |       73.3 |      80.3 | +1.45 [+0.54,+2.41] |
| always-gamma                          |    83.9 |   83.5 |       76.9 |      81.4 | +2.62 [+1.72,+3.55] |
| always-dehaze                         |    83.9 |   83.7 |       70.4 |      79.3 | +0.52 [-0.19,+1.28] |
| router: GBDT (fit on source)          |    83.5 |   83.6 |       76.9 |      81.3 | +2.52 [+1.63,+3.48] |
| router: thresholds (fit on source)    |    83.5 |   83.4 |       76.9 |      81.3 | +2.45 [+1.56,+3.41] |
| oracle: true condition + source table |    83.9 |   83.7 |       76.9 |      81.5 | +2.69 [+1.79,+3.67] |
| oracle: true condition + TARGET table |    84.7 |   83.7 |       76.9 |      81.8 | +2.95 [+2.06,+3.87] |

## Key paired comparisons (router fit on source)
- router: GBDT (fit on source) vs always-gamma: -0.10 [-0.57,+0.36]
- router: GBDT (fit on source) vs always-dehaze: +2.00 [+1.15,+2.87]
- router: thresholds (fit on source) vs always-gamma: -0.17 [-0.64,+0.28]
- router: thresholds (fit on source) vs always-dehaze: +1.93 [+1.08,+2.78]

## By question type
| qtype            |   never-enhance |   always-gamma |   always-dehaze |   router: GBDT (fit on source) |
|:-----------------|----------------:|---------------:|----------------:|-------------------------------:|
| presence_object  |            78.6 |           80.8 |            79.5 |                           80.9 |
| presence_surface |            72.8 |           75.7 |            72.7 |                           75.1 |
| scene_mcq        |            82.9 |           85.7 |            83.7 |                           85.8 |

## By condition and severity
|                 |   never-enhance |   always-gamma |   always-dehaze |   router: GBDT (fit on source) |
|:----------------|----------------:|---------------:|----------------:|-------------------------------:|
| ('haze', 1)     |            84   |           84.2 |            83   |                           82.9 |
| ('haze', 2)     |            80.9 |           83.2 |            82.2 |                           82.2 |
| ('haze', 3)     |            77.7 |           82.9 |            86.4 |                           86.4 |
| ('lowlight', 1) |            81.7 |           86.4 |            80.7 |                           86.4 |
| ('lowlight', 2) |            72.9 |           78.3 |            72.2 |                           78.3 |
| ('lowlight', 3) |            58.1 |           65.8 |            58.1 |                           65.8 |