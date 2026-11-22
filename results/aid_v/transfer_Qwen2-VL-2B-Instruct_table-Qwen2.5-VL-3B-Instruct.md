# Transfer report: ../results/phase1 -> ../results/aid_v | VLM Qwen2-VL-2B-Instruct | lookup table from Qwen2.5-VL-3B-Instruct
target: questions=1400 images=1000 | blank-image acc 40.9% | levels [np.int64(0), np.int64(1), np.int64(2), np.int64(3)]
lookup table (source): {'clean': 'gamma', 'haze': 'dehaze', 'lowlight': 'gamma'} | target-best: {'clean': 'none', 'haze': 'dehaze', 'lowlight': 'gamma'}
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
| never-enhance                         |    85.1 |   82.7 |       79.2 |      82.4 | nan                 |
| always-clahe                          |    85   |   83.3 |       78.6 |      82.3 | -0.05 [-0.79,+0.69] |
| always-gamma                          |    85.1 |   83.7 |       80.2 |      83   | +0.64 [-0.12,+1.40] |
| always-dehaze                         |    84.9 |   84.9 |       79.6 |      83.1 | +0.76 [+0.02,+1.50] |
| router: GBDT (fit on source)          |    84.9 |   84.9 |       80.2 |      83.3 | +0.95 [+0.17,+1.75] |
| router: thresholds (fit on source)    |    84.6 |   84.9 |       80.2 |      83.2 | +0.86 [+0.05,+1.67] |
| oracle: true condition + source table |    85.1 |   84.9 |       80.2 |      83.4 | +1.02 [+0.24,+1.81] |
| oracle: true condition + TARGET table |    85.1 |   84.9 |       80.2 |      83.4 | +1.05 [+0.31,+1.78] |

## Key paired comparisons (router fit on source)
- router: GBDT (fit on source) vs always-dehaze: +0.19 [-0.52,+0.94]
- router: GBDT (fit on source) vs always-gamma: +0.31 [-0.07,+0.69]
- router: thresholds (fit on source) vs always-dehaze: +0.10 [-0.60,+0.82]
- router: thresholds (fit on source) vs always-gamma: +0.21 [-0.21,+0.63]

## By question type
| qtype            |   never-enhance |   always-gamma |   always-dehaze |   router: GBDT (fit on source) |
|:-----------------|----------------:|---------------:|----------------:|-------------------------------:|
| presence_object  |            84.6 |           85.2 |            85.7 |                           85.5 |
| presence_surface |            74.2 |           75.8 |            75.9 |                           76.7 |
| scene_mcq        |            86.3 |           86.4 |            86.2 |                           86.3 |

## By condition and severity
|                 |   never-enhance |   always-gamma |   always-dehaze |   router: GBDT (fit on source) |
|:----------------|----------------:|---------------:|----------------:|-------------------------------:|
| ('haze', 1)     |            85.4 |           85.7 |            85.2 |                           85.4 |
| ('haze', 2)     |            82.6 |           82.4 |            84.7 |                           84.5 |
| ('haze', 3)     |            79.5 |           82.7 |            84.7 |                           84.7 |
| ('lowlight', 1) |            83   |           84.7 |            83.4 |                           84.7 |
| ('lowlight', 2) |            80.7 |           79.4 |            81.1 |                           79.4 |
| ('lowlight', 3) |            73.9 |           76.5 |            74.4 |                           76.5 |