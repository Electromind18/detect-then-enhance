# Transfer report: ../results/phase1 -> ../results/aid_v | VLM Qwen2-VL-2B-Instruct
target: questions=1400 images=1000 | blank-image acc 40.9% | levels [np.int64(0), np.int64(1), np.int64(2), np.int64(3)]
lookup table (source): {'clean': 'none', 'haze': 'dehaze', 'lowlight': 'gamma'} | target-best: {'clean': 'none', 'haze': 'dehaze', 'lowlight': 'gamma'}
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
| router: GBDT (fit on source)          |    84.7 |   84.8 |       80.4 |      83.3 | +0.93 [+0.17,+1.68] |
| router: thresholds (fit on source)    |    84.4 |   84.9 |       80.3 |      83.2 | +0.83 [+0.07,+1.62] |
| oracle: true condition + source table |    85.1 |   84.9 |       80.2 |      83.4 | +1.05 [+0.31,+1.78] |
| oracle: true condition + TARGET table |    85.1 |   84.9 |       80.2 |      83.4 | +1.05 [+0.31,+1.78] |

## Key paired comparisons (router fit on source)
- router: GBDT (fit on source) vs always-dehaze: +0.17 [-0.53,+0.90]
- router: GBDT (fit on source) vs always-gamma: +0.29 [-0.24,+0.79]
- router: thresholds (fit on source) vs always-dehaze: +0.07 [-0.59,+0.76]
- router: thresholds (fit on source) vs always-gamma: +0.19 [-0.33,+0.68]

## By question type
| qtype            |   never-enhance |   always-gamma |   always-dehaze |   router: GBDT (fit on source) |
|:-----------------|----------------:|---------------:|----------------:|-------------------------------:|
| presence_object  |            84.6 |           85.2 |            85.7 |                           85.6 |
| presence_surface |            74.2 |           75.8 |            75.9 |                           76.9 |
| scene_mcq        |            86.3 |           86.4 |            86.2 |                           86   |

## By condition and severity
|                 |   never-enhance |   always-gamma |   always-dehaze |   router: GBDT (fit on source) |
|:----------------|----------------:|---------------:|----------------:|-------------------------------:|
| ('haze', 1)     |            85.4 |           85.7 |            85.2 |                           85.2 |
| ('haze', 2)     |            82.6 |           82.4 |            84.7 |                           84.5 |
| ('haze', 3)     |            79.5 |           82.7 |            84.7 |                           84.7 |
| ('lowlight', 1) |            83   |           84.7 |            83.4 |                           84.9 |
| ('lowlight', 2) |            80.7 |           79.4 |            81.1 |                           79.6 |
| ('lowlight', 3) |            73.9 |           76.5 |            74.4 |                           76.5 |