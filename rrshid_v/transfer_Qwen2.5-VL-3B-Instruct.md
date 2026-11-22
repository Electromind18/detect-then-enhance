# Transfer report: ../results/phase1 -> ../results/rrshid_v | VLM Qwen2.5-VL-3B-Instruct
target: questions=184 images=106 | blank-image acc 65.2% | levels [np.int64(0), np.int64(1), np.int64(2), np.int64(3)]
lookup table (source): {'clean': 'gamma', 'haze': 'dehaze', 'lowlight': 'gamma'} | target-best: {'clean': 'none', 'haze': 'dehaze'}
thresholds (source): lum<0.318 -> lowlight, dark>0.371 -> haze
detector accuracy on target: GBDT 73.4% | thresholds 48.1%
GBDT confusion (rows true, cols predicted):
| cond   |   clean |   haze |   lowlight |
|:-------|--------:|-------:|-----------:|
| clean  |     156 |      3 |         25 |
| haze   |      70 |    114 |          0 |

## Accuracy % on target (image-level 95% CI)
| policy                                |   clean |   haze | lowlight   |   overall | gain_vs_never       |
|:--------------------------------------|--------:|-------:|:-----------|----------:|:--------------------|
| never-enhance                         |   100   |   71.7 |            |      85.9 | nan                 |
| always-clahe                          |    94.6 |   79.3 |            |      87   | +1.09 [-2.99,+5.45] |
| always-gamma                          |    94.6 |   73.4 |            |      84   | -1.90 [-5.22,+1.61] |
| always-dehaze                         |    99.5 |   84.2 |            |      91.8 | +5.98 [+2.19,+9.84] |
| router: GBDT (fit on source)          |    94.6 |   77.7 |            |      86.1 | +0.27 [-3.46,+4.24] |
| router: thresholds (fit on source)    |    94   |   79.3 |            |      86.7 | +0.82 [-3.35,+5.03] |
| oracle: true condition + source table |    94.6 |   84.2 |            |      89.4 | +3.53 [-0.73,+7.87] |
| oracle: true condition + TARGET table |   100   |   84.2 |            |      92.1 | +6.25 [+2.69,+9.94] |

## Key paired comparisons (router fit on source)
- router: GBDT (fit on source) vs always-dehaze: -5.71 [-8.38,-2.87]
- router: GBDT (fit on source) vs always-gamma: +2.17 [-0.30,+4.80]
- router: thresholds (fit on source) vs always-dehaze: -5.16 [-7.34,-3.06]
- router: thresholds (fit on source) vs always-gamma: +2.72 [-0.31,+5.85]

## By question type
| qtype            |   never-enhance |   always-gamma |   always-dehaze |   router: GBDT (fit on source) |
|:-----------------|----------------:|---------------:|----------------:|-------------------------------:|
| presence_object  |            79.5 |           82.6 |            92.4 |                           86.4 |
| presence_surface |            89.4 |           84.7 |            91.5 |                           86   |

## By condition and severity
|             |   never-enhance |   always-gamma |   always-dehaze |   router: GBDT (fit on source) |
|:------------|----------------:|---------------:|----------------:|-------------------------------:|
| ('haze', 1) |            67.7 |           63.1 |            95.4 |                           76.9 |
| ('haze', 2) |            66   |           67.9 |            79.2 |                           75.5 |
| ('haze', 3) |            80.3 |           87.9 |            77.3 |                           80.3 |