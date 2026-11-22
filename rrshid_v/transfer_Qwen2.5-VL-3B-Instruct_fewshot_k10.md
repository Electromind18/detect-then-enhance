few-shot detector: +60 labelled inputs ({'haze': 30, 'clean': 30}), repeated x21
# Transfer report: ../results/phase1 -> ../results/rrshid_v | VLM Qwen2.5-VL-3B-Instruct
target: questions=184 images=106 | blank-image acc 65.2% | levels [np.int64(0), np.int64(1), np.int64(2), np.int64(3)]
lookup table (source): {'clean': 'gamma', 'haze': 'dehaze', 'lowlight': 'gamma'} | target-best: {'clean': 'none', 'haze': 'dehaze'}
thresholds (source): lum<0.293 -> lowlight, dark>0.366 -> haze
detector accuracy on target: GBDT 98.4% | thresholds 49.5%
GBDT confusion (rows true, cols predicted):
| cond   |   clean |   haze |
|:-------|--------:|-------:|
| clean  |     178 |      6 |
| haze   |       0 |    184 |

## Accuracy % on target (image-level 95% CI)
| policy                                |   clean |   haze | lowlight   |   overall | gain_vs_never       |
|:--------------------------------------|--------:|-------:|:-----------|----------:|:--------------------|
| never-enhance                         |   100   |   71.7 |            |      85.9 | nan                 |
| always-clahe                          |    94.6 |   79.3 |            |      87   | +1.09 [-2.99,+5.45] |
| always-gamma                          |    94.6 |   73.4 |            |      84   | -1.90 [-5.22,+1.61] |
| always-dehaze                         |    99.5 |   84.2 |            |      91.8 | +5.98 [+2.19,+9.84] |
| router: GBDT (fit on source)          |    94   |   84.2 |            |      89.1 | +3.26 [-1.06,+7.57] |
| router: thresholds (fit on source)    |    94   |   79.3 |            |      86.7 | +0.82 [-3.35,+5.03] |
| oracle: true condition + source table |    94.6 |   84.2 |            |      89.4 | +3.53 [-0.73,+7.87] |
| oracle: true condition + TARGET table |   100   |   84.2 |            |      92.1 | +6.25 [+2.69,+9.94] |

## Key paired comparisons (router fit on source)
- router: GBDT (fit on source) vs always-dehaze: -2.72 [-4.26,-1.26]
- router: GBDT (fit on source) vs always-gamma: +5.16 [+1.69,+8.47]
- router: thresholds (fit on source) vs always-dehaze: -5.16 [-7.34,-3.06]
- router: thresholds (fit on source) vs always-gamma: +2.72 [-0.31,+5.85]

## By question type
| qtype            |   never-enhance |   always-gamma |   always-dehaze |   router: GBDT (fit on source) |
|:-----------------|----------------:|---------------:|----------------:|-------------------------------:|
| presence_object  |            79.5 |           82.6 |            92.4 |                           92.4 |
| presence_surface |            89.4 |           84.7 |            91.5 |                           87.3 |

## By condition and severity
|             |   never-enhance |   always-gamma |   always-dehaze |   router: GBDT (fit on source) |
|:------------|----------------:|---------------:|----------------:|-------------------------------:|
| ('haze', 1) |            67.7 |           63.1 |            95.4 |                           95.4 |
| ('haze', 2) |            66   |           67.9 |            79.2 |                           79.2 |
| ('haze', 3) |            80.3 |           87.9 |            77.3 |                           77.3 |