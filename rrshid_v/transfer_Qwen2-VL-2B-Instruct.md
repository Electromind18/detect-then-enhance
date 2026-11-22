# Transfer report: ../results/phase1 -> ../results/rrshid_v | VLM Qwen2-VL-2B-Instruct
target: questions=304 images=142 | blank-image acc 40.5% | levels [np.int64(0), np.int64(1), np.int64(2), np.int64(3)]
lookup table (source): {'clean': 'none', 'haze': 'dehaze', 'lowlight': 'gamma'} | target-best: {'clean': 'none', 'haze': 'dehaze'}
thresholds (source): lum<0.318 -> lowlight, dark>0.371 -> haze
detector accuracy on target: GBDT 76.5% | thresholds 46.9%
GBDT confusion (rows true, cols predicted):
| cond   |   clean |   haze |   lowlight |
|:-------|--------:|-------:|-----------:|
| clean  |     265 |      2 |         37 |
| haze   |     104 |    200 |          0 |

## Accuracy % on target (image-level 95% CI)
| policy                                |   clean |   haze | lowlight   |   overall | gain_vs_never       |
|:--------------------------------------|--------:|-------:|:-----------|----------:|:--------------------|
| never-enhance                         |   100   |   68.1 |            |      84   | nan                 |
| always-clahe                          |    98.7 |   71.4 |            |      85   | +0.99 [-0.88,+3.03] |
| always-gamma                          |    99.3 |   72.4 |            |      85.9 | +1.81 [+0.32,+3.44] |
| always-dehaze                         |    99   |   79.9 |            |      89.5 | +5.43 [+2.59,+8.14] |
| router: GBDT (fit on source)          |   100   |   77.3 |            |      88.7 | +4.61 [+2.72,+6.73] |
| router: thresholds (fit on source)    |    99.3 |   78.9 |            |      89.1 | +5.10 [+2.78,+7.45] |
| oracle: true condition + source table |   100   |   79.9 |            |      90   | +5.92 [+3.28,+8.44] |
| oracle: true condition + TARGET table |   100   |   79.9 |            |      90   | +5.92 [+3.28,+8.44] |

## Key paired comparisons (router fit on source)
- router: GBDT (fit on source) vs always-dehaze: -0.82 [-2.62,+1.19]
- router: GBDT (fit on source) vs always-gamma: +2.80 [+0.78,+4.93]
- router: thresholds (fit on source) vs always-dehaze: -0.33 [-1.61,+0.99]
- router: thresholds (fit on source) vs always-gamma: +3.29 [+1.32,+5.27]

## By question type
| qtype            |   never-enhance |   always-gamma |   always-dehaze |   router: GBDT (fit on source) |
|:-----------------|----------------:|---------------:|----------------:|-------------------------------:|
| presence_object  |            84.7 |           85.8 |            93.7 |                           92.1 |
| presence_surface |            83.7 |           85.9 |            87.6 |                           87.1 |

## By condition and severity
|             |   never-enhance |   always-gamma |   always-dehaze |   router: GBDT (fit on source) |
|:------------|----------------:|---------------:|----------------:|-------------------------------:|
| ('haze', 1) |            70.4 |           73   |            81.7 |                           74.8 |
| ('haze', 2) |            69.7 |           70.7 |            75.8 |                           75.8 |
| ('haze', 3) |            63.3 |           73.3 |            82.2 |                           82.2 |