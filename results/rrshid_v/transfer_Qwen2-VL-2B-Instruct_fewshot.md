few-shot detector: +60 labelled inputs ({'haze': 30, 'clean': 30}), repeated x21
# Transfer report: ../results/phase1 -> ../results/rrshid_v | VLM Qwen2-VL-2B-Instruct
target: questions=304 images=142 | blank-image acc 40.5% | levels [np.int64(0), np.int64(1), np.int64(2), np.int64(3)]
lookup table (source): {'clean': 'none', 'haze': 'dehaze', 'lowlight': 'gamma'} | target-best: {'clean': 'none', 'haze': 'dehaze'}
thresholds (source): lum<0.293 -> lowlight, dark>0.366 -> haze
detector accuracy on target: GBDT 99.2% | thresholds 48.0%
GBDT confusion (rows true, cols predicted):
| cond   |   clean |   haze |
|:-------|--------:|-------:|
| clean  |     299 |      5 |
| haze   |       0 |    304 |

## Accuracy % on target (image-level 95% CI)
| policy                                |   clean |   haze | lowlight   |   overall | gain_vs_never       |
|:--------------------------------------|--------:|-------:|:-----------|----------:|:--------------------|
| never-enhance                         |   100   |   68.1 |            |      84   | nan                 |
| always-clahe                          |    98.7 |   71.4 |            |      85   | +0.99 [-0.88,+3.03] |
| always-gamma                          |    99.3 |   72.4 |            |      85.9 | +1.81 [+0.32,+3.44] |
| always-dehaze                         |    99   |   79.9 |            |      89.5 | +5.43 [+2.59,+8.14] |
| router: GBDT (fit on source)          |   100   |   79.9 |            |      90   | +5.92 [+3.28,+8.44] |
| router: thresholds (fit on source)    |    99.3 |   78.9 |            |      89.1 | +5.10 [+2.78,+7.45] |
| oracle: true condition + source table |   100   |   79.9 |            |      90   | +5.92 [+3.28,+8.44] |
| oracle: true condition + TARGET table |   100   |   79.9 |            |      90   | +5.92 [+3.28,+8.44] |

## Key paired comparisons (router fit on source)
- router: GBDT (fit on source) vs always-dehaze: +0.49 [+0.00,+1.12]
- router: GBDT (fit on source) vs always-gamma: +4.11 [+2.02,+6.27]
- router: thresholds (fit on source) vs always-dehaze: -0.33 [-1.61,+0.99]
- router: thresholds (fit on source) vs always-gamma: +3.29 [+1.32,+5.27]

## By question type
| qtype            |   never-enhance |   always-gamma |   always-dehaze |   router: GBDT (fit on source) |
|:-----------------|----------------:|---------------:|----------------:|-------------------------------:|
| presence_object  |            84.7 |           85.8 |            93.7 |                           94.2 |
| presence_surface |            83.7 |           85.9 |            87.6 |                           88   |

## By condition and severity
|             |   never-enhance |   always-gamma |   always-dehaze |   router: GBDT (fit on source) |
|:------------|----------------:|---------------:|----------------:|-------------------------------:|
| ('haze', 1) |            70.4 |           73   |            81.7 |                           81.7 |
| ('haze', 2) |            69.7 |           70.7 |            75.8 |                           75.8 |
| ('haze', 3) |            63.3 |           73.3 |            82.2 |                           82.2 |