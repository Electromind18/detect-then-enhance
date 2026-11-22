# Transfer report: ../results/phase1 -> ../results/unseen_v | VLM Qwen2-VL-2B-Instruct
target: questions=600 images=200 | blank-image acc 46.7% | levels [np.int64(0), np.int64(4)]
lookup table (source): {'clean': 'none', 'haze': 'dehaze', 'lowlight': 'gamma'} | target-best: {'clean': 'dehaze', 'haze': 'dehaze', 'lowlight': 'gamma'}
thresholds (source): lum<0.318 -> lowlight, dark>0.371 -> haze
detector accuracy on target: GBDT 97.2% | thresholds 89.7%
GBDT confusion (rows true, cols predicted):
| cond     |   clean |   haze |   lowlight |
|:---------|--------:|-------:|-----------:|
| clean    |     549 |     36 |         15 |
| haze     |       0 |    600 |          0 |
| lowlight |       0 |      0 |        600 |

## Accuracy % on target (image-level 95% CI)
| policy                                |   clean |   haze |   lowlight |   overall | gain_vs_never       |
|:--------------------------------------|--------:|-------:|-----------:|----------:|:--------------------|
| never-enhance                         |    86   |   75.8 |         64 |      75.3 | nan                 |
| always-clahe                          |    85.3 |   77.2 |         64 |      75.5 | +0.22 [-1.17,+1.67] |
| always-gamma                          |    85.7 |   82   |         66 |      77.9 | +2.61 [+1.11,+4.17] |
| always-dehaze                         |    86.3 |   84.3 |         64 |      78.2 | +2.94 [+1.89,+4.00] |
| router: GBDT (fit on source)          |    86   |   84.3 |         66 |      78.8 | +3.50 [+1.94,+4.95] |
| router: thresholds (fit on source)    |    84.3 |   84.3 |         66 |      78.2 | +2.94 [+1.39,+4.50] |
| oracle: true condition + source table |    86   |   84.3 |         66 |      78.8 | +3.50 [+2.00,+4.94] |
| oracle: true condition + TARGET table |    86.3 |   84.3 |         66 |      78.9 | +3.61 [+2.11,+5.06] |

## Key paired comparisons (router fit on source)
- router: GBDT (fit on source) vs always-dehaze: +0.56 [-0.61,+1.78]
- router: GBDT (fit on source) vs always-gamma: +0.89 [-0.06,+1.72]
- router: thresholds (fit on source) vs always-dehaze: +0.00 [-1.17,+1.33]
- router: thresholds (fit on source) vs always-gamma: +0.33 [-0.50,+1.11]

## By question type
| qtype            |   never-enhance |   always-gamma |   always-dehaze |   router: GBDT (fit on source) |
|:-----------------|----------------:|---------------:|----------------:|-------------------------------:|
| presence_object  |            85.3 |           86.8 |            86.8 |                           86.2 |
| presence_surface |            64.8 |           70.3 |            69.5 |                           72   |
| scene_mcq        |            75.7 |           76.5 |            78.3 |                           78.2 |

## By condition and severity
|                 |   never-enhance |   always-gamma |   always-dehaze |   router: GBDT (fit on source) |
|:----------------|----------------:|---------------:|----------------:|-------------------------------:|
| ('haze', 4)     |            75.8 |             82 |            84.3 |                           84.3 |
| ('lowlight', 4) |            64   |             66 |            64   |                           66   |