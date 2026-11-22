# Transfer report: ../results/phase1 -> ../results/unseen_v | VLM Qwen2.5-VL-3B-Instruct
target: questions=600 images=200 | blank-image acc 49.2% | levels [np.int64(0), np.int64(4)]
lookup table (source): {'clean': 'gamma', 'haze': 'dehaze', 'lowlight': 'gamma'} | target-best: {'clean': 'none', 'haze': 'dehaze', 'lowlight': 'gamma'}
thresholds (source): lum<0.318 -> lowlight, dark>0.371 -> haze
detector accuracy on target: GBDT 97.2% | thresholds 89.7%
GBDT confusion (rows true, cols predicted):
| cond     |   clean |   haze |   lowlight |
|:---------|--------:|-------:|-----------:|
| clean    |     549 |     36 |         15 |
| haze     |       0 |    600 |          0 |
| lowlight |       0 |      0 |        600 |

## Accuracy % on target (image-level 95% CI)
| policy                                |   clean |   haze |   lowlight |   overall | gain_vs_never        |
|:--------------------------------------|--------:|-------:|-----------:|----------:|:---------------------|
| never-enhance                         |    87.3 |   63.2 |       51.7 |      67.4 | nan                  |
| always-clahe                          |    84.2 |   75   |       54.5 |      71.2 | +3.83 [+2.11,+5.56]  |
| always-gamma                          |    86.3 |   77.5 |       57.2 |      73.7 | +6.28 [+4.50,+7.94]  |
| always-dehaze                         |    85   |   85.3 |       51.8 |      74.1 | +6.67 [+5.33,+8.11]  |
| router: GBDT (fit on source)          |    86.5 |   85.3 |       57.2 |      76.3 | +8.94 [+7.00,+10.78] |
| router: thresholds (fit on source)    |    85.5 |   85.3 |       57.2 |      76   | +8.61 [+6.61,+10.50] |
| oracle: true condition + source table |    86.3 |   85.3 |       57.2 |      76.3 | +8.89 [+6.94,+10.72] |
| oracle: true condition + TARGET table |    87.3 |   85.3 |       57.2 |      76.6 | +9.22 [+7.39,+11.00] |

## Key paired comparisons (router fit on source)
- router: GBDT (fit on source) vs always-dehaze: +2.28 [+1.06,+3.44]
- router: GBDT (fit on source) vs always-gamma: +2.67 [+1.56,+3.78]
- router: thresholds (fit on source) vs always-dehaze: +1.94 [+0.72,+3.06]
- router: thresholds (fit on source) vs always-gamma: +2.33 [+1.17,+3.50]

## By question type
| qtype            |   never-enhance |   always-gamma |   always-dehaze |   router: GBDT (fit on source) |
|:-----------------|----------------:|---------------:|----------------:|-------------------------------:|
| presence_object  |            78.2 |           83.3 |            84.5 |                           85.3 |
| presence_surface |            63   |           67.5 |            68.8 |                           71.2 |
| scene_mcq        |            61   |           70.2 |            68.8 |                           72.5 |

## By condition and severity
|                 |   never-enhance |   always-gamma |   always-dehaze |   router: GBDT (fit on source) |
|:----------------|----------------:|---------------:|----------------:|-------------------------------:|
| ('haze', 4)     |            63.2 |           77.5 |            85.3 |                           85.3 |
| ('lowlight', 4) |            51.7 |           57.2 |            51.8 |                           57.2 |