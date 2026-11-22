# Transfer report: ../results/phase1 -> ../results/codrone_v | VLM Qwen2-VL-2B-Instruct
target: questions=1000 images=250 | blank-image acc 52.8% | levels [np.int64(0), np.int64(1)]
lookup table (source): {'clean': 'none', 'haze': 'dehaze', 'lowlight': 'gamma'} | target-best: {'clean': 'clahe', 'lowlight': 'clahe'}
thresholds (source): lum<0.318 -> lowlight, dark>0.371 -> haze
detector accuracy on target: GBDT 40.4% | thresholds 98.4%
GBDT confusion (rows true, cols predicted):
| cond     |   clean |   lowlight |
|:---------|--------:|-----------:|
| clean    |     400 |          0 |
| lowlight |     596 |          4 |

## Accuracy % on target (image-level 95% CI)
| policy                                |   clean | haze   |   lowlight |   overall | gain_vs_never       |
|:--------------------------------------|--------:|:-------|-----------:|----------:|:--------------------|
| never-enhance                         |    66.2 |        |       72   |      69.7 | nan                 |
| always-clahe                          |    67.5 |        |       74   |      71.4 | +1.70 [+0.40,+3.00] |
| always-gamma                          |    67   |        |       74   |      71.2 | +1.50 [+0.10,+2.90] |
| always-dehaze                         |    65.5 |        |       73.5 |      70.3 | +0.60 [-0.50,+1.70] |
| router: GBDT (fit on source)          |    66.2 |        |       72   |      69.7 | +0.00 [+0.00,+0.00] |
| router: thresholds (fit on source)    |    66.2 |        |       74   |      70.9 | +1.20 [+0.00,+2.40] |
| oracle: true condition + source table |    66.2 |        |       74   |      70.9 | +1.20 [+0.00,+2.40] |
| oracle: true condition + TARGET table |    67.5 |        |       74   |      71.4 | +1.70 [+0.40,+3.00] |

## Key paired comparisons (router fit on source)
- router: GBDT (fit on source) vs always-clahe: -1.70 [-3.00,-0.40]
- router: GBDT (fit on source) vs always-gamma: -1.50 [-2.90,-0.10]
- router: GBDT (fit on source) vs always-dehaze: -0.60 [-1.70,+0.50]
- router: thresholds (fit on source) vs always-clahe: -0.50 [-2.00,+0.90]
- router: thresholds (fit on source) vs always-gamma: -0.30 [-1.00,+0.40]
- router: thresholds (fit on source) vs always-dehaze: +0.60 [-0.70,+1.90]

## By question type
| qtype           |   never-enhance |   always-gamma |   always-dehaze |   router: GBDT (fit on source) |
|:----------------|----------------:|---------------:|----------------:|-------------------------------:|
| presence_object |            69.7 |           71.2 |            70.3 |                           69.7 |

## By condition and severity
|                 |   never-enhance |   always-gamma |   always-dehaze |   router: GBDT (fit on source) |
|:----------------|----------------:|---------------:|----------------:|-------------------------------:|
| ('lowlight', 1) |              72 |             74 |            73.5 |                             72 |