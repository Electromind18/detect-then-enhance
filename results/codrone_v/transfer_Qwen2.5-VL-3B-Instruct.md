# Transfer report: ../results/phase1 -> ../results/codrone_v | VLM Qwen2.5-VL-3B-Instruct
target: questions=1000 images=250 | blank-image acc 52.8% | levels [np.int64(0), np.int64(1)]
lookup table (source): {'clean': 'gamma', 'haze': 'dehaze', 'lowlight': 'gamma'} | target-best: {'clean': 'none', 'lowlight': 'none'}
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
| never-enhance                         |    66.8 |        |       72.2 |      70   | nan                 |
| always-clahe                          |    64.8 |        |       72   |      69.1 | -0.90 [-2.50,+0.60] |
| always-gamma                          |    66.8 |        |       70.8 |      69.2 | -0.80 [-2.40,+0.70] |
| always-dehaze                         |    65   |        |       70.7 |      68.4 | -1.60 [-3.10,-0.20] |
| router: GBDT (fit on source)          |    66.8 |        |       70.8 |      69.2 | -0.80 [-2.40,+0.70] |
| router: thresholds (fit on source)    |    66.8 |        |       70.8 |      69.2 | -0.80 [-2.40,+0.70] |
| oracle: true condition + source table |    66.8 |        |       70.8 |      69.2 | -0.80 [-2.40,+0.70] |
| oracle: true condition + TARGET table |    66.8 |        |       72.2 |      70   | +0.00 [+0.00,+0.00] |

## Key paired comparisons (router fit on source)
- router: GBDT (fit on source) vs always-gamma: +0.00 [+0.00,+0.00]
- router: GBDT (fit on source) vs always-dehaze: +0.80 [-1.00,+2.70]
- router: thresholds (fit on source) vs always-gamma: +0.00 [+0.00,+0.00]
- router: thresholds (fit on source) vs always-dehaze: +0.80 [-1.00,+2.70]

## By question type
| qtype           |   never-enhance |   always-gamma |   always-dehaze |   router: GBDT (fit on source) |
|:----------------|----------------:|---------------:|----------------:|-------------------------------:|
| presence_object |              70 |           69.2 |            68.4 |                           69.2 |

## By condition and severity
|                 |   never-enhance |   always-gamma |   always-dehaze |   router: GBDT (fit on source) |
|:----------------|----------------:|---------------:|----------------:|-------------------------------:|
| ('lowlight', 1) |            72.2 |           70.8 |            70.7 |                           70.8 |