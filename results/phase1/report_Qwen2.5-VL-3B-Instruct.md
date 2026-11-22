# Pilot report: Qwen2.5-VL-3B-Instruct

questions=2100  images=700  runs=27300
text-only (grey image) accuracy: 48.2%   mean answer-token prob. mass: 0.999

## Accuracy % (rows=condition, cols=action)
| cond     |   none |   clahe |   gamma |   dehaze |
|:---------|-------:|--------:|--------:|---------:|
| clean    |   86.8 |    83.7 |    86.9 |     85.1 |
| haze     |   80.8 |    81.6 |    83.5 |     86.1 |
| lowlight |   71.3 |    75   |    77.2 |     70.6 |

## Mean P(correct) (soft score, lower variance)
| cond     |   none |   clahe |   gamma |   dehaze |
|:---------|-------:|--------:|--------:|---------:|
| clean    |  0.79  |   0.759 |   0.787 |    0.773 |
| haze     |  0.735 |   0.737 |   0.765 |    0.78  |
| lowlight |  0.657 |   0.679 |   0.694 |    0.645 |

## Help / hurt rate % vs 'none' (by condition x action)
| cond     |   ('helped', 'clahe') |   ('helped', 'dehaze') |   ('helped', 'gamma') |   ('hurt', 'clahe') |   ('hurt', 'dehaze') |   ('hurt', 'gamma') |
|:---------|----------------------:|-----------------------:|----------------------:|--------------------:|---------------------:|--------------------:|
| clean    |                   2.9 |                    2.1 |                   2   |                 5.9 |                  3.8 |                 2   |
| haze     |                   4.7 |                    8.5 |                   5.2 |                 3.9 |                  3.2 |                 2.5 |
| lowlight |                   7.1 |                    1.5 |                  10.1 |                 3.5 |                  2.3 |                 4.2 |

## Hurt rate % by question type x action
| qtype            |   clahe |   dehaze |   gamma |
|:-----------------|--------:|---------:|--------:|
| presence_object  |     2.6 |      2   |     1.3 |
| presence_surface |     4.8 |      3.7 |     3.3 |
| scene_mcq        |     5.9 |      3.6 |     4.1 |

## Evidence drift: mean drift when enhancement HURT vs did not
- drift_dE: not-hurt=34.066  hurt=37.835
- drift_hue_drift: not-hurt=3.496  hurt=3.481
- drift_hf_ratio: not-hurt=1.910  hurt=2.000

## Policies (accuracy %, margin=0.02); gain vs never with 95% paired-bootstrap CI
| policy                                  |   clean |   haze |   lowlight |   overall |   gain | ci          |   pct_enhanced |
|:----------------------------------------|--------:|-------:|-----------:|----------:|-------:|:------------|---------------:|
| never-enhance                           |    86.8 |   80.8 |       71.3 |      79.6 |    0   | [+0.0,+0.0] |            0   |
| always-clahe                            |    83.7 |   81.6 |       75   |      80.1 |    0.5 | [-0.4,+1.2] |          100   |
| always-gamma                            |    86.9 |   83.5 |       77.2 |      82.5 |    2.9 | [+2.2,+3.7] |          100   |
| always-dehaze                           |    85.1 |   86.1 |       70.6 |      80.6 |    1   | [+0.3,+1.7] |          100   |
| qtype-rule (CV)                         |    86.9 |   83.5 |       77.2 |      82.5 |    2.9 | [+2.2,+3.6] |          100   |
| gate: image-only (CV)                   |    86.8 |   84.7 |       75.9 |      82.5 |    2.8 | [+2.3,+3.4] |           38.2 |
| gate: question-only (CV)                |    87   |   81.2 |       72.9 |      80.4 |    0.7 | [+0.3,+1.2] |           40   |
| gate: full (CV)                         |    86.9 |   85.2 |       75.3 |      82.4 |    2.8 | [+2.2,+3.4] |           37   |
| router: degradation (CV)                |    86.8 |   86   |       77.2 |      83.3 |    3.7 | [+2.9,+4.5] |           94   |
| router: degradation+qtype (CV)          |    86.8 |   86   |       76.7 |      83.1 |    3.5 | [+2.7,+4.2] |           85   |
| condition-oracle (knows degradation)    |    86.9 |   86.1 |       77.2 |      83.4 |    3.8 | [+3.0,+4.6] |          100   |
| ORACLE (any action correct; optimistic) |    91.2 |   91.4 |       84.5 |      89   |    9.4 | nan         |          nan   |

Degradation detector accuracy (image stats only, CV): 94.7%
- router vs always-gamma: +0.78 [+0.33,+1.25]
- router vs always-dehaze: +2.71 [+2.08,+3.38]
- router vs gate: full (CV): +0.87 [+0.38,+1.38]

## Accuracy % by question type (key evidence for 'question-aware')
| qtype            |   never-enhance |   always-gamma |   always-clahe |   always-dehaze |   gate: full (CV) |   router: degradation (CV) |   router: degradation+qtype (CV) |
|:-----------------|----------------:|---------------:|---------------:|----------------:|------------------:|---------------------------:|---------------------------------:|
| presence_object  |            85.3 |           88.6 |           86.5 |            86.3 |              87.9 |                       88.9 |                             89   |
| presence_surface |            73.9 |           77.8 |           74.7 |            75.9 |              77.6 |                       79.1 |                             79   |
| scene_mcq        |            79.7 |           81.3 |           79.1 |            79.6 |              81.9 |                       82   |                             81.4 |

## Hurt rate % by question type x action, degraded inputs only
| qtype            |   clahe |   dehaze |   gamma |
|:-----------------|--------:|---------:|--------:|
| presence_object  |     2.4 |      1.6 |     1.6 |
| presence_surface |     3.5 |      3.1 |     3.6 |
| scene_mcq        |     5.2 |      3.4 |     4.9 |

## Accuracy % by degradation severity (1=mild, 3=strong)
|                 |   never-enhance |   always-gamma |   always-dehaze |   router: degradation (CV) |
|:----------------|----------------:|---------------:|----------------:|---------------------------:|
| ('haze', 1)     |            86.1 |           87.2 |            87.4 |                       86.9 |
| ('haze', 2)     |            80.3 |           81.3 |            84.4 |                       84.4 |
| ('haze', 3)     |            76   |           82.1 |            86.6 |                       86.6 |
| ('lowlight', 1) |            81.5 |           83.8 |            80.1 |                       83.7 |
| ('lowlight', 2) |            73   |           80   |            72.7 |                       80   |
| ('lowlight', 3) |            57.4 |           66.5 |            56.9 |                       66.5 |

## GO / NO-GO
- oracle headroom: +9.4 pts (need >= 3)
- router vs never: +3.7 pts, CI lower bound +2.9 (want > 0)
- router effect on CLEAN images: +0.0 pts (want >= -0.5: gate must not hurt clean)
- image dependence: never-enhance clean 86.8% vs text-only 48.2%
=> GO: gating story holds. Scale to Phase 1.