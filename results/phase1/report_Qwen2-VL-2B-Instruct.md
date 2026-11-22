# Pilot report: Qwen2-VL-2B-Instruct

questions=2100  images=700  runs=27300
text-only (grey image) accuracy: 45.2%   mean answer-token prob. mass: 0.998

## Accuracy % (rows=condition, cols=action)
| cond     |   none |   clahe |   gamma |   dehaze |
|:---------|-------:|--------:|--------:|---------:|
| clean    |   88.6 |    85.8 |    87.5 |     87.1 |
| haze     |   85.2 |    84.3 |    86.8 |     87.5 |
| lowlight |   81.8 |    80.5 |    82.5 |     81.8 |

## Mean P(correct) (soft score, lower variance)
| cond     |   none |   clahe |   gamma |   dehaze |
|:---------|-------:|--------:|--------:|---------:|
| clean    |  0.794 |   0.769 |   0.788 |    0.78  |
| haze     |  0.753 |   0.746 |   0.777 |    0.781 |
| lowlight |  0.745 |   0.726 |   0.733 |    0.738 |

## Help / hurt rate % vs 'none' (by condition x action)
| cond     |   ('helped', 'clahe') |   ('helped', 'dehaze') |   ('helped', 'gamma') |   ('hurt', 'clahe') |   ('hurt', 'dehaze') |   ('hurt', 'gamma') |
|:---------|----------------------:|-----------------------:|----------------------:|--------------------:|---------------------:|--------------------:|
| clean    |                   2.1 |                    2.2 |                   1.2 |                 4.9 |                  3.6 |                 2.3 |
| haze     |                   2.9 |                    5.3 |                   3.5 |                 3.8 |                  3   |                 1.9 |
| lowlight |                   3   |                    1.6 |                   5.3 |                 4.3 |                  1.7 |                 4.6 |

## Hurt rate % by question type x action
| qtype            |   clahe |   dehaze |   gamma |
|:-----------------|--------:|---------:|--------:|
| presence_object  |     2.5 |      1.4 |     1.4 |
| presence_surface |     7   |      4.4 |     5   |
| scene_mcq        |     3.5 |      2.5 |     2.4 |

## Evidence drift: mean drift when enhancement HURT vs did not
- drift_dE: not-hurt=34.078  hurt=37.629
- drift_hue_drift: not-hurt=3.468  hurt=4.281
- drift_hf_ratio: not-hurt=1.907  hurt=2.112

## Policies (accuracy %, margin=0.02); gain vs never with 95% paired-bootstrap CI
| policy                                  |   clean |   haze |   lowlight |   overall |   gain | ci          |   pct_enhanced |
|:----------------------------------------|--------:|-------:|-----------:|----------:|-------:|:------------|---------------:|
| never-enhance                           |    88.6 |   85.2 |       81.8 |      85.2 |    0   | [+0.0,+0.0] |            0   |
| always-clahe                            |    85.8 |   84.3 |       80.5 |      83.5 |   -1.7 | [-2.3,-1.0] |          100   |
| always-gamma                            |    87.5 |   86.8 |       82.5 |      85.6 |    0.4 | [-0.2,+1.0] |          100   |
| always-dehaze                           |    87.1 |   87.5 |       81.8 |      85.5 |    0.3 | [-0.3,+0.9] |          100   |
| qtype-rule (CV)                         |    87   |   86.1 |       81.9 |      85   |   -0.2 | [-0.8,+0.4] |           66.7 |
| gate: image-only (CV)                   |    88.6 |   87   |       81.7 |      85.8 |    0.6 | [+0.3,+0.9] |           12.6 |
| gate: question-only (CV)                |    88.6 |   85.2 |       81.8 |      85.2 |    0   | [+0.0,+0.0] |            0   |
| gate: full (CV)                         |    88.6 |   86.5 |       81.8 |      85.6 |    0.4 | [+0.2,+0.7] |           10.6 |
| router: degradation (CV)                |    88.5 |   87.6 |       82.5 |      86.2 |    1   | [+0.4,+1.6] |           67.7 |
| router: degradation+qtype (CV)          |    88.3 |   87.7 |       82.4 |      86.1 |    0.9 | [+0.3,+1.5] |           65.4 |
| condition-oracle (knows degradation)    |    88.6 |   87.5 |       82.5 |      86.2 |    1   | [+0.4,+1.6] |           66.7 |
| ORACLE (any action correct; optimistic) |    92.1 |   92.1 |       89.1 |      91.1 |    5.9 | nan         |          nan   |

Degradation detector accuracy (image stats only, CV): 94.7%
- router vs always-gamma: +0.57 [+0.13,+1.00]
- router vs always-dehaze: +0.71 [+0.14,+1.29]
- router vs gate: full (CV): +0.56 [+0.02,+1.13]

## Accuracy % by question type (key evidence for 'question-aware')
| qtype            |   never-enhance |   always-gamma |   always-clahe |   always-dehaze |   gate: full (CV) |   router: degradation (CV) |   router: degradation+qtype (CV) |
|:-----------------|----------------:|---------------:|---------------:|----------------:|------------------:|---------------------------:|---------------------------------:|
| presence_object  |            90.7 |           91.3 |           90.1 |            91   |              90.8 |                       91.7 |                             91.6 |
| presence_surface |            77.1 |           77.6 |           73.4 |            77.8 |              78.1 |                       78.7 |                             78.7 |
| scene_mcq        |            87.9 |           88   |           87.1 |            87.6 |              88   |                       88.2 |                             88.1 |

## Hurt rate % by question type x action, degraded inputs only
| qtype            |   clahe |   dehaze |   gamma |
|:-----------------|--------:|---------:|--------:|
| presence_object  |     2.1 |      1.1 |     1.4 |
| presence_surface |     6.5 |      3.9 |     5.6 |
| scene_mcq        |     3.6 |      2   |     2.8 |

## Accuracy % by degradation severity (1=mild, 3=strong)
|                 |   never-enhance |   always-gamma |   always-dehaze |   router: degradation (CV) |
|:----------------|----------------:|---------------:|----------------:|---------------------------:|
| ('haze', 1)     |            89.9 |           90.2 |            89.9 |                       90.1 |
| ('haze', 2)     |            83.3 |           85   |            85.5 |                       85.5 |
| ('haze', 3)     |            82.5 |           85.3 |            87.2 |                       87.2 |
| ('lowlight', 1) |            87.6 |           86.1 |            87.3 |                       86.1 |
| ('lowlight', 2) |            84.8 |           84.5 |            84.2 |                       84.5 |
| ('lowlight', 3) |            71.8 |           76.3 |            72.6 |                       76.3 |

## GO / NO-GO
- oracle headroom: +5.9 pts (need >= 3)
- router vs never: +1.0 pts, CI lower bound +0.4 (want > 0)
- router effect on CLEAN images: -0.1 pts (want >= -0.5: gate must not hurt clean)
- image dependence: never-enhance clean 88.6% vs text-only 45.2%
=> GO: gating story holds. Scale to Phase 1.