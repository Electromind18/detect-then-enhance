# RRSHID against blind human labels: 168 questions, 48 images
Qwen2.5-VL-3B-Instruct | zero-shot GBDT: 67.3% of hazy inputs detected as haze
Qwen2.5-VL-3B-Instruct | few-shot GBDT (30 pairs): 100.0% of hazy inputs detected as haze
Qwen2.5-VL-3B-Instruct | thresholds: 86.3% of hazy inputs detected as haze

## Qwen2.5-VL-3B-Instruct
| input / policy | accuracy vs human (%) | gain vs never [95% CI, image-level] |
|---|---|---|
| haze-free image, no enhancement (reference input) | 69.0 | +24.4 [+15.1, +33.9] |
| hazy, never enhance | 44.6 | -- |
| hazy, always clahe | 57.7 | +13.1 [+5.1, +21.7] |
| hazy, always gamma | 47.0 | +2.4 [-1.8, +7.2] |
| hazy, always dehaze | 57.7 | +13.1 [+5.3, +21.3] |
| hazy, router zero-shot GBDT | 54.2 | +9.5 [+2.3, +17.4] |
| hazy, router few-shot GBDT (30 pairs) | 57.7 | +13.1 [+5.3, +21.3] |
| hazy, router thresholds | 54.2 | +9.5 [+1.8, +17.4] |
old reference check: VLM answer on haze-free image agrees with human on 69.0% of questions (83.3% of the 48 confident ones kept in the paper)
Qwen2-VL-2B-Instruct | zero-shot GBDT: 67.3% of hazy inputs detected as haze
Qwen2-VL-2B-Instruct | few-shot GBDT (30 pairs): 100.0% of hazy inputs detected as haze
Qwen2-VL-2B-Instruct | thresholds: 86.3% of hazy inputs detected as haze

## Qwen2-VL-2B-Instruct
| input / policy | accuracy vs human (%) | gain vs never [95% CI, image-level] |
|---|---|---|
| haze-free image, no enhancement (reference input) | 73.8 | +24.4 [+15.1, +34.3] |
| hazy, never enhance | 49.4 | -- |
| hazy, always clahe | 56.5 | +7.1 [+1.8, +12.7] |
| hazy, always gamma | 53.6 | +4.2 [+0.6, +8.6] |
| hazy, always dehaze | 58.3 | +8.9 [+2.9, +15.5] |
| hazy, router zero-shot GBDT | 54.8 | +5.4 [+0.0, +10.9] |
| hazy, router few-shot GBDT (30 pairs) | 58.3 | +8.9 [+2.9, +15.5] |
| hazy, router thresholds | 55.4 | +6.0 [+0.6, +11.6] |
old reference check: VLM answer on haze-free image agrees with human on 73.8% of questions (87.7% of the 73 confident ones kept in the paper)
