# Dataset Rankings

These are the compact tables most suitable for the thesis main text.

## Celeb-DF-v1 (screening subset)

| rank | detector | base_video_auc | snapchat_video_auc | instagram_video_auc | tiktok_video_auc | champion_score | avg_distorted_video_auc | worst_distorted_video_auc | base_to_distorted_drop |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | UCF | 0.8413 | 0.8183 | 0.8379 | 0.7500 | 0.8177 | 0.8021 | 0.7500 | 0.0392 |
| 2 | SPSL | 0.8230 | 0.7861 | 0.8132 | 0.7267 | 0.7944 | 0.7753 | 0.7267 | 0.0477 |
| 3 | F3Net | 0.7997 | 0.6846 | 0.7712 | 0.6014 | 0.7313 | 0.6858 | 0.6014 | 0.1139 |

## FaceForensics++

| rank | detector | base_video_auc | snapchat_video_auc | instagram_video_auc | tiktok_video_auc | champion_score | avg_distorted_video_auc | worst_distorted_video_auc | base_to_distorted_drop |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | UCF | 0.9951 | 0.9679 | 0.9892 | 0.9573 | 0.9809 | 0.9715 | 0.9573 | 0.0236 |
| 2 | F3Net | 0.9943 | 0.9527 | 0.9822 | 0.9337 | 0.9715 | 0.9562 | 0.9337 | 0.0381 |
| 3 | SPSL | 0.9568 | 0.9176 | 0.9505 | 0.9359 | 0.9435 | 0.9346 | 0.9176 | 0.0221 |

## DFDCP

| rank | detector | base_video_auc | snapchat_video_auc | instagram_video_auc | tiktok_video_auc | champion_score | avg_distorted_video_auc | worst_distorted_video_auc | base_to_distorted_drop |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | F3Net | 0.7417 | 0.7042 | 0.7337 | 0.7531 | 0.7349 | 0.7303 | 0.7042 | 0.0113 |
| 2 | SPSL | 0.7478 | 0.6885 | 0.7530 | 0.7135 | 0.7301 | 0.7183 | 0.6885 | 0.0294 |
| 3 | UCF | 0.7161 | 0.6746 | 0.7109 | 0.6597 | 0.6954 | 0.6817 | 0.6597 | 0.0343 |

## Cross-Dataset Full-Evaluation Average

This table averages the full-test-set results only, i.e. `FaceForensics++` and `DFDCP`.

| rank | detector | mean_champion_score | mean_base_video_auc | mean_avg_distorted_video_auc | mean_worst_distorted_video_auc |
|---:|---|---:|---:|---:|---:|
| 1 | F3Net | 0.8532 | 0.8680 | 0.8433 | 0.8190 |
| 2 | UCF | 0.8382 | 0.8556 | 0.8266 | 0.8085 |
| 3 | SPSL | 0.8368 | 0.8523 | 0.8265 | 0.8030 |

