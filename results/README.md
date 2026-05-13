# Thesis Results Package

This folder contains cleaned summaries of the final controlled detector experiment.

## Scope

The summaries focus on the three champion detectors:

- `UCF`
- `SPSL`
- `F3Net`

and the following evaluation sets:

- `Celeb-DF-v1 (screening subset)` using `test_frames = 4`
- `FaceForensics++` full test set using `test_frames = 4`
- `DFDCP` full test set using `test_frames = 4`

## Condition names

- `base`: original clean images
- `snapchat`: Snapchat-style text overlay distortion
- `instagram`: Instagram-style filter distortion
- `tiktok`: TikTok-style hybrid distortion

## Files

- `01_metric_tables.md`: full metric tables grouped by dataset
- `02_dataset_rankings.md`: thesis-friendly ranking tables grouped by dataset
- `03_detector_profiles.md`: detector-centric view across datasets
- `csv/final_results_long.csv`: one row per detector, dataset, and condition
- `csv/final_results_summary.csv`: one row per detector and dataset with the compact summary metrics

## Important note

`Celeb-DF-v1 (screening subset)` is the confirmation-stage screening dataset and should not be described as a full final dataset evaluation. `FaceForensics++` and `DFDCP` are the full final test-set evaluations currently included in this results package.
