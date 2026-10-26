# Weak Supervision Dataset V1

This version contains 1,000 complete question samples and 1,000 matching synthetic
answer samples exported from the `balanced-weak-v1` generation run.

## File layout

```text
weak_supervision_v1/
  all/
    question_ranking_all.csv
    answer_scoring_all.csv
  splits/
    question_train.csv
    question_validation.csv
    question_test.csv
    answer_train.csv
    answer_validation.csv
    answer_test.csv
  manifest.json
```

The CSV files are generated artifacts and are excluded from Git. `manifest.json` keeps
their row counts and SHA-256 checksums so a local export can be verified.

## Split policy

- Random seed: `42`
- Train/validation/test context-group ratio: `70/15/15`
- One `group_id` is assigned to exactly one partition.
- Context groups are stratified by interview stage and target criterion.
- Question and answer rows use the same group assignment.
- The held-out test files must not be used for model selection.

Current row counts are 701 train, 147 validation, and 152 test rows for each task.
The corresponding context-group counts are 168, 35, and 37.

## Rebuild

```powershell
python -B scripts\export_weak_supervision_datasets.py
python -B scripts\split_weak_supervision_datasets.py
python -B scripts\train_weak_supervision_models.py
```

The trained model files and reports are written to `artifacts/weak_supervision/v1/`.
