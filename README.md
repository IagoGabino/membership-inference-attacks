# Membership Inference Attacks and Defenses

An experimental study of how vulnerable neural-network classifiers are to membership inference attacks, and of how differential privacy and restricted API outputs reduce that risk.

Final project for Privacy-Preserving Machine Learning (UFMG).

## Setup

- **Datasets:** CDC Diabetes (tabular, downloaded from UCI) and 20 Newsgroups (text, TF-IDF features, downloaded by scikit-learn).
- **Target models:** configurable PyTorch MLPs, from regularized to deliberately overfitted, trained on 2k to 40k samples.
- **Splits:** four disjoint partitions (fit members, validation, non-members, holdout). Imputers, scalers and vectorizers are fitted only on the training members, to avoid leakage.
- **Attacks:** threshold, logistic regression and random forest on behavioral features of the model output (loss, confidence, entropy and others). Metrics: AUC, TPR at low FPR, top-k precision.
- **Defense:** DP-SGD with Opacus at several noise multipliers.
- **API exposure:** what the attacker sees, from full loss and probabilities down to rounded probabilities, top-1 confidence or label only.

## Results

Mean best attack AUC (0.5 is chance), 3 seeds.

| Scenario | AUC |
|---|---|
| CDC, regularized / baseline model | 0.51 |
| CDC, overfitted, 5k samples | 0.65 |
| CDC, with DP noise 0.6 (from 0.65 without DP) | 0.53 |
| 20 Newsgroups, regularized | 0.69 |
| 20 Newsgroups, baseline | 0.73 |
| 20 Newsgroups, overfitted, 2k samples | 0.94 |
| 20 Newsgroups, overfitted + DP noise 1.0 (from 0.89 without DP) | 0.65 |
| 20 Newsgroups API exposure: full output / label only | 0.80 / 0.67 |

Overfitting drives the risk; regularization, DP noise and exposing less of the model output each reduce it. Per-experiment tables are in `results_final/`; the methodology audit is in `docs/final_experiment_audit.md`.

## Running

```bash
pip install -r requirements.txt opacus
python -m src.run_final_all                  # all suites (add --smoke for a quick run, --only cdc to pick one)
python -m src.build_final_summary            # summary tables
```

All experiment configurations are in `src/config.py`.
