"""Quick attack-feature importance analysis for one final experiment.

This is a lightweight diagnostic script, not part of the final experimental
pipeline. It trains one non-DP target model, builds the standard membership
attack dataset, and ranks the behavioral attack features.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.attack_features import build_attack_dataset
from src.config import CDC_EXPERIMENTS, NEWS20_EXPERIMENTS
from src.data import make_attack_splits_from_features
from src.final_runner_utils import load_final_dataset, resolve_device, train_and_evaluate_target


ATTACK_FEATURE_NAMES = [
    "loss",
    "max_confidence",
    "true_class_confidence",
    "entropy",
    "correctness",
    "top2_confidence_gap",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quick feature-importance diagnostic for membership attacks."
    )
    parser.add_argument("--dataset", choices=["cdc", "news20"], default="news20")
    parser.add_argument("--experiment", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="results_final/feature_importance")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=20,
        help="Permutation importance repeats on attack_test.",
    )
    return parser.parse_args()


def get_config(dataset: str, experiment: str | None) -> dict:
    experiments = CDC_EXPERIMENTS if dataset == "cdc" else NEWS20_EXPERIMENTS
    default_name = "cdc_vulnerable_5k" if dataset == "cdc" else "news20_vulnerable_2k"
    experiment_name = experiment or default_name
    if experiment_name not in experiments:
        available = ", ".join(sorted(experiments))
        raise ValueError(f"Unknown experiment '{experiment_name}'. Available: {available}")
    return copy.deepcopy(experiments[experiment_name])


def train_attack_models(X_train, y_train, X_test, y_test, seed: int):
    logistic = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=seed),
    )
    logistic.fit(X_train, y_train)
    logistic_scores = logistic.predict_proba(X_test)[:, 1]
    logistic_auc = roc_auc_score(y_test, logistic_scores)

    random_forest = RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)
    random_forest.fit(X_train, y_train)
    rf_scores = random_forest.predict_proba(X_test)[:, 1]
    rf_auc = roc_auc_score(y_test, rf_scores)

    return logistic, logistic_auc, random_forest, rf_auc


def build_importance_table(
    logistic,
    logistic_auc: float,
    random_forest,
    rf_auc: float,
    X_test,
    y_test,
    seed: int,
    n_repeats: int,
) -> pd.DataFrame:
    logistic_clf = logistic.named_steps["logisticregression"]
    logistic_abs_coef = abs(logistic_clf.coef_[0])
    logistic_signed_coef = logistic_clf.coef_[0]

    perm_log = permutation_importance(
        logistic,
        X_test,
        y_test,
        scoring="roc_auc",
        n_repeats=n_repeats,
        random_state=seed,
        n_jobs=-1,
    )
    perm_rf = permutation_importance(
        random_forest,
        X_test,
        y_test,
        scoring="roc_auc",
        n_repeats=n_repeats,
        random_state=seed,
        n_jobs=-1,
    )

    rows = []
    for idx, feature_name in enumerate(ATTACK_FEATURE_NAMES):
        rows.append(
            {
                "feature": feature_name,
                "logistic_abs_coef": logistic_abs_coef[idx],
                "logistic_signed_coef": logistic_signed_coef[idx],
                "logistic_auc": logistic_auc,
                "rf_gini_importance": random_forest.feature_importances_[idx],
                "rf_auc": rf_auc,
                "logistic_permutation_auc_drop_mean": perm_log.importances_mean[idx],
                "logistic_permutation_auc_drop_std": perm_log.importances_std[idx],
                "rf_permutation_auc_drop_mean": perm_rf.importances_mean[idx],
                "rf_permutation_auc_drop_std": perm_rf.importances_std[idx],
            }
        )

    df = pd.DataFrame(rows)
    df["mean_rank_score"] = (
        df["logistic_abs_coef"].rank(ascending=False)
        + df["rf_gini_importance"].rank(ascending=False)
        + df["logistic_permutation_auc_drop_mean"].rank(ascending=False)
        + df["rf_permutation_auc_drop_mean"].rank(ascending=False)
    ) / 4.0
    return df.sort_values(["mean_rank_score", "rf_permutation_auc_drop_mean"], ascending=[True, False])


def main():
    args = parse_args()
    config = get_config(args.dataset, args.experiment)
    device = resolve_device(args.device)
    X, y, dataset_name = load_final_dataset(args.dataset)

    target = train_and_evaluate_target(
        args.dataset,
        X,
        y,
        dataset_name,
        config,
        args.seed,
        device,
        use_dp=False,
    )
    if target is None:
        raise RuntimeError("Target training failed.")

    X_attack, y_attack = build_attack_dataset(
        target["model"],
        target["member_loader"],
        target["nonmember_loader"],
        device=device,
    )
    attack_splits = make_attack_splits_from_features(
        X_attack,
        y_attack,
        test_ratio=0.30,
        seed=args.seed,
    )

    logistic, logistic_auc, random_forest, rf_auc = train_attack_models(
        attack_splits["X_attack_train"],
        attack_splits["y_attack_train"],
        attack_splits["X_attack_test"],
        attack_splits["y_attack_test"],
        args.seed,
    )

    importance_df = build_importance_table(
        logistic,
        logistic_auc,
        random_forest,
        rf_auc,
        attack_splits["X_attack_test"],
        attack_splits["y_attack_test"],
        args.seed,
        args.n_repeats,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{config['name']}_seed_{args.seed}_attack_feature_importance.csv"
    importance_df.to_csv(out_path, index=False)

    print("\n[feature_importance] Attack feature ranking")
    print(f"dataset={dataset_name} | experiment={config['name']} | seed={args.seed}")
    print(f"logistic_auc={logistic_auc:.4f} | random_forest_auc={rf_auc:.4f}")
    print(importance_df.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\n[save] {out_path}")


if __name__ == "__main__":
    main()
