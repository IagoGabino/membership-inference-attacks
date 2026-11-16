"""Shared helpers for the final experiment runners."""

from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path
from typing import Iterable
import gc
import torch

def cleanup_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch

from src.api_exposure import build_api_exposure_attack_dataset
from src.attack_features import build_attack_dataset
from src.config import API_EXPOSURE_MODES, DEBUG_CONFIG
from src.data import (
    create_splits,
    load_dataset,
    make_attack_splits_from_features,
    make_dataloader,
    preprocess_splits,
)
from src.data_20newsgroups import (
    create_text_splits_20ng,
    load_20newsgroups_texts,
    preprocess_text_splits_20ng,
)
from src.dp_train import train_model_with_dp
from src.evaluate import compute_confidence_gap, compute_generalization_gap, evaluate_model
from src.membership_attack import run_all_attacks
from src.models import build_model
from src.train import train_model
from src.utils import ensure_dirs, get_device, set_seed


ATTACK_METRIC_COLUMNS = [
    "attack_accuracy",
    "attack_auc",
    "attack_precision",
    "attack_recall",
    "attack_f1",
    "tpr_at_fpr_1",
    "tpr_at_fpr_5",
    "tpr_at_fpr_10",
    "top_1_percent_precision",
    "top_5_percent_precision",
    "top_10_percent_precision",
    "top_1_percent_recall",
    "top_5_percent_recall",
    "top_10_percent_recall",
    "lift_at_1_percent",
    "lift_at_5_percent",
    "lift_at_10_percent",
]

RESULT_COLUMNS = [
    "dataset",
    "model",
    "seed",
    "attack_type",
    "train_acc",
    "test_acc",
    "train_f1",
    "test_f1",
    "train_auc",
    "test_auc",
    "generalization_gap",
    "loss_gap",
    "confidence_gap",
    *ATTACK_METRIC_COLUMNS,
    "train_size",
    "epochs",
    "actual_epochs",
    "regularization_type",
    "hidden_layers",
    "dropout",
    "weight_decay",
    "early_stopping",
    "restore_best_model",
    "epsilon",
    "delta",
    "noise_multiplier",
    "max_grad_norm",
    "status",
]

EXPOSURE_ORDER = [
    "oracle_full",
    "probs_full",
    "probs_rounded_2",
    "probs_rounded_1",
    "top1_confidence",
    "label_only",
]


def parse_common_args(description: str, default_output_dir: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 7])
    parser.add_argument("--output-dir", type=str, default=default_output_dir)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return get_device()
    if device_arg == "cuda" and not torch.cuda.is_available():
        print("[device] CUDA requested but unavailable. Using CPU.")
        return torch.device("cpu")
    device = torch.device(device_arg)
    print(f"[device] Using {device}.")
    return device


def make_dirs(output_dir: str | Path) -> tuple[Path, Path, Path]:
    root = Path(output_dir)
    metrics_dir = root / "metrics"
    figures_dir = root / "figures"
    ensure_dirs(str(metrics_dir), str(figures_dir))
    return root, metrics_dir, figures_dir


def apply_smoke_overrides(config: dict, dataset: str) -> dict:
    cfg = copy.deepcopy(config)
    cfg["name"] = config["name"]
    cfg["experiment_sample_size"] = min(
        int(config.get("experiment_sample_size") or DEBUG_CONFIG["experiment_sample_size"]),
        500 if dataset == "news20" else 800,
    )
    cfg["epochs"] = min(int(config.get("epochs", 1)), int(DEBUG_CONFIG["epochs"]))
    cfg["batch_size"] = min(int(config.get("batch_size", 128)), int(DEBUG_CONFIG["batch_size"]))
    cfg["patience"] = min(int(config.get("patience") or 1), 1)
    if dataset == "news20":
        cfg["max_tfidf_features"] = min(
            int(config.get("max_tfidf_features", 5000)),
            300,
        )
    hidden = cfg.get("hidden_layers", [])
    cfg["hidden_layers"] = [min(int(h), 64) for h in hidden[:2]] or [32]
    return cfg


def load_final_dataset(dataset: str):
    if dataset == "cdc":
        X, y, _, dataset_name = load_dataset("cdc_diabetes")
        return X, y, dataset_name
    if dataset == "news20":
        texts, y, dataset_name, _ = load_20newsgroups_texts()
        return texts, y, dataset_name
    raise ValueError(f"Unknown dataset: {dataset}")


def prepare_splits(dataset: str, X, y, config: dict, seed: int) -> dict:
    if dataset == "cdc":
        raw = create_splits(
            X,
            y,
            seed=seed,
            experiment_sample_size=config.get("experiment_sample_size"),
        )
        return preprocess_splits(raw, binary_target=True)
    if dataset == "news20":
        raw = create_text_splits_20ng(
            X,
            y,
            seed=seed,
            experiment_sample_size=config.get("experiment_sample_size"),
        )
        return preprocess_text_splits_20ng(
            raw,
            max_features=config.get("max_tfidf_features", 5000),
        )
    raise ValueError(f"Unknown dataset: {dataset}")


def _safe_float(value):
    if value is None:
        return None
    try:
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return value
        return float(value)
    except Exception:
        return value


def make_result_row(
    dataset_name: str,
    config: dict,
    seed: int,
    attack_type: str,
    attack_metrics: dict,
    target_metrics: dict,
    status: str = "ok",
) -> dict:
    train_metrics = target_metrics["train_metrics"]
    test_metrics = target_metrics["test_metrics"]
    gap_metrics = target_metrics["gap_metrics"]
    conf_gap = target_metrics["conf_gap"]
    row = {
        "dataset": dataset_name,
        "model": config["name"],
        "seed": seed,
        "attack_type": attack_type,
        "train_acc": train_metrics.get("accuracy"),
        "test_acc": test_metrics.get("accuracy"),
        "train_f1": train_metrics.get("f1"),
        "test_f1": test_metrics.get("f1"),
        "train_auc": train_metrics.get("auc"),
        "test_auc": test_metrics.get("auc"),
        "generalization_gap": gap_metrics.get("gen_gap"),
        "loss_gap": gap_metrics.get("loss_gap"),
        "confidence_gap": conf_gap.get("confidence_gap"),
        "train_size": config.get("experiment_sample_size"),
        "epochs": config.get("epochs"),
        "actual_epochs": target_metrics.get("actual_epochs"),
        "regularization_type": config.get("regularization_type"),
        "hidden_layers": "-".join(str(v) for v in config.get("hidden_layers", [])),
        "dropout": config.get("dropout"),
        "weight_decay": config.get("weight_decay"),
        "early_stopping": config.get("early_stopping"),
        "restore_best_model": config.get("restore_best_model", config.get("early_stopping", False)),
        "epsilon": target_metrics.get("epsilon"),
        "delta": config.get("delta"),
        "noise_multiplier": config.get("noise_multiplier"),
        "max_grad_norm": config.get("max_grad_norm"),
        "status": status,
    }
    for col in ATTACK_METRIC_COLUMNS:
        row[col] = attack_metrics.get(col)
    return row


def save_results(rows: list[dict], path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    cols = columns or RESULT_COLUMNS
    for col in cols:
        if col not in df.columns:
            df[col] = None
    df = df[cols]
    df.to_csv(path, index=False)
    print(f"[save] {path} ({len(df)} rows)")
    return df


def train_and_evaluate_target(
    dataset: str,
    X,
    y,
    dataset_name: str,
    config: dict,
    seed: int,
    device: torch.device,
    use_dp: bool = False,
) -> dict | None:
    set_seed(seed)
    print(
        f"\n[target] dataset={dataset_name} | model={config['name']} | "
        f"seed={seed} | sample_size={config.get('experiment_sample_size')} | dp={use_dp}"
    )
    splits = prepare_splits(dataset, X, y, config, seed)
    batch_size = config.get("batch_size", 256)

    train_loader = make_dataloader(splits["X_fit"], splits["y_fit"], batch_size=batch_size, shuffle=True)
    val_loader = make_dataloader(splits["X_val"], splits["y_val"], batch_size=batch_size, shuffle=False)
    member_loader = make_dataloader(splits["X_fit"], splits["y_fit"], batch_size=batch_size, shuffle=False)
    nonmember_loader = make_dataloader(
        splits["X_nonmember"], splits["y_nonmember"], batch_size=batch_size, shuffle=False
    )

    model = build_model(config, input_dim=splits["X_fit"].shape[1])
    epsilon = None
    if use_dp:
        dp_result = train_model_with_dp(model, train_loader, val_loader, config, device=device, verbose=True)
        if dp_result is None:
            return None
        target_model = dp_result["model"]
        actual_epochs = dp_result.get("actual_epochs")
        epsilon = dp_result.get("epsilon")
    else:
        train_result = train_model(model, train_loader, val_loader, config, device=device, verbose=True)
        target_model = train_result["model"]
        actual_epochs = train_result.get("actual_epochs")

    train_metrics = evaluate_model(target_model, member_loader, device=device)
    test_metrics = evaluate_model(target_model, nonmember_loader, device=device)
    gap_metrics = compute_generalization_gap(train_metrics, test_metrics)
    conf_gap = compute_confidence_gap(train_metrics["probabilities"], test_metrics["probabilities"])

    target_metrics = {
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "gap_metrics": gap_metrics,
        "conf_gap": conf_gap,
        "actual_epochs": actual_epochs,
        "epsilon": epsilon,
    }

    return {
        "model": target_model,
        "member_loader": member_loader,
        "nonmember_loader": nonmember_loader,
        "target_metrics": target_metrics,
        "splits": splits,
    }


def run_current_attacks(
    target: dict,
    model_name: str,
    seed: int,
    device: torch.device,
) -> dict:
    X_attack, y_attack = build_attack_dataset(
        target["model"],
        target["member_loader"],
        target["nonmember_loader"],
        device=device,
    )
    attack_splits = make_attack_splits_from_features(X_attack, y_attack, test_ratio=0.30, seed=seed)
    return run_all_attacks(
        model_name,
        attack_splits["X_attack_train"],
        attack_splits["y_attack_train"],
        attack_splits["X_attack_test"],
        attack_splits["y_attack_test"],
    )


def build_best_attack_summary(results_df: pd.DataFrame, output_path: Path, extra_group_cols: list[str] | None = None) -> pd.DataFrame:
    if results_df.empty:
        summary = pd.DataFrame()
        summary.to_csv(output_path, index=False)
        return summary
    group_cols = ["dataset", "model", "seed"] + (extra_group_cols or [])
    df = results_df.copy()
    df["attack_auc"] = pd.to_numeric(df["attack_auc"], errors="coerce")
    best = df.sort_values("attack_auc", ascending=False).groupby(group_cols, as_index=False).first()
    best = best.rename(
        columns={
            "attack_type": "best_attack_type",
            "attack_auc": "best_attack_auc",
            "attack_accuracy": "best_attack_accuracy",
            "top_1_percent_precision": "best_top_1_percent_precision",
            "top_5_percent_precision": "best_top_5_percent_precision",
            "top_10_percent_precision": "best_top_10_percent_precision",
            "lift_at_1_percent": "best_lift_at_1_percent",
            "lift_at_5_percent": "best_lift_at_5_percent",
            "lift_at_10_percent": "best_lift_at_10_percent",
        }
    )
    keep = [
        *group_cols,
        "best_attack_type",
        "best_attack_auc",
        "best_attack_accuracy",
        "best_top_1_percent_precision",
        "best_top_5_percent_precision",
        "best_top_10_percent_precision",
        "best_lift_at_1_percent",
        "best_lift_at_5_percent",
        "best_lift_at_10_percent",
        "train_acc",
        "test_acc",
        "generalization_gap",
        "loss_gap",
        "confidence_gap",
        "epsilon",
        "noise_multiplier",
    ]
    keep = [col for col in keep if col in best.columns]
    summary = best[keep]
    summary.to_csv(output_path, index=False)
    print(f"[save] {output_path} ({len(summary)} rows)")
    return summary


def plot_standard_results(summary_df: pd.DataFrame, figures_dir: Path, prefix: str) -> None:
    if summary_df.empty:
        return
    sns.set_theme(style="whitegrid")
    df = summary_df.copy()
    df["best_attack_auc"] = pd.to_numeric(df["best_attack_auc"], errors="coerce")
    plt.figure(figsize=(10, 5))
    sns.barplot(data=df, x="model", y="best_attack_auc", errorbar="sd")
    plt.axhline(0.5, color="black", linestyle="--", linewidth=1)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(figures_dir / f"{prefix}_best_attack_auc_by_model.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df, x="generalization_gap", y="best_attack_auc", hue="model")
    plt.axhline(0.5, color="black", linestyle="--", linewidth=1)
    plt.tight_layout()
    plt.savefig(figures_dir / f"{prefix}_gap_vs_attack_auc.png", dpi=180)
    plt.close()


def plot_topk_results(summary_df: pd.DataFrame, figures_dir: Path, prefix: str) -> None:
    if summary_df.empty:
        return
    df = summary_df.copy()
    plt.figure(figsize=(10, 5))
    sns.barplot(data=df, x="model", y="best_top_5_percent_precision", errorbar="sd")
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("Best top-5% precision")
    plt.tight_layout()
    plt.savefig(figures_dir / f"{prefix}_topk_precision_by_model.png", dpi=180)
    plt.close()

    plt.figure(figsize=(10, 5))
    sns.barplot(data=df, x="model", y="best_lift_at_5_percent", errorbar="sd")
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("Best lift at 5%")
    plt.tight_layout()
    plt.savefig(figures_dir / f"{prefix}_topk_lift_by_model.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df, x="best_attack_auc", y="best_top_5_percent_precision", hue="model")
    plt.xlabel("Best attack AUC")
    plt.ylabel("Best top-5% precision")
    plt.tight_layout()
    plt.savefig(figures_dir / f"{prefix}_attack_auc_vs_top5_precision.png", dpi=180)
    plt.close()


def run_standard_suite(
    dataset: str,
    experiments: dict,
    seeds: Iterable[int],
    output_dir: str,
    results_filename: str,
    summary_filename: str,
    figure_prefix: str,
    smoke: bool = False,
    topk_plots: bool = False,
    device_arg: str = "auto",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    _, metrics_dir, figures_dir = make_dirs(output_dir)
    device = resolve_device(device_arg)
    X, y, dataset_name = load_final_dataset(dataset)
    rows = []
    results_path = metrics_dir / results_filename
    experiment_items = list(experiments.items())
    if smoke:
        experiment_items = experiment_items[:2]
    seed_list = list(seeds)
    total = len(experiment_items) * len(seed_list)
    step = 0
    for seed in seed_list:
        for _, base_config in experiment_items:
            step += 1
            config = apply_smoke_overrides(base_config, dataset) if smoke else copy.deepcopy(base_config)
            print(f"\n[run] {step}/{total} {config['name']} seed={seed} smoke={smoke}")
            target = train_and_evaluate_target(dataset, X, y, dataset_name, config, seed, device, use_dp=False)
            attacks = run_current_attacks(target, config["name"], seed, device)
            for attack_type, attack_metrics in attacks.items():
                rows.append(make_result_row(dataset_name, config, seed, attack_type, attack_metrics, target["target_metrics"]))
            save_results(rows, results_path)

    results_df = save_results(rows, results_path)
    summary_df = build_best_attack_summary(results_df, metrics_dir / summary_filename)
    plot_standard_results(summary_df, figures_dir, figure_prefix)
    if topk_plots:
        plot_topk_results(summary_df, figures_dir, figure_prefix)
    return results_df, summary_df


def plot_dp_results(summary_df: pd.DataFrame, figures_dir: Path, prefix: str) -> None:
    if summary_df.empty:
        return
    df = summary_df.copy()
    df["noise_multiplier"] = pd.to_numeric(df["noise_multiplier"], errors="coerce").fillna(0.0)
    df["best_attack_auc"] = pd.to_numeric(df["best_attack_auc"], errors="coerce")
    df["generalization_gap"] = pd.to_numeric(df["generalization_gap"], errors="coerce")
    df = df.sort_values("noise_multiplier")

    plt.figure(figsize=(8, 5))
    sns.lineplot(data=df, x="noise_multiplier", y="best_attack_auc", marker="o", errorbar="sd")
    plt.axhline(0.5, color="black", linestyle="--", linewidth=1)
    plt.tight_layout()
    plt.savefig(figures_dir / f"{prefix}_dp_noise_vs_attack_auc.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.lineplot(data=df, x="noise_multiplier", y="generalization_gap", marker="o", errorbar="sd")
    plt.tight_layout()
    plt.savefig(figures_dir / f"{prefix}_dp_noise_vs_generalization_gap.png", dpi=180)
    plt.close()


def run_dp_noise_suite(
    dataset: str,
    base_config: dict,
    dp_experiments: dict,
    seeds: Iterable[int],
    output_dir: str,
    results_filename: str,
    summary_filename: str,
    figure_prefix: str,
    smoke: bool = False,
    device_arg: str = "auto",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    _, metrics_dir, figures_dir = make_dirs(output_dir)
    device = resolve_device(device_arg)
    X, y, dataset_name = load_final_dataset(dataset)
    rows = []
    results_path = metrics_dir / results_filename

    for seed in seeds:
        reference = apply_smoke_overrides(base_config, dataset) if smoke else copy.deepcopy(base_config)
        reference["name"] = f"{base_config['name']}_no_dp_reference"
        reference["regularization_type"] = "none_reference"
        reference.pop("noise_multiplier", None)
        print(f"\n[dp/reference] {reference['name']} seed={seed}")
        target = train_and_evaluate_target(dataset, X, y, dataset_name, reference, seed, device, use_dp=False)
        attacks = run_current_attacks(target, reference["name"], seed, device)
        for attack_type, attack_metrics in attacks.items():
            rows.append(make_result_row(dataset_name, reference, seed, attack_type, attack_metrics, target["target_metrics"]))
        save_results(rows, results_path)

        cleanup_cuda()

        dp_items = list(dp_experiments.items())
        if smoke:
            dp_items = dp_items[:1]
        for _, base_dp_config in dp_items:
            config = apply_smoke_overrides(base_dp_config, dataset) if smoke else copy.deepcopy(base_dp_config)
            print(
                f"\n[dp/noise] {config['name']} seed={seed} "
                f"noise={config.get('noise_multiplier')} smoke={smoke}"
            )
            target = train_and_evaluate_target(dataset, X, y, dataset_name, config, seed, device, use_dp=True)
            if target is None:
                skipped = make_result_row(
                    dataset_name,
                    config,
                    seed,
                    "dp_sgd_skipped",
                    {},
                    {
                        "train_metrics": {},
                        "test_metrics": {},
                        "gap_metrics": {},
                        "conf_gap": {},
                        "actual_epochs": 0,
                        "epsilon": None,
                    },
                    status="skipped_opacus_missing",
                )
                rows.append(skipped)
                save_results(rows, results_path)
                continue
            attacks = run_current_attacks(target, config["name"], seed, device)
            for attack_type, attack_metrics in attacks.items():
                rows.append(make_result_row(dataset_name, config, seed, attack_type, attack_metrics, target["target_metrics"]))
            save_results(rows, results_path)

    results_df = save_results(rows, results_path)
    summary_df = build_best_attack_summary(results_df, metrics_dir / summary_filename)
    plot_dp_results(summary_df, figures_dir, figure_prefix)
    return results_df, summary_df


def _threshold_settings(exposure_mode: str, feature_names: list[str]) -> dict:
    if exposure_mode == "oracle_full":
        return {
            "threshold_feature_index": 0,
            "threshold_higher_is_member": False,
            "threshold_feature_label": "loss",
            "threshold_attack_name": "threshold_loss",
        }
    return {
        "threshold_feature_index": 0,
        "threshold_higher_is_member": True,
        "threshold_feature_label": feature_names[0],
        "threshold_attack_name": "threshold",
    }


def build_api_impact(summary_df: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    rows = []
    if summary_df.empty:
        impact = pd.DataFrame()
        impact.to_csv(output_path, index=False)
        return impact
    for (model, seed), group in summary_df.groupby(["model", "seed"]):
        oracle = group[group["exposure_mode"] == "oracle_full"]
        if oracle.empty:
            continue
        oracle_row = oracle.iloc[0]
        for _, row in group[group["exposure_mode"] != "oracle_full"].iterrows():
            rows.append(
                {
                    "model": model,
                    "seed": seed,
                    "exposure_mode": row["exposure_mode"],
                    "oracle_best_auc": oracle_row["best_attack_auc"],
                    "restricted_best_auc": row["best_attack_auc"],
                    "auc_drop_from_oracle": oracle_row["best_attack_auc"] - row["best_attack_auc"],
                    "oracle_top_5_precision": oracle_row.get("best_top_5_percent_precision"),
                    "restricted_top_5_precision": row.get("best_top_5_percent_precision"),
                    "top_5_precision_drop_from_oracle": oracle_row.get("best_top_5_percent_precision")
                    - row.get("best_top_5_percent_precision"),
                }
            )
    impact = pd.DataFrame(rows)
    impact.to_csv(output_path, index=False)
    print(f"[save] {output_path} ({len(impact)} rows)")
    return impact


def plot_api_results(summary_df: pd.DataFrame, impact_df: pd.DataFrame, figures_dir: Path) -> None:
    if summary_df.empty:
        return
    summary_plot = summary_df.copy()
    summary_plot["exposure_mode"] = pd.Categorical(summary_plot["exposure_mode"], EXPOSURE_ORDER, ordered=True)
    summary_plot = summary_plot.sort_values(["model", "exposure_mode"])

    plt.figure(figsize=(11, 6))
    sns.lineplot(data=summary_plot, x="exposure_mode", y="best_attack_auc", hue="model", marker="o", errorbar=None)
    plt.axhline(0.5, color="black", linestyle="--", linewidth=1)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(figures_dir / "20news_api_exposure_attack_auc_by_mode.png", dpi=180)
    plt.close()

    if not impact_df.empty:
        impact_plot = impact_df.copy()
        impact_plot["exposure_mode"] = pd.Categorical(impact_plot["exposure_mode"], EXPOSURE_ORDER, ordered=True)
        impact_plot = impact_plot.sort_values(["model", "exposure_mode"])
        plt.figure(figsize=(11, 6))
        sns.lineplot(data=impact_plot, x="exposure_mode", y="auc_drop_from_oracle", hue="model", marker="o", errorbar=None)
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        plt.savefig(figures_dir / "20news_api_exposure_auc_drop_from_oracle.png", dpi=180)
        plt.close()


def run_api_exposure_suite(
    experiments: dict,
    seeds: Iterable[int],
    output_dir: str,
    smoke: bool = False,
    device_arg: str = "auto",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    _, metrics_dir, figures_dir = make_dirs(output_dir)
    device = resolve_device(device_arg)
    X, y, dataset_name = load_final_dataset("news20")
    rows = []
    results_path = metrics_dir / "20news_api_exposure_results.csv"

    experiment_items = list(experiments.items())
    if smoke:
        experiment_items = experiment_items[:1]
    for seed in seeds:
        for _, base_config in experiment_items:
            config = apply_smoke_overrides(base_config, "news20") if smoke else copy.deepcopy(base_config)
            target = train_and_evaluate_target("news20", X, y, dataset_name, config, seed, device, use_dp=False)
            for exposure_cfg in API_EXPOSURE_MODES:
                exposure_mode = exposure_cfg["name"]
                print(f"\n[api] model={config['name']} seed={seed} mode={exposure_mode}")
                X_attack, y_attack, feature_names = build_api_exposure_attack_dataset(
                    target["model"],
                    target["member_loader"],
                    target["nonmember_loader"],
                    exposure_mode=exposure_mode,
                    device=device,
                    round_decimals=exposure_cfg.get("round_decimals"),
                )
                attack_splits = make_attack_splits_from_features(X_attack, y_attack, test_ratio=0.30, seed=seed)
                attacks = run_all_attacks(
                    f"{config['name']}/{exposure_mode}",
                    attack_splits["X_attack_train"],
                    attack_splits["y_attack_train"],
                    attack_splits["X_attack_test"],
                    attack_splits["y_attack_test"],
                    **_threshold_settings(exposure_mode, feature_names),
                )
                for attack_type, attack_metrics in attacks.items():
                    row = make_result_row(
                        dataset_name,
                        config,
                        seed,
                        attack_type,
                        attack_metrics,
                        target["target_metrics"],
                    )
                    row["exposure_mode"] = exposure_mode
                    row["exposure_description"] = exposure_cfg.get("description")
                    row["n_exposed_features"] = len(feature_names)
                    row["feature_names"] = "|".join(feature_names)
                    rows.append(row)
                save_results(rows, results_path, columns=[*RESULT_COLUMNS, "exposure_mode", "exposure_description", "n_exposed_features", "feature_names"])

    api_cols = [*RESULT_COLUMNS, "exposure_mode", "exposure_description", "n_exposed_features", "feature_names"]
    results_df = save_results(rows, results_path, columns=api_cols)
    summary_df = build_best_attack_summary(
        results_df,
        metrics_dir / "20news_api_exposure_best_attack_summary.csv",
        extra_group_cols=["exposure_mode"],
    )
    impact_df = build_api_impact(summary_df, metrics_dir / "20news_api_exposure_impact_vs_oracle.csv")
    plot_api_results(summary_df, impact_df, figures_dir)
    return results_df, summary_df, impact_df
