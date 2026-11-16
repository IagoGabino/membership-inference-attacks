"""Build consolidated tables for final experiment results."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


SUMMARY_INPUTS = {
    "cdc_best_results.csv": "cdc/metrics/cdc_best_attack_summary.csv",
    "cdc_dp_summary.csv": "cdc_dp/metrics/cdc_dp_noise_summary.csv",
    "20news_best_results.csv": "20news/metrics/20news_best_attack_summary.csv",
    "20news_topk_summary.csv": "20news_topk/metrics/20news_topk_best_attack_summary.csv",
    "20news_dp_summary.csv": "20news_dp_overfitted/metrics/20news_dp_noise_summary.csv",
    "20news_api_summary.csv": "20news_api/metrics/20news_api_exposure_best_attack_summary.csv",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Build final summary CSVs and key numbers.")
    parser.add_argument("--input-dir", type=str, default="results_final")
    parser.add_argument("--output-dir", type=str, default="results_final/summary")
    return parser.parse_args()


def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"[summary] Missing: {path}")
        return pd.DataFrame()
    df = pd.read_csv(path)
    print(f"[summary] Read: {path} ({len(df)} rows)")
    return df


def _write_copy(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[summary] Wrote: {output_path}")


def _mean_by_model(df: pd.DataFrame, value_col: str = "best_attack_auc") -> pd.DataFrame:
    if df.empty or "model" not in df.columns or value_col not in df.columns:
        return pd.DataFrame()
    tmp = df.copy()
    tmp[value_col] = pd.to_numeric(tmp[value_col], errors="coerce")
    return (
        tmp.groupby("model", as_index=False)[value_col]
        .agg(["mean", "std", "max"])
        .reset_index()
        .sort_values(("mean" if isinstance(tmp, pd.DataFrame) else value_col), ascending=False)
    )


def _lines_for_auc(title: str, df: pd.DataFrame) -> list[str]:
    lines = [f"## {title}", ""]
    if df.empty or "model" not in df.columns or "best_attack_auc" not in df.columns:
        lines.append("No data found.")
        lines.append("")
        return lines
    tmp = df.copy()
    tmp["best_attack_auc"] = pd.to_numeric(tmp["best_attack_auc"], errors="coerce")
    agg = tmp.groupby("model", as_index=False)["best_attack_auc"].mean().sort_values("best_attack_auc", ascending=False)
    for _, row in agg.iterrows():
        lines.append(f"- {row['model']}: mean best attack AUC = {row['best_attack_auc']:.4f}")
    best = agg.iloc[0]
    lines.append(f"- Best: {best['model']} ({best['best_attack_auc']:.4f})")
    lines.append("")
    return lines


def _lines_for_dp(title: str, df: pd.DataFrame) -> list[str]:
    lines = [f"## {title}", ""]
    if df.empty or "noise_multiplier" not in df.columns:
        lines.append("No data found.")
        lines.append("")
        return lines
    tmp = df.copy()
    tmp["noise_multiplier"] = pd.to_numeric(tmp["noise_multiplier"], errors="coerce").fillna(0.0)
    tmp["best_attack_auc"] = pd.to_numeric(tmp["best_attack_auc"], errors="coerce")
    agg = tmp.groupby("noise_multiplier", as_index=False)["best_attack_auc"].mean().sort_values("noise_multiplier")
    for _, row in agg.iterrows():
        label = "no DP reference" if row["noise_multiplier"] == 0 else f"noise={row['noise_multiplier']:.2f}"
        lines.append(f"- {label}: mean best attack AUC = {row['best_attack_auc']:.4f}")
    lines.append("")
    return lines


def _lines_for_topk(df: pd.DataFrame) -> list[str]:
    lines = ["## 20NG top-k risk", ""]
    col = "best_top_5_percent_precision"
    if df.empty or col not in df.columns:
        lines.append("No data found.")
        lines.append("")
        return lines
    tmp = df.copy()
    tmp[col] = pd.to_numeric(tmp[col], errors="coerce")
    agg = tmp.groupby("model", as_index=False)[col].mean().sort_values(col, ascending=False)
    for _, row in agg.iterrows():
        lines.append(f"- {row['model']}: mean best top-5% precision = {row[col]:.4f}")
    lines.append("")
    return lines


def _lines_for_api(df: pd.DataFrame) -> list[str]:
    lines = ["## 20NG API exposure", ""]
    if df.empty or "exposure_mode" not in df.columns:
        lines.append("No data found.")
        lines.append("")
        return lines
    tmp = df.copy()
    tmp["best_attack_auc"] = pd.to_numeric(tmp["best_attack_auc"], errors="coerce")
    agg = tmp.groupby("exposure_mode", as_index=False)["best_attack_auc"].mean().sort_values("best_attack_auc", ascending=False)
    for _, row in agg.iterrows():
        lines.append(f"- {row['exposure_mode']}: mean best attack AUC = {row['best_attack_auc']:.4f}")
    lines.append("")
    return lines


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    loaded = {}
    for output_name, rel_input in SUMMARY_INPUTS.items():
        df = _read(input_dir / rel_input)
        loaded[output_name] = df
        _write_copy(df, output_dir / output_name)

    lines = [
        "# Final key numbers",
        "",
        "Generated from CSVs under the selected results_final directory.",
        "",
    ]
    lines += _lines_for_auc("CDC best attack AUC", loaded["cdc_best_results.csv"])
    lines += _lines_for_dp("CDC DP noise effect", loaded["cdc_dp_summary.csv"])
    lines += _lines_for_auc("20NG best attack AUC", loaded["20news_best_results.csv"])
    lines += _lines_for_topk(loaded["20news_topk_summary.csv"])
    lines += _lines_for_dp("20NG DP noise effect (overfitted baseline)", loaded["20news_dp_summary.csv"])
    lines += _lines_for_api(loaded["20news_api_summary.csv"])

    key_path = output_dir / "final_key_numbers.md"
    key_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[summary] Wrote: {key_path}")


if __name__ == "__main__":
    main()
