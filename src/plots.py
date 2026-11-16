"""
src/plots.py
Geração de métricas CSV e visualizações dos experimentos.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional

# Estilo global
sns.set_theme(style="whitegrid", palette="tab10", font_scale=1.1)
plt.rcParams.update({"figure.dpi": 120, "figure.figsize": (8, 5)})

# Colunas obrigatórias do CSV (sem NaN permitido)
MANDATORY_COLS = [
    "model", "attack_type",
    "train_size", "epochs", "regularization_type", "dropout", "weight_decay", "early_stopping",
    "actual_epochs", "train_acc", "test_acc", "train_f1", "test_f1", "train_auc", "test_auc",
    "generalization_gap", "loss_gap", "confidence_gap",
    "mean_max_conf_members", "mean_max_conf_nonmembers",
    "mean_train_loss", "mean_test_loss",
    "attack_accuracy", "attack_auc", "attack_precision", "attack_recall", "attack_f1",
]

# Colunas opcionais (NaN permitido)
OPTIONAL_COLS = ["epsilon", "delta", "noise_multiplier", "max_grad_norm", "seed"]


# ============================================================
# 1. CSV de resultados
# ============================================================

def build_results_row(
    exp_config: dict,
    train_metrics: dict,
    test_metrics: dict,
    gap_metrics: dict,
    conf_gap: dict,
    attack_name: str,
    attack_metrics: dict,
    dp_result: Optional[dict] = None,
    actual_epochs: Optional[int] = None,
) -> dict:
    """
    Monta uma linha do CSV de resultados para uma combinação (modelo, ataque).
    """
    if actual_epochs is None:
        if dp_result and dp_result.get("actual_epochs") is not None:
            actual_epochs = dp_result.get("actual_epochs")
        elif dp_result and dp_result.get("history"):
            actual_epochs = len(dp_result["history"].get("train_loss", []))
        else:
            actual_epochs = exp_config.get("epochs", None)

    row = {
        "model":               exp_config.get("name", "unknown"),
        "attack_type":         attack_name,
        "train_size":          exp_config.get("experiment_sample_size", None),
        "epochs":              exp_config.get("epochs", None),
        "actual_epochs":       actual_epochs,
        "regularization_type": exp_config.get("regularization_type", "none"),
        "dropout":             exp_config.get("dropout", 0.0),
        "weight_decay":        exp_config.get("weight_decay", 0.0),
        "early_stopping":      exp_config.get("early_stopping", False),
        # Métricas do modelo
        "train_acc":           train_metrics.get("accuracy"),
        "test_acc":            test_metrics.get("accuracy"),
        "train_f1":            train_metrics.get("f1"),
        "test_f1":             test_metrics.get("f1"),
        "train_auc":           train_metrics.get("auc"),
        "test_auc":            test_metrics.get("auc"),
        "generalization_gap":  gap_metrics.get("gen_gap"),
        "loss_gap":            gap_metrics.get("loss_gap"),
        "confidence_gap":      conf_gap.get("confidence_gap"),
        "mean_max_conf_members":    conf_gap.get("mean_max_conf_members"),
        "mean_max_conf_nonmembers": conf_gap.get("mean_max_conf_nonmembers"),
        "mean_train_loss":     train_metrics.get("mean_loss"),
        "mean_test_loss":      test_metrics.get("mean_loss"),
        # Métricas do ataque
        "attack_accuracy":     attack_metrics.get("attack_accuracy"),
        "attack_auc":          attack_metrics.get("attack_auc"),
        "attack_precision":    attack_metrics.get("attack_precision"),
        "attack_recall":       attack_metrics.get("attack_recall"),
        "attack_f1":           attack_metrics.get("attack_f1"),
        # DP (opcional)
        "epsilon":             dp_result.get("epsilon") if dp_result else None,
        "delta":               dp_result.get("delta")   if dp_result else None,
        "noise_multiplier":    exp_config.get("noise_multiplier"),
        "max_grad_norm":       exp_config.get("max_grad_norm"),
    }
    return row


def save_results_csv(all_rows: List[dict], path: str) -> pd.DataFrame:
    """
    Salva lista de linhas de resultado como CSV.

    Garante que colunas obrigatórias não tenham NaN.
    Colunas opcionais (epsilon, delta, etc.) podem ser NaN.

    Returns
    -------
    DataFrame com os resultados salvos
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    df = pd.DataFrame(all_rows)

    # Garantir colunas na ordem correta
    all_cols = MANDATORY_COLS + OPTIONAL_COLS
    for col in all_cols:
        if col not in df.columns:
            df[col] = None
    df = df[all_cols]

    # AUC None -> "N/A" (string) antes da validacao de obrigatorias.
    for auc_col in ["train_auc", "test_auc", "attack_auc"]:
        df[auc_col] = df[auc_col].apply(lambda v: "N/A" if v is None else v)

    # Validar obrigatórias
    for col in MANDATORY_COLS:
        n_nan = df[col].isna().sum()
        if n_nan > 0:
            warnings.warn(
                f"[plots] Coluna obrigatória '{col}' tem {n_nan} NaN(s).",
                UserWarning,
            )

    # AUC None → "N/A" (string) para evitar coluna de tipo misto no CSV
    df.to_csv(path, index=False)
    print(f"[plots] CSV salvo em: {path} ({len(df)} linhas)")
    return df


# ============================================================
# 2. Gráficos obrigatórios
# ============================================================

def generate_mandatory_plots(results_df: pd.DataFrame, output_dir: str) -> None:
    """
    Gera os 4 gráficos obrigatórios e salva como PNG.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Converter attack_auc para numérico (pode ser "N/A" string)
    df = results_df.copy()
    df["attack_auc"] = pd.to_numeric(df["attack_auc"], errors="coerce")

    # ---- 1. Generalization gap vs Attack AUC ----
    fig, ax = plt.subplots()
    for attack_type, grp in df.groupby("attack_type"):
        ax.scatter(grp["generalization_gap"], grp["attack_auc"],
                   label=attack_type, s=80, alpha=0.8)
        # Anotar modelos
        for _, row in grp.iterrows():
            ax.annotate(row["model"], (row["generalization_gap"], row["attack_auc"]),
                        fontsize=7, alpha=0.7, xytext=(4, 2), textcoords="offset points")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="Chance (0.5)")
    ax.set_xlabel("Generalization Gap (train_acc − test_acc)")
    ax.set_ylabel("Attack AUC")
    ax.set_title("Generalization Gap vs Attack AUC")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "generalization_gap_vs_attack_auc.png"))
    plt.close()
    print("[plots] generalization_gap_vs_attack_auc.png salvo.")

    # ---- 2. Confidence gap vs Attack AUC ----
    fig, ax = plt.subplots()
    for attack_type, grp in df.groupby("attack_type"):
        ax.scatter(grp["confidence_gap"], grp["attack_auc"],
                   label=attack_type, s=80, alpha=0.8)
        for _, row in grp.iterrows():
            ax.annotate(row["model"], (row["confidence_gap"], row["attack_auc"]),
                        fontsize=7, alpha=0.7, xytext=(4, 2), textcoords="offset points")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="Chance (0.5)")
    ax.set_xlabel("Confidence Gap (members − non-members)")
    ax.set_ylabel("Attack AUC")
    ax.set_title("Confidence Gap vs Attack AUC")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confidence_gap_vs_attack_auc.png"))
    plt.close()
    print("[plots] confidence_gap_vs_attack_auc.png salvo.")

    # ---- 3. Attack AUC por modelo (grouped bar) ----
    pivot = df.pivot_table(
        index="model", columns="attack_type", values="attack_auc", aggfunc="mean"
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    pivot.plot(kind="bar", ax=ax, edgecolor="white", width=0.75)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="Chance (0.5)")
    ax.set_xlabel("Modelo")
    ax.set_ylabel("Attack AUC")
    ax.set_title("Attack AUC por Modelo e Tipo de Ataque")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    ax.legend(title="Tipo de Ataque", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "attack_auc_by_model.png"))
    plt.close()
    print("[plots] attack_auc_by_model.png salvo.")

    # ---- 4. Utility vs Privacy (test_acc vs attack_auc) ----
    fig, ax = plt.subplots()
    for attack_type, grp in df.groupby("attack_type"):
        ax.scatter(grp["test_acc"], grp["attack_auc"],
                   label=attack_type, s=80, alpha=0.8)
        for _, row in grp.iterrows():
            ax.annotate(row["model"], (row["test_acc"], row["attack_auc"]),
                        fontsize=7, alpha=0.7, xytext=(4, 2), textcoords="offset points")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="Chance (0.5)")
    ax.set_xlabel("Test Accuracy (utilidade do modelo)")
    ax.set_ylabel("Attack AUC (vulnerabilidade)")
    ax.set_title("Utilidade vs Privacidade (Trade-off)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "utility_vs_privacy_attack.png"))
    plt.close()
    print("[plots] utility_vs_privacy_attack.png salvo.")


# ============================================================
# 3. Gráficos opcionais (condicionais)
# ============================================================

def generate_optional_plots(results_df: pd.DataFrame, output_dir: str) -> None:
    """
    Gera gráficos opcionais SOMENTE se os dados relevantes existirem.
    Não lança exceções se os dados não estiverem presentes.
    """
    os.makedirs(output_dir, exist_ok=True)
    df = results_df.copy()
    df["attack_auc"] = pd.to_numeric(df["attack_auc"], errors="coerce")
    df["epsilon"]    = pd.to_numeric(df.get("epsilon", pd.Series()), errors="coerce")

    # ---- Epsilon vs Attack AUC ----
    if "epsilon" in df.columns and df["epsilon"].notna().any():
        fig, ax = plt.subplots()
        for attack_type, grp in df.dropna(subset=["epsilon"]).groupby("attack_type"):
            ax.scatter(grp["epsilon"], grp["attack_auc"], label=attack_type, s=80, alpha=0.8)
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
        ax.set_xlabel("ε (Privacy Budget)")
        ax.set_ylabel("Attack AUC")
        ax.set_title("DP Privacy Budget (ε) vs Attack AUC")
        ax.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "epsilon_vs_attack_auc.png"))
        plt.close()
        print("[plots] epsilon_vs_attack_auc.png salvo.")

        # ---- Epsilon vs Test Accuracy ----
        fig, ax = plt.subplots()
        for attack_type, grp in df.dropna(subset=["epsilon"]).groupby("attack_type"):
            ax.scatter(grp["epsilon"], grp["test_acc"], label=attack_type, s=80, alpha=0.8)
        ax.set_xlabel("ε (Privacy Budget)")
        ax.set_ylabel("Test Accuracy")
        ax.set_title("DP Privacy Budget (ε) vs Test Accuracy")
        ax.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "epsilon_vs_accuracy.png"))
        plt.close()
        print("[plots] epsilon_vs_accuracy.png salvo.")

    # ---- Train size vs Attack AUC ----
    if "train_size" in df.columns and df["train_size"].nunique() >= 2:
        agg = (
            df.groupby(["train_size", "attack_type"])["attack_auc"]
            .mean().reset_index()
        )
        fig, ax = plt.subplots()
        for attack_type, grp in agg.groupby("attack_type"):
            ax.plot(grp["train_size"].astype(int), grp["attack_auc"],
                    marker="o", label=attack_type)
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Tamanho do Conjunto de Treino")
        ax.set_ylabel("Attack AUC (média)")
        ax.set_title("Tamanho do Treino vs Vulnerabilidade a Membership Inference")
        ax.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "train_size_vs_attack_auc.png"))
        plt.close()
        print("[plots] train_size_vs_attack_auc.png salvo.")

    # ---- Epochs vs Attack AUC ----
    if "epochs" in df.columns and df["epochs"].nunique() >= 2:
        agg = (
            df.groupby(["epochs", "attack_type"])["attack_auc"]
            .mean().reset_index()
        )
        fig, ax = plt.subplots()
        for attack_type, grp in agg.groupby("attack_type"):
            ax.plot(grp["epochs"].astype(int), grp["attack_auc"],
                    marker="o", label=attack_type)
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Número de Épocas")
        ax.set_ylabel("Attack AUC (média)")
        ax.set_title("Épocas de Treino vs Vulnerabilidade a Membership Inference")
        ax.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "epochs_vs_attack_auc.png"))
        plt.close()
        print("[plots] epochs_vs_attack_auc.png salvo.")


# ============================================================
# 4. Gráfico: Epsilon (ε) vs Attack AUC
# ============================================================

def plot_epsilon_vs_attack_auc(
    results_df: pd.DataFrame,
    output_dir: str,
    overfitted_auc: Optional[float] = None,
) -> None:
    """
    Gera gráficos mostrando como o epsilon (ε) da privacidade diferencial
    afeta o sucesso do membership inference attack.

    Gráfico principal: ε no eixo x vs Attack AUC no eixo y.
    Gráfico secundário: ε vs Test Accuracy (utilidade).

    Parameters
    ----------
    results_df    : DataFrame com resultados dos experimentos dp_noise_*
    output_dir    : diretório para salvar os PNGs
    overfitted_auc: AUC do modelo overfitted sem DP (referência), opcional
    """
    os.makedirs(output_dir, exist_ok=True)
    df = results_df.copy()
    df["attack_auc"]       = pd.to_numeric(df["attack_auc"], errors="coerce")
    df["noise_multiplier"] = pd.to_numeric(df.get("noise_multiplier", pd.Series()), errors="coerce")
    df["epsilon"]          = pd.to_numeric(df.get("epsilon", pd.Series()), errors="coerce")

    # Filtrar apenas os experimentos dp_noise_*
    df_noise = df[df["model"].str.startswith("dp_noise_")].copy()
    if df_noise.empty:
        print("[plots] Nenhum experimento dp_noise_* encontrado. Pulando plot.")
        return

    if not df_noise["epsilon"].notna().any():
        print("[plots] Sem valores de epsilon. Pulando plot.")
        return

    # ================================================================
    # Gráfico principal: ε (Epsilon) vs Attack AUC
    # ================================================================
    fig, ax = plt.subplots(figsize=(10, 6))

    # Calcular epsilon médio por noise_multiplier (para agrupar)
    eps_map = df_noise.groupby("noise_multiplier")["epsilon"].mean()

    for attack_type, grp in df_noise.groupby("attack_type"):
        agg = grp.groupby("noise_multiplier").agg(
            attack_auc_mean=("attack_auc", "mean"),
            attack_auc_std=("attack_auc", "std"),
            epsilon_mean=("epsilon", "mean"),
        ).reset_index()
        agg = agg.sort_values("epsilon_mean")
        ax.errorbar(
            agg["epsilon_mean"], agg["attack_auc_mean"],
            yerr=agg["attack_auc_std"],
            marker="o", linewidth=2.5, capsize=5, capthick=1.5,
            label=attack_type, markersize=9, alpha=0.9,
        )

    # Linha de referência: random chance
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1.2, alpha=0.7,
               label="Random Chance (0.5)")

    # Linha de referência: overfitted sem DP
    if overfitted_auc is not None:
        ax.axhline(
            overfitted_auc, color="#e74c3c", linestyle=":",
            linewidth=2, alpha=0.8,
            label=f"Overfitted sem DP (AUC={overfitted_auc:.3f})",
        )

    ax.set_xlabel("ε (Epsilon — Privacy Budget)", fontsize=14)
    ax.set_ylabel("Attack AUC", fontsize=14)
    ax.set_title(
        "Efeito do ε (Privacidade Diferencial) no Membership Inference Attack",
        fontsize=14, fontweight="bold",
    )
    ax.legend(fontsize=10, loc="best")
    ax.set_ylim(0.45, max(0.65, df_noise["attack_auc"].max() + 0.05))
    ax.grid(True, alpha=0.3)

    # Anotações de contexto
    ax.annotate(
        "← Mais privacidade    Menos privacidade →",
        xy=(0.5, 0.02), xycoords="axes fraction",
        fontsize=9, color="#888", ha="center", fontstyle="italic",
    )

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "epsilon_vs_attack_auc_noise_sweep.png"), dpi=150)
    plt.close()
    print("[plots] epsilon_vs_attack_auc_noise_sweep.png salvo.")

    # ================================================================
    # Gráfico combinado: ε vs Attack AUC + ε vs Test Accuracy
    # ================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # --- (a) ε vs Attack AUC (média de todos os ataques) ---
    agg_all = df_noise.groupby("noise_multiplier").agg(
        attack_auc_mean=("attack_auc", "mean"),
        attack_auc_std=("attack_auc", "std"),
        epsilon_mean=("epsilon", "mean"),
    ).reset_index().sort_values("epsilon_mean")

    ax1.errorbar(
        agg_all["epsilon_mean"], agg_all["attack_auc_mean"],
        yerr=agg_all["attack_auc_std"],
        marker="o", linewidth=2.5, capsize=5, capthick=1.5,
        color="#3498db", markersize=9,
    )
    ax1.axhline(0.5, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    if overfitted_auc is not None:
        ax1.axhline(overfitted_auc, color="#e74c3c", linestyle=":", linewidth=1.5, alpha=0.7)
    ax1.set_xlabel("ε (Privacy Budget)", fontsize=13)
    ax1.set_ylabel("Attack AUC (média)", fontsize=13)
    ax1.set_title("Privacidade vs Vulnerabilidade", fontsize=13, fontweight="bold")
    ax1.grid(True, alpha=0.3)

    # Anotar noise_multiplier em cada ponto
    for _, row in agg_all.iterrows():
        ax1.annotate(
            f"σ={row['noise_multiplier']:.1f}",
            (row["epsilon_mean"], row["attack_auc_mean"]),
            textcoords="offset points", xytext=(6, 8),
            fontsize=8, color="#555",
        )

    # --- (b) ε vs Test Accuracy ---
    agg_acc = df_noise.groupby("noise_multiplier").agg(
        test_acc_mean=("test_acc", "mean"),
        test_acc_std=("test_acc", "std"),
        epsilon_mean=("epsilon", "mean"),
    ).reset_index().sort_values("epsilon_mean")

    ax2.errorbar(
        agg_acc["epsilon_mean"], agg_acc["test_acc_mean"],
        yerr=agg_acc["test_acc_std"],
        marker="D", linewidth=2.5, capsize=5, capthick=1.5,
        color="#2ecc71", markersize=9,
    )
    ax2.set_xlabel("ε (Privacy Budget)", fontsize=13)
    ax2.set_ylabel("Test Accuracy", fontsize=13)
    ax2.set_title("Privacidade vs Utilidade do Modelo", fontsize=13, fontweight="bold")
    ax2.grid(True, alpha=0.3)

    for _, row in agg_acc.iterrows():
        ax2.annotate(
            f"σ={row['noise_multiplier']:.1f}",
            (row["epsilon_mean"], row["test_acc_mean"]),
            textcoords="offset points", xytext=(6, 8),
            fontsize=8, color="#555",
        )

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "epsilon_privacy_tradeoff.png"), dpi=150)
    plt.close()
    print("[plots] epsilon_privacy_tradeoff.png salvo.")


# ============================================================
# 5. Tabela de resultados no Colab
# ============================================================

def display_results_table(results_df: pd.DataFrame) -> None:
    """
    Exibe tabela de resultados estilizada no Colab com highlighting.
    Colunas de interesse: attack_auc (max=vulnerável), test_acc (min=pior utilidade).
    """
    try:
        from IPython.display import display
    except ImportError:
        print(results_df.to_string())
        return

    df = results_df.copy()
    df["attack_auc"] = pd.to_numeric(df["attack_auc"], errors="coerce")

    cols_to_show = [
        "model", "attack_type", "train_size", "actual_epochs",
        "train_acc", "test_acc", "train_f1", "test_f1", "test_auc",
        "generalization_gap", "loss_gap", "confidence_gap",
        "mean_max_conf_members", "mean_max_conf_nonmembers",
        "attack_accuracy", "attack_auc",
        "attack_precision", "attack_recall", "attack_f1",
        "epsilon",
    ]
    cols_available = [c for c in cols_to_show if c in df.columns]
    df_show = df[cols_available].round(4)

    styled = (
        df_show.style
        .highlight_max(subset=["attack_auc"], color="#ffcccc")   # mais vulnerável
        .highlight_min(subset=["test_acc"],   color="#ffe0b2")   # pior utilidade
        .highlight_max(subset=["generalization_gap"], color="#fff9c4")
        .format(na_rep="N/A")
        .set_caption("Resumo dos Experimentos de Membership Inference")
    )
    display(styled)


# ============================================================
# 5. Graficos current vs shadow
# ============================================================

def generate_current_vs_shadow_plots(results_df: pd.DataFrame, output_dir: str) -> None:
    """
    Gera graficos comparativos para os CSVs novos do pipeline por script.
    Espera colunas attack_family, attack_name, model_name e metricas de ataque.
    """
    os.makedirs(output_dir, exist_ok=True)
    df = results_df.copy()
    if df.empty:
        print("[plots] Nenhum resultado para plotar.")
        return

    for col in ["attack_auc", "tpr_at_fpr_1", "tpr_at_fpr_5", "tpr_at_fpr_10"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 1. AUC por ataque
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.barplot(
        data=df,
        x="model_name",
        y="attack_auc",
        hue="attack_name",
        errorbar=None,
        ax=ax,
    )
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Modelo")
    ax.set_ylabel("Attack AUC")
    ax.set_title("Current vs Shadow - Attack AUC")
    ax.tick_params(axis="x", rotation=25)
    ax.legend(title="Ataque", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "current_vs_shadow_auc.png"))
    plt.close()
    print("[plots] current_vs_shadow_auc.png salvo.")

    # 2. TPR @ FPR
    tpr_cols = [
        col for col in ["tpr_at_fpr_1", "tpr_at_fpr_5", "tpr_at_fpr_10"]
        if col in df.columns
    ]
    if tpr_cols:
        tpr_df = df.melt(
            id_vars=["model_name", "attack_family", "attack_name"],
            value_vars=tpr_cols,
            var_name="fpr_level",
            value_name="tpr",
        )
        fig, ax = plt.subplots(figsize=(11, 5))
        sns.barplot(
            data=tpr_df,
            x="fpr_level",
            y="tpr",
            hue="attack_name",
            errorbar=None,
            ax=ax,
        )
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.set_title("Current vs Shadow - TPR at Fixed FPR")
        ax.legend(title="Ataque", fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "current_vs_shadow_tpr_at_fpr.png"))
        plt.close()
        print("[plots] current_vs_shadow_tpr_at_fpr.png salvo.")

    # 3. Shadow AUC por n_shadow, se houver multiplos valores
    shadow_df = df[df.get("attack_family") == "shadow"].copy()
    if not shadow_df.empty and shadow_df["n_shadow"].nunique(dropna=True) > 1:
        fig, ax = plt.subplots(figsize=(9, 5))
        sns.lineplot(
            data=shadow_df,
            x="n_shadow",
            y="attack_auc",
            hue="attack_name",
            style="model_name",
            marker="o",
            errorbar=None,
            ax=ax,
        )
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Numero de shadow models")
        ax.set_ylabel("Attack AUC")
        ax.set_title("Shadow AUC by n_shadow")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "shadow_auc_by_n_shadow.png"))
        plt.close()
        print("[plots] shadow_auc_by_n_shadow.png salvo.")

    # 4. Melhor current vs melhor shadow por modelo
    if {"current", "shadow"}.issubset(set(df["attack_family"].dropna().unique())):
        best = (
            df.groupby(["model_name", "attack_family"], as_index=False)["attack_auc"]
            .max()
        )
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(
            data=best,
            x="model_name",
            y="attack_auc",
            hue="attack_family",
            errorbar=None,
            ax=ax,
        )
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Modelo")
        ax.set_ylabel("Melhor Attack AUC")
        ax.set_title("Best Shadow vs Best Current by Model")
        ax.tick_params(axis="x", rotation=25)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "shadow_vs_current_by_model.png"))
        plt.close()
        print("[plots] shadow_vs_current_by_model.png salvo.")


# ============================================================
# 6. Graficos finais do relatorio (final_report)
# ============================================================

def generate_final_report_plots(df: pd.DataFrame, output_dir: str) -> None:
    """
    Gera os 7 graficos finais do relatorio consolidado.

    Espera colunas: model_name, attack_name, attack_family, attack_auc,
    tpr_at_fpr_1/5/10, top_k_percent_precision/recall,
    generalization_gap, confidence_gap, test_accuracy.
    """
    os.makedirs(output_dir, exist_ok=True)
    if df.empty:
        print("[plots/final] DataFrame vazio. Nenhum grafico gerado.")
        return

    df = df.copy()
    for col in [
        "attack_auc", "tpr_at_fpr_1", "tpr_at_fpr_5", "tpr_at_fpr_10",
        "generalization_gap", "confidence_gap", "test_accuracy",
        "top_1_percent_precision", "top_5_percent_precision", "top_10_percent_precision",
        "top_1_percent_recall", "top_5_percent_recall", "top_10_percent_recall",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # ---- 1. Attack AUC por modelo ----
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.barplot(
        data=df, x="model_name", y="attack_auc", hue="attack_name",
        estimator="mean", errorbar="sd", ax=ax,
    )
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="Chance")
    ax.set_xlabel("Modelo")
    ax.set_ylabel("Attack AUC (media)")
    ax.set_title("Attack AUC por Modelo e Tipo de Ataque")
    ax.tick_params(axis="x", rotation=30)
    ax.legend(title="Ataque", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "final_attack_auc_by_model.png"))
    plt.close()
    print("[plots/final] final_attack_auc_by_model.png salvo.")

    # ---- 2. TPR @ FPR por modelo ----
    tpr_cols = [c for c in ["tpr_at_fpr_1", "tpr_at_fpr_5", "tpr_at_fpr_10"] if c in df.columns]
    if tpr_cols:
        tpr_df = df.melt(
            id_vars=["model_name", "attack_name"],
            value_vars=tpr_cols,
            var_name="fpr_level",
            value_name="tpr",
        )
        tpr_label_map = {
            "tpr_at_fpr_1": "FPR=1%", "tpr_at_fpr_5": "FPR=5%", "tpr_at_fpr_10": "FPR=10%",
        }
        tpr_df["fpr_level"] = tpr_df["fpr_level"].map(tpr_label_map)
        fig, ax = plt.subplots(figsize=(11, 5))
        sns.barplot(
            data=tpr_df, x="model_name", y="tpr", hue="fpr_level",
            estimator="mean", errorbar="sd", ax=ax,
        )
        ax.set_xlabel("Modelo")
        ax.set_ylabel("TPR (media)")
        ax.set_title("TPR a FPR Fixo por Modelo")
        ax.tick_params(axis="x", rotation=30)
        ax.legend(title="FPR", fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "final_tpr_at_fpr_by_model.png"))
        plt.close()
        print("[plots/final] final_tpr_at_fpr_by_model.png salvo.")

    # ---- 3. Top-K Privacy Risk ----
    topk_prec_cols = [
        c for c in ["top_1_percent_precision", "top_5_percent_precision", "top_10_percent_precision"]
        if c in df.columns
    ]
    topk_rec_cols = [
        c for c in ["top_1_percent_recall", "top_5_percent_recall", "top_10_percent_recall"]
        if c in df.columns
    ]
    if topk_prec_cols:
        topk_df = df.melt(
            id_vars=["model_name", "attack_name"],
            value_vars=topk_prec_cols + topk_rec_cols,
            var_name="metric",
            value_name="value",
        )
        label_map = {
            "top_1_percent_precision": "Prec@1%",
            "top_5_percent_precision": "Prec@5%",
            "top_10_percent_precision": "Prec@10%",
            "top_1_percent_recall": "Recall@1%",
            "top_5_percent_recall": "Recall@5%",
            "top_10_percent_recall": "Recall@10%",
        }
        topk_df["metric"] = topk_df["metric"].map(label_map)
        fig, ax = plt.subplots(figsize=(13, 5))
        sns.barplot(
            data=topk_df, x="model_name", y="value", hue="metric",
            estimator="mean", errorbar=None, ax=ax,
        )
        ax.set_xlabel("Modelo")
        ax.set_ylabel("Valor (media)")
        ax.set_title("Top-K Privacy Risk (Precision e Recall no Top-K% por Score)")
        ax.tick_params(axis="x", rotation=30)
        ax.legend(title="Metrica", fontsize=7, ncol=2)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "final_topk_privacy_risk.png"))
        plt.close()
        print("[plots/final] final_topk_privacy_risk.png salvo.")

    # ---- 4. Generalization Gap vs Attack AUC ----
    if "generalization_gap" in df.columns and "attack_auc" in df.columns:
        agg = (
            df.groupby(["model_name", "attack_name"], as_index=False)
            [["generalization_gap", "attack_auc"]].mean()
        )
        fig, ax = plt.subplots()
        for attack_name, grp in agg.groupby("attack_name"):
            ax.scatter(grp["generalization_gap"], grp["attack_auc"],
                       label=attack_name, s=80, alpha=0.85)
            for _, row in grp.iterrows():
                ax.annotate(
                    row["model_name"], (row["generalization_gap"], row["attack_auc"]),
                    fontsize=7, alpha=0.7, xytext=(4, 2), textcoords="offset points",
                )
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="Chance")
        ax.set_xlabel("Generalization Gap (train_acc − test_acc)")
        ax.set_ylabel("Attack AUC")
        ax.set_title("Generalization Gap vs Attack AUC")
        ax.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "final_generalization_gap_vs_attack_auc.png"))
        plt.close()
        print("[plots/final] final_generalization_gap_vs_attack_auc.png salvo.")

    # ---- 5. Confidence Gap vs Attack AUC ----
    if "confidence_gap" in df.columns and "attack_auc" in df.columns:
        agg = (
            df.groupby(["model_name", "attack_name"], as_index=False)
            [["confidence_gap", "attack_auc"]].mean()
        )
        fig, ax = plt.subplots()
        for attack_name, grp in agg.groupby("attack_name"):
            ax.scatter(grp["confidence_gap"], grp["attack_auc"],
                       label=attack_name, s=80, alpha=0.85)
            for _, row in grp.iterrows():
                ax.annotate(
                    row["model_name"], (row["confidence_gap"], row["attack_auc"]),
                    fontsize=7, alpha=0.7, xytext=(4, 2), textcoords="offset points",
                )
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="Chance")
        ax.set_xlabel("Confidence Gap (members − non-members)")
        ax.set_ylabel("Attack AUC")
        ax.set_title("Confidence Gap vs Attack AUC")
        ax.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "final_confidence_gap_vs_attack_auc.png"))
        plt.close()
        print("[plots/final] final_confidence_gap_vs_attack_auc.png salvo.")

    # ---- 6. Utility vs Privacy Tradeoff ----
    if "test_accuracy" in df.columns and "attack_auc" in df.columns:
        agg = (
            df.groupby(["model_name", "attack_name"], as_index=False)
            [["test_accuracy", "attack_auc"]].mean()
        )
        fig, ax = plt.subplots()
        for attack_name, grp in agg.groupby("attack_name"):
            ax.scatter(grp["test_accuracy"], grp["attack_auc"],
                       label=attack_name, s=80, alpha=0.85)
            for _, row in grp.iterrows():
                ax.annotate(
                    row["model_name"], (row["test_accuracy"], row["attack_auc"]),
                    fontsize=7, alpha=0.7, xytext=(4, 2), textcoords="offset points",
                )
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="Chance")
        ax.set_xlabel("Test Accuracy (utilidade)")
        ax.set_ylabel("Attack AUC (vulnerabilidade)")
        ax.set_title("Utilidade vs Privacidade (Trade-off)")
        ax.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "final_utility_privacy_tradeoff.png"))
        plt.close()
        print("[plots/final] final_utility_privacy_tradeoff.png salvo.")

    # ---- 7. Current vs Shadow AUC ----
    if "attack_family" in df.columns and df["attack_family"].notna().any():
        families = set(df["attack_family"].dropna().unique())
        if len(families) >= 1:
            best_per_family = (
                df.groupby(["model_name", "attack_family"], as_index=False)["attack_auc"]
                .mean()
            )
            fig, ax = plt.subplots(figsize=(9, 5))
            sns.barplot(
                data=best_per_family,
                x="model_name",
                y="attack_auc",
                hue="attack_family",
                estimator="mean",
                errorbar=None,
                ax=ax,
            )
            ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="Chance")
            ax.set_xlabel("Modelo")
            ax.set_ylabel("Attack AUC (media)")
            ax.set_title("Current vs Shadow: AUC por Modelo")
            ax.tick_params(axis="x", rotation=25)
            ax.legend(title="Familia", fontsize=8)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "final_current_vs_shadow_auc.png"))
            plt.close()
            print("[plots/final] final_current_vs_shadow_auc.png salvo.")


# ============================================================
# 7. Graficos de comparacao entre datasets (CDC Diabetes vs 20 Newsgroups)
# ============================================================

def generate_dataset_comparison_plots(df: pd.DataFrame, output_dir: str) -> None:
    """
    Gera graficos comparativos entre CDC Diabetes e 20 Newsgroups.
    Mostra que datasets mais complexos sao mais vulneraveis.
    """
    os.makedirs(output_dir, exist_ok=True)
    df = df.copy()
    for col in ["attack_auc", "generalization_gap", "confidence_gap", "tpr_at_fpr_5"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "dataset" not in df.columns or df["dataset"].nunique() < 2:
        print("[plots/compare] Menos de 2 datasets. Pulando comparacao.")
        return

    # Label mais legivel por dataset
    dataset_label = {
        "cdc_diabetes":  "CDC Diabetes\n(binario, tabular)",
        "20newsgroups":  "20 Newsgroups\n(20 classes, texto)",
    }
    df["dataset_label"] = df["dataset"].map(lambda x: dataset_label.get(x, x))

    # Usar apenas current attacks para comparacao justa
    df_c = df[df.get("attack_family", pd.Series("current", index=df.index)) == "current"].copy()
    if df_c.empty:
        df_c = df.copy()

    # ---- 1. Attack AUC por modelo e dataset ----
    model_order = ["baseline", "regularized", "overfitted",
                   "baseline_small", "overfitted_small", "vulnerable_2k", "vulnerable_5k", "dp_sgd"]
    existing_models = [m for m in model_order if m in df_c["model_name"].values]

    agg = (
        df_c.groupby(["dataset_label", "model_name"])["attack_auc"]
        .mean()
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.barplot(
        data=agg, x="model_name", y="attack_auc", hue="dataset_label",
        order=[m for m in existing_models if m in agg["model_name"].values],
        ax=ax,
    )
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="Chance (0.5)")
    ax.set_xlabel("Modelo")
    ax.set_ylabel("Attack AUC (media)")
    ax.set_title("Comparacao de Vulnerabilidade: Diabetes vs 20 Newsgroups")
    ax.tick_params(axis="x", rotation=30)
    ax.legend(title="Dataset", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "final_dataset_comparison_auc.png"))
    plt.close()
    print("[plots/compare] final_dataset_comparison_auc.png salvo.")

    # ---- 2. Generalization Gap vs Attack AUC por dataset ----
    if "generalization_gap" in df_c.columns:
        agg2 = (
            df_c.groupby(["dataset_label", "model_name"])
            [["generalization_gap", "attack_auc"]].mean()
            .reset_index()
        )
        fig, ax = plt.subplots(figsize=(9, 6))
        colors = {"CDC Diabetes\n(binario, tabular)": "#2196F3",
                  "20 Newsgroups\n(20 classes, texto)": "#FF5722"}
        for label, grp in agg2.groupby("dataset_label"):
            color = colors.get(label, None)
            ax.scatter(grp["generalization_gap"], grp["attack_auc"],
                       label=label, s=100, alpha=0.85, color=color)
            for _, row in grp.iterrows():
                ax.annotate(row["model_name"],
                            (row["generalization_gap"], row["attack_auc"]),
                            fontsize=7, alpha=0.8,
                            xytext=(4, 3), textcoords="offset points")
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="Chance")
        ax.set_xlabel("Generalization Gap (train_acc - test_acc)")
        ax.set_ylabel("Attack AUC")
        ax.set_title("Generalization Gap vs Vulnerabilidade\nDiabetes x 20 Newsgroups")
        ax.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "final_dataset_gap_vs_auc.png"))
        plt.close()
        print("[plots/compare] final_dataset_gap_vs_auc.png salvo.")

    # ---- 3. Box plot: distribuicao de AUC por dataset ----
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=df_c, x="dataset_label", y="attack_auc", ax=ax, palette="Set2")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="Chance (0.5)")
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Attack AUC")
    ax.set_title("Distribuicao de AUC por Dataset\n(todos modelos e ataques)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "final_dataset_auc_distribution.png"))
    plt.close()
    print("[plots/compare] final_dataset_auc_distribution.png salvo.")

    # ---- 4. Resumo: melhor AUC por dataset ----
    summary = (
        df_c.groupby("dataset")["attack_auc"]
        .agg(["mean", "max", "min"])
        .round(4)
        .reset_index()
    )
    print("\n[plots/compare] Resumo por dataset:")
    print(summary.to_string(index=False))
