"""
src/membership_attack.py
Implementacao dos ataques atuais de membership inference.

Ataques:
  1. Threshold por loss, com score continuo = -loss.
  2. Logistic Regression sobre as 6 features comportamentais.
  3. Random Forest sobre as 6 features comportamentais.

Feature layout em X_attack:
  0: loss, 1: max_confidence, 2: true_class_confidence,
  3: entropy, 4: correctness, 5: top2_confidence_gap.
"""

from typing import Dict

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler


def compute_tpr_at_fpr(
    y_true: np.ndarray,
    y_score: np.ndarray,
    fpr_levels=(0.01, 0.05, 0.10),
) -> Dict:
    """
    Calcula TPR nos maiores thresholds com FPR <= niveis pedidos.

    y_score deve crescer com a probabilidade de membership. Para loss, use -loss.
    """
    result = {}
    try:
        fpr, tpr, _ = roc_curve(y_true, y_score)
        for level in fpr_levels:
            valid = np.where(fpr <= level)[0]
            value = float(tpr[valid].max()) if len(valid) else 0.0
            result[f"tpr_at_fpr_{int(level * 100)}"] = value
    except Exception:
        for level in fpr_levels:
            result[f"tpr_at_fpr_{int(level * 100)}"] = 0.0
    return result


def compute_top_k_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    k_percents: tuple = (1, 5, 10),
) -> Dict:
    """
    Calcula precision e recall para o top-K% de amostras rankeadas por score.

    Score maior = maior probabilidade de ser membro.

    top_k_precision: fracao de reais membros entre os top-K%
    top_k_recall: fracao dos membros reais capturados no top-K%
    """
    result = {}
    n = len(y_true)
    n_members = int(y_true.sum())

    if n_members == 0:
        for k in k_percents:
            result[f"top_{k}_percent_precision"] = 0.0
            result[f"top_{k}_percent_recall"] = 0.0
            result[f"lift_at_{k}_percent"] = 0.0
        return result

    base_rate = n_members / n
    sorted_idx = np.argsort(y_score)[::-1]
    for k in k_percents:
        top_k = max(1, int(np.ceil(k / 100.0 * n)))
        top_k_idx = sorted_idx[:top_k]
        top_k_members = int(y_true[top_k_idx].sum())
        precision = float(top_k_members / top_k)
        result[f"top_{k}_percent_precision"] = precision
        result[f"top_{k}_percent_recall"] = float(top_k_members / n_members)
        result[f"lift_at_{k}_percent"] = float(precision / base_rate) if base_rate > 0 else 0.0
    return result


def _compute_attack_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
) -> Dict:
    """Calcula as metricas padrao do ataque."""
    metrics = {
        "attack_accuracy": float(accuracy_score(y_true, y_pred)),
        "attack_auc": float(roc_auc_score(y_true, y_score)),
        "attack_precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "attack_recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "attack_f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    metrics.update(compute_tpr_at_fpr(y_true, y_score))
    metrics.update(compute_top_k_metrics(y_true, y_score))
    return metrics


def feature_threshold_attack(
    X_attack_train: np.ndarray,
    y_attack_train: np.ndarray,
    X_attack_test: np.ndarray,
    y_attack_test: np.ndarray,
    feature_index: int = 0,
    higher_is_member: bool = False,
    feature_label: str = "loss",
    n_thresholds: int = 100,
) -> Dict:
    """
    Ataque por threshold em uma feature observavel.

    Por padrao preserva a regra historica: feature baixa indica member
    (loss <= threshold). Para features como max_confidence e correctness,
    use higher_is_member=True.
    """
    values_train = X_attack_train[:, feature_index]
    values_test = X_attack_test[:, feature_index]

    thresholds = np.linspace(values_train.min(), values_train.max(), n_thresholds)
    best_threshold = thresholds[0]
    best_acc = 0.0

    for threshold in thresholds:
        if higher_is_member:
            y_pred_train = (values_train >= threshold).astype(int)
        else:
            y_pred_train = (values_train <= threshold).astype(int)
        acc = accuracy_score(y_attack_train, y_pred_train)
        if acc > best_acc:
            best_acc = acc
            best_threshold = threshold

    if higher_is_member:
        y_pred_test = (values_test >= best_threshold).astype(int)
        y_score_test = values_test
    else:
        y_pred_test = (values_test <= best_threshold).astype(int)
        y_score_test = -values_test

    metrics = _compute_attack_metrics(y_attack_test, y_pred_test, y_score_test)
    metrics["threshold_used"] = float(best_threshold)
    metrics["threshold_feature"] = feature_label

    print(
        f"[threshold_attack/{feature_label}] threshold={best_threshold:.4f} | "
        f"acc={metrics['attack_accuracy']:.4f} | "
        f"auc={metrics['attack_auc']:.4f} | "
        f"tpr@5%fpr={metrics['tpr_at_fpr_5']:.4f}"
    )
    return metrics


def loss_threshold_attack(
    X_attack_train: np.ndarray,
    y_attack_train: np.ndarray,
    X_attack_test: np.ndarray,
    y_attack_test: np.ndarray,
    n_thresholds: int = 100,
) -> Dict:
    """Ataque historico por threshold na loss."""
    return feature_threshold_attack(
        X_attack_train,
        y_attack_train,
        X_attack_test,
        y_attack_test,
        feature_index=0,
        higher_is_member=False,
        feature_label="loss",
        n_thresholds=n_thresholds,
    )


def learned_attack(
    X_attack_train: np.ndarray,
    y_attack_train: np.ndarray,
    X_attack_test: np.ndarray,
    y_attack_test: np.ndarray,
    model_type: str = "logistic_regression",
) -> Dict:
    """
    Ataque aprendido com sklearn.

    model_type:
      - logistic_regression: StandardScaler + LogisticRegression.
      - random_forest: RandomForestClassifier.
    """
    if model_type == "logistic_regression":
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_attack_train)
        X_test_scaled = scaler.transform(X_attack_test)

        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(X_train_scaled, y_attack_train)
        y_pred = clf.predict(X_test_scaled)
        y_score = clf.predict_proba(X_test_scaled)[:, 1]

    elif model_type == "random_forest":
        clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        clf.fit(X_attack_train, y_attack_train)
        y_pred = clf.predict(X_attack_test)
        y_score = clf.predict_proba(X_attack_test)[:, 1]

    else:
        raise ValueError(
            f"model_type invalido: '{model_type}'. "
            "Use 'logistic_regression' ou 'random_forest'."
        )

    metrics = _compute_attack_metrics(y_attack_test, y_pred, y_score)
    print(
        f"[learned_attack/{model_type}] "
        f"acc={metrics['attack_accuracy']:.4f} | "
        f"auc={metrics['attack_auc']:.4f} | "
        f"tpr@5%fpr={metrics['tpr_at_fpr_5']:.4f}"
    )
    return metrics


def run_all_attacks(
    model_name: str,
    X_attack_train: np.ndarray,
    y_attack_train: np.ndarray,
    X_attack_test: np.ndarray,
    y_attack_test: np.ndarray,
    threshold_feature_index: int = 0,
    threshold_higher_is_member: bool = False,
    threshold_feature_label: str = "loss",
    threshold_attack_name: str = "threshold_loss",
) -> Dict:
    """
    Executa threshold, Logistic Regression e Random Forest.
    """
    print(f"\n[attacks] === Rodando ataques para: {model_name} ===")

    results = {
        threshold_attack_name: feature_threshold_attack(
            X_attack_train,
            y_attack_train,
            X_attack_test,
            y_attack_test,
            feature_index=threshold_feature_index,
            higher_is_member=threshold_higher_is_member,
            feature_label=threshold_feature_label,
        ),
        "logistic_regression": learned_attack(
            X_attack_train,
            y_attack_train,
            X_attack_test,
            y_attack_test,
            model_type="logistic_regression",
        ),
        "random_forest": learned_attack(
            X_attack_train,
            y_attack_train,
            X_attack_test,
            y_attack_test,
            model_type="random_forest",
        ),
    }

    print(f"\n[attacks] Resumo para {model_name}:")
    print(f"  {'Ataque':<24} {'Acc':>6} {'AUC':>6} {'F1':>6} {'TPR@5':>7}")
    print(f"  {'-' * 54}")
    for name, metrics in results.items():
        print(
            f"  {name:<24} "
            f"{metrics['attack_accuracy']:>6.4f} "
            f"{metrics['attack_auc']:>6.4f} "
            f"{metrics['attack_f1']:>6.4f} "
            f"{metrics['tpr_at_fpr_5']:>7.4f}"
        )

    return results
