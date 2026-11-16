"""
src/evaluate.py
Avaliação do modelo alvo e cálculo de métricas de confiança.

Funções separadas para evitar responsabilidades misturadas:
  - evaluate_model()          : métricas de um único conjunto
  - compute_confidence_metrics(): métricas de confiança de um conjunto
  - compute_confidence_gap()  : diferença member vs non-member (função separada)
  - compute_generalization_gap(): gap treino-teste
"""

import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, Optional

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
)

from src.models import MLPClassifier


def evaluate_model(
    model: MLPClassifier,
    data_loader: DataLoader,
    device: Optional[torch.device] = None,
) -> Dict:
    """
    Avalia o modelo em um DataLoader e retorna métricas completas.

    AUC rule (spec):
      - output_dim == 2: AUC binária com probs[:, 1]
      - output_dim > 2: AUC multiclasse OVR
      - Se falhar: retorna None (não lança exceção)

    Returns
    -------
    dict com: accuracy, f1, auc, mean_loss,
              probabilities (N, C), predictions (N,), true_labels (N,)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="mean")

    all_logits = []
    all_labels = []
    total_loss = 0.0
    n_samples  = 0

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            total_loss += loss.item() * len(y_batch)
            n_samples  += len(y_batch)
            all_logits.append(logits.cpu())
            all_labels.append(y_batch.cpu())

    logits_all = torch.cat(all_logits, dim=0)
    labels_all = torch.cat(all_labels, dim=0).numpy()
    probs_all  = F.softmax(logits_all, dim=1).numpy()   # shape (N, C)
    preds_all  = probs_all.argmax(axis=1)               # shape (N,)
    mean_loss  = total_loss / n_samples

    acc = accuracy_score(labels_all, preds_all)
    f1  = f1_score(labels_all, preds_all, average="macro", zero_division=0)

    # AUC — binário vs multiclasse
    output_dim = probs_all.shape[1]
    auc = None
    try:
        if output_dim == 2:
            auc = roc_auc_score(labels_all, probs_all[:, 1])
        else:
            auc = roc_auc_score(
                labels_all, probs_all,
                multi_class="ovr", average="macro"
            )
    except Exception as e:
        warnings.warn(f"[evaluate] AUC não pôde ser calculada: {e}", UserWarning)
        auc = None

    return {
        "accuracy":     acc,
        "f1":           f1,
        "auc":          auc,
        "mean_loss":    mean_loss,
        "probabilities": probs_all,
        "predictions":  preds_all,
        "true_labels":  labels_all,
    }


def compute_confidence_metrics(
    probs: np.ndarray,
    labels: np.ndarray,
) -> Dict:
    """
    Calcula métricas de confiança para UM conjunto de amostras.

    NÃO calcula confidence_gap (use compute_confidence_gap para isso).

    Parameters
    ----------
    probs  : softmax probabilities, shape (N, C)
    labels : true labels, shape (N,)

    Returns
    -------
    dict com:
      mean_max_confidence      : média da confiança máxima (max softmax)
      mean_true_class_confidence: média da softmax na classe correta
      mean_entropy             : entropia de Shannon média (nats)
      correctness_rate         : fração de predições corretas
    """
    # Máxima confiança por amostra
    max_conf = probs.max(axis=1)               # (N,)

    # Confiança na classe correta
    true_conf = probs[np.arange(len(labels)), labels]  # (N,)

    # Entropia de Shannon (nats)
    eps = 1e-12
    entropy = -(probs * np.log(probs + eps)).sum(axis=1)  # (N,)

    # Acerto
    preds = probs.argmax(axis=1)
    correctness = (preds == labels).astype(float)

    return {
        "mean_max_confidence":       float(max_conf.mean()),
        "mean_true_class_confidence": float(true_conf.mean()),
        "mean_entropy":              float(entropy.mean()),
        "correctness_rate":          float(correctness.mean()),
    }


def compute_confidence_gap(
    member_probs: np.ndarray,
    nonmember_probs: np.ndarray,
) -> Dict:
    """
    Calcula a diferença de confiança entre membros e não-membros.

    Função SEPARADA de compute_confidence_metrics (responsabilidade única).

    Returns
    -------
    dict com:
      confidence_gap: mean_max_confidence(members) - mean_max_confidence(nonmembers)
      Valor positivo indica que o modelo é mais confiante nos membros (vulnerabilidade).
    """
    mean_conf_members    = member_probs.max(axis=1).mean()
    mean_conf_nonmembers = nonmember_probs.max(axis=1).mean()
    gap = float(mean_conf_members - mean_conf_nonmembers)

    return {
        "confidence_gap":             gap,
        "mean_max_conf_members":      float(mean_conf_members),
        "mean_max_conf_nonmembers":   float(mean_conf_nonmembers),
    }


def compute_generalization_gap(
    train_metrics: Dict,
    test_metrics: Dict,
) -> Dict:
    """
    Calcula o generalization gap entre treino (fit_members) e teste (nonmembers).

    Returns
    -------
    dict com:
      gen_gap  : train_acc - test_acc  (positivo = overfitting)
      loss_gap : test_loss - train_loss (positivo = overfitting)
    """
    gen_gap  = train_metrics["accuracy"] - test_metrics["accuracy"]
    loss_gap = test_metrics["mean_loss"] - train_metrics["mean_loss"]

    return {
        "gen_gap":  float(gen_gap),
        "loss_gap": float(loss_gap),
    }
