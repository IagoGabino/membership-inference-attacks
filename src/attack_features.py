"""
src/attack_features.py
Extração de features para o ataque de membership inference.

Cada amostra gera 6 features:
  [0] loss               — cross-entropy por amostra
  [1] max_confidence     — max softmax
  [2] true_class_conf    — softmax na classe correta
  [3] entropy            — entropia de Shannon
  [4] correctness        — 1 se acertou, 0 se errou
  [5] top2_conf_gap      — softmax[top1] - softmax[top2]

IMPORTANTE: member_loader deve usar APENAS target_fit_members (idx_fit).
target_val_selection NÃO deve entrar no dataset do ataque.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Tuple, Optional, Dict

from src.models import MLPClassifier


def extract_attack_features(
    model: MLPClassifier,
    data_loader: DataLoader,
    label: int,
    device: Optional[torch.device] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extrai features de ataque para todas as amostras de um DataLoader.

    Parameters
    ----------
    model       : modelo alvo treinado
    data_loader : DataLoader do conjunto (member ou non-member)
    label       : 1 para members, 0 para non-members
    device      : torch.device

    Returns
    -------
    features : np.ndarray de shape (N, 6)
               [0]=loss, [1]=max_conf, [2]=true_class_conf,
               [3]=entropy, [4]=correctness, [5]=top2_conf_gap
    labels   : np.ndarray de shape (N,) com o valor de `label`
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.eval()

    criterion = nn.CrossEntropyLoss(reduction="none")  # loss por amostra
    eps = 1e-12

    all_features = []
    all_labels   = []

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = model(X_batch)
            probs  = F.softmax(logits, dim=1)  # (batch, C)

            # [0] Loss por amostra
            loss_per_sample = criterion(logits, y_batch)  # (batch,)

            # [1] Max confidence
            max_conf, top1_idx = probs.max(dim=1)  # (batch,)

            # [2] True class confidence
            true_conf = probs[torch.arange(len(y_batch)), y_batch]  # (batch,)

            # [3] Entropy (nats)
            entropy = -(probs * torch.log(probs + eps)).sum(dim=1)  # (batch,)

            # [4] Correctness
            preds       = probs.argmax(dim=1)
            correctness = (preds == y_batch).float()  # (batch,)

            # [5] Top-2 confidence gap
            sorted_probs, _ = probs.sort(dim=1, descending=True)
            top2_gap = sorted_probs[:, 0] - sorted_probs[:, 1]  # (batch,)

            # Montar feature matrix (batch, 6)
            batch_features = torch.stack([
                loss_per_sample,
                max_conf,
                true_conf,
                entropy,
                correctness,
                top2_gap,
            ], dim=1).cpu().numpy()

            all_features.append(batch_features)
            all_labels.extend([label] * len(y_batch))

    features = np.concatenate(all_features, axis=0).astype(np.float32)
    labels   = np.array(all_labels, dtype=np.int64)

    return features, labels


def build_attack_dataset(
    model: MLPClassifier,
    member_loader: DataLoader,
    nonmember_loader: DataLoader,
    device: Optional[torch.device] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Constrói o dataset do ataque concatenando features de members e non-members.

    Feature layout (índices):
      0: loss               — cross-entropy (menor = mais provável ser membro)
      1: max_confidence     — max(softmax) (maior = mais provável ser membro)
      2: true_class_conf    — softmax na classe correta
      3: entropy            — incerteza do modelo (menor = mais provável ser membro)
      4: correctness        — 1 se predição correta, 0 se errada
      5: top2_conf_gap      — gap entre top-1 e top-2 softmax

    IMPORTANTE:
      member_loader   : deve conter APENAS target_fit_members (idx_fit)
      nonmember_loader: deve conter APENAS target_test_nonmembers (idx_nonmember)
      target_val_selection (idx_val) NÃO entra no ataque.

    Returns
    -------
    X_attack : np.ndarray (N_members + N_nonmembers, 6)
    y_attack : np.ndarray (N_members + N_nonmembers,) — 1=member, 0=non-member
    """
    print("[attack] Extraindo features dos members...")
    X_mem, y_mem = extract_attack_features(model, member_loader,    label=1, device=device)

    print("[attack] Extraindo features dos non-members...")
    X_nm,  y_nm  = extract_attack_features(model, nonmember_loader, label=0, device=device)

    X_attack = np.concatenate([X_mem, X_nm], axis=0)
    y_attack = np.concatenate([y_mem, y_nm], axis=0)

    # Assertion: labels corretas
    assert (y_attack[:len(y_mem)] == 1).all(), "[ERRO] Members não estão com label=1"
    assert (y_attack[len(y_mem):] == 0).all(), "[ERRO] Non-members não estão com label=0"

    print(
        f"[attack] Dataset do ataque: {len(X_attack)} amostras "
        f"({y_mem.sum()} members, {(y_nm==0).sum()} non-members)"
    )

    return X_attack, y_attack
