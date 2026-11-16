"""
Feature extraction for the API Exposure / Output Restriction experiment.

Each exposure mode represents what a membership attacker can observe from a
target model API. The target model is trained once; these functions only
transform its outputs into attack features.
"""

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


ORACLE_FEATURE_NAMES = [
    "loss",
    "max_confidence",
    "true_class_confidence",
    "entropy",
    "correctness",
    "top2_confidence_gap",
]

EXPOSURE_MODE_DEFAULT_ROUNDING = {
    "oracle_full": None,
    "probs_full": None,
    "probs_rounded_2": 2,
    "probs_rounded_1": 1,
    "top1_confidence": None,
    "label_only": None,
}


def _resolve_round_decimals(exposure_mode: str, round_decimals: Optional[int]) -> Optional[int]:
    if exposure_mode not in EXPOSURE_MODE_DEFAULT_ROUNDING:
        valid = ", ".join(EXPOSURE_MODE_DEFAULT_ROUNDING)
        raise ValueError(f"Invalid exposure_mode '{exposure_mode}'. Use one of: {valid}")
    if round_decimals is not None:
        return round_decimals
    return EXPOSURE_MODE_DEFAULT_ROUNDING[exposure_mode]


def _entropy_from_probs(probs: torch.Tensor) -> torch.Tensor:
    eps = 1e-12
    return -(probs * torch.log(probs + eps)).sum(dim=1)


def _top2_gap(sorted_probs: torch.Tensor) -> torch.Tensor:
    if sorted_probs.shape[1] < 2:
        return torch.zeros(sorted_probs.shape[0], device=sorted_probs.device)
    return sorted_probs[:, 0] - sorted_probs[:, 1]


def _feature_names_for_mode(exposure_mode: str, n_classes: int) -> list[str]:
    if exposure_mode == "oracle_full":
        return ORACLE_FEATURE_NAMES.copy()
    if exposure_mode in {"probs_full", "probs_rounded_2", "probs_rounded_1"}:
        return [
            "max_confidence",
            "entropy",
            "top2_confidence_gap",
            *[f"sorted_prob_{i}" for i in range(1, n_classes + 1)],
        ]
    if exposure_mode == "top1_confidence":
        return ["max_confidence"]
    if exposure_mode == "label_only":
        return ["correctness"]
    valid = ", ".join(EXPOSURE_MODE_DEFAULT_ROUNDING)
    raise ValueError(f"Invalid exposure_mode '{exposure_mode}'. Use one of: {valid}")


def _extract_features_for_loader(
    model,
    data_loader: DataLoader,
    membership_label: int,
    exposure_mode: str,
    device: torch.device,
    round_decimals: Optional[int],
) -> Tuple[np.ndarray, np.ndarray, list[str]]:
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="none")

    all_features = []
    all_labels = []
    feature_names = None

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(X_batch)
            probs = F.softmax(logits, dim=1)
            if round_decimals is not None and exposure_mode != "oracle_full":
                probs = torch.round(probs * (10**round_decimals)) / (10**round_decimals)

            n_classes = probs.shape[1]
            if feature_names is None:
                feature_names = _feature_names_for_mode(exposure_mode, n_classes)

            max_conf, _ = probs.max(dim=1)
            sorted_probs, _ = probs.sort(dim=1, descending=True)
            top2_gap = _top2_gap(sorted_probs)
            entropy = _entropy_from_probs(probs)
            preds = probs.argmax(dim=1)
            correctness = (preds == y_batch).float()

            if exposure_mode == "oracle_full":
                loss_per_sample = criterion(logits, y_batch)
                true_conf = probs[torch.arange(len(y_batch), device=device), y_batch]
                batch_features = torch.stack(
                    [
                        loss_per_sample,
                        max_conf,
                        true_conf,
                        entropy,
                        correctness,
                        top2_gap,
                    ],
                    dim=1,
                )
            elif exposure_mode in {"probs_full", "probs_rounded_2", "probs_rounded_1"}:
                batch_features = torch.cat(
                    [
                        max_conf.unsqueeze(1),
                        entropy.unsqueeze(1),
                        top2_gap.unsqueeze(1),
                        sorted_probs,
                    ],
                    dim=1,
                )
            elif exposure_mode == "top1_confidence":
                batch_features = max_conf.unsqueeze(1)
            elif exposure_mode == "label_only":
                batch_features = correctness.unsqueeze(1)
            else:
                valid = ", ".join(EXPOSURE_MODE_DEFAULT_ROUNDING)
                raise ValueError(f"Invalid exposure_mode '{exposure_mode}'. Use one of: {valid}")

            all_features.append(batch_features.cpu().numpy())
            all_labels.extend([membership_label] * len(y_batch))

    features = np.concatenate(all_features, axis=0).astype(np.float32)
    labels = np.array(all_labels, dtype=np.int64)
    return features, labels, feature_names or _feature_names_for_mode(exposure_mode, 0)


def build_api_exposure_attack_dataset(
    model,
    member_loader: DataLoader,
    nonmember_loader: DataLoader,
    exposure_mode: str,
    device=None,
    round_decimals: int | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Returns:
      X_attack: matrix of features observable by the attacker
      y_attack: 1 for member, 0 for nonmember
      feature_names: names of generated features
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    resolved_round_decimals = _resolve_round_decimals(exposure_mode, round_decimals)
    model = model.to(device)

    X_mem, y_mem, feature_names = _extract_features_for_loader(
        model=model,
        data_loader=member_loader,
        membership_label=1,
        exposure_mode=exposure_mode,
        device=device,
        round_decimals=resolved_round_decimals,
    )
    X_nm, y_nm, nonmember_feature_names = _extract_features_for_loader(
        model=model,
        data_loader=nonmember_loader,
        membership_label=0,
        exposure_mode=exposure_mode,
        device=device,
        round_decimals=resolved_round_decimals,
    )

    if feature_names != nonmember_feature_names:
        raise RuntimeError(
            "Member and nonmember loaders produced different API exposure feature schemas."
        )

    X_attack = np.concatenate([X_mem, X_nm], axis=0)
    y_attack = np.concatenate([y_mem, y_nm], axis=0)

    assert (y_attack[: len(y_mem)] == 1).all(), "Members must have attack label 1"
    assert (y_attack[len(y_mem) :] == 0).all(), "Nonmembers must have attack label 0"

    return X_attack, y_attack, feature_names
