"""
src/data_20newsgroups.py
Pipeline de dados para o dataset 20 Newsgroups.

Diferenças em relação ao pipeline de diabetes (data.py):
  - Texto → TF-IDF (vetorização feita ANTES do split, mas o fit do scaler
    continua sendo feito apenas em X_fit para evitar leakage)
  - 20 classes (não-binário)
  - Sem necessidade de imputação (TF-IDF não gera NaN)
  - StandardScaler aplicado APENAS em X_fit (mesma lógica de leakage)
"""

import os
import warnings
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.config import FIT_RATIO, VAL_RATIO, NONMEMBER_RATIO, HOLDOUT_RATIO


# ============================================================
# 1. Carregamento do dataset 20 Newsgroups
# ============================================================

def load_20newsgroups(
    max_features: int = 5000,
    subset: str = "all",
    remove_headers: bool = True,
) -> Tuple[np.ndarray, np.ndarray, list, str, list]:
    """
    Carrega o 20 Newsgroups e converte para representação TF-IDF.

    Parameters
    ----------
    max_features   : número máximo de features TF-IDF
    subset         : 'all', 'train', ou 'test' (do split original)
    remove_headers : se True, remove headers/footers/quotes do email

    Returns
    -------
    X              : np.ndarray (N, max_features) — TF-IDF matrix densa
    y              : np.ndarray (N,) — labels inteiras 0..19
    feature_names  : lista com nomes das features TF-IDF
    dataset_name   : '20newsgroups'
    target_names   : lista com nomes das 20 categorias
    """
    remove = ()
    if remove_headers:
        remove = ("headers", "footers", "quotes")

    print(f"[data_20ng] Carregando 20 Newsgroups (subset='{subset}', "
          f"max_features={max_features}, remove={remove})...")

    newsgroups = fetch_20newsgroups(
        subset=subset,
        remove=remove,
        shuffle=True,
        random_state=42,
    )

    texts = newsgroups.data
    y = np.array(newsgroups.target, dtype=np.int64)
    target_names = list(newsgroups.target_names)

    # TF-IDF vectorization
    print(f"[data_20ng] Aplicando TF-IDF com max_features={max_features}...")
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words="english",
        max_df=0.95,
        min_df=2,
        sublinear_tf=True,
    )
    X_tfidf = vectorizer.fit_transform(texts)
    X = X_tfidf.toarray().astype(np.float32)  # Converter para densa
    feature_names = vectorizer.get_feature_names_out().tolist()

    print(f"[data_20ng] Dataset carregado: {X.shape[0]} amostras, "
          f"{X.shape[1]} features TF-IDF, {len(target_names)} classes.")
    print(f"[data_20ng] Classes: {target_names}")

    # Distribuição de classes
    unique, counts = np.unique(y, return_counts=True)
    for cls_idx, count in zip(unique, counts):
        print(f"  {target_names[cls_idx]}: {count} amostras")

    return X, y, feature_names, "20newsgroups", target_names


def load_20newsgroups_texts(
    subset: str = "all",
    remove_headers: bool = True,
) -> Tuple[np.ndarray, np.ndarray, str, list]:
    """
    Carrega o 20 Newsgroups como texto cru.

    Este e o caminho recomendado para os experimentos finais: o TF-IDF deve ser
    fitado depois do split, usando apenas target_fit_members.
    """
    remove = ()
    if remove_headers:
        remove = ("headers", "footers", "quotes")

    print(
        f"[data_20ng] Carregando textos brutos (subset='{subset}', remove={remove})..."
    )
    newsgroups = fetch_20newsgroups(
        subset=subset,
        remove=remove,
        shuffle=True,
        random_state=42,
    )
    texts = np.array(newsgroups.data, dtype=object)
    y = np.array(newsgroups.target, dtype=np.int64)
    target_names = list(newsgroups.target_names)
    print(
        f"[data_20ng] Textos carregados: {len(texts)} amostras, "
        f"{len(target_names)} classes."
    )
    return texts, y, "20newsgroups", target_names


# ============================================================
# 2. Split 4-way (reutiliza a mesma lógica do data.py)
# ============================================================

def create_splits_20ng(
    X: np.ndarray,
    y: np.ndarray,
    seed: int = 42,
    experiment_sample_size: Optional[int] = None,
    fit_ratio: float = FIT_RATIO,
    val_ratio: float = VAL_RATIO,
    nonmember_ratio: float = NONMEMBER_RATIO,
    holdout_ratio: float = HOLDOUT_RATIO,
) -> Dict:
    """
    Cria 4 partições estratificadas com isolamento estrito de non-members.

    Mesma lógica de data.py, mas trabalha com arrays numpy diretamente
    (já que X vem como ndarray do TF-IDF, não DataFrame).

    Returns
    -------
    dict com chaves:
      X_fit, y_fit, idx_fit
      X_val, y_val, idx_val
      X_nonmember, y_nonmember, idx_nonmember
      X_holdout, y_holdout, idx_holdout
    """
    assert abs(fit_ratio + val_ratio + nonmember_ratio + holdout_ratio - 1.0) < 1e-6, \
        "Proporções devem somar 1.0"

    all_indices = np.arange(len(X))

    # --- 1. Subsample estratificado ---
    if experiment_sample_size is not None and experiment_sample_size < len(X):
        sample_idx, _ = train_test_split(
            all_indices,
            train_size=experiment_sample_size,
            stratify=y,
            random_state=seed,
        )
        X = X[sample_idx]
        y = y[sample_idx]
        all_indices = np.arange(len(X))
        print(f"[data_20ng] Subsample: {len(X)} amostras.")
    else:
        print(f"[data_20ng] Usando dataset completo: {len(X)} amostras.")

    # --- 2. Separar fit_members do restante ---
    fit_idx, rest_idx = train_test_split(
        all_indices,
        train_size=fit_ratio,
        stratify=y,
        random_state=seed,
    )

    # --- 3. Dividir o restante em val / nonmember / holdout ---
    rest_total = val_ratio + nonmember_ratio + holdout_ratio
    val_rel = val_ratio / rest_total
    nm_rel = nonmember_ratio / (nonmember_ratio + holdout_ratio)

    y_rest = y[rest_idx]

    val_idx_in_rest, nonholdout_idx = train_test_split(
        rest_idx,
        train_size=val_rel,
        stratify=y_rest,
        random_state=seed + 1,
    )

    y_nonholdout = y[nonholdout_idx]
    nonmember_idx, holdout_idx = train_test_split(
        nonholdout_idx,
        train_size=nm_rel,
        stratify=y_nonholdout,
        random_state=seed + 2,
    )

    # --- 4. Assertions de disjuntividade ---
    fit_set = set(fit_idx.tolist())
    val_set = set(val_idx_in_rest.tolist())
    nm_set = set(nonmember_idx.tolist())
    hold_set = set(holdout_idx.tolist())

    assert fit_set.isdisjoint(nm_set), \
        "[ERRO] fit_members e nonmembers se sobrepõem!"
    assert val_set.isdisjoint(nm_set), \
        "[ERRO] val_selection e nonmembers se sobrepõem!"
    assert fit_set.isdisjoint(val_set), \
        "[ERRO] fit_members e val_selection se sobrepõem!"

    print(
        f"[data_20ng] Splits criados:\n"
        f"  fit_members    : {len(fit_idx):>6} amostras\n"
        f"  val_selection  : {len(val_idx_in_rest):>6} amostras\n"
        f"  nonmembers     : {len(nonmember_idx):>6} amostras\n"
        f"  holdout        : {len(holdout_idx):>6} amostras"
    )

    return {
        "X_fit": X[fit_idx].astype(np.float32),
        "y_fit": y[fit_idx].astype(np.int64),
        "idx_fit": np.array(fit_idx),
        "X_val": X[val_idx_in_rest].astype(np.float32),
        "y_val": y[val_idx_in_rest].astype(np.int64),
        "idx_val": np.array(val_idx_in_rest),
        "X_nonmember": X[nonmember_idx].astype(np.float32),
        "y_nonmember": y[nonmember_idx].astype(np.int64),
        "idx_nonmember": np.array(nonmember_idx),
        "X_holdout": X[holdout_idx].astype(np.float32),
        "y_holdout": y[holdout_idx].astype(np.int64),
        "idx_holdout": np.array(holdout_idx),
    }


def create_text_splits_20ng(
    texts: np.ndarray,
    y: np.ndarray,
    seed: int = 42,
    experiment_sample_size: Optional[int] = None,
    fit_ratio: float = FIT_RATIO,
    val_ratio: float = VAL_RATIO,
    nonmember_ratio: float = NONMEMBER_RATIO,
    holdout_ratio: float = HOLDOUT_RATIO,
) -> Dict:
    """
    Cria os mesmos splits 4-way, mas preserva textos crus.

    Use com preprocess_text_splits_20ng para fit do TF-IDF apenas em X_fit.
    """
    assert abs(fit_ratio + val_ratio + nonmember_ratio + holdout_ratio - 1.0) < 1e-6, \
        "Proporcoes devem somar 1.0"

    all_indices = np.arange(len(texts))

    if experiment_sample_size is not None and experiment_sample_size < len(texts):
        sample_idx, _ = train_test_split(
            all_indices,
            train_size=experiment_sample_size,
            stratify=y,
            random_state=seed,
        )
        texts = texts[sample_idx]
        y = y[sample_idx]
        all_indices = np.arange(len(texts))
        print(f"[data_20ng] Subsample texto: {len(texts)} amostras.")
    else:
        print(f"[data_20ng] Usando textos completos: {len(texts)} amostras.")

    fit_idx, rest_idx = train_test_split(
        all_indices,
        train_size=fit_ratio,
        stratify=y,
        random_state=seed,
    )

    rest_total = val_ratio + nonmember_ratio + holdout_ratio
    val_rel = val_ratio / rest_total
    nm_rel = nonmember_ratio / (nonmember_ratio + holdout_ratio)

    val_idx, nonholdout_idx = train_test_split(
        rest_idx,
        train_size=val_rel,
        stratify=y[rest_idx],
        random_state=seed + 1,
    )
    nonmember_idx, holdout_idx = train_test_split(
        nonholdout_idx,
        train_size=nm_rel,
        stratify=y[nonholdout_idx],
        random_state=seed + 2,
    )

    fit_set = set(fit_idx.tolist())
    val_set = set(val_idx.tolist())
    nm_set = set(nonmember_idx.tolist())
    hold_set = set(holdout_idx.tolist())

    assert fit_set.isdisjoint(val_set)
    assert fit_set.isdisjoint(nm_set)
    assert fit_set.isdisjoint(hold_set)
    assert val_set.isdisjoint(nm_set)
    assert val_set.isdisjoint(hold_set)
    assert nm_set.isdisjoint(hold_set)

    print(
        f"[data_20ng] Splits texto:\n"
        f"  fit_members    : {len(fit_idx):>6} amostras\n"
        f"  val_selection  : {len(val_idx):>6} amostras\n"
        f"  nonmembers     : {len(nonmember_idx):>6} amostras\n"
        f"  holdout        : {len(holdout_idx):>6} amostras"
    )

    return {
        "X_fit": texts[fit_idx], "y_fit": y[fit_idx].astype(np.int64), "idx_fit": np.array(fit_idx),
        "X_val": texts[val_idx], "y_val": y[val_idx].astype(np.int64), "idx_val": np.array(val_idx),
        "X_nonmember": texts[nonmember_idx], "y_nonmember": y[nonmember_idx].astype(np.int64), "idx_nonmember": np.array(nonmember_idx),
        "X_holdout": texts[holdout_idx], "y_holdout": y[holdout_idx].astype(np.int64), "idx_holdout": np.array(holdout_idx),
    }


# ============================================================
# 3. Preprocessamento sem leakage
# ============================================================

def preprocess_splits_20ng(splits: Dict) -> Dict:
    """
    Aplica preprocessamento SEM LEAKAGE para 20 Newsgroups.

    - StandardScaler fitado APENAS em X_fit
    - transform aplicado em X_val, X_nonmember, X_holdout
    - SEM binarização do target (são 20 classes)
    - SEM imputação (TF-IDF não gera NaN)

    Returns
    -------
    dict com splits transformados + 'scaler' para reprodução
    """
    X_fit = splits["X_fit"].copy()

    # --- Scaler ---
    scaler = StandardScaler()
    X_fit = scaler.fit_transform(X_fit).astype(np.float32)

    print(
        f"[preprocess_20ng] Scaler fitado em X_fit ({X_fit.shape[0]} amostras, "
        f"{X_fit.shape[1]} features). mean_ shape: {scaler.mean_.shape}"
    )

    def _transform(X):
        return scaler.transform(X).astype(np.float32)

    X_val = _transform(splits["X_val"])
    X_nonmember = _transform(splits["X_nonmember"])
    X_holdout = _transform(splits["X_holdout"])

    result = {
        "X_fit":         X_fit,
        "y_fit":         splits["y_fit"],
        "idx_fit":       splits["idx_fit"],
        "X_val":         X_val,
        "y_val":         splits["y_val"],
        "idx_val":       splits["idx_val"],
        "X_nonmember":   X_nonmember,
        "y_nonmember":   splits["y_nonmember"],
        "idx_nonmember": splits["idx_nonmember"],
        "X_holdout":     X_holdout,
        "y_holdout":     splits["y_holdout"],
        "idx_holdout":   splits["idx_holdout"],
        # Artefatos de preprocessamento
        "scaler": scaler,
    }

    # Distribuição de classes em fit_members
    unique, counts = np.unique(result["y_fit"], return_counts=True)
    print(f"[preprocess_20ng] Distribuição de classes em fit_members:")
    for cls_idx, count in zip(unique, counts):
        print(f"  Classe {cls_idx}: {count} amostras ({100*count/len(result['y_fit']):.1f}%)")

    return result


def preprocess_text_splits_20ng(
    splits: Dict,
    max_features: int = 5000,
    scale: bool = True,
) -> Dict:
    """
    TF-IDF sem leakage para 20 Newsgroups.

    Fit do TfidfVectorizer ocorre apenas em X_fit. Val, nonmembers e holdout
    sao apenas transformados com o vocabulario aprendido nos members de treino.
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words="english",
        max_df=0.95,
        min_df=2,
        sublinear_tf=True,
    )
    X_fit_sparse = vectorizer.fit_transform(splits["X_fit"])
    feature_names = vectorizer.get_feature_names_out().tolist()

    def _vectorize(texts):
        return vectorizer.transform(texts).toarray().astype(np.float32)

    X_fit = X_fit_sparse.toarray().astype(np.float32)
    X_val = _vectorize(splits["X_val"])
    X_nonmember = _vectorize(splits["X_nonmember"])
    X_holdout = _vectorize(splits["X_holdout"])

    scaler = None
    if scale:
        scaler = StandardScaler()
        X_fit = scaler.fit_transform(X_fit).astype(np.float32)
        X_val = scaler.transform(X_val).astype(np.float32)
        X_nonmember = scaler.transform(X_nonmember).astype(np.float32)
        X_holdout = scaler.transform(X_holdout).astype(np.float32)

    print(
        f"[preprocess_20ng] TF-IDF fitado apenas em X_fit "
        f"({len(splits['X_fit'])} textos, {len(feature_names)} features)."
    )

    return {
        "X_fit": X_fit,
        "y_fit": splits["y_fit"],
        "idx_fit": splits["idx_fit"],
        "X_val": X_val,
        "y_val": splits["y_val"],
        "idx_val": splits["idx_val"],
        "X_nonmember": X_nonmember,
        "y_nonmember": splits["y_nonmember"],
        "idx_nonmember": splits["idx_nonmember"],
        "X_holdout": X_holdout,
        "y_holdout": splits["y_holdout"],
        "idx_holdout": splits["idx_holdout"],
        "vectorizer": vectorizer,
        "feature_names": feature_names,
        "scaler": scaler,
    }
