"""
src/data.py
Pipeline de dados: carregamento, split 4-way com isolamento de non-members,
preprocessamento sem leakage (scaler fitado apenas em target_fit_members).
"""

import os
import warnings
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.config import FIT_RATIO, VAL_RATIO, NONMEMBER_RATIO, HOLDOUT_RATIO


# ============================================================
# 1. Carregamento do dataset
# ============================================================

def load_dataset(name: str = "cdc_diabetes") -> Tuple[pd.DataFrame, pd.Series, list, str]:
    """
    Carrega o dataset pelo nome.

    Ordem de tentativa:
      1. ucimlrepo (online)
      2. CSV local em data/raw/<name>.csv
      3. Fallback para Adult Income (ucimlrepo ID=2)

    Returns
    -------
    X            : DataFrame com features
    y            : Series com target
    feature_names: lista de nomes de features
    dataset_name : nome do dataset carregado
    """
    if name == "cdc_diabetes":
        dataset_id = 891
        local_path = os.path.join("data", "raw", "cdc_diabetes.csv")
        fallback_id = 2
        fallback_name = "adult_income"
    else:
        raise ValueError(f"Dataset desconhecido: {name}. Use 'cdc_diabetes'.")

    # --- Tentativa 1: ucimlrepo ---
    try:
        from ucimlrepo import fetch_ucirepo
        print(f"[data] Baixando dataset '{name}' via ucimlrepo (ID={dataset_id})...")
        repo = fetch_ucirepo(id=dataset_id)
        X = repo.data.features
        y = repo.data.targets.squeeze()
        feature_names = list(X.columns)
        print(f"[data] Dataset carregado: {X.shape[0]} amostras, {X.shape[1]} features.")
        return X, y, feature_names, name
    except Exception as e:
        print(f"[data] ucimlrepo falhou ({e}). Tentando CSV local...")

    # --- Tentativa 2: CSV local ---
    if os.path.exists(local_path):
        try:
            df = pd.read_csv(local_path)
            y = df.iloc[:, -1]
            X = df.iloc[:, :-1]
            feature_names = list(X.columns)
            print(f"[data] Dataset carregado do CSV local: {X.shape[0]} amostras.")
            return X, y, feature_names, name
        except Exception as e:
            print(f"[data] CSV local falhou ({e}). Tentando fallback Adult Income...")

    # --- Tentativa 3: Fallback Adult Income ---
    try:
        from ucimlrepo import fetch_ucirepo
        print(f"[data] Usando fallback: Adult Income (ID={fallback_id})...")
        repo = fetch_ucirepo(id=fallback_id)
        X = repo.data.features
        y = repo.data.targets.squeeze()
        feature_names = list(X.columns)
        print(f"[data] Fallback carregado: {X.shape[0]} amostras, {X.shape[1]} features.")
        return X, y, feature_names, fallback_name
    except Exception as e:
        raise RuntimeError(
            f"Não foi possível carregar nenhum dataset. Erro final: {e}\n"
            "Solução: baixe o CDC Diabetes CSV manualmente em data/raw/cdc_diabetes.csv"
        )


# ============================================================
# 2. Split 4-way com isolamento estrito de non-members
# ============================================================

def create_splits(
    X: pd.DataFrame,
    y: pd.Series,
    seed: int = 42,
    experiment_sample_size: Optional[int] = None,
    fit_ratio: float = FIT_RATIO,
    val_ratio: float = VAL_RATIO,
    nonmember_ratio: float = NONMEMBER_RATIO,
    holdout_ratio: float = HOLDOUT_RATIO,
) -> Dict:
    """
    Cria 4 partições estratificadas com isolamento estrito de non-members.

    Ordem de operações:
      1. Subsample estratificado (se experiment_sample_size não for None)
      2. Divide em fit_members + rest (stratified)
      3. Divide rest em val_selection + nonmembers + holdout (stratified)
      4. Verifica 3 assertions de disjuntividade
      5. Retorna dict com arrays + índices originais

    IMPORTANTE: target_test_nonmembers (idx_nonmember) nunca pode ser usado
    como val_loader, para early stopping, checkpoint ou qualquer etapa que
    influencie o modelo alvo.

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

    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    all_indices = np.arange(len(X))

    # --- 1. Subsample estratificado ---
    if experiment_sample_size is not None and experiment_sample_size < len(X):
        sample_idx, _ = train_test_split(
            all_indices,
            train_size=experiment_sample_size,
            stratify=y.values,
            random_state=seed,
        )
        X = X.iloc[sample_idx].reset_index(drop=True)
        y = y.iloc[sample_idx].reset_index(drop=True)
        all_indices = np.arange(len(X))
        print(f"[data] Subsample: {len(X)} amostras (de {len(sample_idx) + len(_)} originais).")
    else:
        print(f"[data] Usando dataset completo: {len(X)} amostras.")

    # --- 2. Separar fit_members do restante ---
    fit_idx, rest_idx = train_test_split(
        all_indices,
        train_size=fit_ratio,
        stratify=y.values,
        random_state=seed,
    )

    # --- 3. Dividir o restante em val / nonmember / holdout ---
    # Proporções relativas ao bloco "rest"
    rest_total = val_ratio + nonmember_ratio + holdout_ratio
    val_rel   = val_ratio / rest_total
    nm_rel    = nonmember_ratio / (nonmember_ratio + holdout_ratio)

    y_rest = y.iloc[rest_idx].values

    val_idx_in_rest, nonholdout_idx = train_test_split(
        rest_idx,
        train_size=val_rel,
        stratify=y_rest,
        random_state=seed + 1,
    )

    y_nonholdout = y.iloc[nonholdout_idx].values
    nonmember_idx, holdout_idx = train_test_split(
        nonholdout_idx,
        train_size=nm_rel,
        stratify=y_nonholdout,
        random_state=seed + 2,
    )

    # --- 4. Assertions de disjuntividade ---
    fit_set    = set(fit_idx.tolist())
    val_set    = set(val_idx_in_rest.tolist())
    nm_set     = set(nonmember_idx.tolist())
    hold_set   = set(holdout_idx.tolist())

    assert fit_set.isdisjoint(nm_set),  \
        "[ERRO] fit_members e nonmembers se sobrepõem!"
    assert val_set.isdisjoint(nm_set),  \
        "[ERRO] val_selection e nonmembers se sobrepõem!"
    assert fit_set.isdisjoint(val_set), \
        "[ERRO] fit_members e val_selection se sobrepõem!"

    print(
        f"[data] Splits criados:\n"
        f"  fit_members    : {len(fit_idx):>6} amostras\n"
        f"  val_selection  : {len(val_idx_in_rest):>6} amostras\n"
        f"  nonmembers     : {len(nonmember_idx):>6} amostras\n"
        f"  holdout        : {len(holdout_idx):>6} amostras"
    )

    def _extract(idx):
        return X.iloc[idx].values.astype(np.float32), y.iloc[idx].values.astype(np.int64), np.array(idx)

    X_fit, y_fit, idx_fit             = _extract(fit_idx)
    X_val, y_val, idx_val             = _extract(val_idx_in_rest)
    X_nm,  y_nm,  idx_nm              = _extract(nonmember_idx)
    X_hold, y_hold, idx_hold          = _extract(holdout_idx)

    return {
        "X_fit": X_fit, "y_fit": y_fit, "idx_fit": idx_fit,
        "X_val": X_val, "y_val": y_val, "idx_val": idx_val,
        "X_nonmember": X_nm,  "y_nonmember": y_nm,  "idx_nonmember": idx_nm,
        "X_holdout":  X_hold, "y_holdout": y_hold, "idx_holdout": idx_hold,
    }


# ============================================================
# 3. Preprocessamento sem leakage
# ============================================================

def preprocess_splits(splits: Dict, binary_target: bool = True) -> Dict:
    """
    Aplica preprocessamento SEM LEAKAGE:
      - SimpleImputer (mediana) e StandardScaler fitados APENAS em X_fit
      - transform aplicado em X_val, X_nonmember, X_holdout
      - Binarização do target (0 → 0, {1,2} → 1) se binary_target=True

    Returns
    -------
    dict com splits transformados + 'scaler' e 'imputer' para reprodução
    """
    X_fit = splits["X_fit"].copy()

    # --- Imputer ---
    imputer = SimpleImputer(strategy="median")
    X_fit = imputer.fit_transform(X_fit)

    # --- Scaler ---
    scaler = StandardScaler()
    X_fit = scaler.fit_transform(X_fit).astype(np.float32)

    print(
        f"[preprocess] Scaler fitado em X_fit ({X_fit.shape[0]} amostras, "
        f"{X_fit.shape[1]} features). mean_ shape: {scaler.mean_.shape}"
    )

    def _transform(X):
        return scaler.transform(imputer.transform(X)).astype(np.float32)

    X_val      = _transform(splits["X_val"])
    X_nonmember = _transform(splits["X_nonmember"])
    X_holdout  = _transform(splits["X_holdout"])

    # --- Binarização do target ---
    def _binarize(y):
        if binary_target:
            return (y > 0).astype(np.int64)
        return y.astype(np.int64)

    result = {
        "X_fit":        X_fit,
        "y_fit":        _binarize(splits["y_fit"]),
        "idx_fit":      splits["idx_fit"],
        "X_val":        X_val,
        "y_val":        _binarize(splits["y_val"]),
        "idx_val":      splits["idx_val"],
        "X_nonmember":  X_nonmember,
        "y_nonmember":  _binarize(splits["y_nonmember"]),
        "idx_nonmember": splits["idx_nonmember"],
        "X_holdout":    X_holdout,
        "y_holdout":    _binarize(splits["y_holdout"]),
        "idx_holdout":  splits["idx_holdout"],
        # Artefatos de preprocessamento (para reprodução)
        "scaler":  scaler,
        "imputer": imputer,
    }

    if binary_target:
        fit_pos = result["y_fit"].sum()
        print(
            f"[preprocess] Target binarizado. "
            f"Positivos em fit_members: {fit_pos}/{len(result['y_fit'])} "
            f"({100*fit_pos/len(result['y_fit']):.1f}%)"
        )

    return result


# ============================================================
# 4. DataLoaders PyTorch
# ============================================================

def make_dataloader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 256,
    shuffle: bool = True,
) -> DataLoader:
    """
    Converte arrays numpy em DataLoader PyTorch.

    ATENÇÃO: Para val_loader do modelo alvo, passar X_val/y_val (val_selection).
    NUNCA passar X_nonmember/y_nonmember como val_loader do modelo alvo.
    """
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.long)
    dataset = TensorDataset(X_t, y_t)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


# ============================================================
# 5. Splits do ataque (members vs non-members)
# ============================================================

def make_attack_splits(
    X_fit: np.ndarray,
    y_fit: np.ndarray,
    idx_fit: np.ndarray,
    X_nonmember: np.ndarray,
    y_nonmember: np.ndarray,
    idx_nonmember: np.ndarray,
    test_ratio: float = 0.30,
    seed: int = 42,
) -> Dict:
    """
    Cria attack_train e attack_test a partir de members (idx_fit)
    e non-members (idx_nonmember).

    IMPORTANTE: X_fit deve ser target_fit_members (NÃO target_val_selection).
    target_val_selection NÃO deve entrar nem como member nem como non-member
    no dataset do ataque. Ela é usada apenas para model selection (early stopping).

    Os dois conjuntos são balanceados (mesma quantidade de members e non-members)
    em attack_train e attack_test.

    Returns
    -------
    dict com:
      X_attack_train, y_attack_train
      X_attack_test,  y_attack_test
    """
    n_members    = len(X_fit)
    n_nonmembers = len(X_nonmember)
    n = min(n_members, n_nonmembers)  # balancear

    # Subsample do maior conjunto para balancear
    rng = np.random.default_rng(seed)
    member_idx = rng.choice(n_members,    size=n, replace=False)
    nm_idx     = rng.choice(n_nonmembers, size=n, replace=False)

    X_m = X_fit[member_idx]
    y_m = np.ones(n, dtype=np.int64)   # label 1 = membro

    X_nm = X_nonmember[nm_idx]
    y_nm = np.zeros(n, dtype=np.int64)  # label 0 = não-membro

    # Concatenar e embaralhar
    X_all = np.concatenate([X_m, X_nm], axis=0)
    y_all = np.concatenate([y_m, y_nm], axis=0)

    shuffle_idx = rng.permutation(len(X_all))
    X_all = X_all[shuffle_idx]
    y_all = y_all[shuffle_idx]

    # Split attack_train / attack_test
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all,
        test_size=test_ratio,
        stratify=y_all,
        random_state=seed,
    )

    print(
        f"[attack_splits] attack_train: {len(X_train)} amostras "
        f"({y_train.sum()} membros, {(y_train==0).sum()} não-membros)\n"
        f"[attack_splits] attack_test:  {len(X_test)} amostras "
        f"({y_test.sum()} membros, {(y_test==0).sum()} não-membros)"
    )

    return {
        "X_attack_train": X_train,
        "y_attack_train": y_train,
        "X_attack_test":  X_test,
        "y_attack_test":  y_test,
    }


def make_attack_splits_from_features(
    X_attack: np.ndarray,
    y_attack: np.ndarray,
    test_ratio: float = 0.30,
    seed: int = 42,
) -> Dict:
    """
    Cria attack_train e attack_test a partir das features de ataque ja extraidas.

    IMPORTANTE: X_attack deve vir de um extrator de features de ataque. Cada
    linha representa o comportamento observavel do modelo alvo sobre uma
    amostra. O pipeline historico usa 6 features, mas experimentos de
    restricao de API podem usar menos ou mais features observaveis.

    Esta funcao substitui make_attack_splits(...) no notebook principal para
    garantir que o atacante use sinais do modelo, nao atributos tabulares crus.
    """
    assert X_attack.ndim == 2, "X_attack deve ser uma matriz 2D"
    assert X_attack.shape[1] >= 1, "X_attack deve conter ao menos 1 feature de ataque"
    assert set(np.unique(y_attack)).issubset({0, 1}), \
        "y_attack deve conter apenas labels 0=non-member e 1=member"

    rng = np.random.default_rng(seed)

    idx_member = np.where(y_attack == 1)[0]
    idx_nonmember = np.where(y_attack == 0)[0]
    n = min(len(idx_member), len(idx_nonmember))

    idx_member = rng.choice(idx_member, size=n, replace=False)
    idx_nonmember = rng.choice(idx_nonmember, size=n, replace=False)

    idx_all = np.concatenate([idx_member, idx_nonmember])
    rng.shuffle(idx_all)

    X_bal = X_attack[idx_all]
    y_bal = y_attack[idx_all]

    X_train, X_test, y_train, y_test = train_test_split(
        X_bal, y_bal,
        test_size=test_ratio,
        stratify=y_bal,
        random_state=seed,
    )

    print(
        f"[attack_splits/features] attack_train: {len(X_train)} amostras "
        f"({y_train.sum()} membros, {(y_train==0).sum()} nao-membros), "
        f"features={X_train.shape[1]}\n"
        f"[attack_splits/features] attack_test:  {len(X_test)} amostras "
        f"({y_test.sum()} membros, {(y_test==0).sum()} nao-membros), "
        f"features={X_test.shape[1]}"
    )

    return {
        "X_attack_train": X_train,
        "y_attack_train": y_train,
        "X_attack_test":  X_test,
        "y_attack_test":  y_test,
    }
