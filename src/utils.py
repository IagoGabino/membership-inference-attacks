"""
src/utils.py
Funções utilitárias compartilhadas pelo pipeline.
"""

import os
import json
import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Fixa seeds de aleatoriedade para reprodutibilidade."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Garante determinismo nas operações do CUDA (pode ser mais lento)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Retorna GPU se disponível, senão CPU."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[device] Usando GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("[device] GPU não disponível, usando CPU.")
    return device


def save_json(data: dict, path: str) -> None:
    """Salva dicionário como JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    print(f"[save] JSON salvo em: {path}")


def load_json(path: str) -> dict:
    """Carrega JSON de arquivo."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dirs(*paths: str) -> None:
    """Cria diretórios se não existirem."""
    for path in paths:
        os.makedirs(path, exist_ok=True)
