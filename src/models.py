"""
src/models.py
MLP configurável para todos os cenários experimentais.
input_dim é sempre inferido dinamicamente de X_fit.shape[1].
"""

import torch
import torch.nn as nn
from typing import List


class MLPClassifier(nn.Module):
    """
    MLP simples com camadas lineares, ReLU e dropout opcional.

    Parâmetros
    ----------
    input_dim     : número de features de entrada (use X_fit.shape[1])
    hidden_layers : lista de inteiros com tamanho de cada camada oculta
    dropout       : taxa de dropout (0.0 = sem dropout)
    output_dim    : número de classes (2 para classificação binária)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_layers: List[int],
        dropout: float = 0.0,
        output_dim: int = 2,
    ):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

        # Armazenar config para referência
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout
        self.output_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: FloatTensor de shape (batch, input_dim)
        retorna: logits de shape (batch, output_dim)
        """
        return self.network(x)

    def __repr__(self):
        return (
            f"MLPClassifier(input_dim={self.input_dim}, "
            f"hidden_layers={self.hidden_layers}, "
            f"dropout={self.dropout_rate}, "
            f"output_dim={self.output_dim})"
        )


def build_model(config: dict, input_dim: int) -> MLPClassifier:
    """
    Instancia MLPClassifier a partir de um dicionário de configuração.

    Parameters
    ----------
    config    : dicionário de configuração do experimento (de src/config.py)
    input_dim : X_fit.shape[1] — nunca hardcoded

    Returns
    -------
    modelo MLPClassifier inicializado (não treinado)
    """
    model = MLPClassifier(
        input_dim=input_dim,
        hidden_layers=config["hidden_layers"],
        dropout=config.get("dropout", 0.0),
        output_dim=config.get("output_dim", 2),
    )
    print(f"[model] Criado: {model}")
    return model
