"""
src/train.py
Treinamento do modelo alvo com suporte a early stopping, weight decay e tqdm.

IMPORTANTE: val_loader deve ser construído EXCLUSIVAMENTE de target_val_selection.
NUNCA passar target_test_nonmembers como val_loader.
"""

import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, Optional

from src.models import MLPClassifier


def train_model(
    model: MLPClassifier,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: dict,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> Dict:
    """
    Treina o modelo alvo com Adam, CrossEntropyLoss, suporte a early stopping.

    Parameters
    ----------
    model        : MLPClassifier não treinado
    train_loader : DataLoader de target_fit_members
    val_loader   : DataLoader de target_val_selection
                   !!! NUNCA usar target_test_nonmembers aqui !!!
    config       : dicionário de configuração (de src/config.py)
    device       : torch.device (inferido automaticamente se None)
    verbose      : exibir barra de progresso

    Returns
    -------
    dict com:
      "model"   : melhor modelo (checkpoint de menor val_loss)
      "history" : {"train_loss", "val_loss", "train_acc", "val_acc"} como listas
      "actual_epochs": numero real de epocas treinadas
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.get("learning_rate", 1e-3),
        weight_decay=config.get("weight_decay", 0.0),
    )

    epochs        = config.get("epochs", 30)
    early_stop    = config.get("early_stopping", False)
    patience      = config.get("patience", 7)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    best_val_loss  = float("inf")
    best_model_wts = copy.deepcopy(model.state_dict())
    patience_count = 0

    epoch_iter = range(1, epochs + 1)
    if verbose:
        epoch_iter = tqdm(epoch_iter, desc=f"Treinando [{config.get('name', 'model')}]")

    for epoch in epoch_iter:
        # --- Treino ---
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            train_loss    += loss.item() * len(y_batch)
            preds          = logits.argmax(dim=1)
            train_correct += (preds == y_batch).sum().item()
            train_total   += len(y_batch)

        avg_train_loss = train_loss / train_total
        avg_train_acc  = train_correct / train_total

        # --- Validação (target_val_selection) ---
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                logits = model(X_batch)
                loss = criterion(logits, y_batch)

                val_loss    += loss.item() * len(y_batch)
                preds        = logits.argmax(dim=1)
                val_correct += (preds == y_batch).sum().item()
                val_total   += len(y_batch)

        avg_val_loss = val_loss / val_total
        avg_val_acc  = val_correct / val_total

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["train_acc"].append(avg_train_acc)
        history["val_acc"].append(avg_val_acc)

        if verbose:
            epoch_iter.set_postfix({
                "tr_loss": f"{avg_train_loss:.4f}",
                "val_loss": f"{avg_val_loss:.4f}",
                "val_acc": f"{avg_val_acc:.3f}",
            })

        # --- Checkpoint do melhor modelo ---
        if avg_val_loss < best_val_loss:
            best_val_loss  = avg_val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            patience_count = 0
        else:
            patience_count += 1

        # --- Early stopping ---
        if early_stop and patience_count >= patience:
            if verbose:
                tqdm.write(
                    f"[early_stop] Parado na época {epoch} "
                    f"(melhor val_loss={best_val_loss:.4f})"
                )
            break
    actual_epochs = len(history["train_loss"])

    # Restaurar melhor checkpoint só quando fizer sentido.
    # Para cenários overfitted/vulnerable, queremos manter a última época
    # para permitir memorização.
    restore_best_model = config.get(
        "restore_best_model",
        config.get("early_stopping", False)
    )

    if restore_best_model:
        model.load_state_dict(best_model_wts)
        checkpoint_msg = "restaurado melhor checkpoint de validação"
    else:
        checkpoint_msg = "mantidos pesos da última época"

    model.eval()

    print(
        f"[train] Concluído: {actual_epochs} épocas | "
        f"best val_loss={best_val_loss:.4f} | "
        f"final val_acc={history['val_acc'][-1]:.4f} | "
        f"{checkpoint_msg}"
    )

    return {"model": model, "history": history, "actual_epochs": actual_epochs}
