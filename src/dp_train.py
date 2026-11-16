"""
src/dp_train.py
Treinamento com Privacidade Diferencial usando Opacus (OPCIONAL).

Se Opacus não estiver instalado, todas as funções retornam None
com um aviso — sem lançar exceções para o pipeline principal.
"""

import warnings
from typing import Dict, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import copy

from src.models import MLPClassifier


def _opacus_available() -> bool:
    try:
        import opacus  # noqa: F401
        return True
    except ImportError:
        return False


def train_model_with_dp(
    model: MLPClassifier,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: dict,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> Optional[Dict]:
    """
    Treina o modelo com Privacidade Diferencial usando Opacus PrivacyEngine.

    Parameters
    ----------
    model        : MLPClassifier não treinado
    train_loader : DataLoader de target_fit_members
    val_loader   : DataLoader de target_val_selection (NÃO nonmembers)
    config       : dicionário com chaves:
                   epochs, learning_rate, noise_multiplier,
                   max_grad_norm, delta
    device       : torch.device (CPU/GPU)
    verbose      : exibir progresso

    Returns
    -------
    dict com "model", "epsilon" (float), "history", "actual_epochs"
    OU None se Opacus não estiver disponível
    """
    if not _opacus_available():
        warnings.warn(
            "[dp_train] Opacus não está instalado. Pulando experimento DP-SGD.\n"
            "Para instalar: pip install opacus\n"
            "O pipeline continua normalmente sem DP.",
            UserWarning,
            stacklevel=2,
        )
        return None

    from opacus import PrivacyEngine
    from opacus.utils.batch_memory_manager import BatchMemoryManager

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    
    optimizer_name = config.get("optimizer", "sgd")
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.get("learning_rate", 1e-3),
        )
    else:
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config.get("learning_rate", 1e-3),
        )

    noise_multiplier = config.get("noise_multiplier", 1.0)
    max_grad_norm    = config.get("max_grad_norm", 1.0)
    delta            = config.get("delta", 1e-5)
    epochs           = config.get("epochs", 20)

    use_class_weights = config.get("use_class_weights", False)

    if use_class_weights:
        y_train_all = train_loader.dataset.tensors[1]
        class_counts = torch.bincount(y_train_all.long(), minlength=config.get("output_dim", 2)).float()

        class_weights = class_counts.sum() / (len(class_counts) * class_counts.clamp_min(1))
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

        print(f"[dp_train] Usando class_weights: {class_weights.tolist()}")
    else:
        criterion = nn.CrossEntropyLoss()

    privacy_engine = PrivacyEngine(accountant="rdp")
    model, optimizer, train_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
    )

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss  = float("inf")
    best_model_wts = copy.deepcopy(model.state_dict())

    epoch_iter = range(1, epochs + 1)
    if verbose:
        epoch_iter = tqdm(epoch_iter, desc=f"DP-SGD [{config.get('name', 'dp')}]")

    for epoch in epoch_iter:
        # --- Treino com DP ---
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

        # --- Validação ---
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

        if avg_val_loss < best_val_loss:
            best_val_loss  = avg_val_loss
            best_model_wts = copy.deepcopy(model.state_dict())

        epsilon = privacy_engine.get_epsilon(delta=delta)
        if verbose:
            epoch_iter.set_postfix({
                "ε": f"{epsilon:.2f}",
                "val_acc": f"{avg_val_acc:.3f}",
            })

    restore_best_model = config.get("restore_best_model", config.get("early_stopping", False))

    if restore_best_model:
        model.load_state_dict(best_model_wts)
        checkpoint_msg = "restaurado melhor checkpoint de validação"
    else:
        checkpoint_msg = "mantidos pesos da última época"

    model.eval()

    epsilon_final = privacy_engine.get_epsilon(delta=delta)
    print(
        f"[dp_train] Concluído: {epochs} épocas | "
        f"ε={epsilon_final:.4f} (δ={delta}) | "
        f"val_acc={history['val_acc'][-1]:.4f} | "
        f"{checkpoint_msg}"
    )

    actual_epochs = len(history["train_loss"])

    return {
        "model":   model,
        "epsilon": epsilon_final,
        "delta":   delta,
        "history": history,
        "actual_epochs": actual_epochs,
    }
