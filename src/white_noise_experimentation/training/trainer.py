"""Training loop implementation."""

from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .callbacks import CheckpointCallback, EarlyStoppingCallback


def train_autoencoder(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Dict[str, Any],
    device: torch.device,
    run_dir: Path,
) -> Dict[str, Any]:
    """
    Train an autoencoder model.

    Args:
        model: Autoencoder model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Configuration dict with keys: lr, weight_decay, epochs, early_stopping_patience
        device: Device to train on ('cpu' or 'cuda')
        run_dir: Directory to save checkpoints and logs

    Returns:
        Dict with keys:
            - 'model': trained model
            - 'train_loss_history': list of training losses
            - 'val_loss_history': list of validation losses
            - 'best_val_loss': best validation loss achieved
    """
    model = model.to(device)

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["train"]["lr"],
        weight_decay=config["train"]["weight_decay"],
    )

    # Loss function
    criterion = nn.MSELoss()

    # Check if this model has sigma regularisation (NoiseControlledSNN)
    has_sigma_reg = hasattr(model, "sigma_regularisation")

    # Callbacks
    run_dir = Path(run_dir)
    early_stopping = EarlyStoppingCallback(
        patience=config["train"]["early_stopping_patience"]
    )
    checkpoint = CheckpointCallback(run_dir / "checkpoints")

    # Training history
    train_loss_history = []
    val_loss_history = []

    epochs = config["train"]["epochs"]

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")

        for x, _ in train_pbar:
            x = x.to(device)
            optimizer.zero_grad()
            x_recon = model(x)
            loss = criterion(x_recon, x)
            # Add sigma regularisation for NoiseControlledSNN
            if has_sigma_reg:
                loss = loss + model.sigma_regularisation()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            train_pbar.set_postfix({"loss": loss.item()})

        avg_train_loss = train_loss / len(train_loader)
        train_loss_history.append(avg_train_loss)

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
            for x, _ in val_pbar:
                x = x.to(device)
                x_recon = model(x)
                loss = criterion(x_recon, x)
                val_loss += loss.item()
                val_pbar.set_postfix({"loss": loss.item()})

        avg_val_loss = val_loss / len(val_loader)
        val_loss_history.append(avg_val_loss)

        print(
            f"Epoch {epoch+1}/{epochs} - "
            f"Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}"
        )

        # Checkpoint
        checkpoint.step(model, avg_val_loss)

        # Early stopping
        if early_stopping.step(avg_val_loss):
            print(f"Early stopping at epoch {epoch+1}")
            break

    # Load best model
    checkpoint.load_best(model)

    return {
        "model": model,
        "train_loss_history": train_loss_history,
        "val_loss_history": val_loss_history,
        "best_val_loss": min(val_loss_history),
    }
