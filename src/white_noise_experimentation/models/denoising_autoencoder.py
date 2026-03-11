"""Denoising Autoencoder model."""

import torch
import torch.nn as nn

from .ann_autoencoder import ANNAutoencoder


class DenoisingAutoencoder(ANNAutoencoder):
    """
    Denoising autoencoder that extends ANNAutoencoder.

    Supports two noise injection modes controlled by ``noise_where``:

    * ``"input"`` (classic denoising AE) – Gaussian noise is added to the
      input before encoding.  The model learns to reconstruct the *clean*
      input.
    * ``"latent"`` – noise is added to the latent representation after
      encoding.  The decoder must reconstruct from a noisy bottleneck.

    Noise is only injected during training (or when ``add_noise`` is True).
    """

    def __init__(
        self,
        n_channels: int = 64,
        window_size: int = 256,
        latent_dim: int = 64,
        hidden_dims: list = None,
        sigma: float = 0.1,
        noise_where: str = "input",
    ):
        """
        Initialize denoising autoencoder.

        Args:
            n_channels: Number of input channels
            window_size: Length of input signal
            latent_dim: Dimension of latent space
            hidden_dims: List of hidden dimensions for conv filters
            sigma: Standard deviation of Gaussian noise
            noise_where: Where to inject noise – ``"input"`` or ``"latent"``
        """
        super().__init__(
            n_channels=n_channels,
            window_size=window_size,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
        )
        if noise_where not in ("input", "latent"):
            raise ValueError(f"noise_where must be 'input' or 'latent', got '{noise_where}'")
        self.sigma = sigma
        self.noise_where = noise_where
        self.add_noise = True  # Can be toggled at evaluation time

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with optional noise injection.

        Args:
            x: (batch_size, n_channels, window_size)

        Returns:
            reconstruction: (batch_size, n_channels, window_size)
        """
        inject = self.training and self.add_noise

        # --- Input noise ---
        if inject and self.noise_where == "input":
            x_enc = x + self.sigma * torch.randn_like(x)
        else:
            x_enc = x

        # --- Encode ---
        z = self.encode(x_enc)

        # --- Latent noise ---
        if inject and self.noise_where == "latent":
            z = z + self.sigma * torch.randn_like(z)

        # --- Decode ---
        x_recon = self.decode(z)
        return x_recon

    def set_noise_enabled(self, enabled: bool):
        """
        Enable or disable noise injection.

        Args:
            enabled: Whether to add noise
        """
        self.add_noise = enabled
