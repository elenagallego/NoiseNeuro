"""Denoising Autoencoder model."""

import torch
import torch.nn as nn

from .ann_autoencoder import ANNAutoencoder


class DenoisingAutoencoder(ANNAutoencoder):
    """
    Denoising autoencoder that extends ANNAutoencoder.

    During training, injects Gaussian noise into inputs.
    At evaluation, noise injection can be turned off.
    """

    def __init__(
        self,
        n_channels: int = 64,
        window_size: int = 256,
        latent_dim: int = 64,
        hidden_dims: list = None,
        sigma: float = 0.1,
    ):
        """
        Initialize denoising autoencoder.

        Args:
            n_channels: Number of input channels
            window_size: Length of input signal
            latent_dim: Dimension of latent space
            hidden_dims: List of hidden dimensions for conv filters
            sigma: Standard deviation of Gaussian noise
        """
        super().__init__(
            n_channels=n_channels,
            window_size=window_size,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
        )
        self.sigma = sigma
        self.add_noise = True  # Can be toggled at evaluation time

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with optional noise injection.

        Args:
            x: (batch_size, n_channels, window_size)

        Returns:
            reconstruction: (batch_size, n_channels, window_size)
        """
        # Add noise during training if enabled
        if self.training and self.add_noise:
            x_noisy = x + torch.randn_like(x) * self.sigma
        else:
            x_noisy = x

        # Encode and decode
        z = self.encode(x_noisy)
        x_recon = self.decode(z)
        return x_recon

    def set_noise_enabled(self, enabled: bool):
        """
        Enable or disable noise injection.

        Args:
            enabled: Whether to add noise
        """
        self.add_noise = enabled
