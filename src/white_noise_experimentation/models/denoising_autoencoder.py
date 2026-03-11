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
        noise_location: str = "input",
    ):
        """
        Initialize denoising autoencoder.

        Args:
            n_channels: Number of input channels
            window_size: Length of input signal
            latent_dim: Dimension of latent space
            hidden_dims: List of hidden dimensions for conv filters
            sigma: Standard deviation of Gaussian noise
            noise_location: Where to inject noise - "input" or "latent"
        """
        super().__init__(
            n_channels=n_channels,
            window_size=window_size,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
        )
        self.sigma = sigma
        self.noise_location = noise_location  # 'input' or 'latent'
        self.add_noise = True  # Can be toggled at evaluation time

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with optional noise injection.

        Args:
            x: (batch_size, n_channels, window_size)

        Returns:
            reconstruction: (batch_size, n_channels, window_size)
        """
        if self.training and self.add_noise:
            if self.noise_location == "input":
                # Add noise to raw input
                x_noisy = x + torch.randn_like(x) * self.sigma
                z = self.encode(x_noisy)
            else:  # latent
                # Encode clean input, then add noise to latent
                z = self.encode(x)
                z = z + torch.randn_like(z) * self.sigma
        else:
            z = self.encode(x)

        # Decode
        x_recon = self.decode(z)
        return x_recon

    def set_noise_enabled(self, enabled: bool):
        """
        Enable or disable noise injection.

        Args:
            enabled: Whether to add noise
        """
        self.add_noise = enabled
