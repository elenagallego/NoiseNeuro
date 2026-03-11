"""Noise-Controlled SNN autoencoder – Phase 2 novelty model.

Instead of a fixed noise level σ across all layers, a small neural
network (the *NoiseController*) observes channel-level statistics of
the input EEG window and **predicts a per-layer σ**.  This allows the
model to adaptively increase noise on noisy inputs and reduce it on
clean ones.

Architecture::

    Input EEG ──┬── NoiseController ──→ σ_1, σ_2 (per encoder layer)
                │
                └── Poisson rate coding → LIF enc (σ_l per layer) → latent
                     → LIF dec → reconstruction

The controller adds a small regularisation term to prevent σ from
exploding::

    total_loss = MSE_recon + λ * mean(σ²)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .snn_autoencoder import FixedNoiseSNN, poisson_rate_coding


class NoiseController(nn.Module):
    """Predicts per-layer noise levels from input EEG statistics.

    Input features are the per-channel mean and variance of the EEG
    window (2 × n_channels).  Two hidden layers map these to
    ``n_layers`` positive σ values via softplus.
    """

    def __init__(self, n_channels: int, n_layers: int = 2):
        super().__init__()
        self.fc1 = nn.Linear(n_channels * 2, 32)
        self.fc2 = nn.Linear(32, 16)
        self.sigma_out = nn.Linear(16, n_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: ``(batch, n_channels, window_size)``

        Returns:
            sigmas: ``(batch, n_layers)`` with values > 0.
        """
        x_mean = x.mean(dim=-1)   # (B, C)
        x_var = x.var(dim=-1)     # (B, C)
        stats = torch.cat([x_mean, x_var], dim=-1)  # (B, 2C)

        h = F.relu(self.fc1(stats))
        h = F.relu(self.fc2(h))
        sigmas = F.softplus(self.sigma_out(h))  # > 0
        return sigmas


class NoiseControlledSNN(FixedNoiseSNN):
    """SNN autoencoder with a learned noise controller.

    Extends :class:`FixedNoiseSNN` by replacing the fixed ``noise_sigma``
    with per-layer σ values predicted by a :class:`NoiseController`.

    A regularisation term ``sigma_reg_lambda * mean(σ²)`` should be
    added to the loss externally (the :meth:`forward` method stores the
    predicted sigmas in ``self.last_sigmas`` for this purpose).

    Args:
        sigma_reg_lambda: Regularisation weight for the predicted sigmas.
            Stored as an attribute for the training loop to use.
        See :class:`FixedNoiseSNN` for remaining arguments.
    """

    def __init__(
        self,
        n_channels: int = 22,
        window_size: int = 256,
        latent_dim: int = 64,
        hidden_dims: list = None,
        noise_sigma: float = 0.1,   # initial / fallback (not really used)
        tau: float = 20.0,
        vth: float = 1.0,
        sigma_reg_lambda: float = 0.01,
    ):
        super().__init__(
            n_channels=n_channels,
            window_size=window_size,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            noise_sigma=noise_sigma,
            tau=tau,
            vth=vth,
        )
        n_enc_layers = len(self.hidden_dims)
        self.noise_controller = NoiseController(n_channels, n_layers=n_enc_layers)
        self.sigma_reg_lambda = sigma_reg_lambda
        self.last_sigmas: torch.Tensor = None  # set during forward

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T = x.shape

        # --- Predict per-layer noise ---
        if self.training:
            sigmas = self.noise_controller(x)  # (B, n_enc_layers)
            self.last_sigmas = sigmas
        else:
            sigmas = None
            self.last_sigmas = None

        # --- Rate coding ---
        spikes_in = poisson_rate_coding(x)
        h = spikes_in

        # --- Encoder with learned per-layer noise ---
        for layer_idx, (conv, lif) in enumerate(zip(self.enc_convs, self.enc_lifs)):
            h = conv(h)
            spike_out, _ = lif(h)
            if self.training and sigmas is not None:
                # Per-sample, per-layer sigma
                sigma_l = sigmas[:, layer_idx].view(B, 1, 1)
                h = spike_out + sigma_l * torch.randn_like(spike_out)
            else:
                h = spike_out

        # --- Bottleneck ---
        h_flat = h.view(B, -1)
        z = self.fc_encode(h_flat)
        h_dec = self.fc_decode(z)
        h = h_dec.view(B, self.hidden_dims[-1], self._enc_spatial)

        # --- Decoder (no learned noise – only encoder is controlled) ---
        for i, conv in enumerate(self.dec_convs):
            h = conv(h)
            if i < len(self.dec_lifs):
                spike_out, _ = self.dec_lifs[i](h)
                h = spike_out

        if h.shape[-1] != T:
            h = F.pad(h, (0, T - h.shape[-1])) if h.shape[-1] < T else h[:, :, :T]

        return h

    def sigma_regularisation(self) -> torch.Tensor:
        """Return the σ² regularisation term.

        Should be called *after* :meth:`forward` during training and
        added to the reconstruction loss::

            total_loss = mse_loss + model.sigma_regularisation()
        """
        if self.last_sigmas is not None:
            return self.sigma_reg_lambda * (self.last_sigmas ** 2).mean()
        return torch.tensor(0.0)
