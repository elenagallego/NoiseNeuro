"""Spiking Neural Network autoencoder with LIF neurons.

Implements three SNN-based autoencoder variants:

1. **FixedNoiseSNN** – SNN equivalent of the Phase 1 denoising AE. Fixed
   Gaussian noise (σ = 0.1) is injected into spike rates during training.
2. **LatentNoiseSNN** – Noise is injected only into the latent spike
   representation (bottleneck), leaving encoder spikes clean.

All models use Leaky Integrate-and-Fire (LIF) neurons with surrogate
gradients for backpropagation through time (BPTT).  No external SNN
library is required – everything is pure PyTorch.

Architecture overview::

    Input EEG → Poisson rate coding → LIF encoder layers → latent spikes
              → LIF decoder layers → reconstructed spike rates → MSE loss
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Surrogate gradient for spiking
# ---------------------------------------------------------------------------

class _SurrogateSpike(torch.autograd.Function):
    """Heaviside step with fast-sigmoid surrogate gradient."""

    scale = 25.0

    @staticmethod
    def forward(ctx, membrane):
        ctx.save_for_backward(membrane)
        return (membrane > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        (membrane,) = ctx.saved_tensors
        grad = grad_output / (_SurrogateSpike.scale * membrane.abs() + 1.0) ** 2
        return grad


_spike_fn = _SurrogateSpike.apply


# ---------------------------------------------------------------------------
# LIF Neuron
# ---------------------------------------------------------------------------

class LIFNeuron(nn.Module):
    """Leaky Integrate-and-Fire neuron with surrogate gradient.

    Parameters:
        tau: Membrane time constant (ms).  Larger = slower leak.
        vth: Firing threshold voltage.
        dt: Simulation timestep (ms).
    """

    def __init__(self, tau: float = 20.0, vth: float = 1.0, dt: float = 1.0):
        super().__init__()
        self.tau = tau
        self.vth = vth
        self.dt = dt
        # Decay factor (learnable through tau if desired)
        self.decay = math.exp(-dt / tau)

    def forward(self, current: torch.Tensor, v_prev: torch.Tensor = None):
        """Single timestep update.

        Args:
            current: Input current ``(batch, features)``.
            v_prev: Previous membrane voltage (same shape).  Defaults to 0.

        Returns:
            spike: Binary spike tensor ``(batch, features)``.
            v_new: Updated membrane voltage after reset.
        """
        if v_prev is None:
            v_prev = torch.zeros_like(current)

        # Leak + input integration
        v = v_prev * self.decay + self.dt * current

        # Spike generation (surrogate gradient)
        spike = _spike_fn(v - self.vth)

        # Soft reset: subtract threshold on spike
        v_new = v - spike * self.vth

        return spike, v_new


# ---------------------------------------------------------------------------
# Spike-aware Conv layers (thin wrappers for clarity)
# ---------------------------------------------------------------------------

class SpikeConv1D(nn.Conv1d):
    """Conv1d operating on spike trains.  Identical to nn.Conv1d."""
    pass


class SpikeConvTranspose1D(nn.ConvTranspose1d):
    """ConvTranspose1d operating on spike trains."""
    pass


# ---------------------------------------------------------------------------
# Poisson rate coding
# ---------------------------------------------------------------------------

def poisson_rate_coding(x: torch.Tensor, dt: float = 1.0, max_rate: float = 100.0):
    """Convert real-valued signal to Poisson spike train.

    Each sample in ``x`` is mapped to a firing probability, then a Bernoulli
    draw determines whether a spike occurs.  During evaluation the expected
    rate (i.e. the probability itself) is returned so outputs are smooth.

    Args:
        x: Input tensor of arbitrary shape.  Values are sigmoid-squashed
           to [0, 1] to obtain firing probabilities.
        dt: Timestep in ms (used to scale rate).
        max_rate: Maximum firing rate in Hz (normalisation factor).

    Returns:
        Spike tensor of same shape as ``x``.
    """
    rate = torch.sigmoid(x)  # [0, 1] firing probability
    if not torch.is_grad_enabled():
        return rate  # smooth output at eval
    # Training: stochastic Bernoulli spikes with straight-through gradient
    spikes = (torch.rand_like(rate) < rate).float()
    # Straight-through estimator: grad flows through as if spikes == rate
    return spikes + (rate - rate.detach())


# ---------------------------------------------------------------------------
# Fixed-Noise SNN Autoencoder
# ---------------------------------------------------------------------------

class FixedNoiseSNN(nn.Module):
    """SNN autoencoder with fixed Gaussian noise injection.

    Architecture mirrors the ANN Conv autoencoder from Phase 1 but replaces
    ReLU activations with LIF neurons and processes input through Poisson
    rate coding.  Fixed noise σ is added to spike rates during training.

    Args:
        n_channels: Number of input EEG channels.
        window_size: Temporal length of each window.
        latent_dim: Dimension of the latent bottleneck.
        hidden_dims: Channel counts for encoder conv layers.
        noise_sigma: Standard deviation of training noise (0 = no noise).
        tau: LIF membrane time constant.
        vth: LIF firing threshold.
    """

    def __init__(
        self,
        n_channels: int = 22,
        window_size: int = 256,
        latent_dim: int = 64,
        hidden_dims: list = None,
        noise_sigma: float = 0.1,
        tau: float = 20.0,
        vth: float = 1.0,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [128, 64]

        self.n_channels = n_channels
        self.window_size = window_size
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.noise_sigma = noise_sigma

        # ---- Encoder convolutions + LIF ----
        self.enc_convs = nn.ModuleList()
        self.enc_lifs = nn.ModuleList()
        in_ch = n_channels
        for hd in hidden_dims:
            self.enc_convs.append(SpikeConv1D(in_ch, hd, kernel_size=3, stride=2, padding=1))
            self.enc_lifs.append(LIFNeuron(tau=tau, vth=vth))
            in_ch = hd

        # Compute spatial size after conv encoder
        size = window_size
        for _ in hidden_dims:
            size = (size + 2 * 1 - 3) // 2 + 1
        self._enc_spatial = size
        self._enc_flat = hidden_dims[-1] * size

        # Bottleneck
        self.fc_encode = nn.Linear(self._enc_flat, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, self._enc_flat)

        # ---- Decoder convolutions + LIF ----
        self.dec_convs = nn.ModuleList()
        self.dec_lifs = nn.ModuleList()
        hd_rev = list(reversed(hidden_dims))
        for i, hd in enumerate(hd_rev):
            out_ch = n_channels if i == len(hd_rev) - 1 else hd
            in_ch_d = hidden_dims[-1] if i == 0 else hd_rev[i - 1]
            self.dec_convs.append(
                SpikeConvTranspose1D(in_ch_d, out_ch, kernel_size=3, stride=2, padding=1, output_padding=1)
            )
            # No LIF on last decoder layer (output is continuous reconstruction)
            if i < len(hd_rev) - 1:
                self.dec_lifs.append(LIFNeuron(tau=tau, vth=vth))

    # ------------------------------------------------------------------ helpers
    def _reset_membrane(self, batch_size: int, device: torch.device):
        """Return zeroed membrane states for all LIF layers."""
        enc_v = [torch.zeros(batch_size, hd, 1, device=device) for hd in self.hidden_dims]
        dec_v = [torch.zeros(batch_size, hd, 1, device=device) for hd in reversed(self.hidden_dims[:-1])]
        return enc_v, dec_v

    def _add_noise(self, spikes: torch.Tensor, sigma: float):
        """Inject Gaussian noise into spike rates during training."""
        if self.training and sigma > 0:
            spikes = spikes + sigma * torch.randn_like(spikes)
        return spikes

    # ------------------------------------------------------------------ forward
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: ``(batch, n_channels, window_size)``

        Returns:
            Reconstruction ``(batch, n_channels, window_size)``
        """
        B, C, T = x.shape

        # --- Rate coding ---
        spikes_in = poisson_rate_coding(x)  # (B, C, T)

        # We process the full sequence at once through the conv layers
        # (treating the time dimension as Conv1d's spatial dim).
        h = spikes_in

        # --- Encoder ---
        for conv, lif in zip(self.enc_convs, self.enc_lifs):
            h = conv(h)  # (B, hd, T')
            # Apply LIF per-timestep across the spatial dim
            spike_out, _ = lif(h)
            h = self._add_noise(spike_out, self.noise_sigma)

        # --- Bottleneck ---
        h_flat = h.view(B, -1)
        z = self.fc_encode(h_flat)
        h_dec = self.fc_decode(z)
        h = h_dec.view(B, self.hidden_dims[-1], self._enc_spatial)

        # --- Decoder ---
        for i, conv in enumerate(self.dec_convs):
            h = conv(h)
            if i < len(self.dec_lifs):
                spike_out, _ = self.dec_lifs[i](h)
                h = self._add_noise(spike_out, self.noise_sigma)

        # Trim/pad to match original window_size
        if h.shape[-1] != T:
            h = F.pad(h, (0, T - h.shape[-1])) if h.shape[-1] < T else h[:, :, :T]

        return h


# ---------------------------------------------------------------------------
# Latent-Noise SNN Autoencoder
# ---------------------------------------------------------------------------

class LatentNoiseSNN(FixedNoiseSNN):
    """SNN autoencoder with noise injected only in the latent space.

    Encoder layers are noise-free; Gaussian noise is added to the latent
    representation before decoding.  This tests whether noise in the
    bottleneck alone is sufficient for robustness.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T = x.shape

        spikes_in = poisson_rate_coding(x)
        h = spikes_in

        # Encoder (no noise)
        for conv, lif in zip(self.enc_convs, self.enc_lifs):
            h = conv(h)
            spike_out, _ = lif(h)
            h = spike_out  # no noise here

        # Bottleneck with noise
        h_flat = h.view(B, -1)
        z = self.fc_encode(h_flat)
        if self.training and self.noise_sigma > 0:
            z = z + self.noise_sigma * torch.randn_like(z)
        h_dec = self.fc_decode(z)
        h = h_dec.view(B, self.hidden_dims[-1], self._enc_spatial)

        # Decoder (no noise)
        for i, conv in enumerate(self.dec_convs):
            h = conv(h)
            if i < len(self.dec_lifs):
                spike_out, _ = self.dec_lifs[i](h)
                h = spike_out

        if h.shape[-1] != T:
            h = F.pad(h, (0, T - h.shape[-1])) if h.shape[-1] < T else h[:, :, :T]

        return h
