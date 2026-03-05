"""ANN Autoencoder model."""

import torch
import torch.nn as nn


class ANNAutoencoder(nn.Module):
    """
    Simple ANN autoencoder using 1D convolutions.

    Architecture:
        Encoder: Conv1d -> ReLU -> Conv1d -> ReLU -> Flatten -> Linear
        Decoder: Linear -> Reshape -> ConvTranspose1d -> ReLU -> ConvTranspose1d
    """

    def __init__(
        self,
        n_channels: int = 64,
        window_size: int = 256,
        latent_dim: int = 64,
        hidden_dims: list = None,
    ):
        """
        Initialize autoencoder.

        Args:
            n_channels: Number of input channels (e.g., EEG channels)
            window_size: Length of input signal
            latent_dim: Dimension of latent space
            hidden_dims: List of hidden dimensions for conv filters
        """
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [128, 64]

        self.n_channels = n_channels
        self.window_size = window_size
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims

        # Encoder
        encoder_layers = []
        in_channels = n_channels
        for hidden_dim in hidden_dims:
            encoder_layers.append(
                nn.Conv1d(
                    in_channels,
                    hidden_dim,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                )
            )
            encoder_layers.append(nn.ReLU())
            in_channels = hidden_dim

        self.encoder = nn.Sequential(*encoder_layers)

        # Calculate size after encoder convolutions
        # Each conv with stride=2 reduces size by ~2
        latent_size = window_size
        for _ in hidden_dims:
            latent_size = (latent_size + 2 * 1 - 3) // 2 + 1

        self.latent_shape = (hidden_dims[-1], latent_size)
        self.encoder_out_size = hidden_dims[-1] * latent_size

        # Bottleneck to latent
        self.fc_encode = nn.Linear(self.encoder_out_size, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, self.encoder_out_size)

        # Decoder (reverse of encoder)
        decoder_layers = []
        hidden_dims_reversed = list(reversed(hidden_dims))

        for i, hidden_dim in enumerate(hidden_dims_reversed):
            if i == 0:
                in_channels = hidden_dims[-1]
            else:
                in_channels = hidden_dims_reversed[i - 1]

            if i == len(hidden_dims_reversed) - 1:
                # Last layer: output to n_channels
                out_channels = n_channels
            else:
                out_channels = hidden_dim

            decoder_layers.append(
                nn.ConvTranspose1d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                )
            )
            if i < len(hidden_dims_reversed) - 1:
                decoder_layers.append(nn.ReLU())

        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input to latent space.

        Args:
            x: (batch_size, n_channels, window_size)

        Returns:
            z: (batch_size, latent_dim)
        """
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        z = self.fc_encode(h)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode from latent space to reconstruction.

        Args:
            z: (batch_size, latent_dim)

        Returns:
            reconstruction: (batch_size, n_channels, window_size)
        """
        h = self.fc_decode(z)
        h = h.view(-1, *self.latent_shape)
        x_recon = self.decoder(h)
        return x_recon

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: encode then decode.

        Args:
            x: (batch_size, n_channels, window_size)

        Returns:
            reconstruction: (batch_size, n_channels, window_size)
        """
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon
