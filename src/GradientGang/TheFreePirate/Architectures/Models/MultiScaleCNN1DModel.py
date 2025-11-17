import torch
import torch.nn as nn
from typing import List, Tuple


class InceptionBlock1D(nn.Module):
    """Inception block for multiscale feature extraction in 1D"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_sizes: Tuple[int, int, int] = (3, 5, 7),
    ):
        super().__init__()

        branch_channels = out_channels // 4

        # 1x1 conv branch
        self.branch1 = nn.Sequential(
            nn.Conv1d(in_channels, branch_channels, kernel_size=1),
            nn.BatchNorm1d(branch_channels),
            nn.ReLU(),
        )

        # 1x1 -> k1 conv branch
        k1, k2, k3 = kernel_sizes
        self.branch2 = nn.Sequential(
            nn.Conv1d(in_channels, branch_channels, kernel_size=1),
            nn.BatchNorm1d(branch_channels),
            nn.ReLU(),
            nn.Conv1d(
                branch_channels, branch_channels, kernel_size=k1, padding=k1 // 2
            ),
            nn.BatchNorm1d(branch_channels),
            nn.ReLU(),
        )

        # 1x1 -> k2 conv branch
        self.branch3 = nn.Sequential(
            nn.Conv1d(in_channels, branch_channels, kernel_size=1),
            nn.BatchNorm1d(branch_channels),
            nn.ReLU(),
            nn.Conv1d(
                branch_channels, branch_channels, kernel_size=k2, padding=k2 // 2
            ),
            nn.BatchNorm1d(branch_channels),
            nn.ReLU(),
        )

        # Max pooling -> 1x1 conv branch
        self.branch4 = nn.Sequential(
            nn.MaxPool1d(kernel_size=k3, stride=1, padding=k3 // 2),
            nn.Conv1d(in_channels, branch_channels, kernel_size=1),
            nn.BatchNorm1d(branch_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        return torch.cat([b1, b2, b3, b4], dim=1)


class InceptionTransposeBlock1D(nn.Module):
    """Inception block for decoder using transposed convolutions"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_sizes: Tuple[int, int, int] = (3, 5, 7),
    ):
        super().__init__()

        # Adjust branch_channels to ensure the correct number of output channels
        branch_channels = out_channels // 4
        remainder = out_channels % 4

        # 1x1 conv branch
        self.branch1 = nn.Sequential(
            nn.ConvTranspose1d(
                in_channels,
                branch_channels + (1 if remainder > 0 else 0),
                kernel_size=1,
            ),
            nn.BatchNorm1d(branch_channels + (1 if remainder > 0 else 0)),
            nn.ReLU(),
        )

        k1, k2, k3 = kernel_sizes
        # 1x1 -> k1 transposed conv branch
        self.branch2 = nn.Sequential(
            nn.ConvTranspose1d(
                in_channels,
                branch_channels + (1 if remainder > 1 else 0),
                kernel_size=1,
            ),
            nn.BatchNorm1d(branch_channels + (1 if remainder > 1 else 0)),
            nn.ReLU(),
            nn.ConvTranspose1d(
                branch_channels + (1 if remainder > 1 else 0),
                branch_channels + (1 if remainder > 1 else 0),
                kernel_size=k1,
                padding=k1 // 2,
            ),
            nn.BatchNorm1d(branch_channels + (1 if remainder > 1 else 0)),
            nn.ReLU(),
        )

        # 1x1 -> k2 transposed conv branch
        self.branch3 = nn.Sequential(
            nn.ConvTranspose1d(
                in_channels,
                branch_channels + (1 if remainder > 2 else 0),
                kernel_size=1,
            ),
            nn.BatchNorm1d(branch_channels + (1 if remainder > 2 else 0)),
            nn.ReLU(),
            nn.ConvTranspose1d(
                branch_channels + (1 if remainder > 2 else 0),
                branch_channels + (1 if remainder > 2 else 0),
                kernel_size=k2,
                padding=k2 // 2,
            ),
            nn.BatchNorm1d(branch_channels + (1 if remainder > 2 else 0)),
            nn.ReLU(),
        )

        # Upsample -> 1x1 conv branch
        self.branch4 = nn.Sequential(
            nn.Upsample(scale_factor=1, mode="linear", align_corners=False),
            nn.ConvTranspose1d(in_channels, branch_channels, kernel_size=1),
            nn.BatchNorm1d(branch_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        return torch.cat([b1, b2, b3, b4], dim=1)


class MultiscaleInceptionEncoder1D(nn.Module):
    """Encoder with Inception blocks and configurable architecture"""

    @staticmethod
    def calculateBottleneckLength(seq_length: int, num_layers: int) -> int:
        """Calculate the length after all pooling operations"""
        return seq_length // (2**num_layers)

    def __init__(
        self,
        in_channels: int,
        hidden_dims: List[int],
        latent_dim: int,
        seq_length: int,
        kernel_sizes: Tuple[int, int, int] = (3, 5, 7),
        use_pooling: bool = True,
    ):
        """
        Args:
            in_channels: Number of input channels
            hidden_dims: List of channel dimensions for each Inception block (e.g., [64, 128, 256])
            latent_dim: Dimension of the latent space
            seq_length: Length of input sequence
            kernel_sizes: Tuple of kernel sizes for Inception branches (small, medium, large)
            use_pooling: Whether to use max pooling after each block
        """
        super().__init__()

        self.in_channels = in_channels
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        self.seq_length = seq_length
        self.kernel_sizes = kernel_sizes
        self.use_pooling = use_pooling

        # Build encoder blocks
        self.blocks = []
        prev_channels = in_channels

        for i, hidden_dim in enumerate(hidden_dims):
            if use_pooling and i < len(hidden_dims):
                block = nn.Sequential(
                    InceptionBlock1D(prev_channels, hidden_dim, kernel_sizes),
                    nn.MaxPool1d(2),
                )
            else:
                block = InceptionBlock1D(prev_channels, hidden_dim, kernel_sizes)

            self.blocks.append(block)
            prev_channels = hidden_dim
        self.blocksModule = nn.Sequential(*self.blocks)

        # Calculate bottleneck dimensions
        num_pooling_layers = len(hidden_dims) if use_pooling else 0
        self.bottleneck_length = self.calculateBottleneckLength(
            seq_length, num_pooling_layers
        )
        self.bottleneck_channels = hidden_dims[-1]

        # Latent projection
        self.to_latent = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(self.bottleneck_channels, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq_length, in_channels)
        Returns:
            latent: Latent representation of shape (batch, latent_dim)
        """
        # Permute last two dimensions to match correct shape
        x = torch.permute(x, (0, 2, 1))  # (batch, in_channels, seq_length)
        x = self.blocksModule(x)

        latent = self.to_latent(x)
        return latent


class MultiscaleInceptionDecoder1D(nn.Module):
    """Decoder with Inception transpose blocks and configurable architecture"""

    def __init__(
        self,
        latent_dim: int,
        hidden_dims: List[int],
        out_channels: int,
        seq_length: int,
        kernel_sizes: Tuple[int, int, int] = (3, 5, 7),
        use_upsampling: bool = True,
    ):
        """
        Args:
            latent_dim: Dimension of the latent space
            hidden_dims: List of channel dimensions for each Inception block (reversed order, e.g., [256, 128, 64])
            out_channels: Number of output channels
            seq_length: Target length of output sequence
            kernel_sizes: Tuple of kernel sizes for Inception branches (small, medium, large)
            use_upsampling: Whether to use upsampling before each block
        """
        super().__init__()

        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.out_channels = out_channels
        self.seq_length = seq_length
        self.kernel_sizes = kernel_sizes
        self.use_upsampling = use_upsampling

        # Calculate initial bottleneck dimensions
        num_upsampling_layers = len(hidden_dims) if use_upsampling else 0
        self.bottleneck_length = seq_length // (2**num_upsampling_layers)
        self.bottleneck_channels = hidden_dims[0]

        # Latent to bottleneck
        self.from_latent = nn.Sequential(
            nn.Linear(latent_dim, self.bottleneck_channels * self.bottleneck_length),
            nn.ReLU(),
        )

        # Build decoder blocks
        self.blocks = []
        prev_channels = self.bottleneck_channels

        for i, hidden_dim in enumerate(hidden_dims[1:] + [out_channels]):
            if use_upsampling and i < len(hidden_dims):
                block = nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="linear", align_corners=False),
                    InceptionTransposeBlock1D(prev_channels, hidden_dim, kernel_sizes),
                )
            else:
                block = InceptionTransposeBlock1D(
                    prev_channels, hidden_dim, kernel_sizes
                )

            self.blocks.append(block)
            prev_channels = hidden_dim

        self.blocksModule = nn.Sequential(*self.blocks)

        # Final reconstruction layer
        self.final_conv = nn.Conv1d(out_channels, out_channels, kernel_size=1)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Args:
            latent: Latent representation of shape (batch, latent_dim)
        Returns:
            reconstruction: Reconstructed tensor of shape (batch, seq_length, out_channels)
        """
        # Reshape from latent to bottleneck
        x = self.from_latent(latent)
        x = x.view(x.size(0), self.bottleneck_channels, self.bottleneck_length)

        # Pass through decoder blocks
        x = self.blocksModule(x)

        # Final convolution
        x = self.final_conv(x)

        # Permute back to (batch, seq_length, out_channels)
        x = torch.permute(x, (0, 2, 1))
        return x


class MultiscaleInceptionAutoencoder1D(nn.Module):
    """Complete autoencoder combining encoder and decoder"""

    @staticmethod
    def linearlyInterpolateLayers(
        num_layers: int,
        base_channels: int = 64,
    ) -> List[int]:
        """
        Create linearly interpolated hidden dimensions.

        Args:
            in_channels: Input channels (not used, kept for API compatibility)
            out_channels: Output channels (not used, kept for API compatibility)
            num_layers: Number of layers
            base_channels: Starting channel dimension

        Returns:
            List of hidden dimensions (e.g., [64, 128, 256, 512])
        """
        hidden_dims = []
        for i in range(num_layers):
            dim = base_channels * (2**i)
            hidden_dims.append(dim)
        return hidden_dims

    def __init__(
        self,
        in_channels: int,
        latent_dim: int,
        seq_length: int,
        encoder_hidden_dims: List[int] | None = None,
        decoder_hidden_dims: List[int] | None = None,
        num_layers: int = 4,
        base_channels: int = 64,
        kernel_sizes: Tuple[int, int, int] = (3, 5, 7),
        use_pooling: bool = True,
    ):
        """
        Args:
            in_channels: Number of input channels
            latent_dim: Dimension of latent space
            seq_length: Length of input sequence
            encoder_hidden_dims: List of channel dims for encoder (if None, auto-generated)
            decoder_hidden_dims: List of channel dims for decoder (if None, reversed encoder dims)
            num_layers: Number of layers (used if hidden_dims not provided)
            base_channels: Base channel dimension for auto-generation
            kernel_sizes: Tuple of kernel sizes for Inception branches
            use_pooling: Whether to use pooling/upsampling
        """
        super().__init__()

        # Auto-generate hidden dimensions if not provided
        if encoder_hidden_dims is None:
            encoder_hidden_dims = self.linearlyInterpolateLayers(
                num_layers, base_channels
            )

        if decoder_hidden_dims is None:
            decoder_hidden_dims = list(reversed(encoder_hidden_dims))

        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.seq_length = seq_length
        self.encoder_hidden_dims = encoder_hidden_dims
        self.decoder_hidden_dims = decoder_hidden_dims

        # Create encoder and decoder
        self.encoder = MultiscaleInceptionEncoder1D(
            in_channels=in_channels,
            hidden_dims=encoder_hidden_dims,
            latent_dim=latent_dim,
            seq_length=seq_length,
            kernel_sizes=kernel_sizes,
            use_pooling=use_pooling,
        )

        self.decoder = MultiscaleInceptionDecoder1D(
            latent_dim=latent_dim,
            hidden_dims=decoder_hidden_dims,
            out_channels=in_channels,
            seq_length=seq_length,
            kernel_sizes=kernel_sizes,
            use_upsampling=use_pooling,
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent space"""
        return self.encoder(x)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to reconstruction"""
        return self.decoder(latent)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through autoencoder

        Args:
            x: Input tensor of shape (batch, in_channels, seq_length)

        Returns:
            reconstruction: Reconstructed tensor
            latent: Latent representation
        """
        latent = self.encode(x)
        reconstruction = self.decode(latent)
        return reconstruction, latent


# Example usage for hyperparameter optimization
if __name__ == "__main__":
    # Simple usage with auto-generated dimensions
    model = MultiscaleInceptionAutoencoder1D(
        in_channels=41,
        latent_dim=128,
        seq_length=10,
        num_layers=1,  # Will create [64, 128, 256]
        base_channels=64,
        kernel_sizes=(3, 5, 7),
        use_pooling=True,
    )

    # Test
    x = torch.randn(1984, 1000, 41)
    recon, latent = model(x)
    print(f"Input: {x.shape}, Latent: {latent.shape}, Reconstruction: {recon.shape}")

    # Hyperparameters to optimize:
    # - latent_dim: [64, 128, 256, 512]
    # - num_layers: [2, 3, 4, 5]
    # - base_channels: [32, 64, 128]
    # - kernel_sizes: [(3,5,7), (5,7,9), (3,7,11)]
