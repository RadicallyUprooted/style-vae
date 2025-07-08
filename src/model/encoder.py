from typing import Tuple
import torch
import torch.nn as nn
from .blocks import ResidualSEBlock, DownsampleBlock

class BaseEncoder(nn.Module):
    """A base encoder architecture using residual SE blocks and downsampling."""
    def __init__(self, latent_dim: int, output_channels: int = 256):
        super().__init__()
        self.output_channels = output_channels
        self.conv_in = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.actv_in = nn.SiLU(inplace=True)

        self.residual = nn.Sequential(*[ResidualSEBlock(64) for _ in range(4)])

        self.downsample = nn.Sequential(
            DownsampleBlock(64, 128),
            DownsampleBlock(128, 256),
            DownsampleBlock(256, self.output_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.actv_in(self.conv_in(x))
        x = self.residual(x)
        x = self.downsample(x)
        x = x.view(x.size(0), -1)
        return x

class EncoderLR(BaseEncoder):
    """Encoder for the low-resolution image, producing a stochastic latent z."""
    def __init__(self, latent_dim: int):
        super().__init__(latent_dim)
        self.to_latent = nn.Linear(self.output_channels * 8 * 8, 2 * latent_dim)

    def reparametrize(self, param: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Applies the reparametrization trick."""
        mean, log_var = torch.chunk(param, 2, dim=1)
        std = torch.exp(0.5 * log_var)
        z = mean + std * torch.randn_like(std)
        return z, mean, log_var

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = super().forward(x)
        embedding = self.to_latent(x)
        z, mean, log_var = self.reparametrize(embedding)
        return z, mean, log_var

class EncoderHR(BaseEncoder):
    """Encoder for the high-resolution image, producing a deterministic content vector."""
    def __init__(self, latent_dim: int):
        super().__init__(latent_dim)
        self.to_latent = nn.Linear(self.output_channels * 8 * 8, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = super().forward(x)
        z_hr = self.to_latent(x)
        return z_hr
