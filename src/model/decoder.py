import torch
import torch.nn as nn
from .blocks import AdaINResidualBlock, PixelShuffleBlock

class Decoder(nn.Module):
    """
    The decoder network that reconstructs the image with multi-scale style injection.
    """
    def __init__(self, latent_dim: int, initial_channels: int = 256):
        super().__init__()
        self.from_latent = nn.Linear(latent_dim, initial_channels * 8 * 8)
        self.initial_channels = initial_channels

        # Upsampling and style injection stages
        self.up1 = PixelShuffleBlock(initial_channels, 128)
        self.style1 = AdaINResidualBlock(latent_dim, 128)

        self.up2 = PixelShuffleBlock(128, 64)
        self.style2 = AdaINResidualBlock(latent_dim, 64)

        self.up3 = PixelShuffleBlock(64, 64)
        self.style3 = AdaINResidualBlock(latent_dim, 64)
        
        # Final output convolution
        self.conv_out = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, z_hr: torch.Tensor, z_lr: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_hr (torch.Tensor): The content latent vector.
            z_lr (torch.Tensor): The style latent vector.
        Returns:
            torch.Tensor: The reconstructed image.
        """
        # Initial projection and reshape
        x = self.from_latent(z_hr)
        x = x.view(x.size(0), self.initial_channels, 8, 8)

        # Multi-scale style application
        x = self.up1(x)
        x = self.style1(x, z_lr)

        x = self.up2(x)
        x = self.style2(x, z_lr)

        x = self.up3(x)
        x = self.style3(x, z_lr)

        return self.conv_out(x)
