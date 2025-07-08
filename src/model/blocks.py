import torch
import torch.nn as nn

class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel-wise attention."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.SiLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ResidualSEBlock(nn.Module):
    """A residual block with an integrated Squeeze-and-Excitation block."""
    def __init__(self, features: int):
        super().__init__()
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, padding=1)
        self.norm1 = nn.InstanceNorm2d(features)
        self.actv1 = nn.SiLU(inplace=True)
        
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, padding=1)
        self.norm2 = nn.InstanceNorm2d(features)
        self.actv2 = nn.SiLU(inplace=True)
        
        self.se = SEBlock(features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.actv1(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out = self.se(out)
        out = torch.add(out, residual)
        return self.actv2(out)

class AdaIN(nn.Module):
    """Adaptive Instance Normalization layer."""
    def __init__(self, style_dim: int, features: int):
        super().__init__()
        self.norm = nn.InstanceNorm2d(features, affine=False)
        self.fc = nn.Linear(style_dim, features * 2)

    def forward(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1, 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1 + gamma) * self.norm(x) + beta

class AdaINResidualBlock(nn.Module):
    """A residual block that uses AdaIN for style injection and includes SE attention."""
    def __init__(self, latent_dim: int, features: int):
        super().__init__()
        self.norm1 = AdaIN(latent_dim, features)
        self.actv1 = nn.SiLU(inplace=True)
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, padding=1)

        self.norm2 = AdaIN(latent_dim, features)
        self.actv2 = nn.SiLU(inplace=True)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, padding=1)
        
        self.se = SEBlock(features)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.actv1(self.norm1(x, z))
        out = self.conv1(out)
        out = self.actv2(self.norm2(out, z))
        out = self.conv2(out)
        out = self.se(out)
        out = torch.add(out, residual)
        return out

class PixelShuffleBlock(nn.Module):
    """Upsampling block using PixelShuffle."""
    def __init__(self, in_channels: int, out_channels: int, scale_factor: int = 2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels * (scale_factor ** 2), kernel_size=3, padding=1)
        self.shuffle = nn.PixelShuffle(scale_factor)
        self.actv = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.actv(self.shuffle(self.conv(x)))

class DownsampleBlock(nn.Module):
    """Downsampling block using a strided convolution."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 2, padding: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.actv = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.actv(self.conv(x))
