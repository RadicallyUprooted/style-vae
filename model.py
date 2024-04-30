import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.transforms import v2

class AdaIN(nn.Module):

    def __init__(self, style_dim, features):
        super().__init__()
        self.norm = nn.InstanceNorm2d(features, affine=False)
        self.fc = nn.Linear(style_dim, features * 2)

    def forward(self, x, s):

        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1, 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1 + gamma) * self.norm(x) + beta

class ResidualBlock(nn.Module):

    def __init__(self, features):
        super(ResidualBlock, self).__init__()
        self.norm1 = nn.InstanceNorm2d(features)
        self.actv1 = nn.LeakyReLU(0.2)
        self.conv1 = nn.Conv2d(kernel_size=3, in_channels=features, out_channels=features, stride=1, padding=1)
        
        self.norm2 = nn.InstanceNorm2d(features)
        self.actv2 = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv2d(kernel_size=3, in_channels=features, out_channels=features, stride=1, padding=1)

    def forward(self, x):
        residual = x
        out = self.actv1(self.norm1(x))
        out = self.conv1(out)
        out = self.actv2(self.norm2(out))
        out = self.conv2(out)
        out = torch.add(out, residual)

        return out

class AdaINResidualBlock(nn.Module):

    def __init__(self, latent_dim, features):
        super(AdaINResidualBlock, self).__init__()
        self.norm1 = AdaIN(latent_dim, features)
        self.actv1 = nn.LeakyReLU(0.2)
        self.conv1 = nn.Conv2d(kernel_size=3, in_channels=features, out_channels=features, stride=1, padding=1)

        self.norm2 = AdaIN(latent_dim, features)
        self.actv2 = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv2d(kernel_size=3, in_channels=features, out_channels=features, stride=1, padding=1)

    def forward(self, x, z):
        residual = x
        out = self.actv1(self.norm1(x, z))
        out = self.conv1(out)
        out = self.actv2(self.norm2(out, z))
        out = self.conv2(out)
        out = torch.add(out, residual)

        return out

class UpsampleBlock(nn.Module):
    def __init__(self, features):
        super().__init__()

        self.conv = nn.Conv2d(kernel_size=3, in_channels=features, out_channels=features, stride=1, padding=1)
        self.upsample = nn.Upsample(scale_factor=2)

    def forward(self, x):

        x = self.conv(x)

        return self.upsample(x)

class DownsampleBlock(nn.Module):
    def __init__(self, features):
        super().__init__()

        self.conv = nn.Conv2d(kernel_size=3, in_channels=features, out_channels=features, stride=1, padding=1)
        self.downsample = nn.MaxPool2d(2)

    def forward(self, x):

        x = self.conv(x)

        return self.downsample(x)    

class EncoderLR(nn.Module):
    def __init__(self, latent_dim):
        super(EncoderLR, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)

        self.residual = nn.Sequential(
            *[ResidualBlock(64) for _ in range(4)]
        )

        self.downsample = nn.Sequential(
            DownsampleBlock(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),

            DownsampleBlock(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),

            DownsampleBlock(256),
            nn.LeakyReLU(0.2),
        )

        self.to_latent = nn.Linear(256 * 8 * 8, 2 * latent_dim)

    def reparametrize(self, param):
        
        mean, log_var = torch.chunk(param, 2, dim=1)
        std = torch.exp(0.5 * log_var)

        z = mean + std * torch.randn_like(std)

        return z, mean, log_var

    def forward(self, x):

        x = self.conv1(x)
        x = self.residual(x)
        x = self.downsample(x)
        x = x.view(x.size(0), -1)

        embedding = self.to_latent(x)

        z, mean, log_var = self.reparametrize(embedding)

        return z, mean, log_var

class EncoderHR(nn.Module):
    def __init__(self, latent_dim):
        super(EncoderHR, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)

        self.residual = nn.Sequential(
            *[ResidualBlock(64) for _ in range(4)]
        )

        self.downsample = nn.Sequential(
            DownsampleBlock(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),

            DownsampleBlock(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),

            DownsampleBlock(256),
            nn.LeakyReLU(0.2),
        )

        self.to_latent = nn.Linear(256 * 8 * 8, latent_dim)

    def forward(self, x):

        x = self.conv1(x)
        x = self.residual(x)
        x = self.downsample(x)
        x = x.view(x.size(0), -1)

        z_hr = self.to_latent(x)

        return z_hr

class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()

        self.from_latent = nn.Linear(latent_dim, 256 * 8 * 8)
        
        self.upsample = nn.Sequential(
            UpsampleBlock(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),

            UpsampleBlock(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),

            UpsampleBlock(64),
            nn.LeakyReLU(0.2),
        )

        self.residual = nn.ModuleList()
        for _ in range(4):
            self.residual.append(AdaINResidualBlock(latent_dim, 64))
        
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1)
    
    def forward(self, z_hr, z_lr):
        
        x = self.from_latent(z_hr)
        x = x.view(x.size(0), -1, 8, 8)
        x = self.upsample(x)

        for layer in self.residual:
            x = layer(x, z_lr)

        x = self.conv2(x)

        return torch.sigmoid(x)

class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()

        self.vgg = models.vgg19(weights='DEFAULT').features[:29]
        self.indexes = [0, 5, 10, 19, 28]
        self.normalize = v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def forward(self, x):
        x = self.normalize(x)
        features = []
        for index, layer in enumerate(self.vgg):
            x = layer(x)
            if index in self.indexes:
                features.append(x)

        return features

class MINE(nn.Module):
    def __init__(self) -> None:
        super(MINE, self).__init__()

        self.conv1 = nn.Sequential(
            nn.InstanceNorm2d(3, affine=False),
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1))

        self.conv2 = nn.Sequential(
            nn.InstanceNorm2d(3, affine=False),
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1))

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
        )
        self.fc = nn.Linear(256 * 4 * 4, 1)

    def forward(self, x1, x2):
        
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)

        x = torch.add(x1, x2)
        x = self.conv_block(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
