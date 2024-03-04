import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.batch_norm = nn.BatchNorm2d(64)

    def forward(self, x):

        residual = x
        x = nn.ReLU()(self.batch_norm(self.conv1(x)))
        x = self.batch_norm(self.conv2(x))
        x += residual
        x = nn.ReLU()(x)
        
        return x

class EncoderLR(nn.Module):
    def __init__(self, latent_dim):
        super(EncoderLR, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.residual = nn.Sequential(
            ResidualBlock(),
            ResidualBlock(),
            ResidualBlock(),
            ResidualBlock(),
        )
        self.upsample = nn.Upsample(scale_factor=2, mode='bicubic')
        self.conv2 = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)
        self.fc_mu = nn.Linear(3 * 64 * 64, latent_dim)
        self.fc_sigma = nn.Linear(3 * 64 * 64, latent_dim)

    def forward(self, x):

        x = nn.ReLU()(self.conv1(x))
        x = self.residual(x)
        x = self.upsample(x)
        x = nn.ReLU()(self.conv2(x))
        x = x.view(x.size(0), -1)

        mu = self.fc_mu(x)
        log_sigma = self.fc_sigma(x)

        sigma = torch.exp(0.5 * log_sigma)
        eps = torch.randn_like(sigma)
        z_lr = mu + eps * sigma

        return z_lr, mu, log_sigma

class EncoderHR(nn.Module):
    def __init__(self, latent_dim):
        super(EncoderHR, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=1)
        
        self.residual = nn.Sequential(
            ResidualBlock(),
            ResidualBlock(),
            ResidualBlock(),
            ResidualBlock(),
        )
        self.upsample = nn.Upsample(scale_factor=2, mode='bicubic')
        self.conv2 = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)
        
        self.fc_z = nn.Linear(3 * 64 * 64, latent_dim)

    def forward(self, x):

        x = nn.ReLU()(self.conv1(x))
        x = self.residual(x)
        x = self.upsample(x)
        x = nn.ReLU()(self.conv2(x))
        x = x.view(x.size(0), -1)

        z_hr = self.fc_z(x)

        return z_hr

class AdaIN(nn.Module):
    def __init__(self, latent_dim):
        super(AdaIN, self).__init__()

        self.to_alpha = nn.Linear(latent_dim, 1)
        self.to_beta = nn.Linear(latent_dim, 1)

    def adain(self, x, alpha, beta):

        return alpha * (x - torch.mean(x)) / torch.std(x) + beta
    
    def forward(self, x, z_lr):

        alpha = self.to_alpha(z_lr)
        beta = self.to_beta(z_lr)
        x = self.adain(x, alpha.view(alpha.size(0), -1, 1, 1), beta.view(beta.size(0), -1, 1, 1))
        
        return x

class DecoderHR(nn.Module):
    def __init__(self, latent_dim):
        super(DecoderHR, self).__init__()

        self.fc = nn.Linear(latent_dim, 3 * 64 * 64)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.residual = nn.Sequential(
            ResidualBlock(),
            AdaIN(latent_dim),
            ResidualBlock(),
            AdaIN(latent_dim),
            ResidualBlock(),
            AdaIN(latent_dim),
            ResidualBlock(),
            AdaIN(latent_dim),
        )
        self.upsample = nn.Upsample(scale_factor=2, mode='bicubic')
        self.conv2 = nn.Conv2d(64, 3, kernel_size=3, stride=2, padding=1)
    

    def forward(self, z_hr, z_lr):
        x = self.fc(z_hr)
        x = x.view(x.size(0), -1, 64, 64)
        x = nn.ReLU()(self.conv1(x))
        
        for layer in self.residual:
            if isinstance(layer, AdaIN):
                x = layer(x, z_lr)
            else:
                x = layer(x)

        x = self.upsample(x)
        x = nn.ReLU()(self.conv2(x))

        return x
