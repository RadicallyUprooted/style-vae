from typing import Tuple
import torch
import torch.nn as nn
from .encoder import EncoderLR, EncoderHR
from .decoder import Decoder

class VAE(nn.Module):
    """
    The main Variational Autoencoder model that combines the encoders and decoder.
    """
    def __init__(self, latent_dim: int):
        super(VAE, self).__init__()
        self.encoder_lr = EncoderLR(latent_dim)
        self.encoder_hr = EncoderHR(latent_dim)
        self.decoder = Decoder(latent_dim)

    def forward(self, high_res_img: torch.Tensor, low_res_img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Performs a full forward pass through the VAE.
        Args:
            high_res_img (torch.Tensor): The clean, high-resolution image (for content).
            low_res_img (torch.Tensor): The degraded, low-resolution image (for style).
        Returns:
            A tuple containing:
            - reconstructed_img (torch.Tensor): The output image with content from high_res_img and style from low_res_img.
            - mean (torch.Tensor): The mean of the latent distribution.
            - log_var (torch.Tensor): The log variance of the latent distribution.
        """
        z_hr = self.encoder_hr(high_res_img)
        z_lr, mean, log_var = self.encoder_lr(low_res_img)
        reconstructed_img = self.decoder(z_hr, z_lr)
        return reconstructed_img, mean, log_var