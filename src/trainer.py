import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
import os
from typing import Dict

from .model.vae import VAE
from .model.mine import MINE
from .losses import VGGPerceptualLoss, mutual_information_loss, kl_divergence_loss

class Trainer:
    """Handles the model training, optimization, and checkpointing."""
    def __init__(self, config: Dict, train_loader: DataLoader, checkpoint_path: str = None):
        """
        Args:
            config (Dict): A dictionary containing training configuration.
            train_loader (DataLoader): The data loader for training data.
            checkpoint_path (str, optional): Path to a checkpoint to resume training from.
        """
        self.config = config
        self.train_loader = train_loader
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Initialize models
        self.vae = VAE(config['latent_dim']).to(self.device)
        self.mine = MINE().to(self.device)
        self.vgg_loss = VGGPerceptualLoss(self.device)

        # Initialize optimizers
        self.vae_optim = optim.Adam(self.vae.parameters(), lr=config['lr'])
        self.mine_optim = optim.Adam(self.mine.parameters(), lr=config['lr'])
        
        self.checkpoint_dir = config.get('checkpoint_dir', 'checkpoints')
        self.sample_dir = os.path.join(self.checkpoint_dir, 'samples')
        os.makedirs(self.sample_dir, exist_ok=True)

        self.start_epoch = 1
        if checkpoint_path:
            self._load_checkpoint(checkpoint_path)

    def train(self):
        """Runs the main training loop for the specified number of epochs."""
        # Get a fixed batch for generating samples every epoch
        fixed_batch = next(iter(self.train_loader))
        print(f"Obtained a fixed batch of {fixed_batch[0].size(0)} images for sampling.")

        for epoch in range(self.start_epoch, self.config['num_epochs'] + 1):
            self._train_epoch(epoch)
            self._save_checkpoint(epoch)
            self._save_sample_images(epoch, fixed_batch)

    def _load_checkpoint(self, checkpoint_path: str):
        """Loads a model checkpoint."""
        if not os.path.exists(checkpoint_path):
            print(f"Warning: Checkpoint path {checkpoint_path} does not exist. Starting from scratch.")
            return
            
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.vae.load_state_dict(checkpoint['vae_state_dict'])
        self.mine.load_state_dict(checkpoint['mine_state_dict'])
        self.vae_optim.load_state_dict(checkpoint['vae_optim_state_dict'])
        self.mine_optim.load_state_dict(checkpoint['mine_optim_state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1
        
        # Check if config is compatible
        if 'config' in checkpoint and checkpoint['config'] != self.config:
            print("Warning: The loaded checkpoint's config differs from the current config.")

        print(f"Loaded checkpoint from {checkpoint_path}. Resuming training from epoch {self.start_epoch}.")

    def _train_epoch(self, epoch: int):
        """Runs a single training epoch."""
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.config['num_epochs']}")
        total_vae_loss = 0.0

        for img_normal, img_degraded in progress_bar:
            img_normal = img_normal.to(self.device)
            img_degraded = img_degraded.to(self.device)

            # --- Train VAE ---
            self.vae.train()
            self.mine.eval()
            self.vae_optim.zero_grad()
            
            img_reconstructed, mean, log_var = self.vae(img_normal, img_degraded)
            
            content_loss, style_loss = self.vgg_loss(img_reconstructed, img_normal, img_degraded)
            perceptual_loss = content_loss + self.config['style_weight'] * style_loss
            kl_loss = self.config['kl_weight'] * kl_divergence_loss(mean, log_var)
            mi_estimate = mutual_information_loss(self.mine, img_degraded, img_reconstructed)
            mi_loss = self.config['mi_weight'] * mi_estimate
            
            vae_loss = perceptual_loss + kl_loss - mi_loss
            
            vae_loss.backward()
            self.vae_optim.step()

            # --- Train MINE ---
            self.mine.train()
            self.mine_optim.zero_grad()
            mine_loss = -mutual_information_loss(self.mine, img_degraded, img_reconstructed.detach())
            mine_loss.backward()
            self.mine_optim.step()

            total_vae_loss += vae_loss.item()
            progress_bar.set_postfix({
                "VAE Loss": f"{vae_loss.item():.3f}",
                "Content": f"{content_loss.item():.3f}",
                "Style": f"{style_loss.item():.3f}",
                "KL": f"{kl_loss.item():.3f}",
                "MI": f"{mi_estimate.item():.3f}"
            })
        
        print(f"Epoch {epoch} finished. Average VAE Loss: {total_vae_loss / len(self.train_loader):.4f}")

    def _save_checkpoint(self, epoch: int):
        """Saves a model checkpoint."""
        checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
        torch.save({
            'epoch': epoch,
            'vae_state_dict': self.vae.state_dict(),
            'mine_state_dict': self.mine.state_dict(),
            'vae_optim_state_dict': self.vae_optim.state_dict(),
            'mine_optim_state_dict': self.mine_optim.state_dict(),
            'config': self.config
        }, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

    def _save_sample_images(self, epoch: int, fixed_batch: tuple):
        """Saves a grid of generated images for visual inspection."""
        self.vae.eval()
        with torch.no_grad():
            img_normal, img_degraded = fixed_batch
            img_normal = img_normal.to(self.device)
            img_degraded = img_degraded.to(self.device)
            
            img_reconstructed, _, _ = self.vae(img_normal, img_degraded)
            
            # Arrange images in a grid: [Originals, Degraded (Style), Generated]
            comparison_grid = torch.cat([img_normal, img_degraded, img_reconstructed])
            
            save_path = os.path.join(self.sample_dir, f'epoch_{epoch}.png')
            save_image(comparison_grid, save_path, nrow=img_normal.size(0))
        print(f"Saved sample images to {save_path}")
