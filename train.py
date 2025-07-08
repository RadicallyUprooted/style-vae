import argparse
import yaml
from torch.utils.data import DataLoader
from src.data import ImageArtifactsDataset
from src.trainer import Trainer

def main(config):
    # Create dataset and dataloader
    dataset = ImageArtifactsDataset(
        folder_path=config['data_path'],
        high_res_size=config.get('high_res_size', 64),
        low_res_size=config.get('low_res_size', 16)
    )
    train_loader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config.get('num_workers', 4),
        pin_memory=True
    )

    # Initialize and run trainer
    trainer = Trainer(config, train_loader)
    trainer.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train VAE for image artifact generation.')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the config file.')
    args = parser.parse_args()

    # Load config from YAML file
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Config file not found at {args.config}. Using default settings.")
        # Define a default config if the file is not found
        config = {
            'data_path': 'data',
            'num_epochs': 30,
            'latent_dim': 128,
            'batch_size': 8,
            'lr': 1e-4,
            'style_weight': 1e-2,
            'kl_weight': 1e-4,
            'mi_weight': 1e-1,
            'checkpoint_dir': 'checkpoints'
        }
        # Create a default config.yaml for the user
        with open('config.yaml', 'w') as f:
            yaml.dump(config, f)
        print("Created a default 'config.yaml'. Please review and edit it.")

    main(config)