import os
import random
from typing import Tuple

import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2, InterpolationMode
from PIL import Image

class ImageArtifactsDataset(Dataset):
    """
    A dataset that provides pairs of high-resolution images and synthetically degraded low-resolution images.
    The degradation is created by resizing a randomly selected image down and then up, simulating artifacts.
    """
    def __init__(self, folder_path: str, high_res_size: int = 64, low_res_size: int = 16, use_augmentation: bool = True):
        """
        Args:
            folder_path (str): Path to the directory containing the images.
            high_res_size (int): The size of the high-resolution output images.
            low_res_size (int): The intermediate low-resolution size used to create artifacts.
            use_augmentation (bool): Whether to apply data augmentation (random horizontal flips).
        """
        super().__init__()
        self.folder_path = folder_path
        self.image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

        hr_transforms = [
            v2.Resize((high_res_size, high_res_size), interpolation=InterpolationMode.BICUBIC),
        ]
        if use_augmentation:
            hr_transforms.append(v2.RandomHorizontalFlip(p=0.5))
        
        hr_transforms.extend([
            v2.ToImage(), 
            v2.ToDtype(torch.float32, scale=True)
        ])
        self.hr_transform = v2.Compose(hr_transforms)
        
        self.lr_degradation = v2.Compose([
            v2.Resize((low_res_size, low_res_size), interpolation=InterpolationMode.BICUBIC),
            v2.Resize((high_res_size, high_res_size), interpolation=InterpolationMode.BICUBIC),
            v2.ToImage(), 
            v2.ToDtype(torch.float32, scale=True)
        ])

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns a tuple of (high_res_image, low_res_image).
        """
        original_image_path = self.image_files[idx]
        distorted_image_path = random.choice(self.image_files)

        try:
            original_image = Image.open(original_image_path).convert("RGB")
            distorted_image = Image.open(distorted_image_path).convert("RGB")
        except Exception as e:
            print(f"Warning: Could not load image {original_image_path} or {distorted_image_path}. Skipping. Error: {e}")
            return self.__getitem__((idx + 1) % len(self))

        high_res_image = self.hr_transform(original_image)
        low_res_image = self.lr_degradation(distorted_image)

        return high_res_image, low_res_image