import os
import random
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2, InterpolationMode
from PIL import Image


class Loader(Dataset):
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.image_files = [file for file in os.listdir(folder_path)]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        original_image = os.path.join(self.folder_path, self.image_files[idx])
        
        random_index = random.randint(0, len(self.image_files) - 1)

        distorted_image = os.path.join(self.folder_path, self.image_files[random_index])

        original_image = Image.open(original_image).convert("RGB")
        distorted_image = Image.open(distorted_image).convert("RGB")

        degradation = v2.Compose([
                v2.Resize((16, 16), interpolation=InterpolationMode.BICUBIC),
                v2.Resize((64, 64), interpolation=InterpolationMode.BICUBIC),
                v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])

        transform = v2.Compose([
            v2.Resize((64, 64)),
            v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])  
        
        original_image = transform(original_image)
        distorted_image = degradation(distorted_image)

        return original_image, distorted_image
