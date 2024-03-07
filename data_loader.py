import os
import random
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

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

        degradation = transforms.Compose([
                transforms.Resize((64, 64)),
                transforms.ElasticTransform(alpha=150.0, sigma=7.0),
                transforms.GaussianBlur(kernel_size=3, sigma=(1.0, 2.0)),
                transforms.ToTensor(),])

        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),])        
        original_image = transform(original_image)
        distorted_image = degradation(distorted_image)

        return original_image, distorted_image
