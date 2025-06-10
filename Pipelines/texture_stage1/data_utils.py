import os
import json
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def get_default_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


class TextureDataset(Dataset):
    def __init__(self, image_dir, metadata_path, transform=None):
        self.image_dir = image_dir
        self.transform = transform or get_default_transforms()
        with open(metadata_path, 'r') as f:
            self.samples = json.load(f)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        entry = self.samples[idx]
        img_path = os.path.join(self.image_dir, entry['filename'])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, entry['filename']
