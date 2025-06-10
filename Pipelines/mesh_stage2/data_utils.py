import os
import json
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

def get_image_transforms():
    return transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

def get_mask_transforms():
    return transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.ToTensor()
    ])

class MeshRefinementDataset(Dataset):
    """
    Expects metadata JSON list of {"image":"img.png","mask":"mask.png"} entries.
    """
    def __init__(self, image_dir, mask_dir, metadata_path,
                 img_transform=None, mask_transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.img_transform = img_transform or get_image_transforms()
        self.mask_transform = mask_transform or get_mask_transforms()
        with open(metadata_path, 'r') as f:
            self.samples = json.load(f)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        entry = self.samples[idx]
        img_path  = os.path.join(self.image_dir, entry['image'])
        mask_path = os.path.join(self.mask_dir,  entry['mask'])
        image = Image.open(img_path).convert('RGB')
        mask  = Image.open(mask_path).convert('L')
        if self.img_transform:
            image = self.img_transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)
        return image, mask
