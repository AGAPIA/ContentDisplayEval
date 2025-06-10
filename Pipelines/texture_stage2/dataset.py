import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class TextureStage2Dataset(Dataset):
    def __init__(self, img_dir):
        self.img_dir = img_dir
        self.img_files = [f for f in os.listdir(img_dir) if f.endswith('.png')]
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        image = Image.open(img_path).convert("RGB")
        return self.transform(image), self.img_files[idx]
