import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_utils import MeshRefinementDataset, get_image_transforms, get_mask_transforms
from model_utils import load_model

def dice_loss(pred, target, smooth=1.):
    pred = torch.sigmoid(pred)
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    return 1 - ((2. * intersection + smooth) /
                (pred.sum() + target.sum() + smooth))

def main(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ds = MeshRefinementDataset(
        image_dir=cfg['train']['image_dir'],
        mask_dir=cfg['train']['mask_dir'],
        metadata_path=cfg['train']['metadata_path'],
        img_transform=get_image_transforms(),
        mask_transform=get_mask_transforms()
    )
    loader = DataLoader(ds, batch_size=cfg['train']['batch_size'], shuffle=True)

    model = load_model(device=device)
    model.train()

    bce = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg['train']['learning_rate'])

    os.makedirs(cfg['train']['output_dir'], exist_ok=True)

    for epoch in range(cfg['train']['epochs']):
        total_loss = 0
        for imgs, masks in tqdm(loader, desc=f"Epoch {epoch+1}/{cfg['train']['epochs']}"):
            imgs, masks = imgs.to(device), masks.to(device)

            optimizer.zero_grad()
            logits = model(imgs)['out']  # DeepLab returns a dict
            loss = bce(logits, masks) + dice_loss(logits, masks)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg = total_loss / len(loader)
        print(f"Epoch {epoch+1}: Loss={avg:.4f}")

    save_path = os.path.join(cfg['train']['output_dir'], "mesh_stage2.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    main(args.config)
