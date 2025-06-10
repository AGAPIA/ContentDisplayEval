import os
import json
import yaml
import torch
from torch.utils.data import DataLoader
from data_utils import MeshSegmentationDataset, get_default_transforms, get_mask_transforms
from model_utils import load_model

def run_inference(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(model_path=cfg['inference']['model_path'], device=device)
    model.eval()

    dataset = MeshSegmentationDataset(
        image_dir=cfg['inference']['image_dir'],
        mask_dir =cfg['inference']['mask_dir'],  # if available
        metadata_path=cfg['inference']['metadata_path'],
        img_transform=get_default_transforms(),
        mask_transform=get_mask_transforms()
    )
    loader = DataLoader(dataset, batch_size=cfg['inference']['batch_size'], shuffle=False)

    os.makedirs(cfg['inference']['output_dir'], exist_ok=True)
    results = []

    with torch.no_grad():
        for imgs, filenames in loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            masks = torch.sigmoid(logits).cpu()  # (B,1,H,W)

            for i, name in enumerate(filenames):
                mask = masks[i,0].numpy()
                # Save mask as image
                out_path = os.path.join(cfg['inference']['output_dir'], name)
                # Scale to [0,255]
                from PIL import Image
                img = Image.fromarray((mask * 255).astype('uint8'))
                img.save(out_path)

                # Derive binary flag
                flag = int((mask > cfg['inference']['threshold']).any())
                results.append({'filename': name, 'anomaly_flag': flag})

    # Save JSON
    with open(os.path.join(cfg['inference']['output_dir'], 'inference_results.json'), 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    run_inference(args.config)
