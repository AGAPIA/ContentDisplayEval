import os
import json
import yaml
import torch
from torch.utils.data import DataLoader
from data_utils import MeshRefinementDataset, get_image_transforms, get_mask_transforms
from model_utils import load_model
from PIL import Image

def run_inference(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(cfg['inference']['model_path'], device=device)
    model.eval()

    ds = MeshRefinementDataset(
        image_dir=cfg['inference']['image_dir'],
        mask_dir=cfg['inference']['mask_dir'],
        metadata_path=cfg['inference']['metadata_path'],
        img_transform=get_image_transforms(),
        mask_transform=get_mask_transforms()
    )
    loader = DataLoader(ds, batch_size=cfg['inference']['batch_size'], shuffle=False)

    os.makedirs(cfg['inference']['output_dir'], exist_ok=True)
    results = []

    with torch.no_grad():
        for imgs, filenames in loader:
            imgs = imgs.to(device)
            outputs = model(imgs)['out']  # (B,1,320,320)
            probs = torch.sigmoid(outputs).cpu()

            for i, name in enumerate(filenames):
                mask = probs[i, 0].numpy()
                # Save mask image
                img = Image.fromarray((mask * 255).astype('uint8'))
                mask_path = os.path.join(cfg['inference']['output_dir'], name)
                img.save(mask_path)

                # Derive binary flag
                flag = int((mask > cfg['inference']['threshold']).any())
                results.append({'filename': name, 'anomaly_flag': flag})

    with open(os.path.join(cfg['inference']['output_dir'], 'inference_results.json'), 'w') as f:
        json.dump(results, f, indent=4)

    print(f"Inference completed. Results saved to {cfg['inference']['output_dir']}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    run_inference(args.config)
