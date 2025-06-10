import torch
import torch.nn as nn
from torchvision import models
from torch.utils.data import DataLoader
from data_utils import TextureDataset, get_default_transforms
import os
import json
import yaml
import argparse
from model_utils import load_model

def run_inference(config_path):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = get_default_transforms()

    dataset = TextureDataset(
        image_dir=config['inference']['image_dir'],
        metadata_path=config['inference']['metadata_path'],
        transform=transform
    )
    dataloader = DataLoader(dataset, batch_size=config['inference']['batch_size'], shuffle=False)

    model = load_model(config['inference']['model_path'], device)
    threshold = config['inference'].get('threshold', 0.5)

    os.makedirs(config['inference']['output_dir'], exist_ok=True)
    results = []

    model.eval()
    with torch.no_grad():
        for images, filenames in dataloader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)[:, 1]  # probability of class 1
            preds = (probs > threshold).int().cpu().tolist()

            for filename, score, pred in zip(filenames, probs.cpu().tolist(), preds):
                results.append({
                    'filename': filename,
                    'anomaly_score': round(score, 4),
                    'prediction': pred
                })

    output_path = os.path.join(config['inference']['output_dir'], 'inference_results.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"Inference completed. Results saved to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    run_inference(args.config)
