import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
from data_utils import TextureDataset, get_default_transforms
import os
import yaml
import argparse
from tqdm import tqdm
from model_utils import load_model

def train_model(config_path):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = get_default_transforms()

    train_dataset = TextureDataset(
        image_dir=config['train']['image_dir'],
        metadata_path=config['train']['metadata_path'],
        transform=transform
    )
    train_loader = DataLoader(train_dataset, batch_size=config['train']['batch_size'], shuffle=True)

    model = load_model(config['model'])

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['train']['learning_rate'])

    num_epochs = config['train']['epochs']
    os.makedirs(config['train']['output_dir'], exist_ok=True)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

    model_path = os.path.join(config['train']['output_dir'], 'texture_stage1.pth')
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    train_model(args.config)
