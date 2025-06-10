import torch
import torch.nn as nn
from torchvision import models

def load_model(model_path=None, device=None):
    # Load DeepLabV3+ with a ResNet101 backbone
    model = models.segmentation.deeplabv3_resnet101(pretrained=True, progress=True)
    # Change classifier to output 1 channel (binary mask)
    model.classifier = nn.Sequential(
        nn.Conv2d(2048, 256, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(256),
        nn.ReLU(),
        nn.Conv2d(256, 1, kernel_size=1)
    )
    if model_path:
        state = torch.load(model_path, map_location=device or 'cpu')
        model.load_state_dict(state)
    if device:
        model.to(device)
    return model
