import torch
import torch.nn as nn
from torchvision import models

def load_model(model_path=None, device=None, num_classes=2):
    model = models.densenet121(pretrained=True)
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_classes)

    if model_path:
        state_dict = torch.load(model_path, map_location=device or 'cpu')
        model.load_state_dict(state_dict)

    if device:
        model.to(device)
    return model
