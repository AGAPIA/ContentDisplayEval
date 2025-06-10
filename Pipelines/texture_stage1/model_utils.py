import torch
import torch.nn as nn
from torchvision import models

def load_model(model_path=None, device=None, num_classes=2):
    # Load pretrained ShuffleNetV2
    model = models.shufflenet_v2_x1_0(pretrained=True)

    # Replace final classification layer
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    # If a model path is provided, load the state_dict
    if model_path:
        state_dict = torch.load(model_path, map_location=device or 'cpu')
        model.load_state_dict(state_dict)

    if device:
        model.to(device)

    return model
