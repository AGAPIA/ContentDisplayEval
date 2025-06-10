import torch
import torch.nn as nn
from torchvision import models

class UNet(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        # Use a pretrained encoder (ResNet34) and simple decoder
        self.encoder = models.resnet34(pretrained=True)
        self.enc_layers = list(self.encoder.children())
        self.layer0 = nn.Sequential(*self.enc_layers[:3])   # 64
        self.layer1 = nn.Sequential(*self.enc_layers[3:5])  # 64
        self.layer2 = self.enc_layers[5]                    # 128
        self.layer3 = self.enc_layers[6]                    # 256
        self.layer4 = self.enc_layers[7]                    # 512

        # Decoder
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = nn.Sequential(nn.Conv2d(512, 256, 3, padding=1), nn.ReLU())
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = nn.Sequential(nn.Conv2d(256, 128, 3, padding=1), nn.ReLU())
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1), nn.ReLU())
        self.up0 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.dec0 = nn.Sequential(nn.Conv2d(64 + 64, 64, 3, padding=1), nn.ReLU())

        self.final = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        # Encoder
        x0 = self.layer0(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        # Decoder
        d3 = self.up3(x4)
        d3 = self.dec3(torch.cat([d3, x3], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, x2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, x1], dim=1))
        d0 = self.up0(d1)
        d0 = self.dec0(torch.cat([d0, x0], dim=1))

        out = self.final(d0)
        return out  # shape (B,1,256,256)

def load_model(model_path=None, device=None):
    model = UNet(num_classes=1)
    if model_path:
        state = torch.load(model_path, map_location=device or 'cpu')
        model.load_state_dict(state)
    if device:
        model.to(device)
    return model
