import torch.nn as nn
import torch
from torchvision import models

SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

class ViTTargetModel(nn.Module):
    def __init__(self, num_classes=5):
        super(ViTTargetModel, self).__init__()
        self.vit = models.vit_b_16(weights=None)
        
        self.vit.heads.head = nn.Linear(self.vit.heads.head.in_features, 128)

        self.fc = nn.Linear(128, 64)

        self.output = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.vit(x)
        x = self.fc(x)
        y = self.output(x)
        return y, x


class AttackModel(nn.Module):
    def __init__(self, dimension=64):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(dimension, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 5)
        )

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
