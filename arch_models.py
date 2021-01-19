import torch
import torch.nn as nn
from torchvision import models

__all__ = ['CNN', 'ResNet50', 'ResNest50']


class CNN(nn.Module):
    def __init__(self, n_classes=24):
        super(CNN, self).__init__()
        self.model = torch.nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=5),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),
            nn.Dropout2d(0.25),

            nn.Conv2d(16, 16, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=5),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),
            nn.Dropout2d(0.25),

            nn.Conv2d(32, 32, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.25),

            nn.Flatten(),
            nn.Linear(30976, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, n_classes))

    def forward(self, inputs):
        x = self.model(inputs)
        return x


class ResNet50(nn.Module):
    def __init__(self, n_classes=24):
        super(ResNet50, self).__init__()
        self.model = models.resnet50(pretrained=True)
        num_fc_ftr = self.model.fc.in_features
        self.model.fc = nn.Linear(num_fc_ftr, n_classes)

    def forward(self, inputs):
        x = self.model(inputs)
        return x


class ResNest50(nn.Module):
    def __init__(self, n_classes=24):
        super(ResNest50, self).__init__()
        self.model = torch.hub.load('zhanghang1989/ResNeSt', 'resnest50', pretrained=True)
        num_fc_ftr = self.model.fc.in_features
        self.model.fc = nn.Linear(num_fc_ftr, n_classes)

    def forward(self, inputs):
        x = self.model(inputs)
        return x