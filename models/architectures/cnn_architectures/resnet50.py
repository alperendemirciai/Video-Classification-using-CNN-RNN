import torch.nn as nn
from torchvision import models

class ResNet50Model(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(ResNet50Model, self).__init__()
        self.resnet50 = models.resnet50(pretrained=pretrained)
        
        if pretrained:
            for param in self.resnet50.parameters():
                param.requires_grad = False
        
        num_ftrs = self.resnet50.fc.in_features
        self.resnet50.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.resnet50(x)