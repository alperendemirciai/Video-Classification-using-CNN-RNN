import torch.nn as nn
from torchvision import models

class VGGModel(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(VGGModel, self).__init__()
        self.vgg = models.vgg16(pretrained=pretrained)
        
        if pretrained:
            for param in self.vgg.parameters():
                param.requires_grad = False
        
        num_ftrs = self.vgg.classifier[6].in_features
        self.vgg.classifier[6] = nn.Sequential(
            nn.Linear(num_ftrs, 2048),
            nn.ReLU(),
            nn.Linear(2048, num_classes)
        )

    def forward(self, x):
        return self.vgg(x)