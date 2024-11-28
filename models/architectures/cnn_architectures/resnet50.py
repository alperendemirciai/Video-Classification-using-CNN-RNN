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
    
    def feature_extractor(self, x, until_layer='avgpool'):
        """
        Perform a forward pass up to a specified layer.
        
        Args:
            x: Input tensor.
            until_layer (str): The layer name until which to forward the input.
                               Default is 'avgpool', which extracts features
                               before the classification layer.
                               You can also select any other layer to extract, just print the layer name and use it.
        
        Returns:
            Tensor: Extracted features.
        """
        features = x
        for name, layer in self.resnet50._modules.items():
            features = layer(features)
            if name == until_layer:
                break
        return features