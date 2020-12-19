
import torch
import torch.nn as nn
import torchvision as tv
from modules import NET_REGISTRY

@NET_REGISTRY.register()
class ResNet50(nn.Module):
    def __init__(self, num_classes, pretrain=None):
        super().__init__()
        self.net = tv.models.resnet50(num_classes=num_classes)
        if pretrain is not None:
            weights = torch.load(pretrain)
            for key in list(weights.keys()):
                if 'fc' in key.split('.'):
                    weights.pop(key)
            self.net.load_state_dict(weights, strict=False)
    
    def forward(self, x):
        outputs = self.net(x)
        return {
            'outputs': outputs
        }


@NET_REGISTRY.register()
class ResNet18(nn.Module):
    def __init__(self, num_classes, pretrain=None):
        super().__init__()
        self.net = tv.models.resnet18(num_classes=num_classes)
        if pretrain is not None:
            weights = torch.load(pretrain)
            for key in list(weights.keys()):
                if 'fc' in key.split('.'):
                    weights.pop(key)
            self.net.load_state_dict(weights, strict=False)
    
    def forward(self, x):
        outputs = self.net(x)
        return {
            'outputs': outputs
        }
