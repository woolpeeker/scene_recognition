
import torch
import torch.nn as nn
import torchvision as tv
from modules import NET_REGISTRY

@NET_REGISTRY.register()
class InceptionV3(nn.Module):
    def __init__(self, num_classes, pretrain=None):
        self.net = tv.models.inception_v3(
            pretrained=False,
            transform_input=False,
            num_classes=num_classes)
        
        if pretrain is not None:
            weights = torch.load(pretrain)
            for key, v in list(weights.items()):
                if 'fc' in key.split('.'):
                    weights.pop(key)
            self.net.load_state_dict(weights, strict=False)
    
    def forward(self, x):
        outputs, aux_outputs = self.net(x)
        return {
            'outputs': outputs,
            'aux_outputs': aux_outputs
        }