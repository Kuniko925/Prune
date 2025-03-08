import torch
import torch.nn as nn
from torchvision import models

class VGG11(nn.Module):
    def __init__(self, num_class):
        super().__init__()
        self.base_model = models.vgg11(weights=models.VGG11_Weights.IMAGENET1K_V1)
        in_features = self.base_model.classifier[6].in_features
        self.base_model.classifier[6] = nn.Linear(in_features, num_class)
        self.feature_maps = []
        self.hook = self.base_model.features[18].register_forward_hook(self.hook_fn)
    
    def forward(self, x):
        x = self.base_model(x)
        return x

    def forward(self, x):
        x = self.base_model(x)
        return x

    def hook_fn(self, module, input, output):
        self.feature_maps = [output.detach().cpu()]

    def remove_hook(self):
        self.hook.remove()

class ViT16(nn.Module):
    def __init__(self, num_class):
        super().__init__()
        self.base_model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        in_features = self.base_model.heads[0].in_features
        self.base_model.heads = nn.Sequential(
            nn.Linear(in_features, num_class)
        )
        self.feature_maps = []
        self.hook = self.base_model.conv_proj.register_forward_hook(self.hook_fn) # To take filter of conv2

    def forward(self, x):
        x = self.base_model(x)
        return x

    def forward(self, x):
        x = self.base_model(x)
        return x

    def hook_fn(self, module, input, output):
        self.feature_maps = [output.detach().cpu()] # Reduce memory

    def remove_hook(self):
        self.hook.remove()