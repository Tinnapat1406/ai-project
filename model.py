# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchvision import models   # Not necessary if not using EfficientNet
from facenet_pytorch import InceptionResnetV1

class SiameseEfficientNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.backbone = InceptionResnetV1(
            pretrained='vggface2',  
            classify=False
        )

        
        for p in self.backbone.parameters():
            p.requires_grad = False

    def forward_once(self, x):
        x = self.backbone(x)          
        x = F.normalize(x, p=2, dim=1)
        return x

    def forward(self, img1, img2):
        feat1 = self.forward_once(img1)
        feat2 = self.forward_once(img2)
        return feat1, feat2
