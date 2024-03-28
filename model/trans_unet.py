import sys
sys.path.append('./model')
from model.transunet_module.vit_seg_modeling import VisionTransformer as ViT_seg
from model.transunet_module.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

class TransUnet(nn.Module):
    def __init__(self):
        super(TransUnet, self).__init__()
        vit_name = "R50-ViT-B_16"
        config_vit = CONFIGS_ViT_seg["R50-ViT-B_16"]  # R50-ViT-B_16
        config_vit.n_classes = 2  # 2
        config_vit.n_skip = 3  # 3
        if vit_name.find('R50') != -1:
            config_vit.patches.grid = (
                int(512 / 16), int(512 / 16))
        self.net = ViT_seg(config_vit, img_size=512, num_classes=config_vit.n_classes).cuda()
    def forward(self, x):

        out = self.net(x)
        return torch.sigmoid(out)

