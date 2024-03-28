import torch
import torch.nn as nn
from torch.autograd import Variable as V

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss, DataParallel
import cv2
import numpy as np

def mask2onehot(mask):
    assert len(mask.shape) == 3
    c = torch.max(mask)
    b, h, w = mask.shape
    onehot = torch.zeros((b, c+1, h, w))
    for i in range(c+1):
        onehot[:,i,...] = torch.where(mask==i, 1 ,0 )
    return onehot.int().long()

class BaseBCELoss(nn.Module):
    def __init__(self, sigmoid=False):
        """
        构造函数
        参数:
            sigmoid (bool): 如果为True，那么在计算损失之前会对网络的输出应用sigmoid函数。
                            如果网络的最后一层已经包含了sigmoid激活函数，那么应该设置为False。
        """
        super(BaseBCELoss, self).__init__()
        self.sigmoid = sigmoid

    def forward(self, y_pred, y_true):
        """
        前向传播函数
        参数:
            y_pred (Tensor): 网络的输出，形状为(N, *)，其中N是批量大小，*表示任意其他维度。
            y_true (Tensor): 真实标签，形状和y_pred相同。
        返回:
            bce (Tensor): 计算得到的二元交叉熵损失，形状和输入相同。
        """
        # 如果需要，对网络的输出应用sigmoid函数
        if self.sigmoid:
            y_pred = torch.sigmoid(y_pred)
        # 计算二元交叉熵损失
        bce = F.binary_cross_entropy(y_pred, y_true, reduction='none')
        # 检查损失是否包含NaN或无穷大的值
        assert torch.isnan(bce).any() != True and torch.isinf(bce).any() != True
        return bce

class BaseCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True, ignore_index=-100):
        super(BaseCELoss, self).__init__()
        self.weight = weight
        self.size_average = size_average
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        logits = logits.permute(0, 2, 3, 1).contiguous().view(-1, 2)  # (N, 2, H, W) -> (N*H*W, 2)
        targets = targets.squeeze(1).long().view(-1)  # (N, 1, H, W) -> (N*H*W)

        # 计算交叉熵损失
        loss = F.cross_entropy(logits, targets, weight=self.weight, ignore_index=self.ignore_index, reduction='none')

        # 如果size_average=True，计算均值；否则计算总和
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()
