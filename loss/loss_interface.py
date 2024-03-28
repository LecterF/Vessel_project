from importlib import import_module
import pytorch_lightning as pl
import torch.nn as nn

class LInterface(nn.modules.loss._Loss):
    def __init__(self, loss_name, **kwargs):
        super(LInterface, self).__init__()
        self.loss_name = loss_name
        self.kwargs = kwargs
        if loss_name.lower() == 'bce':
            module = import_module('loss.bceloss')
            self.loss_function = getattr(module, 'BaseBCELoss')(**self.kwargs)
        elif loss_name.lower() == 'ce':
            module = import_module('loss.bceloss')
            self.loss_function = getattr(module, 'BaseCELoss')(**self.kwargs)
        else:
            raise ValueError("Invalid Loss Type!")
    def forward(self, pred, target):
        return self.loss_function(pred, target)