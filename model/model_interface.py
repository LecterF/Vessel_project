# Copyright 2021 Zhongyang Zhang
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
from prettytable import PrettyTable
import pandas as pd
import numpy as np
import torch
import importlib
from torch.nn import functional as F
import torch.optim.lr_scheduler as lrs
import pytorch_lightning as pl
from loss import LInterface
from metrics import MeInterface
from PIL import Image
import albumentations as A
import cv2
import wandb
class MInterface(pl.LightningModule):
    def __init__(self, model_name, loss, lr, **kargs):
        super().__init__()
        self.save_hyperparameters()
        self.load_model()
        self.configure_loss()
        self.configure_metrics()

    def forward(self, img):
        return self.model(img)

    def training_step(self, batch, batch_idx):
        img, labels, filename = batch
        out = self(img)
        loss = self.loss_interface(out, labels)
        self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        img, labels, filename = batch
        # image进行反normalization

        out = self(img)
        loss = self.loss_interface(out, labels)
        img_digit = img.cpu().numpy()
        label_digit = labels.squeeze(1).cpu().numpy()
        out_digit = out.argmax(axis=1).cpu().numpy()

        shape = img_digit.shape
        mean = np.array(self.hparams.mean_sen).reshape(1,-1,1,1)
        std = np.array(self.hparams.std_sen).reshape(1,-1,1,1)
        img_digit = img_digit * std + mean
        min_val = np.min(img_digit)
        max_val = np.max(img_digit)
        img_digit = (img_digit - min_val) / (max_val - min_val) * 255
        self.logger.log_image('image', [Image.fromarray(img_digit[0,...].transpose(1,2,0).astype(np.uint8))], self.current_epoch)
        results = self.metrics_interface.calc_metrics(out, labels)
        for key, value in results.items():
            self.log(key, value, on_step=False, on_epoch=True, prog_bar=True)
            self
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        output = [img, labels, out]
        return output

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)

    def on_validation_epoch_end(self):
        metrics = self.trainer.callback_metrics
        table = PrettyTable(['Metric', 'Value'])
        for key, value in metrics.items():
            table.add_row([key, f'{value:.4f}'])
        df = pd.DataFrame(table.rows, columns=table.field_names)
        wandb.log({'validation_metrics': wandb.Table(dataframe=df)})        # Make the Progress Bar leave there
        self.print('')

    def configure_optimizers(self):
        if hasattr(self.hparams, 'weight_decay'):
            weight_decay = self.hparams.weight_decay
        else:
            weight_decay = 0
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.hparams.lr, betas=(0.9, 0.999), eps=1e-8)

        if self.hparams.lr_scheduler is None:
            return optimizer
        else:
            if self.hparams.lr_scheduler == 'step':
                scheduler = lrs.StepLR(optimizer,
                                       step_size=self.hparams.lr_decay_steps,
                                       gamma=self.hparams.lr_decay_rate)
            elif self.hparams.lr_scheduler == 'cosine':
                scheduler = lrs.CosineAnnealingLR(optimizer,
                                                  T_max=self.hparams.lr_decay_steps,
                                                  eta_min=self.hparams.lr_decay_min_lr)
            else:
                raise ValueError('Invalid lr_scheduler type!')
            return [optimizer], [scheduler]

    def configure_loss(self):
        self.loss_interface = LInterface(self.hparams.loss)
        return self.loss_interface.loss_function

    def configure_metrics(self):
        self.metrics_interface = MeInterface(self.hparams.metrics)
        self.metrics = self.metrics_interface.metric_functions
        return self.metrics

    def load_model(self):
        name = self.hparams.model_name
        # Change the `snake_case.py` file name to `CamelCase` class name.
        # Please always name your model file name as `snake_case.py` and
        # class name corresponding `CamelCase`.
        camel_name = ''.join([i.capitalize() for i in name.split('_')])
        try:
            Model = getattr(importlib.import_module(
                '.' + name, package=__package__), camel_name)
        except:
            raise ValueError(
                f'Invalid Module File Name or Invalid Class Name {name}.{camel_name}!')
        self.model = self.instancialize(Model)

    def instancialize(self, Model, **other_args):
        """ Instancialize a model using the corresponding parameters
            from self.hparams dictionary. You can also input any args
            to overwrite the corresponding value in self.hparams.
        """
        class_args = inspect.getargspec(Model.__init__).args[1:]
        inkeys = self.hparams.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = getattr(self.hparams, arg)
        args1.update(other_args)
        return Model(**args1)
