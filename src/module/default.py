from typing import Any

from omegaconf import OmegaConf
from hydra.utils import instantiate

import torch
from torch import nn

# import pytorch_lightning as pl
import lightning as L

from torchmetrics import Accuracy


class LitCifar10(L.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        loss_module: nn.Module,
        metric_module: nn.Module,
    ) -> None:
        super().__init__()

        self.model = model

        self.loss_module = loss_module
        self.metric_module = metric_module

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        img, labels = batch
        pred = self(img)

        loss = self.loss_module(pred, labels)
        self.log("train/loss", loss.item())

        return loss

    def validation_step(self, batch, batch_idx):
        img, labels = batch
        pred = self(img)

        loss = self.loss_module(pred, labels)
        self.log("val/loss", loss.item())

        acc = self.metric_module(pred, labels)
        self.log("val/acc", acc)

    def test_step(self, batch, batch_idx):
        img, labels = batch
        pred = self(img)

        acc = self.metric_module(pred, labels)
        self.log("test/acc", acc)
