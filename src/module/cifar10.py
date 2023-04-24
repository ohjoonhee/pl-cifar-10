from typing import Any

from omegaconf import OmegaConf
from hydra.utils import instantiate

import torch
from torch import nn

import pytorch_lightning as pl

from torchmetrics import Accuracy

from utils import buildlib


class LitCifar10(buildlib.BuiltLightningModule):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)

        self.save_hyperparameters(
            OmegaConf.to_container(self.cfg, resolve=True, throw_on_missing=True)
        )

        self.loss_func = nn.CrossEntropyLoss()
        self.metric_func = Accuracy()

    def training_step(self, batch, batch_idx):
        img, labels = batch
        pred = self(img)

        loss = self.loss_func(pred, labels)
        self.log("train/loss", loss.item())

        return loss

    def validation_step(self, batch, batch_idx):
        img, labels = batch
        pred = self(img)

        loss = self.loss_func(pred, labels)
        self.log("val/loss", loss.item())

        acc = self.metric_func(pred, labels)
        self.log("val/acc", acc)
        self.log("val_acc", acc)

    def test_step(self, batch, batch_idx):
        img, labels = batch
        pred = self(img)

        acc = self.metric_func(pred, labels)
        self.log("test/acc", acc)
