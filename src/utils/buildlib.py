from typing import Any

from omegaconf import OmegaConf
from omegaconf import DictConfig
from hydra.utils import instantiate

from torch import nn

import pytorch_lightning as pl
from pytorch_lightning.loggers.wandb import WandbLogger


def build_callbacks(cfg):
    if "callbacks" not in cfg:
        print("No Callbacks found!!! Skipping Instantiation...")
        return []

    callbacks = []
    for _, module in cfg.callbacks.items():
        callbacks.append(instantiate(module))

    return callbacks


def build_transforms(cfg):
    assert "transforms" in cfg

    transform_dict = {}
    for stage, module in cfg.transforms.items():
        transform_dict[stage + "_transform"] = instantiate(module)

    return transform_dict


def build_datamodule(cfg):
    assert "dataset" in cfg

    tf_dict = build_transforms(cfg)
    datamodule = instantiate(cfg.dataset, **tf_dict)

    return datamodule


def build_trainer(cfg):
    assert "trainer" in cfg

    callbacks = build_callbacks(cfg)
    logger = instantiate(cfg.logger)

    trainer = instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    return trainer


class BuiltLightningModule(pl.LightningModule):
    def __init__(self, cfg) -> None:
        super().__init__()
        if not isinstance(cfg, DictConfig):
            cfg = OmegaConf.create(cfg)
        self.cfg = cfg

        self.model: nn.Module = instantiate(self.cfg.model)
        self.batch_size = self.cfg.dataset.batch_size
        self.lr = self.cfg.optimizer.optimizer.lr

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self) -> Any:
        optimizer = instantiate(self.cfg.optimizer.optimizer, self.model.parameters())
        if "lr_scheduler" not in self.cfg.optimizer:
            return optimizer

        scheduler = instantiate(self.cfg.optimizer.lr_scheduler.scheduler, optimizer)

        opt_cfg = OmegaConf.to_object(self.cfg.optimizer)
        opt_cfg["optimizer"] = optimizer
        opt_cfg["lr_scheduler"]["scheduler"] = scheduler

        return opt_cfg
