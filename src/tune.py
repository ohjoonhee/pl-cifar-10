import os
import os.path as osp
import sys

import torch

import pytorch_lightning as pl

import hydra
from hydra.utils import instantiate, get_original_cwd
from omegaconf import DictConfig
from omegaconf import OmegaConf

import pytorch_lightning as pl
from pytorch_lightning.loggers.wandb import WandbLogger
import wandb

from utils import buildlib


@hydra.main(version_base="1.1", config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    seed = cfg.get("seed", 42)
    pl.seed_everything(seed)

    torch.set_float32_matmul_precision("medium")

    data_module = buildlib.build_datamodule(cfg)
    module: pl.LightningModule = instantiate(cfg.module, cfg)

    trainer: pl.Trainer = buildlib.build_trainer(cfg)
    lr_finder = trainer.tuner.lr_find(module, data_module)
    # trainer.tune(module, data_module)
    fig = lr_finder.plot(suggest=True)
    if isinstance(trainer.logger, WandbLogger):
        wandb.log({"lr_finder": fig})

    print(lr_finder.suggestion())


if __name__ == "__main__":
    main()
