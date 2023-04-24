import os
import sys

import torch

import pytorch_lightning as pl

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from omegaconf import OmegaConf

import pytorch_lightning as pl

from utils import buildlib


@hydra.main(version_base="1.1", config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    seed = cfg.get("seed", 42)
    pl.seed_everything(seed)

    torch.set_float32_matmul_precision("medium")

    data_module = buildlib.build_datamodule(cfg)
    module = instantiate(cfg.module, cfg)

    trainer = buildlib.build_trainer(cfg)
    trainer.fit(module, data_module)



    


if __name__ == "__main__":
    main()

