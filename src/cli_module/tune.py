import os
import os.path as osp

import json

import torch

from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    RichProgressBar,
)
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme
from lightning.pytorch.tuner import Tuner

from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.cli import LightningArgumentParser
from jsonargparse import lazy_instance


class TuneCLI(LightningCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        parser.add_lightning_class_args(RichProgressBar, "rich_progress")
        parser.set_defaults({"rich_progress.theme.progress_bar": "purple"})
        parser.add_lightning_class_args(ModelCheckpoint, "model_ckpt")
        parser.set_defaults(
            {
                "model_ckpt.monitor": "val/loss",
                "model_ckpt.mode": "min",
                "model_ckpt.save_last": True,
                "model_ckpt.filename": "best-{epoch:03d}",
            }
        )

        parser.add_lightning_class_args(LearningRateMonitor, "lr_monitor")
        parser.set_defaults({"lr_monitor.logging_interval": "epoch"})

        parser.set_defaults(
            {
                "trainer.logger": {
                    "class_path": "lightning.pytorch.loggers.TensorBoardLogger",
                    "init_args": {"save_dir": "logs"},
                },
            }
        )

        # add `-n` argument linked with trainer.logger.name for easy cmdline access
        parser.add_argument(
            "--name", "-n", dest="name", action="store", default="default_name"
        )
        parser.add_argument(
            "--version", "-v", dest="version", action="store", default="version_0"
        )

    def before_instantiate_classes(self) -> None:
        assert "subcommand" not in self.config

        self.config["trainer"]["logger"]["init_args"]["name"] = self.config["name"]
        self.config["trainer"]["logger"]["init_args"]["version"] = self.config[
            "version"
        ]
        self.config["trainer"]["logger"]["init_args"]["sub_dir"] = "tune"

        save_dir = self.config["trainer"]["logger"]["init_args"]["save_dir"]
        name = self.config["name"]
        version = self.config["version"]
        sub_dir = self.config["trainer"]["logger"]["init_args"]["sub_dir"]

        save_dir = osp.join(save_dir, name, version, sub_dir)

        self.config["model_ckpt"]["dirpath"] = osp.join(save_dir, "checkpoints")
