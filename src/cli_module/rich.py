import os
import os.path as osp

import json

from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    RichProgressBar,
)
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme

from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.cli import LightningArgumentParser
from jsonargparse import lazy_instance


class RichCLI(LightningCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        parser.add_lightning_class_args(RichProgressBar, "rich_progress")
        parser.set_defaults({"rich_progress.theme.progress_bar": "purple"})
        parser.add_lightning_class_args(ModelCheckpoint, "model_ckpt")
        parser.set_defaults(
            {
                "model_ckpt.monitor": "val/acc",
                "model_ckpt.mode": "max",
                "model_ckpt.save_last": True,
                "model_ckpt.filename": "{epoch:03d}",
            }
        )

        parser.add_lightning_class_args(LearningRateMonitor, "lr_monitor")
        parser.set_defaults({"lr_monitor.logging_interval": "epoch"})

        # add `-n` argument linked with trainer.logger.name for easy cmdline access
        parser.add_argument(
            "--name", "-n", dest="name", action="store", default="default_name"
        )
        parser.add_argument(
            "--version", "-v", dest="version", action="store", default="version_0"
        )

        parser.link_arguments("name", "trainer.logger.init_args.name")
