import os
import os.path as osp

import json
from lightning.pytorch.loggers import WandbLogger

from lightning.pytorch.cli import LightningArgumentParser

from .rich import RichCLI


class RichWandbCLI(RichCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        super().add_arguments_to_parser(parser)

        parser.set_defaults(
            {
                "trainer.logger": {
                    "class_path": "lightning.pytorch.loggers.WandbLogger",
                    "init_args": {
                        "project": "pl-cifar-10",
                        "config_exclude_keys": [
                            "rich_progress",
                            "model_ckpt",
                        ],  # Callbacks configs are excluded for readability
                    },
                },
            }
        )
        parser.link_arguments(
            ("name", "version"),
            "trainer.logger.init_args.save_dir",
            compute_fn=lambda *args: osp.join("logs", *args),
        )
        parser.link_arguments(
            ("name", "version"),
            "model_ckpt.dirpath",
            compute_fn=lambda *args: osp.join("logs", *args, "checkpoints"),
        )

    def before_fit(self):
        if not isinstance(self.trainer.logger, WandbLogger):
            print("WandbLogger not found! Skipping config upload...")

        else:
            subcommand = self.config["subcommand"]
            dict_config = json.loads(
                json.dumps(self.config[subcommand], default=lambda s: vars(s))
            )
            self.trainer.logger.experiment.config.update(dict_config)
            # self.trainer.logger.experiment.group = self.config["subcommand"]
            print("Config uploaded to Wandb!!!")

    def before_instantiate_classes(self) -> None:
        subcommand = self.config["subcommand"]
        name = self.config[subcommand]["name"]
        version = self.config[subcommand]["version"]
        log_dir = osp.join("logs", name, version)
        if not osp.exists(log_dir):
            os.makedirs(log_dir)

        self.config[subcommand]["trainer"]["logger"]["init_args"].job_type = subcommand
