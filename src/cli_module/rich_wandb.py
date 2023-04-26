import os
import os.path as osp
import inspect

import json
import wandb
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
            print("Config uploaded to Wandb!!!")

            run_id = self.trainer.logger.version
            artifacts = wandb.Artifact(f"src-{run_id}", type="source-code")

            if hasattr(self.datamodule, "transforms"):
                transform_module = self.datamodule.transforms.__class__
                transform_filepath = osp.abspath(
                    inspect.getsourcefile(transform_module)
                )
                artifacts.add_file(transform_filepath, f"src-{run_id}/transforms.py")

            wandb.log_artifact(artifacts)

    def before_instantiate_classes(self) -> None:
        # Dividing directories into subcommand (e.g. fit, validate, test, etc...)
        subcommand = self.config["subcommand"]
        save_dir = self.config[subcommand]["trainer"]["logger"]["init_args"]["save_dir"]
        self.config[subcommand]["trainer"]["logger"]["init_args"][
            "save_dir"
        ] = osp.join(save_dir, subcommand)

        save_dir = self.config[subcommand]["trainer"]["logger"]["init_args"]["save_dir"]

        self.config[subcommand]["model_ckpt"]["dirpath"] = osp.join(
            save_dir, "checkpoints"
        )

        # Making logger save_dir to prevent wandb using /tmp/wandb
        if not osp.exists(save_dir):
            os.makedirs(save_dir)

        # Specifying job_type of wandb.init() for quick grouping
        self.config[subcommand]["trainer"]["logger"]["init_args"].job_type = subcommand
