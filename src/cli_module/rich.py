from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    RichProgressBar,
)
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme

from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.cli import LightningArgumentParser


class RichCLI(LightningCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        parser.add_lightning_class_args(RichProgressBar, "rich_progress")
        # parser.set_defaults(
        #     {
        #         "rich_progress.theme": {
        #             "class_path": "lightning.pytorch.callbacks.progress.rich_progress.RichProgressBarTheme",
        #             "init_args": {"progress_bar": "purple"},
        #         }
        #     }
        # )
        parser.add_lightning_class_args(ModelCheckpoint, "model_ckpt")
        parser.set_defaults(
            {
                "model_ckpt.monitor": "val/acc",
                "model_ckpt.mode": "max",
                "model_ckpt.save_last": True,
            }
        )

        parser.add_lightning_class_args(LearningRateMonitor, "lr_monitor")
        parser.set_defaults({"lr_monitor.logging_interval": "epoch"})
        

        
