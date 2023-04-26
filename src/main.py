import torch

from lightning.pytorch.cli import LightningCLI

from cli_module.rich import RichCLI
from cli_module.rich_wandb import RichWandbCLI
from module.default import LitCifar10
from dataset.cifar10 import Cifar10DataModule

import model
import transforms


def cli_main():
    torch.set_float32_matmul_precision("medium")

    cli = RichWandbCLI(
        LitCifar10,
        Cifar10DataModule,
        parser_kwargs={
            "parser_mode": "omegaconf",
        },
    )


if __name__ == "__main__":
    cli_main()
