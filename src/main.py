import torch

from lightning.pytorch.cli import LightningCLI

from cli_module.rich import RichCLI
from module.default import LitCifar10
from dataset.cifar10 import Cifar10DataModule

import model


def cli_main():
    torch.set_float32_matmul_precision("medium")

    cli = RichCLI(LitCifar10, Cifar10DataModule)


if __name__ == "__main__":
    cli_main()
