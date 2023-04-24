import torch

from lightning.pytorch.cli import LightningCLI

from module.default import LitCifar10
from dataset.cifar10 import Cifar10DataModule


def cli_main():
    torch.set_float32_matmul_precision("medium")

    cli = LightningCLI(LitCifar10, Cifar10DataModule)


if __name__ == "__main__":
    cli_main()
