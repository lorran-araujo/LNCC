from __future__ import annotations

import argparse

import torch
from torch import nn

from src.datasets import create_dataloaders
from src.models import create_model
from src.trainers import SupervisedTrainer
from src.utils.config import load_config, prepare_output_dir
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Vision Transformer classifier.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to a YAML config file.")
    return parser.parse_args()


def create_optimizer(config: dict, model: nn.Module) -> torch.optim.Optimizer:
    training_config = config["training"]
    optimizer_name = training_config.get("optimizer", "adamw").lower()
    lr = float(training_config.get("lr", 3e-4))
    weight_decay = float(training_config.get("weight_decay", 0.05))

    if optimizer_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if optimizer_name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=float(training_config.get("momentum", 0.9)),
            weight_decay=weight_decay,
        )

    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def create_scheduler(config: dict, optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler.LRScheduler | None:
    training_config = config["training"]
    scheduler_name = training_config.get("scheduler", "cosine").lower()
    epochs = int(training_config.get("epochs", 10))

    if scheduler_name == "none":
        return None
    if scheduler_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    if scheduler_name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(training_config.get("step_size", 30)),
            gamma=float(training_config.get("gamma", 0.1)),
        )

    raise ValueError(f"Unsupported scheduler: {scheduler_name}")


def resolve_device(config: dict) -> torch.device:
    requested_device = config["training"].get("device", "auto")
    if requested_device == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(requested_device)

    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device requested ({device}), but CUDA is not available.")

        cuda_index = 0 if device.index is None else device.index
        device_count = torch.cuda.device_count()
        if cuda_index >= device_count:
            raise RuntimeError(
                f"CUDA device {cuda_index} requested, but only {device_count} CUDA device(s) are available."
            )

        torch.cuda.set_device(cuda_index)
        device = torch.device(f"cuda:{cuda_index}")

    return device


def main() -> None:
    args = parse_args()
    print(f"[1/8] Carregando configuracao: {args.config}")
    config = load_config(args.config)

    print("[2/8] Configurando seed")
    set_seed(int(config.get("seed", 42)))

    print("[3/8] Preparando diretorio de saida")
    output_dir = prepare_output_dir(config, args.config)

    print("[4/8] Baixando/carregando dataset")
    train_loader, val_loader, test_loader, dataset_info = create_dataloaders(config)

    print("[5/8] Criando modelo")
    model = create_model(config, num_classes=dataset_info.num_classes, image_size=dataset_info.input_size)

    print("[6/8] Criando otimizador")
    optimizer = create_optimizer(config, model)

    print("[7/8] Criando scheduler")
    scheduler = create_scheduler(config, optimizer)

    print("[8/8] Selecionando dispositivo")
    device = resolve_device(config)
    print(f"Using device: {device}")

    print("Iniciando treinamento")
    trainer = SupervisedTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        criterion=nn.CrossEntropyLoss(),
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config=config,
        output_dir=output_dir,
        class_names=dataset_info.class_names,
    )
    trainer.train()


if __name__ == "__main__":
    main()
