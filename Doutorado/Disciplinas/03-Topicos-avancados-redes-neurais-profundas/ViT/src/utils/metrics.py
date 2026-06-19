from __future__ import annotations

import torch


def accuracy(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    predictions = outputs.argmax(dim=1)
    return (predictions == targets).float().mean().item()
