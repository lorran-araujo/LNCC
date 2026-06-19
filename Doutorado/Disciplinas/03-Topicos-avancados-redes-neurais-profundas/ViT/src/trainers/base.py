from __future__ import annotations

from abc import ABC, abstractmethod


class BaseTrainer(ABC):
    @abstractmethod
    def train(self) -> None:
        raise NotImplementedError
