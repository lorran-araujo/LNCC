from __future__ import annotations

from pimucs.models import AIModel


class DataPreprocessedRepository:
    """Repositório intermediário dos dados pré-processados."""

    def __init__(self) -> None:
        self.data: dict | None = None

    def save(self, data: dict) -> None:
        self.data = data

    def load(self) -> dict | None:
        return self.data


class ModelRepository:
    """Repositório do modelo de IA vigente."""

    def __init__(self) -> None:
        self.model: AIModel = AIModel(version=1)

    def load(self) -> AIModel:
        return self.model

    def save(self, model: AIModel) -> None:
        self.model = model
