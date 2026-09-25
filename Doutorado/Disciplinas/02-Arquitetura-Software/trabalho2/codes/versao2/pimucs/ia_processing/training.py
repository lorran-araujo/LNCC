from __future__ import annotations

from pimucs.ia_processing.pipeline_stage import PipelineStage
from pimucs.ia_processing.repositories import (
    DataPreprocessedRepository,
    ModelRepository,
)
from pimucs.models import AIModel


class Training(PipelineStage):
    def __init__(
        self,
        preprocessed_repository: DataPreprocessedRepository,
        model_repository: ModelRepository,
    ) -> None:
        self._preprocessed_repository = preprocessed_repository
        self._model_repository = model_repository

    def execute(self) -> bool:
        data = self._preprocessed_repository.load()
        model = self._model_repository.load()
        updated_model = self.train(model, data)
        self._model_repository.save(updated_model)
        return True

    def train(self, model: AIModel, data: dict | None) -> AIModel:
        return AIModel(version=model.version + 1)
