from __future__ import annotations

from pimucs.data_management import LandUseCoverManagement
from pimucs.ia_processing.pipeline_stage import PipelineStage
from pimucs.ia_processing.repositories import (
    DataPreprocessedRepository,
    ModelRepository,
)
from pimucs.models import AIModel, LandUseCoverMap


class Generate(PipelineStage):
    def __init__(
        self,
        preprocessed_repository: DataPreprocessedRepository,
        model_repository: ModelRepository,
        land_use_management: LandUseCoverManagement,
    ) -> None:
        self._preprocessed_repository = preprocessed_repository
        self._model_repository = model_repository
        self._land_use_management = land_use_management

    def execute(self) -> bool:
        data = self._preprocessed_repository.load()
        model = self._model_repository.load()
        new_map = self.generate_map(model, data)
        # O mapa anterior só é substituído após a geração bem-sucedida (RNF3).
        self._land_use_management.update_map(new_map)
        return True

    def generate_map(self, model: AIModel, data: dict | None) -> LandUseCoverMap:
        current_map = self._land_use_management.get_current_map()
        new_version = 1 if current_map is None else current_map.version + 1
        return LandUseCoverMap(version=new_version, model_version=model.version)
