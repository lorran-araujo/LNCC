from __future__ import annotations

from pimucs.external import ExternalDataRepository
from pimucs.ia_processing.pipeline_manager import PipelineManager


class IngestionScheduler:
    def __init__(
        self,
        external_repository: ExternalDataRepository,
        pipeline_manager: PipelineManager,
    ) -> None:
        self._external_repository = external_repository
        self._pipeline_manager = pipeline_manager

    def check_for_new_data(self) -> bool:
        """Retorna True somente se havia dados novos e o pipeline concluiu com sucesso."""
        if not self._external_repository.check_new_data():
            return False
        return self._pipeline_manager.execute_pipeline()
