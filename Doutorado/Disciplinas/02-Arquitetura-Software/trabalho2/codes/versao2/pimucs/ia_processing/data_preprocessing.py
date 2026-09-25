from __future__ import annotations

from pimucs.external import ExternalDataRepository
from pimucs.ia_processing.pipeline_stage import PipelineStage
from pimucs.ia_processing.repositories import DataPreprocessedRepository
from pimucs.models import GeoReferencedImage


class DataPreProcessing(PipelineStage):
    def __init__(
        self,
        external_repository: ExternalDataRepository,
        preprocessed_repository: DataPreprocessedRepository,
    ) -> None:
        self._external_repository = external_repository
        self._preprocessed_repository = preprocessed_repository

    def execute(self) -> bool:
        images = self._external_repository.get_new_images()
        processed_data = self.preprocess(images)
        self._preprocessed_repository.save(processed_data)
        return True

    def preprocess(self, images: list[GeoReferencedImage]) -> dict:
        return {"processed": True, "number_of_images": len(images)}
