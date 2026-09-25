from __future__ import annotations

import logging
from collections.abc import Sequence

from pimucs.ia_processing.pipeline_stage import PipelineStage

logger = logging.getLogger(__name__)


class PipelineManager:
    """Executa as etapas em sequência e interrompe na primeira falha (RNF3)."""

    def __init__(self, stages: Sequence[PipelineStage]) -> None:
        self._stages = list(stages)
        self.failed_stage: str | None = None

    def execute_pipeline(self) -> bool:
        self.failed_stage = None
        for stage in self._stages:
            name = type(stage).__name__
            try:
                success = stage.execute()
            except Exception:
                logger.exception("Falha inesperada na etapa %s", name)
                success = False
            if not success:
                self.failed_stage = name
                return False
        return True
