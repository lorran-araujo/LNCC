from __future__ import annotations

from abc import ABC, abstractmethod


class PipelineStage(ABC):
    """Contrato comum das etapas do pipeline (RNF2 - modificabilidade)."""

    @abstractmethod
    def execute(self) -> bool:
        """Executa a etapa. Retorna True em caso de sucesso."""
