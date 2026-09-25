from __future__ import annotations

from pimucs.data_management import (
    AlertManagement,
    LandUseCoverManagement,
    ZoningManagement,
)
from pimucs.ia_processing.pipeline_stage import PipelineStage
from pimucs.models import IrregularityAlert, LandUseCoverMap, ZoningMap


class Monitoring(PipelineStage):
    def __init__(
        self,
        land_use_management: LandUseCoverManagement,
        zoning_management: ZoningManagement,
        alert_management: AlertManagement,
        simulate_irregularity: bool = True,
    ) -> None:
        self._land_use_management = land_use_management
        self._zoning_management = zoning_management
        self._alert_management = alert_management
        self.simulate_irregularity = simulate_irregularity

    def execute(self) -> bool:
        land_use_map = self._land_use_management.get_current_map()
        if land_use_map is None:
            return False  # não há mapa a monitorar

        zoning_map = self._zoning_management.get_zoning_map()
        for alert in self.compare_maps(land_use_map, zoning_map):
            self._alert_management.register_alert(alert)
        return True

    def compare_maps(
        self, land_use_map: LandUseCoverMap, zoning_map: ZoningMap
    ) -> list[IrregularityAlert]:
        if not self.simulate_irregularity:
            return []
        return [IrregularityAlert(description="Possível uso incompatível detectado.")]
