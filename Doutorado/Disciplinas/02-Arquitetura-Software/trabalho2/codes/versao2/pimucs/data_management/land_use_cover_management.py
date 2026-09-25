from __future__ import annotations

from pimucs.models import LandUseCoverMap


class LandUseCoverManagement:
    """Fachada de acesso ao mapa de uso e cobertura do solo vigente."""

    def __init__(self) -> None:
        self._current_map: LandUseCoverMap | None = None

    def update_map(self, land_use_map: LandUseCoverMap) -> None:
        self._current_map = land_use_map

    def get_current_map(self) -> LandUseCoverMap | None:
        return self._current_map
