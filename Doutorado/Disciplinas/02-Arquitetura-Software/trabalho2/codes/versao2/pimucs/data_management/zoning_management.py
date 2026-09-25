from __future__ import annotations

from pimucs.models import ZoningMap


class ZoningManagement:
    """Fachada de acesso ao mapa oficial de zoneamento ambiental."""

    def __init__(self) -> None:
        self._zoning_map = ZoningMap(version=1)

    def get_zoning_map(self) -> ZoningMap:
        return self._zoning_map
