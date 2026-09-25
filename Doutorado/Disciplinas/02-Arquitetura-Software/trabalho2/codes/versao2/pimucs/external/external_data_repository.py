"""Fachada para o repositório externo de dados de sensoriamento remoto."""
from __future__ import annotations

from datetime import datetime

from pimucs.models import GeoReferencedImage


class ExternalDataRepository:
    """Sistema externo: informa se há dados novos e fornece as imagens."""

    def __init__(self, has_new_data: bool = True) -> None:
        self.has_new_data = has_new_data

    def check_new_data(self) -> bool:
        return self.has_new_data

    def get_new_images(self) -> list[GeoReferencedImage]:
        return [GeoReferencedImage("image_001.tif", datetime.now())]
