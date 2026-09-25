"""Entidades de domínio compartilhadas entre os módulos (apenas dados)."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass
class GeoReferencedImage:
    name: str
    acquisition_date: datetime


@dataclass
class AIModel:
    version: int


@dataclass
class LandUseCoverMap:
    version: int
    model_version: int  # versão do AIModel usado na geração (não é referência de objeto)


@dataclass
class ZoningMap:
    version: int


@dataclass
class IrregularityAlert:
    description: str
