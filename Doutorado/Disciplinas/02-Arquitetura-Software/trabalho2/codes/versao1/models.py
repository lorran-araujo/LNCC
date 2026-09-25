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
    model_version: int

@dataclass
class ZoningMap:
    version: int

@dataclass
class IrregularityAlert:
    description: str
