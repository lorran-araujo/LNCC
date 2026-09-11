from datetime import datetime
from models import AIModel, GeoReferencedImage, ZoningMap

class ExternalDataRepository:
    def __init__(self, has_new_data: bool = True):
        self.has_new_data = has_new_data

    def check_new_data(self) -> bool:
        return self.has_new_data

    def get_new_images(self):
        return [GeoReferencedImage("image_001.tif", datetime.now())]

class DataPreprocessedRepository:
    def __init__(self):
        self.data = None

    def save(self, data):
        self.data = data

    def load(self):
        return self.data

class ModelRepository:
    def __init__(self):
        self.model = AIModel(version=1)

    def load(self):
        return self.model

    def save(self, model):
        self.model = model

class LandUseCoverManagement:
    def __init__(self):
        self.current_map = None

    def update_map(self, land_use_map):
        self.current_map = land_use_map

    def get_current_map(self):
        return self.current_map

class ZoningManagement:
    def __init__(self):
        self.zoning_map = ZoningMap(version=1)

    def get_zoning_map(self):
        return self.zoning_map

class ControlledAccess:
    def __init__(self):
        self.alerts = []

    def register_alert(self, alert):
        self.alerts.append(alert)
