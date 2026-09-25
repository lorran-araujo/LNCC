from models import LandUseCoverMap

class Generate:
    def __init__(self, preprocessed_repository, model_repository, land_use_management):
        self.preprocessed_repository = preprocessed_repository
        self.model_repository = model_repository
        self.land_use_management = land_use_management

    def execute(self):
        data = self.preprocessed_repository.load()
        model = self.model_repository.load()
        new_map = self.generate_map(model, data)
        self.land_use_management.update_map(new_map)
        return True

    def generate_map(self, model, data):
        current_map = self.land_use_management.get_current_map()
        new_version = 1 if current_map is None else current_map.version + 1
        return LandUseCoverMap(
            version=new_version,
            model_version=model.version
        )
