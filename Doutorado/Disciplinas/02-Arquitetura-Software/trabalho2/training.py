from models import AIModel

class Training:
    def __init__(self, preprocessed_repository, model_repository):
        self.preprocessed_repository = preprocessed_repository
        self.model_repository = model_repository

    def execute(self):
        data = self.preprocessed_repository.load()
        model = self.model_repository.load()
        updated_model = self.train(model, data)
        self.model_repository.save(updated_model)
        return True

    def train(self, model, data):
        return AIModel(version=model.version + 1)
