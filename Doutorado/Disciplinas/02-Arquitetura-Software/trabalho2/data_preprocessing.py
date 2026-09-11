class DataPreProcessing:
    def __init__(self, external_repository, preprocessed_repository):
        self.external_repository = external_repository
        self.preprocessed_repository = preprocessed_repository

    def execute(self):
        images = self.external_repository.get_new_images()
        processed_data = self.preprocess(images)
        self.preprocessed_repository.save(processed_data)
        return True

    def preprocess(self, images):
        return {
            "processed": True,
            "number_of_images": len(images)
        }
