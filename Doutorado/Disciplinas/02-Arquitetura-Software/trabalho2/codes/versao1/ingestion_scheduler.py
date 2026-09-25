class IngestionScheduler:
    def __init__(self, external_repository, pipeline_manager):
        self.external_repository = external_repository
        self.pipeline_manager = pipeline_manager

    def check_for_new_data(self, preprocessor, trainer, generator, monitoring):
        has_new_data = self.external_repository.check_new_data()

        if not has_new_data:
            return False

        self.pipeline_manager.execute_pipeline(
            preprocessor,
            trainer,
            generator,
            monitoring
        )
        return True
