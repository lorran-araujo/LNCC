class PipelineManager:
    def execute_pipeline(self, preprocessor, trainer, generator, monitoring):
        self.execute_preprocessing(preprocessor)
        self.execute_training(trainer)
        self.execute_generation(generator)
        self.execute_monitoring(monitoring)
        return True

    def execute_preprocessing(self, preprocessor):
        return preprocessor.execute()

    def execute_training(self, trainer):
        return trainer.execute()

    def execute_generation(self, generator):
        return generator.execute()

    def execute_monitoring(self, monitoring):
        return monitoring.execute()
