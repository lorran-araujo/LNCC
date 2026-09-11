from data_preprocessing import DataPreProcessing
from generate import Generate
from ingestion_scheduler import IngestionScheduler
from monitoring import Monitoring
from pipeline_manager import PipelineManager
from repositories import (
    ControlledAccess,
    DataPreprocessedRepository,
    ExternalDataRepository,
    LandUseCoverManagement,
    ModelRepository,
    ZoningManagement,
)
from training import Training

def main():
    external_repository = ExternalDataRepository(has_new_data=True)
    preprocessed_repository = DataPreprocessedRepository()
    model_repository = ModelRepository()

    land_use_management = LandUseCoverManagement()
    zoning_management = ZoningManagement()
    controlled_access = ControlledAccess()

    preprocessor = DataPreProcessing(external_repository, preprocessed_repository)
    trainer = Training(preprocessed_repository, model_repository)
    generator = Generate(preprocessed_repository, model_repository, land_use_management)
    monitoring = Monitoring(
        land_use_management,
        zoning_management,
        controlled_access,
        simulate_irregularity=True
    )

    pipeline_manager = PipelineManager()
    scheduler = IngestionScheduler(external_repository, pipeline_manager)

    scheduler.check_for_new_data(
        preprocessor,
        trainer,
        generator,
        monitoring
    )

    print("Modelo:", model_repository.load())
    print("Mapa:", land_use_management.get_current_map())
    print("Alertas:", controlled_access.alerts)

if __name__ == "__main__":
    main()
