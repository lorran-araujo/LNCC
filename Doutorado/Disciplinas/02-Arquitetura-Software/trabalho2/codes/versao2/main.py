from pimucs.data_management import (
    AlertManagement,
    LandUseCoverManagement,
    ZoningManagement,
)
from pimucs.external import ExternalDataRepository
from pimucs.ia_processing import (
    DataPreProcessing,
    DataPreprocessedRepository,
    Generate,
    IngestionScheduler,
    ModelRepository,
    Monitoring,
    PipelineManager,
    Training,
)
from pimucs.web_application import ControlledAccess


def main() -> None:
    # Composição (injeção de dependências)
    external_repository = ExternalDataRepository(has_new_data=True)
    preprocessed_repository = DataPreprocessedRepository()
    model_repository = ModelRepository()

    land_use_management = LandUseCoverManagement()
    zoning_management = ZoningManagement()
    alert_management = AlertManagement()
    controlled_access = ControlledAccess(alert_management)

    stages = [
        DataPreProcessing(external_repository, preprocessed_repository),
        Training(preprocessed_repository, model_repository),
        Generate(preprocessed_repository, model_repository, land_use_management),
        Monitoring(
            land_use_management,
            zoning_management,
            alert_management,
            simulate_irregularity=True,
        ),
    ]
    scheduler = IngestionScheduler(external_repository, PipelineManager(stages))

    success = scheduler.check_for_new_data()

    print("Pipeline concluído com sucesso:", success)
    print("Modelo:", model_repository.load())
    print("Mapa:", land_use_management.get_current_map())
    print("Alertas:", controlled_access.view_alerts())


if __name__ == "__main__":
    main()
