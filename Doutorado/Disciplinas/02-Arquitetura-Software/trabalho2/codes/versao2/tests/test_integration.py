import unittest
from unittest.mock import patch

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


class TestIAProcessingIntegration(unittest.TestCase):
    def build_system(self, has_new_data=True, simulate_irregularity=True):
        external = ExternalDataRepository(has_new_data=has_new_data)
        preprocessed = DataPreprocessedRepository()
        model_repository = ModelRepository()
        land_use_management = LandUseCoverManagement()
        alert_management = AlertManagement()

        stages = [
            DataPreProcessing(external, preprocessed),
            Training(preprocessed, model_repository),
            Generate(preprocessed, model_repository, land_use_management),
            Monitoring(
                land_use_management,
                ZoningManagement(),
                alert_management,
                simulate_irregularity=simulate_irregularity,
            ),
        ]
        pipeline_manager = PipelineManager(stages)
        return {
            "external": external,
            "scheduler": IngestionScheduler(external, pipeline_manager),
            "pipeline_manager": pipeline_manager,
            "preprocessed_repository": preprocessed,
            "model_repository": model_repository,
            "land_use_management": land_use_management,
            "controlled_access": ControlledAccess(alert_management),
        }

    def test_complete_flow_with_new_data_and_irregularity(self):
        system = self.build_system(has_new_data=True, simulate_irregularity=True)
        initial_model_version = system["model_repository"].load().version

        result = system["scheduler"].check_for_new_data()

        self.assertTrue(result)
        processed = system["preprocessed_repository"].load()
        self.assertIsNotNone(processed)
        self.assertTrue(processed["processed"])

        updated_model = system["model_repository"].load()
        self.assertEqual(updated_model.version, initial_model_version + 1)

        generated_map = system["land_use_management"].get_current_map()
        self.assertIsNotNone(generated_map)
        self.assertEqual(generated_map.model_version, updated_model.version)

        # O alerta gerado no IAProcessing chega à WebApplication via AlertManagement.
        self.assertEqual(len(system["controlled_access"].view_alerts()), 1)

    def test_complete_flow_without_irregularity(self):
        system = self.build_system(has_new_data=True, simulate_irregularity=False)

        result = system["scheduler"].check_for_new_data()

        self.assertTrue(result)
        self.assertIsNotNone(system["land_use_management"].get_current_map())
        self.assertEqual(system["controlled_access"].view_alerts(), [])

    def test_pipeline_is_not_executed_without_new_data(self):
        system = self.build_system(has_new_data=False, simulate_irregularity=True)
        initial_model_version = system["model_repository"].load().version

        result = system["scheduler"].check_for_new_data()

        self.assertFalse(result)
        self.assertIsNone(system["preprocessed_repository"].load())
        self.assertEqual(system["model_repository"].load().version, initial_model_version)
        self.assertIsNone(system["land_use_management"].get_current_map())
        self.assertEqual(system["controlled_access"].view_alerts(), [])

    def test_failure_keeps_previously_published_map_and_model(self):
        """RNF3: falha em uma atualização não afeta o mapa já disponibilizado."""
        system = self.build_system(has_new_data=True, simulate_irregularity=True)
        self.assertTrue(system["scheduler"].check_for_new_data())  # 1ª atualização
        map_before = system["land_use_management"].get_current_map()
        model_before = system["model_repository"].load()
        alerts_before = len(system["controlled_access"].view_alerts())

        with patch.object(
            system["external"], "get_new_images", side_effect=RuntimeError("falha de coleta")
        ):
            with self.assertLogs("pimucs.ia_processing.pipeline_manager", level="ERROR"):
                result = system["scheduler"].check_for_new_data()  # 2ª atualização

        self.assertFalse(result)
        self.assertEqual(system["pipeline_manager"].failed_stage, "DataPreProcessing")
        self.assertEqual(system["land_use_management"].get_current_map(), map_before)
        self.assertEqual(system["model_repository"].load(), model_before)
        self.assertEqual(len(system["controlled_access"].view_alerts()), alerts_before)


if __name__ == "__main__":
    unittest.main()
