import unittest

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


class TestIAProcessingIntegration(unittest.TestCase):
    def build_system(self, has_new_data=True, simulate_irregularity=True):
        external_repository = ExternalDataRepository(
            has_new_data=has_new_data
        )
        preprocessed_repository = DataPreprocessedRepository()
        model_repository = ModelRepository()

        land_use_management = LandUseCoverManagement()
        zoning_management = ZoningManagement()
        controlled_access = ControlledAccess()

        preprocessor = DataPreProcessing(
            external_repository,
            preprocessed_repository
        )
        trainer = Training(
            preprocessed_repository,
            model_repository
        )
        generator = Generate(
            preprocessed_repository,
            model_repository,
            land_use_management
        )
        monitoring = Monitoring(
            land_use_management,
            zoning_management,
            controlled_access,
            simulate_irregularity=simulate_irregularity
        )

        pipeline_manager = PipelineManager()
        scheduler = IngestionScheduler(
            external_repository,
            pipeline_manager
        )

        return {
            "scheduler": scheduler,
            "preprocessor": preprocessor,
            "trainer": trainer,
            "generator": generator,
            "monitoring": monitoring,
            "preprocessed_repository": preprocessed_repository,
            "model_repository": model_repository,
            "land_use_management": land_use_management,
            "controlled_access": controlled_access,
        }

    def test_complete_flow_with_new_data_and_irregularity(self):
        system = self.build_system(
            has_new_data=True,
            simulate_irregularity=True
        )

        initial_model_version = system["model_repository"].load().version

        result = system["scheduler"].check_for_new_data(
            system["preprocessor"],
            system["trainer"],
            system["generator"],
            system["monitoring"],
        )

        self.assertTrue(result)

        processed_data = system["preprocessed_repository"].load()
        self.assertIsNotNone(processed_data)
        self.assertTrue(processed_data["processed"])

        updated_model = system["model_repository"].load()
        self.assertEqual(
            updated_model.version,
            initial_model_version + 1
        )

        generated_map = system["land_use_management"].get_current_map()
        self.assertIsNotNone(generated_map)
        self.assertEqual(
            generated_map.model_version,
            updated_model.version
        )

        self.assertEqual(
            len(system["controlled_access"].alerts),
            1
        )

    def test_complete_flow_without_irregularity(self):
        system = self.build_system(
            has_new_data=True,
            simulate_irregularity=False
        )

        result = system["scheduler"].check_for_new_data(
            system["preprocessor"],
            system["trainer"],
            system["generator"],
            system["monitoring"],
        )

        self.assertTrue(result)
        self.assertIsNotNone(
            system["land_use_management"].get_current_map()
        )
        self.assertEqual(
            system["controlled_access"].alerts,
            []
        )

    def test_pipeline_is_not_executed_without_new_data(self):
        system = self.build_system(
            has_new_data=False,
            simulate_irregularity=True
        )

        initial_model_version = system["model_repository"].load().version

        result = system["scheduler"].check_for_new_data(
            system["preprocessor"],
            system["trainer"],
            system["generator"],
            system["monitoring"],
        )

        self.assertFalse(result)
        self.assertIsNone(
            system["preprocessed_repository"].load()
        )
        self.assertEqual(
            system["model_repository"].load().version,
            initial_model_version
        )
        self.assertIsNone(
            system["land_use_management"].get_current_map()
        )
        self.assertEqual(
            system["controlled_access"].alerts,
            []
        )


if __name__ == "__main__":
    unittest.main()
