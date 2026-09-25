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


class TestDataPreProcessing(unittest.TestCase):
    def test_execute_preprocesses_and_saves_data(self):
        external = ExternalDataRepository(has_new_data=True)
        preprocessed = DataPreprocessedRepository()
        module = DataPreProcessing(external, preprocessed)

        result = module.execute()

        self.assertTrue(result)
        data = preprocessed.load()
        self.assertIsNotNone(data)
        self.assertTrue(data["processed"])
        self.assertEqual(data["number_of_images"], 1)


class TestTraining(unittest.TestCase):
    def test_execute_updates_model_version(self):
        preprocessed = DataPreprocessedRepository()
        preprocessed.save({"processed": True, "number_of_images": 1})
        model_repository = ModelRepository()
        module = Training(preprocessed, model_repository)

        old_version = model_repository.load().version
        result = module.execute()
        new_version = model_repository.load().version

        self.assertTrue(result)
        self.assertEqual(new_version, old_version + 1)


class TestGenerate(unittest.TestCase):
    def test_execute_generates_and_publishes_map(self):
        preprocessed = DataPreprocessedRepository()
        preprocessed.save({"processed": True, "number_of_images": 1})
        model_repository = ModelRepository()
        land_use_management = LandUseCoverManagement()

        module = Generate(
            preprocessed,
            model_repository,
            land_use_management
        )

        result = module.execute()

        self.assertTrue(result)
        generated_map = land_use_management.get_current_map()
        self.assertIsNotNone(generated_map)
        self.assertEqual(generated_map.version, 1)
        self.assertEqual(
            generated_map.model_version,
            model_repository.load().version
        )


class TestMonitoring(unittest.TestCase):
    def test_execute_registers_alert_when_irregularity_exists(self):
        land_use_management = LandUseCoverManagement()
        zoning_management = ZoningManagement()
        controlled_access = ControlledAccess()

        # É suficiente haver um mapa disponível para exercitar o fluxo dummy.
        from models import LandUseCoverMap
        land_use_management.update_map(
            LandUseCoverMap(version=1, model_version=1)
        )

        module = Monitoring(
            land_use_management,
            zoning_management,
            controlled_access,
            simulate_irregularity=True
        )

        result = module.execute()

        self.assertTrue(result)
        self.assertEqual(len(controlled_access.alerts), 1)

    def test_execute_does_not_register_alert_without_irregularity(self):
        land_use_management = LandUseCoverManagement()
        zoning_management = ZoningManagement()
        controlled_access = ControlledAccess()

        from models import LandUseCoverMap
        land_use_management.update_map(
            LandUseCoverMap(version=1, model_version=1)
        )

        module = Monitoring(
            land_use_management,
            zoning_management,
            controlled_access,
            simulate_irregularity=False
        )

        result = module.execute()

        self.assertTrue(result)
        self.assertEqual(controlled_access.alerts, [])


class FakeStage:
    def __init__(self, name, execution_log):
        self.name = name
        self.execution_log = execution_log

    def execute(self):
        self.execution_log.append(self.name)
        return True


class TestPipelineManager(unittest.TestCase):
    def test_executes_pipeline_in_expected_order(self):
        execution_log = []
        manager = PipelineManager()

        preprocessor = FakeStage("preprocessing", execution_log)
        trainer = FakeStage("training", execution_log)
        generator = FakeStage("generation", execution_log)
        monitoring = FakeStage("monitoring", execution_log)

        result = manager.execute_pipeline(
            preprocessor,
            trainer,
            generator,
            monitoring
        )

        self.assertTrue(result)
        self.assertEqual(
            execution_log,
            ["preprocessing", "training", "generation", "monitoring"]
        )


class FakePipelineManager:
    def __init__(self):
        self.called = False

    def execute_pipeline(self, *args):
        self.called = True
        return True


class TestIngestionScheduler(unittest.TestCase):
    def test_starts_pipeline_when_new_data_exists(self):
        external = ExternalDataRepository(has_new_data=True)
        pipeline = FakePipelineManager()
        scheduler = IngestionScheduler(external, pipeline)

        result = scheduler.check_for_new_data(
            object(), object(), object(), object()
        )

        self.assertTrue(result)
        self.assertTrue(pipeline.called)

    def test_does_not_start_pipeline_when_no_new_data_exists(self):
        external = ExternalDataRepository(has_new_data=False)
        pipeline = FakePipelineManager()
        scheduler = IngestionScheduler(external, pipeline)

        result = scheduler.check_for_new_data(
            object(), object(), object(), object()
        )

        self.assertFalse(result)
        self.assertFalse(pipeline.called)


if __name__ == "__main__":
    unittest.main()
