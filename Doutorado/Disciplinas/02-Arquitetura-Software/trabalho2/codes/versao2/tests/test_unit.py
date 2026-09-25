import unittest

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
    PipelineStage,
    Training,
)
from pimucs.models import IrregularityAlert, LandUseCoverMap
from pimucs.web_application import ControlledAccess


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
        module = Generate(preprocessed, model_repository, land_use_management)

        result = module.execute()

        self.assertTrue(result)
        generated_map = land_use_management.get_current_map()
        self.assertIsNotNone(generated_map)
        self.assertEqual(generated_map.version, 1)
        self.assertEqual(generated_map.model_version, model_repository.load().version)


class TestMonitoring(unittest.TestCase):
    def build(self, simulate_irregularity, with_map=True):
        land_use_management = LandUseCoverManagement()
        alert_management = AlertManagement()
        if with_map:
            land_use_management.update_map(LandUseCoverMap(version=1, model_version=1))
        module = Monitoring(
            land_use_management,
            ZoningManagement(),
            alert_management,
            simulate_irregularity=simulate_irregularity,
        )
        return module, alert_management

    def test_execute_registers_alert_when_irregularity_exists(self):
        module, alerts = self.build(simulate_irregularity=True)

        self.assertTrue(module.execute())
        self.assertEqual(len(alerts.get_alerts()), 1)

    def test_execute_does_not_register_alert_without_irregularity(self):
        module, alerts = self.build(simulate_irregularity=False)

        self.assertTrue(module.execute())
        self.assertEqual(alerts.get_alerts(), [])

    def test_execute_fails_when_no_map_is_available(self):
        module, alerts = self.build(simulate_irregularity=True, with_map=False)

        self.assertFalse(module.execute())
        self.assertEqual(alerts.get_alerts(), [])


class FakeStage(PipelineStage):
    def __init__(self, name, execution_log, succeeds=True, raises=False):
        self.name = name
        self.execution_log = execution_log
        self.succeeds = succeeds
        self.raises = raises

    def execute(self):
        self.execution_log.append(self.name)
        if self.raises:
            raise RuntimeError("falha simulada")
        return self.succeeds


class TestPipelineManager(unittest.TestCase):
    def test_executes_stages_in_the_given_order(self):
        log = []
        manager = PipelineManager([FakeStage(n, log) for n in ("a", "b", "c", "d")])

        self.assertTrue(manager.execute_pipeline())
        self.assertEqual(log, ["a", "b", "c", "d"])
        self.assertIsNone(manager.failed_stage)

    def test_stops_at_first_failed_stage(self):
        log = []
        manager = PipelineManager(
            [FakeStage("a", log), FakeStage("b", log, succeeds=False), FakeStage("c", log)]
        )

        self.assertFalse(manager.execute_pipeline())
        self.assertEqual(log, ["a", "b"])
        self.assertEqual(manager.failed_stage, "FakeStage")

    def test_treats_unexpected_exception_as_failure(self):
        log = []
        manager = PipelineManager([FakeStage("a", log, raises=True), FakeStage("b", log)])

        with self.assertLogs("pimucs.ia_processing.pipeline_manager", level="ERROR"):
            result = manager.execute_pipeline()

        self.assertFalse(result)
        self.assertEqual(log, ["a"])


class FakePipelineManager:
    def __init__(self, result=True):
        self.called = False
        self.result = result

    def execute_pipeline(self):
        self.called = True
        return self.result


class TestIngestionScheduler(unittest.TestCase):
    def test_starts_pipeline_when_new_data_exists(self):
        pipeline = FakePipelineManager()
        scheduler = IngestionScheduler(ExternalDataRepository(has_new_data=True), pipeline)

        self.assertTrue(scheduler.check_for_new_data())
        self.assertTrue(pipeline.called)

    def test_does_not_start_pipeline_when_no_new_data_exists(self):
        pipeline = FakePipelineManager()
        scheduler = IngestionScheduler(ExternalDataRepository(has_new_data=False), pipeline)

        self.assertFalse(scheduler.check_for_new_data())
        self.assertFalse(pipeline.called)

    def test_reports_failure_when_pipeline_fails(self):
        pipeline = FakePipelineManager(result=False)
        scheduler = IngestionScheduler(ExternalDataRepository(has_new_data=True), pipeline)

        self.assertFalse(scheduler.check_for_new_data())
        self.assertTrue(pipeline.called)


class TestControlledAccess(unittest.TestCase):
    def test_view_alerts_returns_alerts_registered_in_alert_management(self):
        alert_management = AlertManagement()
        alert = IrregularityAlert(description="teste")
        alert_management.register_alert(alert)

        access = ControlledAccess(alert_management)

        self.assertEqual(access.view_alerts(), [alert])


if __name__ == "__main__":
    unittest.main()
