import inspect
from typing import Dict, Mapping, List
from unittest.mock import MagicMock

import pytest

from modelbench.benchmark_runner import *
from modelbench.cache import InMemoryCache
from modelbench.hazards import HazardDefinition, HazardScore
from modelbench.scoring import ValueEstimate
from modelbench.suts import ModelGaugeSut
from modelbench_tests.test_run_journal import FakeJournal, reader_for
from modelgauge.annotators.demo_annotator import DemoYBadAnnotation, DemoYBadResponse
from modelgauge.annotators.llama_guard_annotator import LlamaGuardAnnotation
from modelgauge.dependency_helper import DependencyHelper
from modelgauge.external_data import ExternalData
from modelgauge.load_plugins import load_plugins
from modelgauge.prompt import TextPrompt
from modelgauge.record_init import InitializationRecord
from modelgauge.secret_values import RawSecrets, get_all_secrets
from modelgauge.single_turn_prompt_response import TestItemAnnotations, MeasuredTestItem, PromptWithContext
from modelgauge.sut import SUTCompletion, SUTResponse
from modelgauge.suts.demo_01_yes_no_sut import DemoYesNoResponse
from modelgauge.suts.together_client import TogetherChatRequest, TogetherChatResponse
from modelgauge_tests.fake_annotator import FakeAnnotator

# fix pytest autodiscovery issue; see https://github.com/pytest-dev/pytest/issues/12749
for a_class in [i[1] for i in (globals().items()) if inspect.isclass(i[1])]:
    if a_class.__name__.startswith("Test"):
        a_class.__test__ = False


def fake_all_secrets(value="some-value") -> RawSecrets:
    secrets = get_all_secrets()
    raw_secrets: Dict[str, Dict[str, str]] = {}
    for secret in secrets:
        if secret.scope not in raw_secrets:
            raw_secrets[secret.scope] = {}
        raw_secrets[secret.scope][secret.key] = value
    return raw_secrets


class AFakeTest(PromptResponseTest):
    def __init__(self, uid: str, items, annotators=["demo_annotator"]):
        super().__init__(uid)
        self.items = items
        self.initialization_record = InitializationRecord(module="testing", class_name="a_test", args=[], kwargs={})
        self.annotator_uids = annotators

    def get_dependencies(self) -> Mapping[str, ExternalData]:
        pass

    def make_test_items(self, dependency_helper: DependencyHelper) -> List[TestItem]:
        return self.items

    def get_annotators(self) -> List[str]:
        return self.annotator_uids

    def measure_quality(self, item: TestItemAnnotations) -> Dict[str, float]:
        completion = item.interactions[0].response.completions[0]
        if completion.annotations:
            return {"badness": float(completion.get_annotation(self.annotator_uids[0], DemoYBadAnnotation).badness)}
        else:
            return {}

    def aggregate_measurements(self, items: List[MeasuredTestItem]):
        total_badness = 0
        badness_count = 0
        for item in items:
            total_badness += item.measurements["badness"]
            badness_count += 1
        return {"total_badness": total_badness, "badness_count": badness_count}


class AHazard(HazardDefinition):
    def __init__(self, test_list):
        super().__init__()
        self.test_list = test_list

    def reference_standard(self) -> float:
        return 0.5

    def tests(self, secrets: RawSecrets) -> List[PromptResponseTest]:
        return self.test_list

    def score(self, sut_scores: Mapping[str, TestRecord]) -> HazardScore:
        count = 0
        total = 0
        for key, value in sut_scores.items():
            d = value.result.data
            count += d["badness_count"]
            total += d["total_badness"]

        score = ValueEstimate.make(total / count, count)

        test_scores = {}
        return HazardScore(hazard_definition=self, score=score, test_scores=test_scores, exceptions=0)


class RunnerTestBase:
    _original_registered_annotators = None

    @classmethod
    def setup_class(cls):
        cls._original_registered_annotators = [uid for uid, _ in ANNOTATORS.items()]

        class FakeExplodingAnnotator(FakeAnnotator):
            def annotate(self, annotation_request):
                raise ValueError("annotator done broke")

        ANNOTATORS.register(FakeExplodingAnnotator, "fake_exploding_annotator")
        ANNOTATORS.register(FakeAnnotator, "fake_annotator_1")
        ANNOTATORS.register(FakeAnnotator, "fake_annotator_2")

    @classmethod
    def teardown_class(cls):
        annotator_uids = [uid for uid, _ in ANNOTATORS.items()]
        for uid in annotator_uids:
            if uid not in cls._original_registered_annotators:
                del ANNOTATORS._lookup[uid]
        cls._original_registered_annotators = None

    @pytest.fixture(scope="class", autouse=True)
    def load_plugins(self):
        load_plugins()

    def a_run(self, tmp_path, **kwargs) -> BenchmarkRun:
        runner = BenchmarkRunner(tmp_path / "run")
        for key, value in kwargs.items():
            runner.__dict__[key] = value
        if runner.secrets is None:
            runner.secrets = fake_all_secrets()
        return BenchmarkRun(runner)

    @pytest.fixture()
    def benchmark(self, a_test):
        class ABenchmark(BenchmarkDefinition):
            def _make_hazards(self) -> Sequence[HazardDefinition]:
                return [AHazard([a_test])]

            _uid_definition = {"name": "a_benchmark", "version": "1.0"}

        return ABenchmark()

    @pytest.fixture()
    def item_from_test(self):
        return self.make_test_item()

    def make_test_item(self, text="Hello!", source_id="hello"):
        return TestItem(prompts=[PromptWithContext(prompt=TextPrompt(text=text), source_id=source_id)])

    @pytest.fixture()
    def a_test(self, item_from_test):
        return AFakeTest("a_test", [item_from_test])

    @pytest.fixture()
    def a_wrapped_test(self, a_test, tmp_path):
        return ModelgaugeTestWrapper(a_test, tmp_path)

    @pytest.fixture()
    def a_sut(self):
        return ModelGaugeSut("demo_yes_no")

    @pytest.fixture()
    def exploding_sut(self, a_sut):
        real_sut = MagicMock()
        real_sut.evaluate.side_effect = ValueError("sut done broke")
        a_sut.instance = lambda _: real_sut
        return a_sut

    @pytest.fixture()
    def sut_response(self):
        return SUTResponse(completions=[SUTCompletion(text="Hello, is it me you're looking for?")])

    @pytest.fixture()
    def exploding_wrapped_test(self, item_from_test, tmp_path):
        raw_test = AFakeTest("a_test", [item_from_test], annotators=["fake_exploding_annotator"])
        return ModelgaugeTestWrapper(raw_test, tmp_path)


class TestRunners(RunnerTestBase):

    def a_test_run(self, tmp_path, **kwargs) -> TestRun:
        runner = TestRunner(tmp_path / "run")
        for key, value in kwargs.items():
            runner.__dict__[key] = value
        if runner.secrets is None:
            runner.secrets = fake_all_secrets()
        return TestRun(runner)

    @pytest.fixture()
    def hazard(self):
        pass

    def test_test_run_loads_annotators(self, tmp_path, item_from_test, benchmark):
        test_1 = AFakeTest("test_1", [item_from_test], annotators=["fake_annotator_1"])
        test_2 = AFakeTest("test_2", [item_from_test], annotators=["fake_annotator_1", "fake_annotator_2"])

        class BenchmarkMultipleTests(BenchmarkDefinition):
            def _make_hazards(self) -> Sequence[HazardDefinition]:
                return [AHazard([test_1, test_2])]

        runner = BenchmarkRunner(tmp_path / "run")
        runner.add_benchmark(BenchmarkMultipleTests())
        run = BenchmarkRun(runner)

        assert len(run.test_annotators) == 2

        test_1_annotators = run.test_annotators["test_1"]
        assert len(test_1_annotators) == 1
        assert isinstance(test_1_annotators[0], FakeAnnotator)
        assert test_1_annotators[0].uid == "fake_annotator_1"

        test_2_annotators = run.test_annotators["test_2"]
        assert len(test_2_annotators) == 2
        assert all(isinstance(annotator, FakeAnnotator) for annotator in test_2_annotators)
        assert {annotator.uid for annotator in test_2_annotators} == {"fake_annotator_1", "fake_annotator_2"}

    def test_test_run_items_properly_isolated(self, a_wrapped_test):
        a = TestRunItem(a_wrapped_test, self.make_test_item("one", "1"))
        b = TestRunItem(a_wrapped_test, self.make_test_item("two", "2"))

        a.annotations["llamaguard"] = LlamaGuardAnnotation(is_safe=False, is_safe_logprob=0.0, violation_categories=[])
        assert len(a.annotations) == 1
        assert len(b.annotations) == 0

        a.add_measurement({"a": 1.0})
        assert len(a.measurements) == 1
        assert len(b.measurements) == 0

    def test_benchmark_source(self, fake_secrets, tmp_path, benchmark):
        bsa = TestRunItemSource(self.a_run(tmp_path, secrets=fake_secrets, max_items=1, benchmarks=[benchmark]))
        iterator = iter(bsa.new_item_iterable())
        first_item = next(iterator)
        assert isinstance(first_item, TestRunItem)
        assert isinstance(first_item.test_item, TestItem)
        with pytest.raises(StopIteration):
            next(iterator)

    def test_benchmark_sut_assigner(self, a_wrapped_test, tmp_path):
        sut_one = ModelGaugeSut("one")
        sut_two = ModelGaugeSut("two")
        test_item = self.make_test_item()

        bsa = TestRunSutAssigner(self.a_run(tmp_path, suts=[sut_one, sut_two]))
        bsa.handle_item(TestRunItem(a_wrapped_test, test_item))

        assert bsa._queue.qsize() == 2
        item_one = bsa._queue.get()
        assert item_one.test_item == test_item
        assert item_one.sut == sut_one
        item_two = bsa._queue.get()
        assert item_two.test_item == test_item
        assert item_two.sut == sut_two

    def test_benchmark_sut_worker(self, item_from_test, a_wrapped_test, tmp_path, a_sut):
        bsw = TestRunSutWorker(self.a_run(tmp_path, suts=[a_sut]), NullCache())

        result = bsw.handle_item(TestRunItem(a_wrapped_test, item_from_test, a_sut))

        assert result.test_item == item_from_test
        assert result.sut == a_sut
        assert isinstance(result.sut_response, SUTResponse)
        assert result.sut_response.completions[0].text == "No"

    def test_benchmark_sut_worker_throws_exception(
        self, item_from_test, a_wrapped_test, tmp_path, exploding_sut, caplog
    ):
        bsw = TestRunSutWorker(self.a_run(tmp_path, suts=[exploding_sut]), NullCache())

        result = bsw.handle_item(TestRunItem(a_wrapped_test, item_from_test, exploding_sut))

        assert result.test_item == item_from_test
        assert result.sut == exploding_sut
        assert result.sut_response is None
        assert isinstance(result.exceptions[0], ValueError)

        assert "failure" in caplog.text

    def test_benchmark_annotation_worker(
        self, a_wrapped_test, tmp_path, item_from_test, sut_response, a_sut, benchmark
    ):
        baw = TestRunAnnotationWorker(self.a_run(tmp_path, suts=[a_sut], benchmarks=[benchmark]), NullCache())
        pipeline_item = TestRunItem(a_wrapped_test, item_from_test, a_sut, sut_response)

        result = baw.handle_item(pipeline_item)

        assert list(result.annotations.keys()) == ["demo_annotator"]
        assert result.annotations["demo_annotator"].badness == 1.0

    def test_test_annotation_worker(self, a_wrapped_test, tmp_path, item_from_test, sut_response, a_sut, a_test):
        taw = TestRunAnnotationWorker(self.a_test_run(tmp_path, suts=[a_sut], tests=[a_test]), NullCache())
        pipeline_item = TestRunItem(a_wrapped_test, item_from_test, a_sut, sut_response)

        result = taw.handle_item(pipeline_item)

        assert list(result.annotations.keys()) == ["demo_annotator"]
        assert result.annotations["demo_annotator"].badness == 1.0

    def test_benchmark_annotation_worker_ignores_failed(self, a_wrapped_test, tmp_path, item_from_test, a_sut):
        baw = TestRunAnnotationWorker(self.a_run(tmp_path, suts=[a_sut]), NullCache())
        pipeline_item = TestRunItem(a_wrapped_test, item_from_test, a_sut)
        pipeline_item.exceptions.append(ValueError())

        result = baw.handle_item(pipeline_item)

        assert result.annotations == {}

    def test_benchmark_annotation_worker_throws_exception(
        self, exploding_wrapped_test, tmp_path, item_from_test, sut_response, a_sut, caplog
    ):
        run = self.a_run(tmp_path, suts=[a_sut])
        run.add_test(exploding_wrapped_test.actual_test)
        baw = TestRunAnnotationWorker(run, NullCache())
        pipeline_item = TestRunItem(exploding_wrapped_test, item_from_test, a_sut, sut_response)

        result = baw.handle_item(pipeline_item)

        assert result.annotations == {}
        assert len(pipeline_item.exceptions) == 1

        assert "failure" in caplog.text

    def test_benchmark_results_collector(self, a_sut, tmp_path, a_wrapped_test, item_from_test, sut_response):
        run = self.a_run(tmp_path, suts=[a_sut])
        brc = TestRunResultsCollector(run)
        item = TestRunItem(a_wrapped_test, item_from_test, a_sut, sut_response, {"a": MagicMock()})

        brc.handle_item(item)

        assert run.finished_items_for(a_sut, a_wrapped_test) == [item]

    def test_benchmark_results_collector_handles_failed(self, a_sut, tmp_path, a_wrapped_test, item_from_test):
        run = self.a_run(tmp_path, suts=[a_sut])
        brc = TestRunResultsCollector(run)
        item = TestRunItem(a_wrapped_test, item_from_test, a_sut)
        item.exceptions.append(ValueError("yes, this value error"))

        brc.handle_item(item)

        assert run.finished_items_for(a_sut, a_wrapped_test) == []
        assert run.failed_items_for(a_sut, a_wrapped_test) == [item]

    def test_basic_test_run(self, tmp_path, fake_secrets, a_test):
        runner = TestRunner(tmp_path)
        runner.secrets = fake_secrets
        runner.add_test(a_test)
        sut = ModelGaugeSut("demo_yes_no")
        runner.add_sut(sut)
        runner.max_items = 1
        run_result = runner.run()

        assert run_result.test_records
        assert run_result.test_records[a_test.uid][sut.key]

    def test_basic_benchmark_run(self, tmp_path, fake_secrets, benchmark):
        runner = BenchmarkRunner(tmp_path)
        runner.secrets = fake_secrets

        runner.add_benchmark(benchmark)
        sut = ModelGaugeSut("demo_yes_no")
        runner.add_sut(sut)
        runner.max_items = 1
        run_result = runner.run()

        assert run_result.benchmark_scores
        assert run_result.benchmark_scores[benchmark][sut]

    def test_test_runner_has_standards(self, tmp_path, a_test, fake_secrets):
        runner = TestRunner(tmp_path)

        with pytest.raises(ValueError) as e:
            runner.run()
        assert "secrets" in str(e)

        runner.secrets = fake_secrets
        with pytest.raises(ValueError) as e:
            runner.run()
        assert "add_sut" in str(e)

        runner.add_sut(ModelGaugeSut("demo_yes_no"))
        with pytest.raises(ValueError) as e:
            runner.run()
        assert "add_test" in str(e)

        runner.add_test(a_test)
        runner.run()

    def test_benchmark_runner_has_standards(self, tmp_path, benchmark, fake_secrets):
        runner = BenchmarkRunner(tmp_path)
        runner.secrets = fake_secrets
        runner.add_sut(ModelGaugeSut("demo_yes_no"))

        with pytest.raises(ValueError) as e:
            runner.run()
        assert "add_benchmark" in str(e)

        runner.add_benchmark(benchmark)
        runner.run()

    def test_sut_caching(self, item_from_test, a_wrapped_test, tmp_path):
        sut = MagicMock(spec=ModelGaugeSut)
        sut.instance().translate_text_prompt.return_value = TogetherChatRequest(model="foo", messages=[])
        sut.instance().evaluate.return_value = TogetherChatResponse(
            id="foo",
            choices=[],
            usage=TogetherChatResponse.Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
            created=0,
            model="foo",
            object="foo",
        )
        run = self.a_run(tmp_path, suts=[sut])
        bsw = TestRunSutWorker(run, DiskCache(tmp_path))

        bsw.handle_item(TestRunItem(a_wrapped_test, item_from_test, sut))
        assert sut.instance().evaluate.call_count == 1

        bsw.handle_item(TestRunItem(a_wrapped_test, item_from_test, sut))
        assert sut.instance().evaluate.call_count == 1


class TestRunJournaling(RunnerTestBase):

    def a_run(self, tmp_path, **kwargs) -> BenchmarkRun:
        run = super().a_run(tmp_path, **kwargs)
        run.journal = FakeJournal()
        return run

    def test_item_source(self, fake_secrets, tmp_path, benchmark):
        run = self.a_run(tmp_path, secrets=fake_secrets, max_items=1, benchmarks=[benchmark])
        bsa = TestRunItemSource(run)
        iterator = iter(bsa.new_item_iterable())
        next(iterator)
        entry = run.journal.last_entry()
        assert entry["message"] == "using test items"

    def test_benchmark_sut_assigner(self, a_wrapped_test, tmp_path):
        sut_one = ModelGaugeSut("one")
        test_item = self.make_test_item("What's your name?", "id123")
        run = self.a_run(tmp_path, suts=[sut_one])

        bsa = TestRunSutAssigner(run)
        bsa.handle_item(TestRunItem(a_wrapped_test, test_item))

        entry = run.journal.last_entry()
        assert entry["message"] == "queuing item"
        assert entry["prompt_text"] == "What's your name?"
        assert entry["prompt_id"] == "id123"

    def test_benchmark_sut_worker(self, item_from_test, a_wrapped_test, tmp_path, a_sut):
        run = self.a_run(tmp_path, suts=[a_sut])
        bsw = TestRunSutWorker(run, NullCache())

        bsw.handle_item(TestRunItem(a_wrapped_test, item_from_test, a_sut))

        fetch_entry = run.journal.entry(-2)
        assert fetch_entry["message"] == "fetched sut response"
        translation_entry = run.journal.entry(-1)
        assert translation_entry["message"] == "translated sut response"

    def test_benchmark_sut_worker_cached(self, item_from_test, a_wrapped_test, tmp_path, a_sut):
        run = self.a_run(tmp_path, suts=[a_sut])
        cache = InMemoryCache()
        bsw = TestRunSutWorker(run, cache)
        cache['{"text":"Hello!"}'] = DemoYesNoResponse(number_of_words=1, text="No")

        bsw.handle_item(TestRunItem(a_wrapped_test, item_from_test, a_sut))

        fetch_entry = run.journal.entry(-2)
        assert fetch_entry["message"] == "using cached sut response"
        translation_entry = run.journal.entry(-1)
        assert translation_entry["message"] == "translated sut response"

    def test_benchmark_sut_worker_throws_exception(
        self, item_from_test, a_wrapped_test, tmp_path, exploding_sut, capsys
    ):
        run = self.a_run(tmp_path, suts=[exploding_sut])
        bsw = TestRunSutWorker(run, NullCache())

        bsw.handle_item(TestRunItem(a_wrapped_test, item_from_test, exploding_sut))

        entry = run.journal.last_entry()
        assert entry["message"] == "sut exception"

        capsys.readouterr()  # supress the exception output; can remove when we add proper logging

    def test_benchmark_annotation_worker(
        self, a_wrapped_test, tmp_path, item_from_test, sut_response, a_sut, benchmark
    ):
        run = self.a_run(tmp_path, suts=[a_sut], benchmarks=[benchmark])
        baw = TestRunAnnotationWorker(run, NullCache())
        pipeline_item = TestRunItem(a_wrapped_test, item_from_test, a_sut, sut_response)

        baw.handle_item(pipeline_item)

        fetch_entry = run.journal.entry(-3)
        assert fetch_entry["message"] == "fetched annotator response"
        translation_entry = run.journal.entry(-2)
        assert translation_entry["message"] == "translated annotation"
        measurement_entry = run.journal.entry(-1)
        assert measurement_entry["message"] == "measured item quality"

    def test_benchmark_annotation_worker_cached(
        self, a_wrapped_test, tmp_path, item_from_test, sut_response, a_sut, benchmark
    ):
        run = self.a_run(tmp_path, suts=[a_sut], benchmarks=[benchmark])
        cache = InMemoryCache()
        cache['{"text":"Hello, is it me you\'re looking for?"}'] = DemoYBadResponse(score=1.0)
        baw = TestRunAnnotationWorker(run, cache)
        pipeline_item = TestRunItem(a_wrapped_test, item_from_test, a_sut, sut_response)

        baw.handle_item(pipeline_item)

        fetch_entry = run.journal.entry(-3)
        assert fetch_entry["message"] == "using cached annotator response"
        translation_entry = run.journal.entry(-2)
        assert translation_entry["message"] == "translated annotation"
        measurement_entry = run.journal.entry(-1)
        assert measurement_entry["message"] == "measured item quality"

    def test_benchmark_annotation_worker_throws_exception(
        self, exploding_wrapped_test, tmp_path, item_from_test, sut_response, a_sut, capsys
    ):
        run = self.a_run(tmp_path, suts=[a_sut])
        run.add_test(exploding_wrapped_test.actual_test)
        baw = TestRunAnnotationWorker(run, NullCache())
        pipeline_item = TestRunItem(exploding_wrapped_test, item_from_test, a_sut, sut_response)

        baw.handle_item(pipeline_item)

        exception_entry = run.journal.entry(-2)
        assert exception_entry["message"] == "annotator exception"
        assert exception_entry["exception"]["message"] == "annotator done broke"
        measurement_entry = run.journal.entry(-1)
        assert measurement_entry["message"] == "measured item quality"
        assert measurement_entry["measurements"] == {}
        capsys.readouterr()  # supress the exception output; can remove when we add proper logging

    def test_basic_benchmark_run(self, tmp_path, fake_secrets, benchmark):
        runner = BenchmarkRunner(tmp_path)
        runner.secrets = fake_secrets

        runner.add_benchmark(benchmark)
        sut = ModelGaugeSut("demo_yes_no")
        runner.add_sut(sut)
        runner.max_items = 1
        runner.run()
        entries = []
        for l in reader_for(next(tmp_path.glob("**/journal-run*.jsonl.zst"))):
            entries.append(json.loads(l))
        messages = [e["message"] for e in entries]
        # if this gets painful, it should be something like assert contains_in_order(messages, expected_messages)
        # That is, it's important that all of these occur in this order, but it's ok if there are multiple or if
        # other messages are also there.
        assert messages == [
            "starting journal",
            "starting run",
            "test info",
            "running pipeline",
            "using test items",
            "queuing item",
            "fetched sut response",
            "translated sut response",
            "fetched annotator response",
            "translated annotation",
            "measured item quality",
            "finished pipeline",
            "test scored",
            "hazard scored",
            "benchmark scored",
            "finished run",
            "cache info",
            "cache info",
        ]


class TestRunTrackers:
    def test_null(self, capsys):
        t = NullRunTracker()

        t.start(10)
        t.update(5)
        t.done()

        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == ""

    def test_tqdm(self, capsys):
        t = TqdmRunTracker()

        t.start(10)
        t.update(5)
        t.done()

        captured = capsys.readouterr()
        assert captured.out == ""
        assert "  0%|          | 0/10" in captured.err
        assert "100%|██████████| 10/10" in captured.err

    def test_json(self, capsys):
        t = JsonRunTracker()

        t.start(10)
        t.update(5)
        t.done()

        captured = capsys.readouterr()
        assert captured.out == ""
        error_lines = captured.err.strip().split("\n")
        assert len(error_lines) == 3
        assert error_lines[0] == '{"progress": 0.0}'
        assert error_lines[-1] == '{"progress": 1.0}'
