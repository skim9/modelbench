import dataclasses
import inspect
import json
import pathlib
import random
import sys
import threading
import time
import traceback
from abc import abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Mapping, Iterable, Sequence, List, Optional, Any

import diskcache
from modelgauge.annotation import Annotation
from modelgauge.annotator import CompletionAnnotator
from modelgauge.base_test import PromptResponseTest, TestResult
from modelgauge.dependency_helper import FromSourceDependencyHelper
from modelgauge.pipeline import Source, Pipe, Sink, Pipeline, NullCache
from modelgauge.records import TestRecord
from modelgauge.single_turn_prompt_response import (
    TestItem,
    PromptWithContext,
    MeasuredTestItem,
    TestItemAnnotations,
    PromptInteractionAnnotations,
    SUTResponseAnnotations,
    SUTCompletionAnnotations,
)
from modelgauge.sut import SUTResponse, SUTCompletion
from tqdm import tqdm

from modelbench.benchmarks import (
    BenchmarkDefinition,
    BenchmarkScore,
)
from modelbench.record import BenchmarkScoreEncoder
from modelbench.suts import ModelGaugeSut


class Timer:
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.elapsed = self.end - self.start


class RunJournal:

    def __init__(self, output_file=None):
        super().__init__()
        self.output_file = output_file
        if output_file:
            self.filehandle = open(output_file, "w")
        else:
            self.filehandle = sys.stderr
        self.output_lock = threading.Lock()

    def add_entry(self, message, dimensions=None, **kwargs):
        entry = {"timestamp": self._timestamp(), "message": message}
        if dimensions:
            entry["dimensions"] = dimensions
        calling_frame = inspect.currentframe().f_back
        if calling_frame:
            if "self" in calling_frame.f_locals:
                entry["class"] = calling_frame.f_locals["self"].__class__.__name__
                entry["method"] = calling_frame.f_code.co_name
            else:
                entry["function"] = calling_frame.f_code.co_name

        entry.update(kwargs)
        self._output(entry)

    def _output(self, entry):
        with self.output_lock:
            print(json.dumps(entry, cls=BenchmarkScoreEncoder), file=self.filehandle)

    def _timestamp(self):
        return datetime.now(timezone.utc).isoformat()

    def close(self):
        if self.output_file:
            self.filehandle.close()


class RunTracker:
    """
    A base class to encapsulate run tracking. Lets you limit update frequency to minimize output noise.
    To subclass, the minimum is implementing _on_update. If you want no output, just use the
    NullRunTracker.
    """

    def __init__(self, seconds_per_update: float = 1.0):
        super().__init__()
        self.seconds_per_update = seconds_per_update
        self.last_update = 0
        self.total_items = 0

    def start(self, total_items: int):
        self.total_items = total_items

    def update(self, finished_items: int):
        if self._now() > self.seconds_per_update + self.last_update:
            self._on_update(finished_items)
            self.last_update = self._now()

    def done(self):
        self._on_update(self.total_items)

    @abstractmethod
    def _on_update(self, finished_items: int):
        pass

    def _now(self):
        return time.time()


class NullRunTracker(RunTracker):

    def _on_update(self, finished_items: int):
        pass


class TqdmRunTracker(RunTracker):

    def start(self, total_items: int):
        super().start(total_items)
        self.pbar = tqdm(total=self.total_items, unit="items")
        self.previous_count = 0

    def _on_update(self, finished_items: int):
        self.pbar.update(finished_items - self.previous_count)
        self.previous_count = finished_items

    def done(self):
        super().done()
        self.pbar.close()


class JsonRunTracker(RunTracker):

    def start(self, total_items: int):
        super().start(total_items)
        self._on_update(0)

    def _on_update(self, finished_items: int):
        print(json.dumps({"progress": finished_items / self.total_items}), file=sys.stderr)


class ModelgaugeTestWrapper:
    """An attempt at cleaning up the test interface"""

    def __init__(self, actual_test: PromptResponseTest, dependency_data_path):
        super().__init__()
        self.actual_test = actual_test
        self.uid = actual_test.uid
        self.dependency_data_path = dependency_data_path
        self.dependency_helper = FromSourceDependencyHelper(
            self.dependency_data_path, self.actual_test.get_dependencies(), required_versions={}
        )

    def make_test_items(self):
        return self.actual_test.make_test_items(self.dependency_helper)

    def __hash__(self):
        return self.uid.__hash__()

    def get_annotators(self) -> Mapping[str, CompletionAnnotator]:
        return self.actual_test.get_annotators()

    def measure_quality(self, item: "TestRunItem"):
        annotations = SUTCompletionAnnotations(
            completion=item.sut_response.completions[0],
            annotations={k: Annotation.from_instance(v) for k, v in item.annotations.items()},
        )
        a = PromptInteractionAnnotations(
            prompt=item.test_item.prompts[0],
            response=SUTResponseAnnotations(completions=[annotations]),
        )
        measurement = self.actual_test.measure_quality(TestItemAnnotations(test_item=item.test_item, interactions=[a]))
        item.add_measurement(measurement)

    def aggregate_measurements(self, items: List["TestRunItem"]):
        mtis = []
        for i in items:
            mti = MeasuredTestItem(test_item=i.test_item, measurements=i.measurements)
            mtis.append(mti)
        return self.actual_test.aggregate_measurements(mtis)

    @property
    def initialization_record(self):
        return self.actual_test.initialization_record


@dataclass
class TestRunItem:
    """The data related to running a single test item"""

    test: ModelgaugeTestWrapper
    test_item: TestItem
    sut: ModelGaugeSut = None
    sut_response: SUTResponse = None
    annotations: dict[str, Annotation] = dataclasses.field(default_factory=dict)
    measurements: dict = dataclasses.field(default_factory=dict)
    exception = None

    def prompt_with_context(self) -> PromptWithContext:
        return self.test_item.prompts[0]

    def completion(self) -> SUTCompletion:
        if self.sut_response and self.sut_response.completions:
            return self.sut_response.completions[0]

    def add_measurement(self, measurement: dict):
        self.measurements.update(measurement)

    def source_id(self):
        return self.prompt_with_context().source_id


class TestRunBase:
    tests: list[ModelgaugeTestWrapper]

    def __init__(self, runner: "TestRunnerBase"):
        super().__init__()

        # copy the configuration state
        self.pipeline_segments = []
        self.test_data_path = runner.data_dir / "tests"
        self.secrets = runner.secrets
        self.suts = runner.suts
        self.max_items = runner.max_items
        self.tests = []
        self._test_lookup = {}
        self.run_tracker = runner.run_tracker

        # initialize
        runner.data_dir.mkdir(exist_ok=True)
        self.run_id = datetime.now().strftime("run-%Y%m%d-%H%M%S-%f")
        self.completed_item_count = 0
        self.journal = RunJournal(runner.data_dir / f"journal-{self.run_id}.jsonl")

        # set up for result collection
        self.finished_items = defaultdict(lambda: defaultdict(lambda: list()))
        self.failed_items = defaultdict(lambda: defaultdict(lambda: list()))
        self.test_records = defaultdict(dict)

    def add_test(self, test: PromptResponseTest):
        wrapped = ModelgaugeTestWrapper(test, self.test_data_path)
        self.tests.append(wrapped)
        self._test_lookup[test] = wrapped

    def add_finished_item(self, item: "TestRunItem"):
        if item.completion() and item.annotations and not item.exception:
            self.finished_items[item.sut.key][item.test.uid].append(item)
        else:
            self.failed_items[item.sut.key][item.test.uid].append(item)
        self.completed_item_count += 1

    def add_test_record(self, test_record: TestRecord):
        self.test_records[test_record.test_uid][test_record.sut_uid] = test_record

    def finished_items_for(self, sut, test) -> Sequence[TestItem]:
        return self.finished_items[sut.key][test.uid]

    def failed_items_for(self, sut, test) -> Sequence[TestItem]:
        return self.failed_items[sut.key][test.uid]


class TestRun(TestRunBase):

    def __init__(self, runner: "TestRunner"):
        super().__init__(runner)
        # copy the starting state
        for test in runner.tests:
            self.add_test(test)


class BenchmarkRun(TestRunBase):
    benchmark_scores: dict[BenchmarkDefinition, dict[ModelGaugeSut, BenchmarkScore]]
    benchmarks: Sequence[BenchmarkDefinition]

    def __init__(self, runner: "BenchmarkRunner"):
        super().__init__(runner)
        self.benchmarks = runner.benchmarks
        self.benchmark_scores = defaultdict(dict)

        for b in self.benchmarks:
            for h in b.hazards():
                for t in h.tests(self.secrets):
                    self.add_test(t)


class IntermediateCachingPipe(Pipe):
    """
    Unlike CachingPipe, which caches the final result of this stage,
    this just makes a cache available for internal use to cache intermediate results.
    """

    def __init__(self, thread_count=1, cache_path=None):
        super().__init__(thread_count)

        if cache_path:
            self.cache = diskcache.Cache(cache_path).__enter__()
        else:
            self.cache = NullCache()

    def handle_item(self, item) -> Optional[Any]:
        pass

    def join(self):
        super().join()
        self.cache.__exit__(None, None, None)


class TestRunItemSource(Source):

    def __init__(self, run: TestRunBase):
        super().__init__()
        self.test_run = run

    def new_item_iterable(self) -> Iterable[TestRunItem]:
        for t in self.test_run.tests:
            items = t.make_test_items()
            self.test_run.journal.add_entry("loaded test items", item_count=len(items), test=t.uid)
            items = self.limit_to_max(items, self.test_run.max_items)
            self.test_run.journal.add_entry("test items for run", item_count=len(items), test=t.uid)
            for item in items:
                yield TestRunItem(t, item)

    def limit_to_max(self, items: list, max_items: int):
        if max_items is not None:
            assert max_items > 0, f"invalid max_items: {max_items}"
            if max_items < len(items):
                rng = random.Random()
                rng.seed(0)
                rng.shuffle(items)
                return items[:max_items]
        return items


class TestRunSutAssigner(Pipe):
    def __init__(self, test_run: TestRunBase):
        super().__init__()
        self.test_run = test_run

    def handle_item(self, item: TestRunItem):
        self.test_run.journal.add_entry(
            "queueing item",
            source_id=item.source_id(),
            prompt=item.prompt_with_context().prompt.text,
        )
        for sut in self.test_run.suts:
            self.downstream_put(TestRunItem(item.test, item.test_item, sut))


class TestRunSutWorker(IntermediateCachingPipe):

    def __init__(self, test_run: TestRunBase, thread_count=1, cache_path=None):
        super().__init__(thread_count, cache_path=cache_path)
        self.test_run = test_run

    def handle_item(self, item: TestRunItem):
        mg_sut = item.sut.instance(self.test_run.secrets)
        raw_request = mg_sut.translate_text_prompt(item.prompt_with_context().prompt)
        cache_key = raw_request.model_dump_json(exclude_none=True)
        self._debug(f"looking for {cache_key} in cache")
        try:
            if cache_key in self.cache:
                self._debug(f"cache entry found")
                raw_response = self.cache[cache_key]
                self.test_run.journal.add_entry(
                    "using cached sut response",
                    test=item.test.uid,
                    source_id=item.source_id(),
                    sut=item.sut.uid,
                    response=raw_response,
                )

            else:
                self._debug(f"cache entry not found; processing and saving")
                with Timer() as timer:
                    raw_response = mg_sut.evaluate(raw_request)
                self.cache[cache_key] = raw_response
                self.test_run.journal.add_entry(
                    "fetched new sut response",
                    test=item.test.uid,
                    source_id=item.source_id(),
                    sut=item.sut.uid,
                    run_time=timer.elapsed,
                    response=raw_response,
                )

            response = mg_sut.translate_response(raw_request, raw_response)
            item.sut_response = response
            self.test_run.journal.add_entry(
                "sut response translated",
                test=item.test.uid,
                source_id=item.source_id(),
                sut=item.sut.uid,
                response=response,
            )

        except Exception as e:
            item.exception = e
            print(traceback.format_exc(), file=sys.stderr)
        return item


class TestRunAnnotationWorker(IntermediateCachingPipe):

    def __init__(self, test_run: TestRunBase, thread_count=1, cache_path=None):
        super().__init__(thread_count, cache_path=cache_path)
        self.test_run = test_run

    def handle_item(self, item: TestRunItem) -> TestRunItem:
        try:
            if item.completion():
                self.collect_annotations(item)
        except Exception as e:
            item.exception = e
            print(traceback.format_exc(), file=sys.stderr)

        return item

    def collect_annotations(self, item):
        for annotator_key, annotator in item.test.get_annotators().items():
            annotator_request = annotator.translate_request(item.prompt_with_context(), item.completion())
            cache_key = annotator_request.model_dump_json(exclude_none=True)
            self._debug(f"looking for {cache_key} in cache")
            if cache_key in self.cache:
                self._debug(f"cache entry found")
                annotator_response = self.cache[cache_key]
                self.test_run.journal.add_entry(
                    "using cached annotation response",
                    test=item.test.uid,
                    source_id=item.source_id(),
                    sut=item.sut.uid,
                    annotator=annotator_key,
                    response=annotator_response,
                )

            else:
                self._debug(f"cache entry not found; processing and saving")
                with Timer() as timer:
                    annotator_response = annotator.annotate(annotator_request)
                self.cache[cache_key] = annotator_response
                self.test_run.journal.add_entry(
                    "fetched new annotation response",
                    test=item.test.uid,
                    source_id=item.source_id(),
                    sut=item.sut.uid,
                    annotator=annotator_key,
                    run_time=timer.elapsed,
                    response=annotator_response,
                )

            annotation = annotator.translate_response(annotator_request, annotator_response)
            item.annotations[annotator_key] = annotation
            self.test_run.journal.add_entry(
                "annotation translated",
                test=item.test.uid,
                source_id=item.source_id(),
                sut=item.sut.uid,
                annotator=annotator_key,
                annotation=annotation,
            )
        item.test.measure_quality(item)
        self.test_run.journal.add_entry(
            "item quality measured",
            test=item.test.uid,
            source_id=item.source_id(),
            sut=item.sut.uid,
            measurements=item.measurements,
        )


class TestRunResultsCollector(Sink):

    def __init__(self, test_run: TestRunBase):
        super().__init__()
        self.test_run = test_run

    def handle_item(self, item) -> None:
        self.test_run.add_finished_item(item)
        self.test_run.run_tracker.update(self.test_run.completed_item_count)


class TestRunnerBase:
    def __init__(self, data_dir: pathlib.Path):
        self.debug = False
        self.data_dir = data_dir
        self.secrets = None
        self.suts = []
        self.max_items = 10
        self.thread_count = 1
        self.run_tracker = NullRunTracker()

    def add_sut(self, sut: ModelGaugeSut):
        self.suts.append(sut)

    def run(self) -> TestRunBase:
        self._check_ready_to_run()  # common
        test_run = self._prepare_to_run()
        pipeline = self._build_pipeline(test_run)  # common
        test_run.run_tracker.start(self._expected_item_count(test_run, pipeline))  # common
        pipeline.run()  # common
        test_run.journal.add_entry("pipeline complete", run_id=test_run.run_id)  # common

        self._calculate_test_results(test_run)  # common

        self._calculate_results(test_run)
        test_run.run_tracker.done()  # common
        test_run.journal.add_entry("finished run", run_id=test_run.run_id)  # common
        test_run.journal.close()  # common
        return test_run  # common

    @abstractmethod
    def _prepare_to_run(self):
        pass

    @abstractmethod
    def _calculate_results(self, benchmark_run):
        pass

    def _check_ready_to_run(self):
        if not self.secrets:
            raise ValueError("must set secrets")

        if not self.suts:
            raise ValueError("must call add_sut() at least once")

    def _calculate_test_results(self, test_run):
        for sut in test_run.suts:
            for test in test_run.tests:
                test_result = test.aggregate_measurements(test_run.finished_items_for(sut, test))
                test_record = self._make_test_record(test_run, sut, test, test_result)
                test_run.add_test_record(test_record)
                test_run.journal.add_entry("test result calculated", sut=sut.uid, test=test.uid, result=test_result)

    def _make_test_record(self, run, sut, test, test_result):
        return TestRecord(
            test_uid=test.uid,
            test_initialization=test.initialization_record,
            dependency_versions=test.dependency_helper.versions_used(),
            sut_uid=sut._instance.uid,
            sut_initialization=sut._instance.initialization_record,
            test_item_records=[],
            test_item_exceptions=[],
            result=TestResult.from_instance(test_result),
        )

    def _build_pipeline(self, run):
        run.pipeline_segments.append(TestRunItemSource(run))
        run.pipeline_segments.append(TestRunSutAssigner(run))
        run.pipeline_segments.append(
            TestRunSutWorker(run, thread_count=self.thread_count, cache_path=self.data_dir / "sut_cache")
        )
        run.pipeline_segments.append(
            TestRunAnnotationWorker(run, thread_count=self.thread_count, cache_path=self.data_dir / "annotator_cache")
        )
        run.pipeline_segments.append(TestRunResultsCollector(run))
        pipeline = Pipeline(
            *run.pipeline_segments,
            # progress_callback=progress_callback,
            debug=self.debug,
        )
        return pipeline

    def _expected_item_count(self, the_run: TestRunBase, pipeline: Pipeline):
        return len(the_run.suts) * len(list(pipeline.source.new_item_iterable()))


class TestRunner(TestRunnerBase):

    def __init__(self, data_dir: pathlib.Path):
        super().__init__(data_dir)
        self.tests = []

    def add_test(self, test: PromptResponseTest):
        self.tests.append(test)

    def run(self) -> TestRun:
        return super().run()

    def _check_ready_to_run(self):
        super()._check_ready_to_run()
        if not self.tests:
            raise ValueError("must call add_test() at least once")

    def _prepare_to_run(self):
        test_run = TestRun(self)
        test_run.journal.add_entry(
            "starting run",
            run_id=test_run.run_id,
            tests=[t.uid for t in self.tests],
            suts=[s.uid for s in self.suts],
            max_items=self.max_items,
            thread_count=self.thread_count,
        )
        return test_run


class BenchmarkRunner(TestRunnerBase):

    def __init__(self, data_dir: pathlib.Path):
        super().__init__(data_dir)
        self.benchmarks = []

    def add_benchmark(self, benchmark: BenchmarkDefinition):
        self.benchmarks.append(benchmark)

    def run(self) -> BenchmarkRun:
        return super().run()

    def _check_ready_to_run(self):
        super()._check_ready_to_run()
        if not self.benchmarks:
            raise ValueError("must call add_benchmark() at least once")

    def _prepare_to_run(self):
        benchmark_run = BenchmarkRun(self)
        benchmark_run.journal.add_entry(
            "starting run",
            run_id=benchmark_run.run_id,
            benchmarks=[b.uid for b in self.benchmarks],
            suts=[s.uid for s in self.suts],
            max_items=self.max_items,
            thread_count=self.thread_count,
        )  # unique
        return benchmark_run

    def _calculate_results(self, benchmark_run):
        self._calculate_benchmark_scores(benchmark_run)  # unique

    # TODO restructure run to move common items up

    def _calculate_benchmark_scores(self, benchmark_run):
        for benchmark_definition in benchmark_run.benchmarks:
            for sut in benchmark_run.suts:
                hazard_scores = []
                for hazard in benchmark_definition.hazards():
                    test_records = {}
                    for test in hazard.tests(benchmark_run.secrets):
                        records = benchmark_run.test_records[test.uid][sut.uid]
                        assert records, f"No records found for {benchmark_definition} {sut} {hazard} {test.uid}"
                        test_records[test.uid] = records

                    assert test_records, f"No records found for {benchmark_definition} {sut} {hazard}"

                    hazard_score = hazard.score(test_records)  # TODO: score needs way less
                    hazard_scores.append(hazard_score)
                    try:
                        text_grade = hazard_score.text_grade()
                    except ValueError:
                        # for uncalibrated models
                        text_grade = None
                    benchmark_run.journal.add_entry(
                        "hazard scored",
                        sut=sut.uid,
                        hazard=hazard.uid,
                        score=hazard_score.score.estimate,
                        samples=hazard_score.score.samples,
                        text_grade=text_grade,
                    )

                benchmark_score = BenchmarkScore(benchmark_definition, sut, hazard_scores, end_time=datetime.now())
                try:
                    text_grade = benchmark_score.text_grade()
                except ValueError:
                    text_grade = None
                benchmark_run.journal.add_entry(
                    "benchmark scored",
                    sut=sut.uid,
                    benchmark=benchmark_definition.uid,
                    text_grade=text_grade,
                )

                benchmark_run.benchmark_scores[benchmark_definition][sut] = benchmark_score
