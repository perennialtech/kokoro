import json

import pytest

from kokoro.telemetry import (InMemoryMetrics, InMemoryTraceSink,
                              JsonlTraceSink, NoOpProfileContext,
                              ProfilerConfig, Telemetry, normalize_voice_label)


def test_span_records_cpu_duration():
    sink = InMemoryTraceSink()
    telemetry = Telemetry(ProfilerConfig(enabled=True), [sink])
    request = telemetry.start_request(language="a", voice="af_heart", precision="fp32")

    with request.span("unit.stage"):
        pass

    request.finalize("ok")
    assert sink.traces[-1].stages[0].name == "unit.stage"
    assert sink.traces[-1].stages[0].cpu_ms >= 0


def test_nested_spans_keep_parent_names():
    telemetry = Telemetry(ProfilerConfig(enabled=True))
    request = telemetry.start_request(language="a", voice="af_heart", precision="fp32")

    with request.span("outer"):
        with request.span("inner"):
            pass

    request.finalize("ok")
    stages = request.trace.stages
    assert stages[0].name == "inner"
    assert stages[0].parent == "outer"
    assert stages[1].name == "outer"
    assert stages[1].parent is None


def test_exception_records_stage_and_request_error():
    telemetry = Telemetry(ProfilerConfig(enabled=True))
    request = telemetry.start_request(language="a", voice="af_heart", precision="fp32")

    with pytest.raises(ValueError):
        with request.span("boom"):
            raise ValueError("bad")

    request.finalize("error", ValueError("bad"))
    assert request.trace.status == "error"
    assert request.trace.stages[0].error_type == "ValueError"


def test_jsonl_sink_writes_valid_schema(tmp_path):
    path = tmp_path / "trace.jsonl"
    telemetry = Telemetry(ProfilerConfig(enabled=True), [JsonlTraceSink(path)])
    request = telemetry.start_request(language="a", voice="af_heart", precision="fp32")
    with request.span("x"):
        pass
    request.finalize("ok")

    payload = json.loads(path.read_text().splitlines()[0])
    assert payload["schema_version"] == 1
    assert payload["kind"] == "request"
    assert payload["stages"][0]["name"] == "x"


def test_in_memory_metrics_receives_counters_and_histograms():
    metrics = InMemoryMetrics()
    telemetry = Telemetry(ProfilerConfig(enabled=True), metrics=metrics)
    request = telemetry.start_request(language="a", voice="af_heart", precision="fp32")
    with request.span("x"):
        pass
    request.finalize("ok")

    assert any(key[0] == "requests_total" for key in metrics.counters)
    assert any(key[0] == "stage_latency_seconds" for key in metrics.histograms)


def test_voice_label_normalization_does_not_leak_local_paths():
    assert normalize_voice_label("/tmp/private/voice.pt") == ("external", "local_file")
    assert normalize_voice_label("af_heart") == ("af_heart", "artifact")
    assert normalize_voice_label("af_heart,bf_voice") == ("mixed", "mixed")


def test_disabled_telemetry_is_noop():
    telemetry = Telemetry()
    request = telemetry.start_request(language="a", voice="af_heart")
    assert isinstance(request, NoOpProfileContext)
    with request.span("not-recorded"):
        pass
    request.finalize("ok")
    assert telemetry.last_request_trace is None


def test_generator_cancellation_marks_request_cancelled():
    telemetry = Telemetry(ProfilerConfig(enabled=True))

    def generator():
        request = telemetry.start_request(language="a", voice="af_heart")
        status = "cancelled"
        try:
            yield 1
            status = "ok"
        finally:
            request.finalize(status)

    gen = generator()
    assert next(gen) == 1
    gen.close()
    assert telemetry.last_request_trace.status == "cancelled"
