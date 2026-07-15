# kokoro

Native TensorRT Runtime API inference tooling for Kokoro-82M.

This fork has one supported public runtime:

```python
from kokoro import KokoroTRT
```

The public compile API is:

```python
from kokoro import compile_artifact
```

Artifacts contain a native TensorRT serialized plan loaded with `tensorrt.Runtime` and executed through `IExecutionContext.execute_async_v3`. Torch-TensorRT is not used.

## Install

```bash
git clone https://github.com/perennialtech/kokoro.git
cd kokoro
uv sync
```

Install TensorRT Python bindings matching your CUDA stack. ONNX export also requires `onnx` and `onnxscript`:

```bash
uv pip install ".[trt]"
```

You also need `espeak-ng`:

```bash
sudo apt-get install espeak-ng
```

Optional Misaki extras:

```bash
uv pip install "misaki[ja]"
uv pip install "misaki[zh]"
```

## Compile a TensorRT artifact

CLI:

```bash
uv run python -m kokoro.compile \
  --output-dir ./build \
  --repo-id hexgrad/Kokoro-82M \
  --include-voice af_heart
```

Python:

```python
from kokoro import compile_artifact

compile_artifact(
    "./build",
    repo_id="hexgrad/Kokoro-82M",
    precision="fp16",
    include_voices=["af_heart"],
)
```

The artifact contains:

```text
artifact/
  config.json
  host_state.pt
  metadata.json
  generator_with_source_pyramid.plan
  generator_with_source_pyramid.onnx
  voices/
    af_heart.pt
```

TensorRT engines use automatically selected dynamic shape ranges.

Compilation exports the generator boundary to ONNX and parses that ONNX with TensorRT. The compile step requires full TensorRT parser/operator coverage for the generator graph; parser errors are treated as hard failures.

## Native TensorRT inference: Python

```python
import soundfile as sf

from kokoro import KokoroTRT

tts = KokoroTRT("./build")

for i, result in enumerate(
    tts.synthesize(
        text="Hello from Kokoro running through TensorRT.",
        voice="af_heart",
        language="a",
        speed=1.0,
    )
):
    sf.write(f"kokoro_trt_{i}.wav", result.audio.detach().cpu().numpy(), 24000)
```

## Native TensorRT inference: CLI

```bash
uv run kokoro \
  --artifact-dir ./build \
  --voice af_heart \
  --language a \
  --text "Hello from TensorRT Kokoro." \
  --output-file hello.wav
```

Read from file:

```bash
uv run kokoro \
  --artifact-dir ./build \
  --voice af_heart \
  --language a \
  --input-file input.txt \
  --output-file hello.wav
```

Read from stdin:

```bash
uv run kokoro \
  --artifact-dir ./build \
  --voice af_heart \
  --language a \
  --output-file hello.wav < input.txt
```

## Public API

```python
from kokoro import KokoroTRT, compile_artifact
```

## Web UI

A simple Gradio UI is provided in `app.py`. Install `gradio` and run:

```bash
uv pip install gradio
uv run python app.py
```

By default it loads the artifact from `./build`. You can override this using the `KOKORO_TRT_ARTIFACT_DIR` environment variable.

## Profiling and production telemetry

Kokoro TensorRT includes explicit first-class telemetry. It instruments the stable runtime boundaries: frontend preparation, voice loading/cache behavior, host PyTorch stages, native TensorRT execution, postprocessing, CLI file I/O, and benchmark runs.

Python:

```python
from kokoro import KokoroTRT
from kokoro.telemetry import (
    JsonlTraceSink,
    PrometheusMetrics,
    ProfilerConfig,
    Telemetry,
)

metrics = PrometheusMetrics(prefix="kokoro_tts")
metrics.start_http_server(8000)

telemetry = Telemetry(
    profiler_config=ProfilerConfig(
        enabled=True,
        cuda_timing=True,
        synchronize_cuda=True,
        emit_nvtx=False,
    ),
    trace_sinks=[JsonlTraceSink("kokoro-profile.jsonl")],
    metrics=metrics,
)

tts = KokoroTRT("./build", telemetry=telemetry)
results = list(tts.synthesize("Hello from telemetry.", "af_heart", "a"))
print(results[0].profile)
print(tts.telemetry.last_request_trace)
```

Telemetry has two outputs:

- detailed request/chunk traces for debugging and benchmark analysis
- low-cardinality production metrics for continuous scraping

Trace JSONL records use `schema_version = 1` and include stage timing, request/chunk status, shapes, TensorRT byte counts, profile bounds, audio duration, RTF, and errors by exception class. Raw text and phonemes are omitted unless `ProfilerConfig(include_text=True)` is set.

### Submit latency versus ready latency

The runtime returns CUDA tensors. CPU wall time to enqueue work is **submit latency**. Time until GPU work is complete is **ready latency**. Ready latency requires a synchronization boundary.

Set:

```python
ProfilerConfig(cuda_timing=True, synchronize_cuda=True)
```

to record CUDA event elapsed times and ready latency. Synchronization is done once per chunk/request finalization, not after every small span.

### Prometheus metrics

Install metrics support:

```bash
uv pip install ".[metrics]"
```

Start a metrics endpoint:

```python
from kokoro.telemetry import PrometheusMetrics

metrics = PrometheusMetrics(prefix="kokoro_tts")
metrics.start_http_server(8000)
```

Key metric families include:

- `kokoro_tts_requests_total`
- `kokoro_tts_chunks_total`
- `kokoro_tts_errors_total`
- `kokoro_tts_out_of_profile_total`
- `kokoro_tts_stage_latency_seconds`
- `kokoro_tts_stage_cuda_seconds`
- `kokoro_tts_request_submit_latency_seconds`
- `kokoro_tts_request_ready_latency_seconds`
- `kokoro_tts_chunk_submit_latency_seconds`
- `kokoro_tts_chunk_ready_latency_seconds`
- `kokoro_tts_audio_duration_seconds`
- `kokoro_tts_rtf_submit`
- `kokoro_tts_rtf_ready`
- `kokoro_tts_synthesis_frames`
- `kokoro_tts_sample_length`
- `kokoro_tts_trt_input_bytes`
- `kokoro_tts_trt_output_bytes`
- `kokoro_tts_voice_cache_events_total`
- `kokoro_tts_voice_loads_total`
- `kokoro_tts_cuda_memory_allocated_bytes`
- `kokoro_tts_cuda_memory_reserved_bytes`
- `kokoro_tts_cuda_max_memory_allocated_bytes`

Labels intentionally avoid raw text, phonemes, request IDs, and local voice paths. Local `.pt` voices are labeled as `voice_kind="local_file"` with a safe external label; tensor voices are `voice_kind="tensor"`.

### CLI profiling

The `kokoro` CLI supports profiling and metrics flags:

```bash
uv run kokoro \
  --artifact-dir ./build \
  --voice af_heart \
  --language a \
  --text "Hello from profiled Kokoro." \
  --output-file hello.wav \
  --profile \
  --profile-jsonl traces.jsonl \
  --profile-cuda \
  --profile-sync-cuda \
  --profile-nvtx \
  --metrics-prometheus-port 8000
```

CLI-specific stages include:

- `cli.read_input`
- `cli.synthesize`
- `cli.cpu_copy`
- `cli.wav_write`
- `cli.total`

### Gradio telemetry

The app can enable telemetry with environment variables:

```bash
KOKORO_TRT_PROFILE=1 \
KOKORO_TRT_PROFILE_JSONL=app-traces.jsonl \
KOKORO_TRT_PROFILE_CUDA=1 \
KOKORO_TRT_PROFILE_SYNC_CUDA=1 \
KOKORO_TRT_PROFILE_NVTX=1 \
uv run --extra ui,metrics python app.py
```

The UI displays telemetry-derived audio duration, latency, RTF, and a compact stage table.

### Benchmark command

A dedicated benchmark runner is available:

```bash
uv run python -m kokoro.profile \
  --artifact-dir ./build \
  --voice af_heart \
  --language a \
  --text "Hello from Kokoro running through TensorRT." \
  --warmups 5 \
  --iterations 50 \
  --profile-cuda \
  --profile-sync-cuda \
  --profile-nvtx \
  --json results.json \
  --jsonl traces.jsonl
```

It reports p50/p90/p95/p99 latency, RTF, audio seconds per wall second, chars per second, stage CPU totals, stage CUDA totals, peak GPU memory, and errors by stage. Use `--input-file corpus.txt --split-pattern "\n+"` for corpus-style workloads.

## Tests

Artifact/runtime tests require a compiled TensorRT artifact:

```bash
KOKORO_TRT_ARTIFACT_DIR=./build \
KOKORO_TRT_VOICE=af_heart \
KOKORO_TRT_LANG=a \
uv run pytest tests/test_trt_artifact.py tests/test_trt_runtime.py
```

Telemetry unit tests do not require CUDA:

```bash
uv run pytest tests/test_telemetry.py
```

## Upstream project

Kokoro-82M is an open-weight TTS model by Hexgrad:

<https://huggingface.co/hexgrad/Kokoro-82M>

The text frontend uses [Misaki](https://github.com/hexgrad/misaki).
