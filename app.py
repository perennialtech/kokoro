import os

import gradio as gr
import numpy as np

from kokoro import KokoroTRT
from kokoro.pipeline import LANGUAGE_CODES
from kokoro.telemetry import (JsonlTraceSink, ProfilerConfig,
                              PrometheusMetrics, Telemetry)

ARTIFACT_DIR = os.environ.get("KOKORO_TRT_ARTIFACT_DIR", "./build")


def build_telemetry() -> Telemetry:
    profile_enabled = os.environ.get("KOKORO_TRT_PROFILE") == "1"
    jsonl_path = os.environ.get("KOKORO_TRT_PROFILE_JSONL")
    sinks = [JsonlTraceSink(jsonl_path)] if jsonl_path else []

    metrics = None
    metrics_port = os.environ.get("KOKORO_TRT_METRICS_PORT")
    if metrics_port:
        metrics = PrometheusMetrics()
        metrics.start_http_server(int(metrics_port))

    return Telemetry(
        profiler_config=ProfilerConfig(
            enabled=profile_enabled or bool(jsonl_path),
            cuda_timing=os.environ.get("KOKORO_TRT_PROFILE_CUDA") == "1",
            synchronize_cuda=os.environ.get("KOKORO_TRT_PROFILE_SYNC_CUDA") == "1",
        ),
        trace_sinks=sinks,
        metrics=metrics,
    )


telemetry = build_telemetry()
tts = KokoroTRT(ARTIFACT_DIR, telemetry=telemetry)


def _stage_table(trace) -> str:
    if trace is None:
        return ""

    totals = {}

    # Helper to accumulate timings
    def _add_stages(stages):
        for stage in stages:
            count, cpu, cuda = totals.get(stage.name, (0, 0.0, 0.0))
            totals[stage.name] = (
                count + 1,
                cpu + float(stage.cpu_ms),
                cuda + float(stage.cuda_ms or 0.0),
            )

    # 1. Aggregate request level (frontend preprocessing)
    if hasattr(trace, "stages"):
        _add_stages(trace.stages)

    # 2. Aggregate chunk level (neural network inference & trt)
    if hasattr(trace, "chunks"):
        for chunk in trace.chunks:
            _add_stages(chunk.stages)

    if not totals:
        return ""

    rows = ["", "Stage | Count | CPU ms | CUDA ms", "--- | ---: | ---: | ---:"]
    for name, (count, cpu, cuda) in sorted(totals.items()):
        rows.append(f"{name} | {count} | {cpu:.3f} | {cuda:.3f}")
    return "\n".join(rows)


def generate_audio(text, language, voice, speed):
    chunks = [
        result.audio.detach().cpu().numpy()
        for result in tts.synthesize(text, voice, language, speed)
    ]

    if not chunks:
        return None, "No audio generated."

    audio = np.concatenate(chunks)
    trace = tts.telemetry.last_request_trace
    if trace is None:
        return (24000, audio), "Profiling disabled."

    latency = trace.ready_latency_s or trace.submit_latency_s
    rtf = trace.rtf_ready or trace.rtf_submit or 0.0
    timing_info = (
        f"Generated {trace.audio_duration_s:.2f}s of audio in "
        f"{latency:.3f}s (RTF: {rtf:.3f})" + _stage_table(trace)
    )

    return (24000, audio), timing_info


with gr.Blocks() as app:
    gr.Markdown("# Kokoro TensorRT TTS")

    with gr.Row():
        with gr.Column():
            text_in = gr.Textbox(
                label="Text",
                lines=5,
                value="Hello from Kokoro running through TensorRT.",
            )
            lang_in = gr.Dropdown(
                label="Language",
                choices=[(name, code) for code, name in LANGUAGE_CODES.items()],
                value="a",
            )
            voice_in = gr.Textbox(label="Voice", value="af_heart")
            speed_in = gr.Slider(
                label="Speed", minimum=0.5, maximum=2.0, value=1.0, step=0.1
            )
            submit_btn = gr.Button("Generate")

        with gr.Column():
            audio_out = gr.Audio(label="Synthesized Audio", type="numpy")
            timing_out = gr.Textbox(label="Timings", interactive=False)

    submit_btn.click(
        generate_audio,
        inputs=[text_in, lang_in, voice_in, speed_in],
        outputs=[audio_out, timing_out],
    )

if __name__ == "__main__":
    app.launch()
