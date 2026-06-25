# kokoro

Inference tooling for Kokoro-82M.

This fork uses an explicit frontend/backend split and exact internal inference:

- `KPipeline` prepares text, phonemes, token IDs, voice style, and speed tensors.
- `KModel` owns neural module construction and PyTorch checkpoint loading.
- `KokoroInferenceBackend`, `KokoroONNXBackend`, and `KokoroTRTBackend` implement the same small backend interface.
- `Synthesizer` performs the shared runtime flow for every backend:
  1. normalize inputs into exact single-utterance requests
  2. run text duration
  3. expand token features to acoustic frames
  4. render audio
  5. trim end silence
  6. return `KModelOutput`

Public batch-like inputs are still accepted, but they are sliced and synthesized one utterance at a time. Internally there is no padded/masked neural inference path.

Older examples that iterate over `KPipeline(...)` as `(graphemes, phonemes, audio)` are not valid for this fork. `KPipeline` yields prepared inputs; pass those inputs to a backend to synthesize audio.

## Install

Install [uv](https://docs.astral.sh/uv/) if you do not already have it:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

To run the examples locally, clone this repository:

```bash
git clone https://github.com/perennialtech/kokoro.git
cd kokoro
uv sync
uv pip install soundfile
```

The examples use `soundfile` only for writing WAV files.

If you prefer a single editable-install command instead of `uv sync`:

```bash
uv venv
uv pip install -e . soundfile
```

Alternatively, to install the package directly into another environment without cloning:

```bash
uv pip install git+https://github.com/perennialtech/kokoro.git
```

Install `espeak-ng` with your OS package manager if you need English fallback pronunciation or languages that use the espeak frontend:

```bash
sudo apt-get install espeak-ng
```

On Windows, use the official espeak-ng installer from the [espeak-ng releases page](https://github.com/espeak-ng/espeak-ng/releases).

Some language frontends are optional Misaki extras. Install the relevant extra before constructing those pipelines, for example:

```bash
uv pip install "misaki[ja]"
uv pip install "misaki[zh]"
```

For the complete set of language codes supported by the installed version, use `kokoro.pipeline.LANG_CODES`.

## PyTorch inference

Save this as `pytorch_infer.py` and run it with `uv run python pytorch_infer.py`.

```python
from pathlib import Path

import soundfile as sf
import torch

from kokoro import KModel, KPipeline

MODEL_REPO = "hexgrad/Kokoro-82M"
SAMPLE_RATE = 24000

text = """
Kokoro is an open-weight text-to-speech model. This example uses the explicit
frontend/backend API in this fork.
"""

out_dir = Path("kokoro_out")
out_dir.mkdir(exist_ok=True)

model = KModel(repo_id=MODEL_REPO).eval()
if torch.cuda.is_available():
    model = model.to("cuda")

pipeline = KPipeline(
    lang_code="a",
    repo_id=model.repo_id,
    vocab=model.vocab,
    context_length=model.context_length,
)
backend = model.inference_backend()

for i, prepared in enumerate(
    pipeline.prepare(text, voice="af_heart", speed=1.0, split_pattern=r"\n+")
):
    output = backend(prepared=prepared)
    audio = output.utterances[0].audio.detach().cpu().reshape(-1).numpy()

    print(prepared.graphemes)
    print(prepared.phonemes)

    sf.write(out_dir / f"{i}.wav", audio, SAMPLE_RATE)
```

`KPipeline.prepare(...)` may yield multiple chunks for long input. Save each chunk separately, or concatenate the returned audio arrays if you want one output file.

The pipeline downloads built-in voice tensors from the model repository. You may also pass a loaded voice tensor instead of a voice name:

```python
voice = torch.load("path/to/voice.pt", weights_only=True)
prepared_items = pipeline.prepare(text, voice=voice)
```

## CLI

The package exposes a simple CLI. The default backend is PyTorch:

```bash
uv run kokoro --backend pytorch -m af_heart -t "Hello from Kokoro." -o hello.wav
```

`--backend pytorch` can be omitted.

To run through ONNX Runtime, point the CLI at a self-contained ONNX export directory containing:

```text
text_duration.onnx
acoustic_vocoder.onnx
metadata.json
config.json
```

```bash
uv run kokoro --backend onnx --onnx-model-dir onnx -m af_heart -t "Hello from ONNX Kokoro." -o hello_onnx.wav
```

For ONNX GPU execution, pass providers in ONNX Runtime order:

```bash
uv run kokoro --backend onnx --onnx-model-dir onnx \
  --onnx-provider CUDAExecutionProvider \
  --onnx-provider CPUExecutionProvider \
  -m af_heart \
  -t "Hello from GPU ONNX." \
  -o hello_onnx.wav
```

Use `--repo-id` for PyTorch/TensorRT when the config, weights, and voices should come from a repository other than the default base model. ONNX exports carry their own config and metadata, including the source `repo_id` used for voice loading.

Standard input works with any backend:

```bash
uv run kokoro --backend pytorch -m af_heart -o hello.wav < input.txt
```

Run:

```bash
uv run kokoro --help
```

for the installed CLI options.

## ONNX Runtime inference

Pre-exported ONNX models for this fork are hosted at:

<https://huggingface.co/8q0sb/Kokoro-82M-v1.0-ONNX>

Install ONNX Runtime first:

```bash
uv pip install onnxruntime soundfile
```

Use `onnxruntime-gpu` instead of `onnxruntime` if you want GPU execution and have a compatible ONNX Runtime setup:

```bash
uv pip install onnxruntime-gpu soundfile
```

Save this as `onnx_infer.py` and run it with `uv run python onnx_infer.py`.

```python
import soundfile as sf
from huggingface_hub import snapshot_download

from kokoro import KONNXModel, KPipeline

ONNX_REPO = "8q0sb/Kokoro-82M-v1.0-ONNX"
SAMPLE_RATE = 24000

onnx_dir = snapshot_download(
    repo_id=ONNX_REPO,
    allow_patterns=["*.onnx", "metadata.json", "config.json"],
)

model = KONNXModel(
    onnx_dir,
    # providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)

pipeline = KPipeline(
    lang_code="a",
    repo_id=model.repo_id,
    vocab=model.vocab,
    context_length=model.context_length,
)

text = "Hello from Kokoro running through ONNX Runtime."

for i, prepared in enumerate(pipeline.prepare(text, voice="af_heart")):
    output = model(prepared=prepared)
    audio = output.utterances[0].audio.detach().cpu().reshape(-1).numpy()
    sf.write(f"onnx_{i}.wav", audio, SAMPLE_RATE)
```

`KONNXModel` no longer downloads model config from Hugging Face. It reads `config.json` and `metadata.json` from the ONNX export directory.

## Export ONNX yourself

If you need a custom export, install the ONNX export dependencies required by your PyTorch version:

```bash
uv pip install onnxscript
```

Save this as `export_onnx.py` and run it with `uv run python export_onnx.py`.

```python
from kokoro import KModel, export_onnx

model = KModel(repo_id="hexgrad/Kokoro-82M").eval()
export_onnx(model, "onnx")
```

The resulting directory is self-contained for model metadata:

```text
onnx/
  text_duration.onnx
  acoustic_vocoder.onnx
  metadata.json
  config.json
```

Load it with:

```python
from kokoro import KONNXModel

model = KONNXModel("onnx")
```

## TensorRT

TensorRT compilation targets only `KokoroDecodeGenerateWithHar`, the decoder/generator stage.

Host/PyTorch stages remain explicit:

- text duration
- frame expansion
- F0/noise prediction
- harmonic feature generation

Compile:

```bash
uv run python -m kokoro.trt_compile \
  --output-dir ./build \
  --repo-id hexgrad/Kokoro-82M \
  --precision fp16 \
  --min-frames 16 \
  --opt-frames 256 \
  --max-frames 1024
```

Run:

```bash
uv run kokoro --backend tensorrt \
  --trt-artifact-dir ./build \
  -m af_heart \
  -t "Hello from TensorRT Kokoro." \
  -o hello_trt.wav
```

If a predicted utterance exceeds the TensorRT profile max frame length, the runtime falls back to the PyTorch decoder by default. Pass `--no-trt-pytorch-fallback` to raise instead.

## Upstream project

Kokoro-82M is an open-weight TTS model by Hexgrad. See the upstream model page for model details, samples, and weights:

<https://huggingface.co/hexgrad/Kokoro-82M>

The text frontend uses [Misaki](https://github.com/hexgrad/misaki).
