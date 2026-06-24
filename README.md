# kokoro

Inference tooling for Kokoro-82M.

This fork keeps the text frontend and neural inference backend explicit:

- `KPipeline` prepares text, phonemes, token IDs, voice style, and speed tensors.
- `KModel` / `KokoroInferenceBackend` run the PyTorch model.
- `KONNXModel` / `KokoroONNXBackend` run exported ONNX models with ONNX Runtime.

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

For the complete set of language codes supported by the installed version, use the source of truth in `kokoro.pipeline.LANG_CODES`.

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
    audio = output.audio.detach().cpu().reshape(-1).numpy()

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

The package also exposes a simple CLI.

After installing locally, run the CLI through uv:

```bash
uv run kokoro -m af_heart -t "Hello from Kokoro." -o hello.wav
```

or from standard input:

```bash
uv run kokoro -m af_heart -o hello.wav < input.txt
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
MODEL_REPO = "hexgrad/Kokoro-82M"
SAMPLE_RATE = 24000

onnx_dir = snapshot_download(
    repo_id=ONNX_REPO,
    allow_patterns=["*.onnx"],
)

model = KONNXModel(
    onnx_dir,
    repo_id=MODEL_REPO,
    # providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)

pipeline = KPipeline(
    lang_code="a",
    repo_id=model.repo_id,
    vocab=model.vocab,
    context_length=model.context_length,
    text_buckets=model.text_buckets,
)

text = "Hello from Kokoro running through ONNX Runtime."

for i, prepared in enumerate(pipeline.prepare(text, voice="af_heart")):
    output = model(prepared=prepared)
    audio = output.audio.detach().cpu().reshape(-1).numpy()
    sf.write(f"onnx_{i}.wav", audio, SAMPLE_RATE)
```

Use the same base model repository for config and voices as the ONNX export was built from.

## Export ONNX yourself

If you need a custom export, install the ONNX export dependencies required by your PyTorch version and export from a compatible PyTorch checkpoint:

```bash
uv pip install onnxscript
```

Save this as `export_onnx.py` and run it with `uv run python export_onnx.py`.

```python
from kokoro import KModel

model = KModel(repo_id="hexgrad/Kokoro-82M").eval()
model.export_onnx("onnx_out")
```

The resulting directory can be loaded with `KONNXModel("onnx_out", repo_id="hexgrad/Kokoro-82M")`.

## Upstream project

Kokoro-82M is an open-weight TTS model by Hexgrad. See the upstream model page for model details, samples, and weights:

<https://huggingface.co/hexgrad/Kokoro-82M>

The text frontend uses [Misaki](https://github.com/hexgrad/misaki).
