# kokoro

TensorRT-first inference tooling for Kokoro-82M.

This fork has one supported public runtime:

```python
from kokoro import KokoroTRT
```

The public compile API is:

```python
from kokoro import Profile, compile_artifact
```

The supported Torch-TensorRT compiler API level is Torch-TensorRT >= 2.12.1.

## Install

```bash
git clone https://github.com/perennialtech/kokoro.git
cd kokoro
uv sync
```

Install TensorRT / Torch-TensorRT >= 2.12.1 matching your CUDA and PyTorch stack.

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
  --precision fp16 \
  --min-frames 16 \
  --opt-frames 256 \
  --max-frames 1024 \
  --include-voice af_heart
```

Python:

```python
from kokoro import Profile, compile_artifact

compile_artifact(
    "./build",
    repo_id="hexgrad/Kokoro-82M",
    precision="fp16",
    profile=Profile(min_frames=16, opt_frames=256, max_frames=1024),
    include_voices=["af_heart"],
)
```

The artifact contains:

```text
artifact/
  config.json
  host_state.pt
  metadata.json
  generator_with_source_pyramid.pt2
  voices/
    af_heart.pt
```

TensorRT engines are profile-bound. If runtime predicts a synthesis frame length outside the compiled profile, synthesis raises a hard error. Recompile with a wider profile.

## TensorRT inference: Python

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

## TensorRT inference: CLI

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
from kokoro import KokoroTRT, Profile, compile_artifact
```

## Tests

Artifact/runtime tests require a compiled TensorRT artifact:

```bash
KOKORO_TRT_ARTIFACT_DIR=./build \
KOKORO_TRT_VOICE=af_heart \
KOKORO_TRT_LANG=a \
uv run pytest tests/test_trt_artifact.py tests/test_trt_runtime.py
```

## Upstream project

Kokoro-82M is an open-weight TTS model by Hexgrad:

<https://huggingface.co/hexgrad/Kokoro-82M>

The text frontend uses [Misaki](https://github.com/hexgrad/misaki).
