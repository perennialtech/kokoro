# Short-input latency evaluation

Measured on 2026-10-21 against baseline `88eeaf4` and runtime change `7d70241`, using the same production FP32 TensorRT artifact. These are Kokoro synthesis measurements, not end-to-end service or voice-conversion measurements.

## Results

Each configuration ran in a fresh process with 25 English inputs of 3–50 characters, 30 shuffled passes, and 750 independently synthesized requests. All configurations returned identical sample counts for every input.

| Configuration | TTFA p50 | TTFA p95 | TTFA p99 | Sampled process peak | Startup and warmup |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline, production behavior | 28.80 ms | 33.01 ms | 35.64 ms | 2.34 GiB | 17.27 s |
| Baseline, host in evaluation mode | 28.45 ms | 32.59 ms | 34.11 ms | 2.34 GiB | 16.84 s |
| New runtime, both graph caches disabled | 28.03 ms | 32.25 ms | 34.80 ms | 1.18 GiB | 17.07 s |
| New runtime, text graphs only | 22.87 ms | 27.30 ms | 28.30 ms | 1.79 GiB | 19.04 s |
| New runtime, default graph limits | 22.82 ms | 27.32 ms | 28.56 ms | 2.26 GiB | 21.39 s |

The default runtime reduces corpus p50 by 20.8% and p95 by 17.2% relative to production behavior. Against the evaluation-mode baseline, p50 improves by 19.8%, so disabling dropout does not account for most of the gain.

| Input | Characters | Baseline p50 | New p50 | Reduction |
| --- | ---: | ---: | ---: | ---: |
| `Hi.` | 3 | 26.23 ms | 14.02 ms | 46.6% |
| `Hello, world!` | 13 | 26.22 ms | 20.21 ms | 22.9% |
| `Please turn off all the lights when you head home.` | 50 | 31.96 ms | 26.10 ms | 18.3% |

Each input's first request is included, not discarded as another warmup. First-use latency across the corpus ranged from 26.53–37.29 ms for the baseline and 14.60–31.71 ms for the default runtime. Only the unrelated utterance `Warm up.` was synthesized before measurement. No response cache was involved.

The decoder cache mainly benefits very short utterances. Disabling it with `render_graph_max_frames=0` saves about 0.47 GiB in this experiment while retaining almost the same corpus median, but increases `Hi.` p50 from 14.02 to 19.81 ms.

## Reproduce

Requires the normal runtime dependencies, a compatible compiled artifact, and an NVIDIA GPU. Run the benchmark in a fresh process:

```bash
uv run python -m kokoro.benchmark \
  --artifact-dir /path/to/artifact \
  --iterations 30 \
  --tf32 \
  --json ttfa.json
```

`--tf32` matches the math setting enabled by Chatterbox in the production service. Omit it when measuring a standalone deployment that does not enable TF32. See `python -m kokoro.benchmark --help` for corpus, voice, speed, and graph-limit options.

To compare an older checkout with the same harness, copy `kokoro/benchmark.py` to that checkout's root as `run_benchmark.py`, then run it there with the same environment and arguments. Do not execute it directly from inside the `kokoro/` package directory: that puts `kokoro/profile.py` ahead of Python's standard `profile` module. Use `--eval-host` for the evaluation-mode baseline. To reproduce the ablations on this revision, use `--render-graph-max-frames 0`, or additionally `--text-graph-max-tokens 0`.

The measured environment was an RTX 3080, driver 595.99.02, PyTorch 2.14.1+cu130, TensorRT 11.4.0.106, one PyTorch CPU thread, voice `af_heart`, speed 1.0, shuffle seed 7, and five generic warmups. Model and compiler caches already existed. Production workloads remained running; no exclusive GPU access or clock locking was used.

The engine SHA-256 was `814d6d90840959357cf80d7089daa470f56ecf4703c89cf0f68905ef40ddc0e2`. Its existing maximum profile, precision, weights, and builder settings were unchanged. No artifact rebuild or training was needed.

## Measurement boundaries

TTFA starts before text preparation and ends after the first nonempty audio tensor has been copied to CPU. Each line in a custom corpus is a separate request. The default short inputs each produce one synthesis chunk, so first audio requires that whole chunk. Streaming a precomputed waveform is not counted as neural streaming.

Startup and warmup timing includes model construction and five warmup syntheses, but excludes Python imports and package or model downloads.

The process-memory sampler uses `nvidia-smi` and excludes processes present before model load. It includes initialization, CUDA graph metadata, TensorRT allocations, and PyTorch reservations. It is a sampled peak, not a guarantee against every transient allocation. PyTorch allocator peaks are reported separately because they do not include all driver and TensorRT allocations. Concurrent new GPU processes can contaminate the process-memory measurement.

These results do not establish peak memory for the artifact's longest supported utterances, multilingual latency, other voices, other GPUs, or the combined Kokoro/Chatterbox service. TensorRT workspace retains its largest allocation, so long requests can raise steady memory use. The benchmark bypasses the service's waveform cache and does not include voice conversion, WAV encoding, transport, or client playback.

## Validation

The CPU suite passed 24 tests. Isolated GPU/artifact runs passed 41 integration tests plus the native-engine profile-point and static-workspace equivalence tests. The waveform regression covers all 25 corpus inputs in production's TF32 mode and requires bitwise equality with eager evaluation-mode synthesis. It also covers changing inputs, speed, output ownership, long-token fallback, concurrent threads/streams, telemetry, and math-mode changes.

Direct comparison with the original native TensorRT runner produced bitwise-identical output for all 153,600 preferred-shape samples. The new regression also compares shape-sized workspace with TensorRT's default static workspace at the lower and preferred profile points.

The existing `test_native_engine_matches_pytorch_generator_at_preferred_shape` fails its random-input tolerance check with the supplied FP32 production artifact. This was reproduced on untouched `88eeaf4` with `torch.manual_seed(7)`. Its tolerance was not changed. The native profile-point test passes, and the direct old/new native-output comparison has zero error. Three opt-in artifact-compilation test cases were not run; the benchmark reuses the existing engine rather than rebuilding it.

## Implementation choices

Exact-shape CUDA graphs remove Python and kernel-launch gaps without padding, truncating, or changing the neural computation. The bounded text cache covers token lengths; a smaller rendering cache targets very short utterances. Shapes outside either range use eager inference. Rendering graphs share external I/O slots, and temporary features do not survive capture. Replays run serially on one inference stream; returned tensors own their storage.

Graph capture happens at warmup or first synthesis, after the application's math settings are active. Changing tracked TF32 or cuDNN settings rebuilds the graphs. This matters because the production service constructs Kokoro before Chatterbox enables TF32.

The host model uses evaluation mode. The baseline left dropout enabled during inference, so its waveform is not an appropriate bitwise reference. Accuracy comparisons use the evaluation-mode eager runtime instead.

TensorRT uses user-managed, shape-sized activation memory. The production engine otherwise reserves approximately 1.41 GiB for the maximum profile even when synthesizing a short utterance. The runtime grows one workspace as needed and records its use on the TensorRT stream so replacing it cannot free memory still in use.

## Rejected approaches

Replacing the custom recurrent kernels with cuDNN LSTMs reduced some memory but increased short-input latency in the probe, so that substitution is not included.

A fused adaptive-normalization/Snake kernel passed individual-layer tolerances but failed the waveform comparison. Tiny pitch differences crossed the harmonic phase's positive/negative pi boundary and became much larger source-feature differences. On `Hello, world!`, waveform SNR was approximately 26 dB with TF32 and 34 dB without it. The kernel is not shipped; the normalization, activation, harmonic, and export code remain unchanged.

A much larger rendering-graph range exhausted the scratch process's available GPU memory while production remained resident. The conservative rendering limit preserves the strongest improvement for tiny utterances without keeping a large decoder graph for every longer frame count.

No neural streaming is added. Bidirectional recurrence and utterance-wide normalization make naive prefix streaming change the model's outputs. The optimization instead preserves the existing whole-chunk synthesis path.
