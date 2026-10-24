"""Bounded, exact-shape CUDA graphs for the host inference stages."""

import torch

from .model import KokoroHostStages


class TextDurationGraphs:
    # Replays must be serialized on one stream. Every graph reads only its
    # current inputs and weights; no temporary value survives between calls.
    @torch.inference_mode()
    def __init__(self, host: KokoroHostStages, max_tokens: int):
        self.host = host
        self.input_ids = torch.zeros(
            (1, max_tokens), device=host.device, dtype=torch.long
        )
        self.ref_s = torch.zeros((1, 256), device=host.device)
        self.speed = torch.ones((1,), device=host.device)
        self.graphs: dict[
            int, tuple[torch.cuda.CUDAGraph, tuple[torch.Tensor, ...]]
        ] = {}

        current = torch.cuda.current_stream(host.device)
        warmup = torch.cuda.Stream(device=host.device)
        warmup.wait_stream(current)
        pool = torch.cuda.graph_pool_handle()
        with torch.cuda.stream(warmup):
            for length in range(3, max_tokens + 1):
                ids = self.input_ids[:, :length]
                # Initialize libraries and LSTM workspaces outside capture.
                host.text_duration(ids, self.ref_s, self.speed)
                warmup.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, pool=pool, stream=warmup):
                    outputs = host.text_duration(ids, self.ref_s, self.speed)
                self.graphs[length] = (graph, outputs)
        warmup.synchronize()

    def __call__(self, input_ids, ref_s, speed):
        length = input_ids.shape[1]
        entry = self.graphs.get(length)
        if entry is None:
            return tuple(
                output.clone()
                for output in self.host.text_duration(input_ids, ref_s, speed)
            )

        self.input_ids[:, :length].copy_(input_ids)
        self.ref_s.copy_(ref_s)
        self.speed.copy_(speed)
        graph, outputs = entry
        graph.replay()
        # Only durations escape the synthesis call. The two hidden tensors are
        # consumed by expand_frames before the next serialized text replay, so
        # copying them only adds device allocations and memory traffic.
        return outputs[0].clone(), outputs[1], outputs[2]


class RenderGraphs:
    """Exact host-render graphs with shared, maximum-sized I/O slots."""

    @torch.inference_mode()
    def __init__(self, host: KokoroHostStages, max_frames: int):
        self.host = host
        channels = host.config_data["hidden_dim"]
        self.asr = torch.zeros((1, channels, max_frames), device=host.device)
        self.en = torch.zeros(
            (1, host.model.predictor.shared.input_dim, max_frames), device=host.device
        )
        self.ref_s = torch.zeros((1, 256), device=host.device)
        self.outputs = tuple(
            torch.empty_like(value) for value in self._compute(max_frames)
        )
        self.graphs: dict[int, tuple[torch.cuda.CUDAGraph, tuple[int, ...]]] = {}

        current = torch.cuda.current_stream(host.device)
        warmup = torch.cuda.Stream(device=host.device)
        warmup.wait_stream(current)
        pool = torch.cuda.graph_pool_handle()
        with torch.cuda.stream(warmup):
            for frames in range(2, max_frames + 1):
                self._compute(frames)
                warmup.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, pool=pool, stream=warmup):
                    lengths = self._save_outputs(frames)
                self.graphs[frames] = (graph, lengths)
        warmup.synchronize()

    def _compute(self, frames):
        # Contiguous inputs preserve the eager convolution layouts and tactics.
        asr = self.asr[..., :frames].contiguous()
        en = self.en[..., :frames].contiguous()
        f0, noise = self.host.predict_f0n(en, self.ref_s)
        harmonic = self.host.compute_harmonic_features(f0)
        sources = self.host.compute_source_pyramid(harmonic, self.ref_s)
        decoded = self.host.decode_features(asr, f0, noise, self.ref_s)
        return f0, noise, harmonic, decoded, *sources

    def _save_outputs(self, frames):
        values = self._compute(frames)
        lengths = tuple(value.shape[-1] for value in values)
        for index, length in enumerate(lengths):
            self.outputs[index][..., :length].copy_(values[index])
        # Only the external I/O slots survive capture. All temporary features
        # are dead here, so independent, serialized graphs can share the pool.
        return lengths

    def __call__(self, asr, en, ref_s):
        frames = asr.shape[-1]
        self.asr[..., :frames].copy_(asr)
        self.en[..., :frames].copy_(en)
        self.ref_s.copy_(ref_s)
        graph, lengths = self.graphs[frames]
        graph.replay()
        # The render results go straight into TensorRT on the same ordered
        # inference stream. No result escapes render_frame, and the runtime
        # serializes graph replays; keeping these views avoids copying the
        # large harmonic/source buffers a second time.
        return tuple(
            output[..., :length] for output, length in zip(self.outputs, lengths)
        )
