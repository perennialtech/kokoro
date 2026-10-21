from unittest.mock import MagicMock

import pytest
import torch

from kokoro.exceptions import TensorRTShapeError
from kokoro.native_trt import NativeTRTEngine
from kokoro.telemetry import NoOpProfileContext


def test_context_workspace_grows_and_reuses_allocation(monkeypatch):
    engine = NativeTRTEngine.__new__(NativeTRTEngine)
    engine.context = MagicMock()
    engine.context.update_device_memory_size_for_shapes.side_effect = [256, 128, 512]
    engine.stream = object()
    engine._context_memory = None
    allocations = []

    def allocate(size, *, dtype, device):
        assert dtype == torch.uint8
        assert device == "cuda"
        memory = MagicMock()
        memory.numel.return_value = size
        memory.data_ptr.return_value = 4096 + size
        allocations.append(memory)
        return memory

    monkeypatch.setattr(torch, "empty", allocate)
    profile = NoOpProfileContext()
    engine._prepare_context_memory(profile)
    first = engine._context_memory
    engine._prepare_context_memory(profile)
    assert engine._context_memory is first
    engine._prepare_context_memory(profile)
    assert len(allocations) == 2
    assert engine._context_memory is allocations[-1]
    for memory in allocations:
        memory.record_stream.assert_called_once_with(engine.stream)
    assert engine.context.set_device_memory.call_args_list[1].args == (4352, 256)
    assert engine.context.set_device_memory.call_args_list[2].args == (4608, 512)


@pytest.mark.parametrize("size", [0, -1])
def test_context_workspace_empty_or_invalid(size):
    engine = NativeTRTEngine.__new__(NativeTRTEngine)
    engine.context = MagicMock()
    engine.context.update_device_memory_size_for_shapes.return_value = size
    engine._context_memory = None
    if size < 0:
        with pytest.raises(TensorRTShapeError):
            engine._prepare_context_memory(NoOpProfileContext())
        engine.context.set_device_memory.assert_not_called()
    else:
        engine._prepare_context_memory(NoOpProfileContext())
        engine.context.set_device_memory.assert_called_once_with(0, 0)
