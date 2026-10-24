import os
from pathlib import Path

import pytest
import torch

from kokoro import compile_artifact
from kokoro.artifact import TensorRTArtifact
from kokoro.model import GeneratorExportBuilder, KokoroModelLoader
from kokoro.native_trt import NativeTRTEngine
from kokoro.shapes import ShapePlan
from kokoro.telemetry import InMemoryTraceSink, ProfilerConfig, Telemetry
from kokoro.trt_builder import build_engine_from_onnx


def existing_artifact_dir() -> Path:
    value = os.getenv("KOKORO_TRT_ARTIFACT_DIR")
    if not value:
        pytest.skip("KOKORO_TRT_ARTIFACT_DIR is required")
    return Path(value)


def load_model_for_artifact(artifact: TensorRTArtifact):
    loader = KokoroModelLoader(
        repo_id=artifact.metadata.repo_id,
        config=artifact.load_config(),
        model=None,
    )
    model = loader.load(load_weights=False)
    model.load_host_state(artifact.paths.host_state_path)
    return model


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="native TensorRT tests require CUDA"
)
def test_native_compile_builds_artifact_and_metadata_loads(tmp_path):
    if os.getenv("KOKORO_NATIVE_TRT_BUILD_TESTS") != "1":
        pytest.skip("set KOKORO_NATIVE_TRT_BUILD_TESTS=1 to run native compile tests")

    output_dir = tmp_path / "artifact-fp16"
    compile_artifact(
        output_dir,
        repo_id=os.getenv("KOKORO_TRT_REPO_ID", "hexgrad/Kokoro-82M"),
        model=os.getenv("KOKORO_TRT_MODEL"),
        include_voices=[],
    )

    artifact = TensorRTArtifact.load(output_dir)
    assert artifact.metadata.precision == "fp16"
    assert artifact.paths.engine_path.is_file()
    assert artifact.paths.onnx_path.is_file()


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="native TensorRT tests require CUDA"
)
def test_build_engine_parses_onnx_with_external_weights(tmp_path):
    if os.getenv("KOKORO_NATIVE_TRT_BUILD_TESTS") != "1":
        pytest.skip("set KOKORO_NATIVE_TRT_BUILD_TESTS=1 to run native compile tests")

    numpy = pytest.importorskip("numpy")
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("tensorrt")

    from onnx import helper, numpy_helper

    onnx_path = tmp_path / "external-data.onnx"
    weights_path = tmp_path / "external-data.weights"
    engine_path = tmp_path / "external-data.plan"

    x = helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT16, [1, 1])
    audio = helper.make_tensor_value_info("audio", onnx.TensorProto.FLOAT16, [1, 1])
    weight = numpy_helper.from_array(
        numpy.asarray([[2.0]], dtype=numpy.float16),
        name="weight",
    )
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["x", "weight"], ["audio"])],
        "external-data",
        [x],
        [audio],
        initializer=[weight],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_operatorsetid("", 18)],
    )
    onnx.save_model(
        model,
        onnx_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=weights_path.name,
        size_threshold=0,
        convert_attribute=False,
    )

    assert onnx_path.is_file()
    assert weights_path.is_file()

    shapes = {
        "lower": {"x": (1, 1)},
        "preferred": {"x": (1, 1)},
        "upper": {"x": (1, 1)},
    }
    build_engine_from_onnx(
        onnx_path=onnx_path,
        engine_path=engine_path,
        shapes=shapes,
        input_order=("x",),
        workspace_size=None,
        builder_optimization_level=None,
    )

    assert engine_path.is_file()
    assert engine_path.stat().st_size > 0


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="native TensorRT tests require CUDA"
)
def test_native_engine_runs_random_profile_points():
    artifact = TensorRTArtifact.load(existing_artifact_dir())
    engine = NativeTRTEngine(artifact.paths.engine_path)
    assert set(engine.input_names) == set(artifact.metadata.input_names)
    assert all(
        engine.tensor_dtypes[name] == torch.float16 for name in engine.input_names
    )
    assert engine.tensor_dtypes["audio"] == torch.float32
    sink = InMemoryTraceSink()
    telemetry = Telemetry(ProfilerConfig(enabled=True), [sink])

    for group in ("lower", "preferred", "upper"):
        chunk = telemetry.start_chunk()
        inputs = {
            name: torch.randn(
                tuple(shape),
                device="cuda",
                dtype=torch.float16,
            ).contiguous()
            for name, shape in artifact.metadata.shapes[group].items()
        }

        outputs = engine.run(inputs, profile=chunk)
        chunk.finalize("ok")
        assert set(outputs) == {"audio"}
        assert outputs["audio"].is_cuda
        assert outputs["audio"].numel() > 0
        assert torch.isfinite(outputs["audio"]).all()

    stage_names = {stage.name for trace in sink.traces for stage in trace.stages}
    assert "trt.validate_inputs" in stage_names
    assert "trt.set_input_shapes" in stage_names
    assert "trt.infer_shapes" in stage_names
    assert "trt.allocate_outputs" in stage_names
    assert "trt.set_tensor_addresses" in stage_names
    assert "trt.execute_async_v3" in stage_names


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="native TensorRT tests require CUDA"
)
def test_native_engine_matches_pytorch_generator_at_preferred_shape():
    artifact = TensorRTArtifact.load(existing_artifact_dir())
    model = load_model_for_artifact(artifact).to("cuda").eval()
    plan = ShapePlan.from_model(model)
    shapes = plan.engine_shapes(model)["preferred"]

    dtype = torch.float16
    torch_generator = (
        GeneratorExportBuilder.build(model).to(device="cuda", dtype=dtype).eval()
    )
    torch_generator.generator.stft.float()
    native_engine = NativeTRTEngine(artifact.paths.engine_path)

    # Fix the input sample: the generator's exponentials amplify some random
    # draws enough to make the FP32/FP16 comparison flaky.
    rng = torch.Generator(device="cuda").manual_seed(1234)
    inputs = {
        name: torch.randn(
            tuple(shapes[name]), device="cuda", dtype=dtype, generator=rng
        ).contiguous()
        for name in plan.input_order()
    }

    with torch.inference_mode():
        expected = torch_generator(
            inputs["x"],
            inputs["ref_s"],
            *(inputs[f"source_{i}"] for i in range(len(plan.source_channels))),
        ).float()

    actual = native_engine.run(inputs)["audio"].float()
    torch.cuda.current_stream().synchronize()

    assert torch.isfinite(actual).all()
    assert torch.allclose(actual, expected, rtol=5e-2, atol=5e-2)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="native TensorRT tests require CUDA"
)
def test_shape_sized_workspace_matches_static_context():
    artifact = TensorRTArtifact.load(existing_artifact_dir())
    engine = NativeTRTEngine(artifact.paths.engine_path)
    free_bytes, _ = torch.cuda.mem_get_info()
    static_workspace = int(engine.engine.device_memory_size_v2)
    if free_bytes < static_workspace + 256 * 1024 * 1024:
        pytest.skip("Not enough free GPU memory for an additional static context")
    static = engine.engine.create_execution_context()
    assert static is not None

    for group in ("lower", "preferred"):
        inputs = {
            name: torch.randn(
                tuple(artifact.metadata.shapes[group][name]),
                device="cuda",
                dtype=engine.tensor_dtypes[name],
            )
            * 0.1
            for name in engine.input_names
        }
        for name, tensor in inputs.items():
            assert static.set_input_shape(name, tuple(tensor.shape))
            static.set_tensor_address(name, tensor.data_ptr())
        assert not static.infer_shapes()
        expected = {
            name: torch.empty(
                tuple(static.get_tensor_shape(name)),
                device="cuda",
                dtype=engine.tensor_dtypes[name],
            )
            for name in engine.output_names
        }
        for name, tensor in expected.items():
            static.set_tensor_address(name, tensor.data_ptr())
        assert static.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        actual = engine.run(inputs)
        torch.cuda.synchronize()
        for name in engine.output_names:
            assert torch.isfinite(actual[name]).all()
            torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)
