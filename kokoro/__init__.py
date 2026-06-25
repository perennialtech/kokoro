__version__ = "0.9.5"

from .config import (CONFIG_FILENAME, DEFAULT_REPO_ID, MODEL_FILENAMES,
                     ONNX_ACOUSTIC_VOCODER_PREFIX, ONNX_METADATA_FILENAME,
                     ONNX_TEXT_DURATION_PREFIX, TRT_METADATA_FILENAME,
                     get_context_length, load_artifact_metadata,
                     load_config_data, load_exported_config, load_trt_metadata,
                     onnx_export_path, resolve_model_path, resolve_repo_id,
                     save_artifact_metadata, save_trt_metadata)
from .model import (KModel, KokoroAcousticVocoder, KokoroDecodeGenerateWithHar,
                    KokoroDecodeGenerateWithSourcePyramid,
                    KokoroGenerateWithSourcePyramid, KokoroInferenceBackend,
                    KokoroTextDuration, remove_weight_norm_parametrizations)
from .onnx import KokoroONNXBackend, KONNXModel
from .export_onnx import export_onnx
from .pipeline import KPipeline
from .runtime import Synthesizer, expand_frames, normalize_requests
from .trt import KokoroTRTBackend, TensorRTDynamicShapeProfile
from .types import FrameItem, InferenceRequest, KModelOutput, UtteranceOutput
