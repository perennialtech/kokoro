from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Optional, Union

import torch
import torch.nn as nn
from loguru import logger
from torch.nn.utils import parametrize
from transformers import AlbertConfig

from .config import (
    MODEL_FILENAMES,
    get_context_length,
    load_config_data,
    resolve_model_path,
    resolve_repo_id,
)
from .istftnet import Decoder, replace_conv_transpose1d_with_static_phase
from .modules import CustomAlbert, ProsodyPredictor, TextEncoder


def remove_weight_norm_parametrizations(module: torch.nn.Module) -> None:
    for m in module.modules():
        if parametrize.is_parametrized(m, "weight"):
            parametrize.remove_parametrizations(m, "weight", leave_parametrized=True)


class KokoroModel(torch.nn.Module):
    MODEL_NAMES = MODEL_FILENAMES

    def __init__(self, repo_id: str, config_data: dict[str, Any]):
        super().__init__()

        self.repo_id = repo_id
        self.config_data = config_data
        self.vocab: Optional[dict[str, int]] = self.config_data.get("vocab")

        self.bert = CustomAlbert(
            AlbertConfig(
                vocab_size=self.config_data["n_token"],
                **self.config_data["plbert"],
            )
        )
        self.bert_encoder = torch.nn.Linear(
            self.bert.config.hidden_size,
            self.config_data["hidden_dim"],
        )
        self.context_length: int = get_context_length(self.config_data)

        self.predictor = ProsodyPredictor(
            style_dim=self.config_data["style_dim"],
            d_hid=self.config_data["hidden_dim"],
            nlayers=self.config_data["n_layer"],
            max_dur=self.config_data["max_dur"],
            dropout=self.config_data["dropout"],
        )
        self.text_encoder = TextEncoder(
            channels=self.config_data["hidden_dim"],
            kernel_size=self.config_data["text_encoder_kernel_size"],
            depth=self.config_data["n_layer"],
            n_symbols=self.config_data["n_token"],
        )
        self.decoder = Decoder(
            dim_in=self.config_data["hidden_dim"],
            style_dim=self.config_data["style_dim"],
            resblock_kernel_sizes=self.config_data["istftnet"]["resblock_kernel_sizes"],
            upsample_rates=self.config_data["istftnet"]["upsample_rates"],
            upsample_initial_channel=self.config_data["istftnet"][
                "upsample_initial_channel"
            ],
            resblock_dilation_sizes=self.config_data["istftnet"][
                "resblock_dilation_sizes"
            ],
            upsample_kernel_sizes=self.config_data["istftnet"]["upsample_kernel_sizes"],
            gen_istft_n_fft=self.config_data["istftnet"]["gen_istft_n_fft"],
            gen_istft_hop_size=self.config_data["istftnet"]["gen_istft_hop_size"],
        )

        self.model_path: Optional[str] = None

    @property
    def device(self):
        return next(self.parameters()).device

    def remove_weight_norm(self) -> None:
        remove_weight_norm_parametrizations(self)

    def load_host_state(self, path: Union[str, Path]) -> None:
        state = torch.load(path, map_location="cpu", weights_only=True)
        self.load_state_dict(state, strict=True)

    def save_host_state(self, path: Union[str, Path]) -> None:
        cpu_state = {k: v.detach().cpu() for k, v in self.state_dict().items()}
        torch.save(cpu_state, path)


class KokoroModelLoader:
    def __init__(
        self,
        repo_id: Optional[str] = None,
        config: Union[dict[str, Any], str, Path, None] = None,
        model: Optional[Union[str, Path]] = None,
    ):
        self.repo_id = resolve_repo_id(repo_id)
        self.config_data = load_config_data(self.repo_id, config)
        self.model_path_arg = model

    def load(self, load_weights: bool = True) -> KokoroModel:
        model = KokoroModel(self.repo_id, self.config_data)

        if load_weights:
            model.model_path = resolve_model_path(self.repo_id, self.model_path_arg)
            checkpoint = torch.load(
                model.model_path,
                map_location="cpu",
                weights_only=True,
            )
            self._load_checkpoint(model, checkpoint)

        model.remove_weight_norm()
        return model

    def _load_checkpoint(
        self,
        model: KokoroModel,
        checkpoint: dict[str, dict[str, torch.Tensor]],
    ) -> None:
        for key, state_dict in checkpoint.items():
            if not hasattr(model, key):
                raise KeyError(f"Checkpoint contains unknown submodule {key!r}")
            self._load_submodule_state(getattr(model, key), key, state_dict)

    @staticmethod
    def _load_submodule_state(
        module: torch.nn.Module,
        key: str,
        state_dict: dict[str, torch.Tensor],
    ) -> None:
        try:
            module.load_state_dict(state_dict, strict=True)
            return
        except RuntimeError as first_error:
            if state_dict and all(k.startswith("module.") for k in state_dict):
                stripped = {k[7:]: v for k, v in state_dict.items()}
                try:
                    module.load_state_dict(stripped, strict=True)
                    return
                except RuntimeError:
                    pass

            raise RuntimeError(
                f"Strict checkpoint load failed for submodule {key!r}. "
                "Only official checkpoint keys and optional module. prefixes are supported."
            ) from first_error


class KokoroHostStages(torch.nn.Module):
    def __init__(self, model: KokoroModel):
        super().__init__()
        self.model = model

    @property
    def repo_id(self) -> str:
        return self.model.repo_id

    @property
    def config_data(self) -> dict[str, Any]:
        return self.model.config_data

    @property
    def vocab(self) -> Optional[dict[str, int]]:
        return self.model.vocab

    @property
    def context_length(self) -> int:
        return self.model.context_length

    @property
    def decoder(self) -> Decoder:
        return self.model.decoder

    @property
    def device(self):
        return self.model.device

    def text_duration(
        self,
        input_ids: torch.Tensor,
        ref_s: torch.Tensor,
        speed: torch.Tensor,
    ):
        if input_ids.dim() != 2 or input_ids.shape[0] != 1:
            raise ValueError(
                f"text_duration expects input_ids [1,T], got {tuple(input_ids.shape)}"
            )
        if ref_s.dim() != 2 or ref_s.shape[0] != 1:
            raise ValueError(
                f"text_duration expects ref_s [1,256], got {tuple(ref_s.shape)}"
            )

        model = self.model
        positions = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(
            0
        )

        bert_dur = model.bert(
            input_ids,
            attention_mask=torch.ones_like(input_ids, dtype=torch.int32),
            token_type_ids=torch.zeros_like(input_ids),
            position_ids=positions,
        )
        d_en = model.bert_encoder(bert_dur).transpose(-1, -2)

        duration_style = ref_s[:, 128:]
        duration_hidden = model.predictor.text_encoder(d_en, duration_style)

        if not torch.jit.is_scripting():
            model.predictor.lstm.flatten_parameters()
        x, _ = model.predictor.lstm(duration_hidden)

        duration = torch.sigmoid(model.predictor.duration_proj(x)).sum(dim=-1)
        duration = duration / speed.reshape(-1, 1).to(duration.dtype)

        text_hidden = model.text_encoder(input_ids)
        return duration, duration_hidden, text_hidden

    def predict_f0n(self, en: torch.Tensor, ref_s: torch.Tensor):
        return self.model.predictor.F0Ntrain(en, ref_s[:, 128:])

    def decode_features(
        self,
        asr: torch.Tensor,
        f0: torch.Tensor,
        noise: torch.Tensor,
        ref_s: torch.Tensor,
    ):
        return self.model.decoder.decode_features(asr, f0, noise, ref_s[:, :128])

    def compute_harmonic_features(self, f0: torch.Tensor):
        return self.model.decoder.generator.compute_harmonic_features(f0)

    def compute_source_pyramid(
        self,
        har: torch.Tensor,
        ref_s: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        return self.model.decoder.compute_source_pyramid(har, ref_s[:, :128])


class GeneratorEngineModule(nn.Module):
    def __init__(self, generator: nn.Module):
        super().__init__()
        self.generator = generator

    def forward(
        self,
        x: torch.Tensor,
        ref_s: torch.Tensor,
        *source_pyramid: torch.Tensor,
    ) -> torch.Tensor:
        return self.generator.forward_with_source_pyramid(
            x,
            ref_s[:, :128],
            source_pyramid,
        )


class GeneratorExportBuilder:
    @staticmethod
    def build(model: KokoroModel) -> GeneratorEngineModule:
        generator = copy.deepcopy(model.decoder.generator).eval()
        remove_weight_norm_parametrizations(generator)

        replacements = replace_conv_transpose1d_with_static_phase(generator)
        remaining = sum(
            1 for module in generator.modules() if isinstance(module, nn.ConvTranspose1d)
        )
        if remaining:
            raise RuntimeError(
                "TensorRT export graph still contains "
                f"{remaining} ConvTranspose1d module(s) after replacement"
            )

        logger.debug(
            "Prepared generator export graph: replaced {} ConvTranspose1d module(s) "
            "with static phase Conv1d equivalents.",
            replacements,
        )

        return GeneratorEngineModule(generator).eval()
