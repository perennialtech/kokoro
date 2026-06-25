from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class Profile:
    min_frames: int
    opt_frames: int
    max_frames: int

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Profile":
        profile = cls(
            min_frames=int(data["min_frames"]),
            opt_frames=int(data["opt_frames"]),
            max_frames=int(data["max_frames"]),
        )
        profile.validate()
        return profile

    def validate(self) -> None:
        if self.min_frames < 1:
            raise ValueError("min_frames must be positive")
        if self.opt_frames < self.min_frames:
            raise ValueError("opt_frames must be >= min_frames")
        if self.max_frames < self.opt_frames:
            raise ValueError("max_frames must be >= opt_frames")

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class Affine:
    slope: int
    intercept: int

    @classmethod
    def from_function(
        cls,
        name: str,
        fn: Callable[[int], int],
        points: tuple[int, ...],
    ) -> "Affine":
        y1 = int(fn(1))
        y2 = int(fn(2))
        slope = y2 - y1
        intercept = y1 - slope

        if slope <= 0:
            raise ValueError(f"{name} must have positive slope, got {slope}")

        check_points = {1, 2, 3, 8}
        check_points.update(int(point) for point in points)

        for x in check_points:
            actual = int(fn(x))
            expected = slope * x + intercept
            if actual != expected:
                raise ValueError(
                    f"{name} is not affine: {name}({x})={actual}, "
                    f"expected {expected} from {slope} * x + {intercept}"
                )

        return cls(slope=slope, intercept=intercept)

    def apply(self, value: int) -> int:
        return self.slope * int(value) + self.intercept

    def apply_dim(self, dim):
        value = dim if self.slope == 1 else self.slope * dim
        return value + self.intercept if self.intercept else value


@dataclass(frozen=True)
class ShapePlan:
    profile: Profile
    input_channels: int
    source_channels: tuple[int, ...]
    source_relations: tuple[Affine, ...]

    @classmethod
    def from_model(cls, model, profile: Profile) -> "ShapePlan":
        profile.validate()

        generator = model.decoder.generator
        if not generator.ups:
            raise ValueError("Generator exposes no upsampling layers")

        input_channels = int(generator.ups[0].in_channels)
        source_channels = tuple(int(c) for c in generator.source_channels())

        min_generator = int(model.decoder.generator_input_frame_length(profile.min_frames))
        opt_generator = int(model.decoder.generator_input_frame_length(profile.opt_frames))
        max_generator = int(model.decoder.generator_input_frame_length(profile.max_frames))
        generator_points = (min_generator, opt_generator, max_generator)

        relations = tuple(
            Affine.from_function(
                f"source_{i}_frame_count_from_generator_frames",
                lambda generator_frames, i=i: generator.source_frame_lengths(
                    int(generator_frames)
                )[i],
                generator_points,
            )
            for i in range(len(source_channels))
        )

        if len(relations) != len(source_channels):
            raise ValueError("Source channel/relation metadata mismatch")

        return cls(
            profile=profile,
            input_channels=input_channels,
            source_channels=source_channels,
            source_relations=relations,
        )

    def input_order(self) -> tuple[str, ...]:
        return (
            "x",
            "ref_s",
            *(f"source_{i}" for i in range(len(self.source_channels))),
        )

    def generator_frames(self, model, synthesis_frames: int) -> int:
        return int(model.decoder.generator_input_frame_length(int(synthesis_frames)))

    def harmonic_frames(self, model, synthesis_frames: int) -> int:
        generator_frames = self.generator_frames(model, synthesis_frames)
        return int(model.decoder.generator.output_frame_length(generator_frames))

    def source_lengths_from_generator_frames(self, generator_frames: int) -> tuple[int, ...]:
        return tuple(relation.apply(generator_frames) for relation in self.source_relations)

    def source_lengths(self, model, synthesis_frames: int) -> tuple[int, ...]:
        return self.source_lengths_from_generator_frames(
            self.generator_frames(model, synthesis_frames)
        )

    def shapes_for(self, model, synthesis_frames: int) -> dict[str, tuple[int, ...]]:
        generator_frames = self.generator_frames(model, synthesis_frames)
        shapes: dict[str, tuple[int, ...]] = {
            "x": (1, self.input_channels, generator_frames),
            "ref_s": (1, 256),
        }

        for i, (channels, source_frames) in enumerate(
            zip(
                self.source_channels,
                self.source_lengths_from_generator_frames(generator_frames),
            )
        ):
            shapes[f"source_{i}"] = (1, int(channels), int(source_frames))

        return shapes

    def profile_shapes(self, model) -> dict[str, dict[str, tuple[int, ...]]]:
        return {
            "min": self.shapes_for(model, self.profile.min_frames),
            "opt": self.shapes_for(model, self.profile.opt_frames),
            "max": self.shapes_for(model, self.profile.max_frames),
        }

    def tensorrt_inputs(
        self,
        model,
        dtype,
        torch_tensorrt,
    ) -> list[Any]:
        shapes = self.profile_shapes(model)

        return [
            torch_tensorrt.Input(
                min_shape=shapes["min"][name],
                opt_shape=shapes["opt"][name],
                max_shape=shapes["max"][name],
                dtype=dtype,
            )
            for name in self.input_order()
        ]

    def example_tensors(self, model, dtype):
        import torch

        shapes = self.profile_shapes(model)
        return tuple(
            torch.empty(shapes["opt"][name], device="cuda", dtype=dtype)
            for name in self.input_order()
        )

    def export_dynamic_shapes(self, model):
        from torch.export import Dim

        min_generator = self.generator_frames(model, self.profile.min_frames)
        max_generator = self.generator_frames(model, self.profile.max_frames)

        if min_generator < 3:
            required_min = self.profile.min_frames
            while self.generator_frames(model, required_min) < 3:
                required_min += 1

            raise ValueError(
                "TensorRT export requires generator input length >= 3. "
                f"With min_frames={self.profile.min_frames}, generator-frame length is "
                f"{min_generator}. Use --min-frames {required_min} or higher."
            )

        generator_frames = Dim(
            "generator_frames",
            min=int(min_generator),
            max=int(max_generator),
        )

        return (
            {2: generator_frames},
            {},
            *({2: relation.apply_dim(generator_frames)} for relation in self.source_relations),
        )
