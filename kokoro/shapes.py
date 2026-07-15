from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

AUTOMATIC_SYNTHESIS_FRAME_POINTS = (2, 256, 2048)


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
    input_channels: int
    source_channels: tuple[int, ...]
    source_relations: tuple[Affine, ...]

    @classmethod
    def from_model(cls, model) -> "ShapePlan":
        generator = model.decoder.generator
        if not generator.ups:
            raise ValueError("Generator exposes no upsampling layers")

        input_channels = int(generator.ups[0].in_channels)
        source_channels = tuple(int(c) for c in generator.source_channels())

        generator_points = tuple(
            int(model.decoder.generator_input_frame_length(point))
            for point in AUTOMATIC_SYNTHESIS_FRAME_POINTS
        )

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

    def source_lengths_from_generator_frames(
        self, generator_frames: int
    ) -> tuple[int, ...]:
        return tuple(
            relation.apply(generator_frames) for relation in self.source_relations
        )

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

    def engine_shapes(self, model) -> dict[str, dict[str, tuple[int, ...]]]:
        lower, preferred, upper = AUTOMATIC_SYNTHESIS_FRAME_POINTS
        return {
            "lower": self.shapes_for(model, lower),
            "preferred": self.shapes_for(model, preferred),
            "upper": self.shapes_for(model, upper),
        }

    def example_tensors(self, model, dtype):
        import torch

        shapes = self.engine_shapes(model)
        return tuple(
            torch.empty(shapes["preferred"][name], device="cuda", dtype=dtype)
            for name in self.input_order()
        )

    def _generator_frames_dim(self, model):
        from torch.export import Dim

        lower, _, upper = AUTOMATIC_SYNTHESIS_FRAME_POINTS
        lower_generator = self.generator_frames(model, lower)
        upper_generator = self.generator_frames(model, upper)

        if lower_generator < 3:
            raise ValueError(
                "TensorRT export requires a generator input length of at least 3, "
                f"but the automatic lower shape point produces {lower_generator}."
            )

        return Dim(
            "generator_frames",
            min=int(lower_generator),
            max=int(upper_generator),
        )

    def export_dynamic_shapes(self, model):
        generator_frames = self._generator_frames_dim(model)
        return {
            "x": {2: generator_frames},
            "ref_s": None,
            "source_pyramid": tuple(
                {2: relation.apply_dim(generator_frames)}
                for relation in self.source_relations
            ),
        }

    def export_dynamic_shapes_trt_save(self, model):
        generator_frames = self._generator_frames_dim(model)
        return (
            {2: generator_frames},
            {},
            tuple(
                {2: relation.apply_dim(generator_frames)}
                for relation in self.source_relations
            ),
        )
