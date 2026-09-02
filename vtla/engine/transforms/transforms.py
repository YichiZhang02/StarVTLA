#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import collections
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from torchvision.transforms import v2
from torchvision.transforms.v2 import (
    Transform,
    functional as F,  # noqa: N812
)


class RandomSubsetApply(Transform):
    """Apply a random subset of N transformations from a list of transformations.

    Args:
        transforms: list of transformations.
        p: represents the multinomial probabilities (with no replacement) used for sampling the transform.
            If the sum of the weights is not 1, they will be normalized. If ``None`` (default), all transforms
            have the same probability.
        n_subset: number of transformations to apply. If ``None``, all transforms are applied.
            Must be in [1, len(transforms)].
        random_order: apply transformations in a random order.
    """

    def __init__(
        self,
        transforms: Sequence[Callable],
        p: list[float] | None = None,
        n_subset: int | None = None,
        random_order: bool = False,
    ) -> None:
        super().__init__()
        if not isinstance(transforms, Sequence):
            raise TypeError("Argument transforms should be a sequence of callables")
        if p is None:
            p = [1] * len(transforms)
        elif len(p) != len(transforms):
            raise ValueError(
                f"Length of p doesn't match the number of transforms: {len(p)} != {len(transforms)}"
            )

        if n_subset is None:
            n_subset = len(transforms)
        elif not isinstance(n_subset, int):
            raise TypeError("n_subset should be an int or None")
        elif not (1 <= n_subset <= len(transforms)):
            raise ValueError(f"n_subset should be in the interval [1, {len(transforms)}]")

        self.transforms = transforms
        total = sum(p)
        self.p = [prob / total for prob in p]
        self.n_subset = n_subset
        self.random_order = random_order

        self.selected_transforms = None

    def forward(self, *inputs: Any) -> Any:
        needs_unpacking = len(inputs) > 1

        selected_indices = torch.multinomial(torch.tensor(self.p), self.n_subset)
        if not self.random_order:
            selected_indices = selected_indices.sort().values

        self.selected_transforms = [self.transforms[i] for i in selected_indices]

        for transform in self.selected_transforms:
            outputs = transform(*inputs)
            inputs = outputs if needs_unpacking else (outputs,)

        return outputs

    def extra_repr(self) -> str:
        return (
            f"transforms={self.transforms}, "
            f"p={self.p}, "
            f"n_subset={self.n_subset}, "
            f"random_order={self.random_order}"
        )


class SharpnessJitter(Transform):
    """Randomly change the sharpness of an image or video.

    Similar to a v2.RandomAdjustSharpness with p=1 and a sharpness_factor sampled randomly.
    While v2.RandomAdjustSharpness applies — with a given probability — a fixed sharpness_factor to an image,
    SharpnessJitter applies a random sharpness_factor each time. This is to have a more diverse set of
    augmentations as a result.

    A sharpness_factor of 0 gives a blurred image, 1 gives the original image while 2 increases the sharpness
    by a factor of 2.

    If the input is a :class:`torch.Tensor`,
    it is expected to have [..., 1 or 3, H, W] shape, where ... means an arbitrary number of leading dimensions.

    Args:
        sharpness: How much to jitter sharpness. sharpness_factor is chosen uniformly from
            [max(0, 1 - sharpness), 1 + sharpness] or the given
            [min, max]. Should be non negative numbers.
    """

    def __init__(self, sharpness: float | Sequence[float]) -> None:
        super().__init__()
        self.sharpness = self._check_input(sharpness)

    def _check_input(self, sharpness):
        if isinstance(sharpness, (int | float)):
            if sharpness < 0:
                raise ValueError("If sharpness is a single number, it must be non negative.")
            sharpness = [1.0 - sharpness, 1.0 + sharpness]
            sharpness[0] = max(sharpness[0], 0.0)
        elif isinstance(sharpness, collections.abc.Sequence) and len(sharpness) == 2:
            sharpness = [float(v) for v in sharpness]
        else:
            raise TypeError(f"{sharpness=} should be a single number or a sequence with length 2.")

        if not 0.0 <= sharpness[0] <= sharpness[1]:
            raise ValueError(f"sharpness values should be between (0., inf), but got {sharpness}.")

        return float(sharpness[0]), float(sharpness[1])

    def make_params(self, flat_inputs: list[Any]) -> dict[str, Any]:
        sharpness_factor = torch.empty(1).uniform_(self.sharpness[0], self.sharpness[1]).item()
        return {"sharpness_factor": sharpness_factor}

    def transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        sharpness_factor = params["sharpness_factor"]
        return self._call_kernel(F.adjust_sharpness, inpt, sharpness_factor=sharpness_factor)


class ColorTemperatureJitter(Transform):
    """Randomly shift the color temperature (white balance) of an image or video.

    Simulates warm / cool lighting by applying per-channel multiplicative gains to
    the RGB channels — the physical model of white balance. This is distinct from
    everything ColorJitter offers: brightness (uniform scale), contrast, saturation
    (pull toward / away from grey) and hue (rotate the whole color wheel). None of
    those model a warm/cool cast, which is what causes a train/deploy domain gap
    when the environment lighting is more yellow (or more blue) than the data.

    A temperature is sampled uniformly from ``temperature`` (a [min, max] range):
      - temperature > 0 -> warmer (boost R, cut B)  ==  偏黄 / 暖
      - temperature < 0 -> cooler (cut R, boost B)  ==  偏蓝 / 冷
      - temperature = 0 -> unchanged

    The input is expected to have shape [..., 3, H, W] in RGB channel order. Both
    float tensors (0.0-1.0) and uint8 tensors (0-255) are handled; non-RGB inputs
    are returned untouched.

    Args:
        temperature: [min, max] range to sample from. A single number ``t`` is
            treated as the symmetric range [-t, t]. Magnitudes are typically in
            [0, 1]; at |t| = 1 the R/B channels are scaled by ``max_gain``.
        max_gain: relative R (and inverse B) channel gain at |temperature| = 1.
    """

    def __init__(self, temperature: float | Sequence[float], max_gain: float = 0.30) -> None:
        super().__init__()
        self.temperature = self._check_input(temperature)
        self.max_gain = float(max_gain)

    def _check_input(self, temperature):
        if isinstance(temperature, (int | float)):
            temperature = [-abs(float(temperature)), abs(float(temperature))]
        elif isinstance(temperature, collections.abc.Sequence) and len(temperature) == 2:
            temperature = [float(v) for v in temperature]
        else:
            raise TypeError(f"{temperature=} should be a single number or a sequence with length 2.")

        if temperature[0] > temperature[1]:
            raise ValueError(f"temperature range should be (min, max), but got {temperature}.")

        return float(temperature[0]), float(temperature[1])

    def make_params(self, flat_inputs: list[Any]) -> dict[str, Any]:
        temperature = torch.empty(1).uniform_(self.temperature[0], self.temperature[1]).item()
        return {"temperature": temperature}

    def transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        t = params["temperature"]
        # Fast paths: no-op temperature, or an input that isn't an RGB [..., 3, H, W] tensor.
        if t == 0.0 or not isinstance(inpt, torch.Tensor):
            return inpt
        if inpt.ndim < 3 or inpt.shape[-3] != 3:
            return inpt

        # Per-channel gains: warm boosts R and cuts B; a small G tweak keeps it natural.
        gains = torch.tensor(
            [1.0 + self.max_gain * t, 1.0 + 0.05 * t, 1.0 - self.max_gain * t],
            dtype=torch.float32,
            device=inpt.device,
        ).view(3, 1, 1)

        is_uint8 = inpt.dtype == torch.uint8
        max_val = 255.0 if is_uint8 else 1.0
        x = (inpt.float() * gains).clamp_(0.0, max_val)
        if is_uint8:
            return x.round_().to(torch.uint8)
        return x.to(inpt.dtype)


@dataclass
class ImageTransformConfig:
    """
    For each transform, the following parameters are available:
      weight: This represents the multinomial probability (with no replacement)
            used for sampling the transform. If the sum of the weights is not 1,
            they will be normalized.
      type: The name of the class used. This is either a class available under torchvision.transforms.v2 or a
            custom transform defined here.
      kwargs: Lower & upper bound respectively used for sampling the transform's parameter
            (following uniform distribution) when it's applied.
    """

    weight: float = 1.0
    type: str = "Identity"
    kwargs: dict[str, Any] = field(default_factory=dict)


_AUGMENTATION_PRESETS: dict[str, dict] = {
    # preset name -> overrides applied in ImageTransformsConfig.__post_init__
    # Presets can override ranges and optionally restrict the active transform set.
    "none":   {"enable": False},
    "color":  {
        "enable": True,
        "brightness": (0.8, 1.2),
        "contrast": (0.8, 1.2),
        "only": ("brightness", "contrast", "saturation"),
    },
    "mild":   {"enable": True, "brightness": (0.8, 1.2), "contrast": (0.8, 1.2)},
    "strong": {"enable": True, "brightness": (0.5, 1.5), "contrast": (0.5, 1.5)},
}


@dataclass
class ImageTransformsConfig:
    """
    These transforms are all using standard torchvision.transforms.v2
    You can find out how these transformations affect images here:
    https://pytorch.org/vision/0.18/auto_examples/transforms/plot_transforms_illustrations.html
    We use a custom RandomSubsetApply container to sample them.

    Use `preset` to pick a named brightness/contrast level:
      - "none"    : augmentation disabled
      - "color"   : brightness/contrast/saturation only
      - "mild"    : brightness=(0.8, 1.2), contrast=(0.8, 1.2)
      - "strong"  : brightness=(0.5, 1.5), contrast=(0.5, 1.5)
    Setting `preset` overrides both `enable` and the brightness/contrast kwargs.
    Leave `preset` empty ("") to configure `enable` and `tfs` manually.
    """

    # Named preset — takes precedence over `enable` and brightness/contrast ranges when non-empty.
    # Allowed values: "none" | "color" | "mild" | "strong" | "" (manual).
    preset: str = "none"
    # Color-temperature (white-balance) augmentation range as (min, max), sampled uniformly.
    #   temp > 0 -> warmer (boost R, cut B, "偏黄/暖");  temp < 0 -> cooler ("偏蓝/冷").
    # When set (non-None), this enables the color_temp transform and injects the range,
    # regardless of `preset`/`enable` — set e.g. (0.0, 0.6) to cover a yellow-tinted
    # deployment environment. Leave None to keep color-temp augmentation off.
    color_temp: tuple[float, float] | None = None
    # Set this flag to `true` to enable transforms during training (ignored when preset != "")
    enable: bool = False
    # This is the maximum number of transforms (sampled from these below) that will be applied to each frame.
    # It's an integer in the interval [1, number_of_available_transforms].
    max_num_transforms: int = 3
    # By default, transforms are applied in Torchvision's suggested order (shown below).
    # Set this to True to apply them in a random order.
    random_order: bool = False
    tfs: dict[str, ImageTransformConfig] = field(
        default_factory=lambda: {
            "brightness": ImageTransformConfig(
                weight=1.0,
                type="ColorJitter",
                kwargs={"brightness": (0.8, 1.2)},
            ),
            "contrast": ImageTransformConfig(
                weight=1.0,
                type="ColorJitter",
                kwargs={"contrast": (0.8, 1.2)},
            ),
            "saturation": ImageTransformConfig(
                weight=1.0,
                type="ColorJitter",
                kwargs={"saturation": (0.5, 1.5)},
            ),
            "hue": ImageTransformConfig(
                weight=1.0,
                type="ColorJitter",
                kwargs={"hue": (-0.05, 0.05)},
            ),
            "sharpness": ImageTransformConfig(
                weight=1.0,
                type="SharpnessJitter",
                kwargs={"sharpness": (0.5, 1.5)},
            ),
            "affine": ImageTransformConfig(
                weight=1.0,
                type="RandomAffine",
                kwargs={"degrees": (-5.0, 5.0), "translate": (0.05, 0.05)},
            ),
            # Disabled by default (weight=0.0). Enabled + configured via `color_temp`
            # (see __post_init__) so existing runs are unaffected until you opt in.
            "color_temp": ImageTransformConfig(
                weight=0.0,
                type="ColorTemperatureJitter",
                kwargs={"temperature": (-0.5, 0.5)},
            ),
        }
    )

    def __post_init__(self) -> None:
        # 1) Resolve the named preset (brightness/contrast + enable). Empty preset = manual mode.
        if self.preset:
            if self.preset not in _AUGMENTATION_PRESETS:
                raise ValueError(
                    f"augmentation preset must be one of {list(_AUGMENTATION_PRESETS)}, got '{self.preset}'"
                )
            p = _AUGMENTATION_PRESETS[self.preset]
            self.enable = p["enable"]
            if "brightness" in p:
                self.tfs["brightness"].kwargs["brightness"] = p["brightness"]
            if "contrast" in p:
                self.tfs["contrast"].kwargs["contrast"] = p["contrast"]
            if "only" in p:
                enabled_names = set(p["only"])
                for name, tf_cfg in self.tfs.items():
                    if name not in enabled_names:
                        tf_cfg.weight = 0.0

        # 2) Wire up color-temperature augmentation independently of the preset.
        #    When set, it always turns on the color_temp transform. If the base
        #    augmentation is otherwise off (preset="none"/enable=False), we isolate
        #    color_temp by zeroing the other weights so ONLY color-temp is applied;
        #    when base augmentation is on, color_temp joins the sampling pool.
        if self.color_temp is not None:
            if len(self.color_temp) != 2 or self.color_temp[0] > self.color_temp[1]:
                raise ValueError(
                    f"color_temp must be a (min, max) range with min <= max, got {self.color_temp}"
                )
        if self.color_temp is not None and tuple(self.color_temp) != (0, 0):
            base_enabled = self.enable
            self.tfs["color_temp"].weight = 1.0
            self.tfs["color_temp"].kwargs["temperature"] = tuple(self.color_temp)
            self.enable = True
            if not base_enabled:
                for name, tf_cfg in self.tfs.items():
                    if name != "color_temp":
                        tf_cfg.weight = 0.0


def make_transform_from_config(cfg: ImageTransformConfig):
    if cfg.type == "SharpnessJitter":
        return SharpnessJitter(**cfg.kwargs)
    if cfg.type == "ColorTemperatureJitter":
        return ColorTemperatureJitter(**cfg.kwargs)

    transform_cls = getattr(v2, cfg.type, None)
    if isinstance(transform_cls, type) and issubclass(transform_cls, Transform):
        return transform_cls(**cfg.kwargs)

    raise ValueError(
        f"Transform '{cfg.type}' is not valid. It must be a class in "
        f"torchvision.transforms.v2 or 'SharpnessJitter'."
    )


class ImageTransforms(Transform):
    """A class to compose image transforms based on configuration."""

    def __init__(self, cfg: ImageTransformsConfig) -> None:
        super().__init__()
        self._cfg = cfg

        self.weights = []
        self.transforms = {}
        for tf_name, tf_cfg in cfg.tfs.items():
            if tf_cfg.weight <= 0.0:
                continue

            self.transforms[tf_name] = make_transform_from_config(tf_cfg)
            self.weights.append(tf_cfg.weight)

        n_subset = min(len(self.transforms), cfg.max_num_transforms)
        if n_subset == 0 or not cfg.enable:
            self.tf = v2.Identity()
        else:
            self.tf = RandomSubsetApply(
                transforms=list(self.transforms.values()),
                p=self.weights,
                n_subset=n_subset,
                random_order=cfg.random_order,
            )

    def forward(self, *inputs: Any) -> Any:
        return self.tf(*inputs)
