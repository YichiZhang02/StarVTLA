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

import abc
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import draccus


@dataclass(kw_only=True)
class RobotConfig(draccus.ChoiceRegistry, abc.ABC):
    # Concrete robot configs own their physical kinematics identity. Generic
    # dataset/policy code queries it through the registry methods below.
    kinematics_force_type: ClassVar[str | None] = None
    kinematics_sides: ClassVar[tuple[str, ...]] = ()
    teleop_type: ClassVar[str | None] = None
    flange_tcp_xyz_m: ClassVar[dict[str, tuple[float, float, float]]] = {}
    flange_tcp_rpy_deg: ClassVar[dict[str, tuple[float, float, float]]] = {}

    # Runtime EE command contract. Inference overwrites this from policy.ee_frame.
    ee_frame: str = "flange"

    # Directory to store calibration file
    calibration_dir: Path | None = None

    def __post_init__(self):
        if self.ee_frame not in ("tcp", "flange"):
            raise ValueError(
                f"Invalid robot ee_frame={self.ee_frame!r}; expected 'tcp' or 'flange'."
            )
        if hasattr(self, "cameras") and self.cameras:
            for _, config in self.cameras.items():
                for attr in ["width", "height", "fps"]:
                    if getattr(config, attr) is None:
                        raise ValueError(
                            f"Specifying '{attr}' is required for the camera to be used in a robot"
                        )

    @classmethod
    def get_kinematics_config_class(cls, robot_type: str | None) -> type["RobotConfig"]:
        supported = cls.get_kinematics_robot_types()
        if not robot_type:
            raise ValueError(f"Missing robot_type; expected one of {supported}")
        try:
            config_cls = cls.get_choice_class(robot_type)
        except KeyError as exc:
            raise ValueError(
                f"Unsupported robot_type={robot_type!r}; expected one of {supported}"
            ) from exc
        if config_cls.kinematics_force_type is None:
            raise ValueError(
                f"robot_type={robot_type!r} does not define RealMan kinematics; "
                f"expected one of {supported}"
            )
        if not config_cls.kinematics_sides:
            raise ValueError(f"robot_type={robot_type!r} does not define an arm layout")
        return config_cls

    @classmethod
    def get_kinematics_force_type(cls, robot_type: str | None) -> str:
        force_type = cls.get_kinematics_config_class(robot_type).kinematics_force_type
        assert force_type is not None
        return force_type

    @classmethod
    def get_flange_tcp_calibration(
        cls, robot_type: str | None, side: str
    ) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        config_cls = cls.get_kinematics_config_class(robot_type)
        try:
            return config_cls.flange_tcp_xyz_m[side], config_cls.flange_tcp_rpy_deg[side]
        except KeyError as exc:
            raise ValueError(
                f"robot_type={robot_type!r} has no flange-to-TCP calibration for side={side!r}"
            ) from exc

    @classmethod
    def validate_kinematics_sides(
        cls, robot_type: str | None, sides: tuple[str, ...]
    ) -> type["RobotConfig"]:
        config_cls = cls.get_kinematics_config_class(robot_type)
        if sides != config_cls.kinematics_sides:
            raise ValueError(
                f"Dataset arm layout={sides} does not match robot_type={robot_type!r} "
                f"layout={config_cls.kinematics_sides}."
            )
        return config_cls

    @classmethod
    def get_kinematics_robot_types(cls) -> list[str]:
        return sorted(
            choice
            for choice in cls.get_known_choices()
            if cls.get_choice_class(choice).kinematics_force_type is not None
        )

    @property
    def type(self) -> str:
        return self.get_choice_name(self.__class__)
