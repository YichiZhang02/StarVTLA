import copy
import json
import unittest
from pathlib import Path

import numpy as np

from vtla.engine.configs import FeatureType
from vtla.engine.utils.constants import ACTION, OBS_STATE
from vtla.engine.utils.feature_utils import dataset_to_policy_features
from vtla.frameworks.act.configuration_act import ACTConfig
from vtla.frameworks.diffusion.configuration_diffusion import DiffusionConfig
from vtla.frameworks.ee_processor_utils import remap_ee_dataset_stats
from vtla.frameworks.fastwam.configuration_fastwam import FastWAMConfig
from vtla.frameworks.pi05.configuration_pi05 import PI05Config
from vtla.frameworks.starvla_groot.configuration_starvla_groot import StarvlaGrootConfig
from vtla.frameworks.starvla_groot_dinoalign.configuration_starvla_groot_dinoalign import (
    StarvlaGrootDinoAlignConfig,
)


DATASET_ROOT = Path(
    "playground/data/20260819_183840_rm_tactile_demo1_undist_uint8_256"
)
STATE_MODES = (
    "none",
    "absolute_joint",
    "episode_joint",
    "absolute_rot6d",
    "episode_rot6d",
    "absolute_quat",
    "episode_quat",
)
ACTION_MODES = (
    "absolute_joint",
    "relative_joint",
    "absolute_rot6d",
    "relative_rot6d",
    "absolute_quat",
    "relative_quat",
)
FRAMEWORK_CONFIGS = (
    ACTConfig,
    DiffusionConfig,
    PI05Config,
    StarvlaGrootConfig,
    StarvlaGrootDinoAlignConfig,
    FastWAMConfig,
)


@unittest.skipUnless(DATASET_ROOT.exists(), f"processed smoke-test dataset missing: {DATASET_ROOT}")
class ModeFrameworkMatrixTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.info = json.loads((DATASET_ROOT / "meta/info.json").read_text())
        cls.stats = {
            key: {stat: np.asarray(value) for stat, value in values.items()}
            for key, values in json.loads((DATASET_ROOT / "meta/stats.json").read_text()).items()
        }
        cls.features = dataset_to_policy_features(cls.info["features"])
        cls.wrist_keys = [
            key
            for key, feature in cls.info["features"].items()
            if feature.get("dtype") == "video" and "wrist" in key
        ]
        cls.state_names = list(cls.info["features"][OBS_STATE]["names"])
        cls.action_names = list(cls.info["features"][ACTION]["names"])

    def make_config(self, config_cls, state_mode, action_mode):
        config = config_cls(
            state_mode=state_mode,
            action_mode=action_mode,
            wrist_only=True,
            wrist_camera_keys=self.wrist_keys,
            tactile_mode="none",
            device="cpu",
        )
        config.ee_num_arms = 1
        config.state_feature_names = self.state_names
        config.action_feature_names = self.action_names
        config.input_features = {
            key: copy.deepcopy(feature)
            for key, feature in self.features.items()
            if feature.type is not FeatureType.ACTION
        }
        config.output_features = {
            key: copy.deepcopy(feature)
            for key, feature in self.features.items()
            if feature.type is FeatureType.ACTION
        }
        config.validate_features()
        return config

    def test_all_framework_mode_combinations_route_features_and_stats(self):
        checked = 0
        expected_state_dims = {
            "none": None,
            "absolute_joint": 8,
            "episode_joint": 8,
            "absolute_rot6d": 10,
            "episode_rot6d": 10,
            "absolute_quat": 8,
            "episode_quat": 8,
        }
        expected_action_dims = {"joint": 8, "rot6d": 10, "quat": 8}

        for config_cls in FRAMEWORK_CONFIGS:
            for state_mode in STATE_MODES:
                for action_mode in ACTION_MODES:
                    with self.subTest(
                        framework=config_cls.__name__,
                        state_mode=state_mode,
                        action_mode=action_mode,
                    ):
                        config = self.make_config(config_cls, state_mode, action_mode)
                        state_feature = config.input_features.get(OBS_STATE)
                        expected_state_dim = expected_state_dims[state_mode]
                        if expected_state_dim is None:
                            self.assertIsNone(state_feature)
                        else:
                            self.assertEqual(int(state_feature.shape[0]), expected_state_dim)

                        action_feature = config.output_features[ACTION]
                        self.assertEqual(
                            int(action_feature.shape[0]),
                            expected_action_dims[config.action_representation],
                        )
                        mapped_stats = remap_ee_dataset_stats(self.stats, config)
                        self.assertEqual(
                            int(np.asarray(mapped_stats[ACTION]["mean"]).shape[0]),
                            expected_action_dims[config.action_representation],
                        )
                        checked += 1

        self.assertEqual(checked, len(FRAMEWORK_CONFIGS) * len(STATE_MODES) * len(ACTION_MODES))


if __name__ == "__main__":
    unittest.main()
