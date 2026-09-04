from types import SimpleNamespace

import numpy as np
import pytest

from deployment.hardware.wrist_cameras.undistort import WristUndistorter
from deployment.inference import _resolve_dataset_fps, _resolve_undistort
from deployment.visual_preprocess import apply_visual_preprocess
from vtla.datasets.visual_preprocess import make_visual_preprocess


def test_online_visual_preprocess_resizes_rgb_and_tactile():
    contract = make_visual_preprocess(
        size=224, wrist_undistort=True, tactile_encoding="tactile_u8_linear_v1"
    )
    observation = {
        "observation.images.left_cam_wrist": np.zeros((1080, 1920, 3), dtype=np.uint8),
        "observation.images.left_cam_finger0": np.zeros((96, 128, 3), dtype=np.uint8),
        "observation.state": np.zeros(16, dtype=np.float32),
    }

    result = apply_visual_preprocess(observation, contract)

    assert result["observation.images.left_cam_wrist"].shape == (224, 224, 3)
    assert result["observation.images.left_cam_finger0"].shape == (224, 224, 3)
    assert result["observation.state"].shape == (16,)


def test_online_visual_preprocess_rejects_old_checkpoint_contract():
    with pytest.raises(ValueError, match="missing visual_preprocess"):
        apply_visual_preprocess({}, None)


def test_checkpoint_contract_enables_full_frame_undistort():
    contract = make_visual_preprocess(size=224, wrist_undistort=True, tactile_encoding=None)
    cfg = SimpleNamespace(
        robot=SimpleNamespace(undistort_wrist="auto", undistort_crop=896),
        policy=SimpleNamespace(visual_preprocess=contract),
    )

    _resolve_undistort(cfg)

    assert cfg.robot.undistort_wrist == "true"
    assert cfg.robot.undistort_crop is None


def test_checkpoint_fps_overrides_inference_default():
    cfg = SimpleNamespace(
        dataset=SimpleNamespace(fps=15),
        policy=SimpleNamespace(dataset_fps=30),
    )

    _resolve_dataset_fps(cfg)

    assert cfg.dataset.fps == 30


def test_wrist_undistorter_can_keep_full_frame(tmp_path):
    calibration = tmp_path / "calibration.json"
    calibration.write_text(
        '{"distortion_model":"equidistant","camera_matrix":'
        '[[4,0,3.5],[0,4,2.5],[0,0,1]],"distortion_coeffs":[0,0,0,0],'
        '"resolution":[8,6]}',
        encoding="utf-8",
    )
    undistorter = WristUndistorter(calibration, crop=None)

    output = undistorter(np.zeros((6, 8, 3), dtype=np.uint8))

    assert undistorter.out_shape == (6, 8, 3)
    assert output.shape == (6, 8, 3)
