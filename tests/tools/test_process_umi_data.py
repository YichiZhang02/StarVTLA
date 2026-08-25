import json

import numpy as np

from tools.convert_umi_to_eepose import (
    pose_indices,
    set_gripper_calibration,
    to_absolute_ee_umi,
    to_absolute_quat_umi,
)
from tools.process_umi_data import CAMERA_KEY_MAP, rewrite_info


def _unified_names():
    return [
        "left_x", "left_y", "left_z", "left_qx", "left_qy", "left_qz", "left_qw",
        "right_x", "right_y", "right_z", "right_qx", "right_qy", "right_qz", "right_qw",
        "gripper", "gripper_left", "gripper_right",
    ]


def test_unified_pose_names_and_grippers_are_packed_right_then_left():
    indices = pose_indices(_unified_names())
    set_gripper_calibration(indices, {"left": (-0.4, -0.3), "right": (-0.35, -0.15)})
    vector = np.array(
        [
            1, 2, 3, 0, 0, 0, 2,
            4, 5, 6, 0, 0, 0, 3,
            9930, -0.4, -0.15,
        ],
        dtype=np.float64,
    )

    rot6d = to_absolute_ee_umi(vector, indices)
    quat = to_absolute_quat_umi(vector, indices)

    np.testing.assert_allclose(rot6d[:3], [4, 5, 6])
    np.testing.assert_allclose(rot6d[10:13], [1, 2, 3])
    np.testing.assert_allclose(rot6d[[9, 19]], [0, 1])
    np.testing.assert_allclose(np.linalg.norm(quat[[3, 4, 5, 6]]), 1)
    np.testing.assert_allclose(np.linalg.norm(quat[[11, 12, 13, 14]]), 1)


def test_rewrite_info_canonicalizes_camera_axes_keys_and_robot_type(tmp_path):
    meta = tmp_path / "meta"
    meta.mkdir()
    video_feature = {
        "dtype": "video",
        "shape": [3, 480, 640],
        "names": ["channels", "height", "width"],
        "intrinsics": {"640x480": {"fx": 200, "fy": 300, "ppx": 320, "ppy": 240}},
        "info": {},
    }
    source_info = {
        "fps": 30,
        "features": {key: json.loads(json.dumps(video_feature)) for key in CAMERA_KEY_MAP},
    }

    result = rewrite_info(tmp_path, source_info, size=256)

    assert result["robot_type"] == "umi"
    assert result["ee_num_arms"] == 2
    assert result["ee_arm_sides"] == ["right", "left"]
    assert set(CAMERA_KEY_MAP.values()).issubset(result["features"])
    for key in CAMERA_KEY_MAP.values():
        feature = result["features"][key]
        assert feature["shape"] == [256, 256, 3]
        assert feature["names"] == ["height", "width", "channels"]
        assert feature["info"]["video.width"] == 256
