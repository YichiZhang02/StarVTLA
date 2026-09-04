import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from tools.convert_umi_to_eepose import (
    pose_indices,
    set_gripper_calibration,
    to_absolute_ee_umi,
    to_absolute_quat_umi,
)
from tools.process_umi_data import (
    CAMERA_KEY_MAP,
    OUTPUT_VIDEO_KEYS,
    TACTILE_KEY_MAP,
    resolve_gripper_calibration,
    rewrite_episode_metadata,
    rewrite_global_stats,
    rewrite_info,
)
from tools.tactile_uint16_to_uint8 import TACTILE_UINT16_ENCODING


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
    tactile_feature = {
        "dtype": "tactile",
        "shape": [3, 96, 128],
        "names": ["channels", "height", "width"],
        "info": {"video.codec": "ffv1", "video.pix_fmt": "gbrp16le"},
    }
    features = {key: json.loads(json.dumps(video_feature)) for key in CAMERA_KEY_MAP}
    features.update(
        {key: json.loads(json.dumps(tactile_feature)) for key in TACTILE_KEY_MAP}
    )
    features["observation.images.unused_rgb"] = json.loads(json.dumps(video_feature))
    features["observation.state"] = {"dtype": "float32", "shape": [17], "names": _unified_names()}
    features["action"] = {"dtype": "float32", "shape": [17], "names": _unified_names()}
    source_info = {
        "fps": 30,
        "features": features,
    }

    result = rewrite_info(tmp_path, source_info, size=256)

    assert result["robot_type"] == "umi"
    assert result["ee_num_arms"] == 2
    assert result["ee_arm_sides"] == ["right", "left"]
    assert result["undistort"] == {"source_preprocessed": True, "crop": None}
    assert result["visual_preprocess"]["wrist_undistort"] is True
    assert result["visual_preprocess"]["wrist_crop"] is None
    assert result["visual_preprocess"]["resize"] == {
        "height": 256,
        "width": 256,
        "mode": "stretch",
        "interpolation": "lanczos",
    }
    assert set(OUTPUT_VIDEO_KEYS).issubset(result["features"])
    assert "observation.images.unused_rgb" not in result["features"]
    for key in CAMERA_KEY_MAP.values():
        feature = result["features"][key]
        assert feature["shape"] == [256, 256, 3]
        assert feature["names"] == ["height", "width", "channels"]
        assert feature["info"]["video.width"] == 256
    for key in TACTILE_KEY_MAP.values():
        feature = result["features"][key]
        assert feature["dtype"] == "video"
        assert feature["shape"] == [96, 128, 3]
        assert feature["names"] == ["height", "width", "channels"]
        assert feature["tactile_encoding"] == TACTILE_UINT16_ENCODING
        assert feature["storage_dtype"] == "uint16"
        assert feature["video_path"].endswith(".mkv")


def test_auto_gripper_calibration_uses_dataset_min_and_zero_closed(tmp_path):
    data_dir = tmp_path / "data" / "chunk-000"
    data_dir.mkdir(parents=True)
    state = np.zeros((2, len(_unified_names())), dtype=np.float32)
    action = state.copy()
    for vectors in (state, action):
        vectors[:, 6] = 1
        vectors[:, 13] = 1
    state[:, 15] = [-0.4, -0.3]
    state[:, 16] = [-0.3, -0.2]
    action[:, 15] = [-0.5, -0.25]
    action[:, 16] = [-0.35, -0.15]
    pq.write_table(
        pa.table({"observation.state": state.tolist(), "action": action.tolist()}),
        data_dir / "file-000.parquet",
    )
    info = {
        "features": {
            "observation.state": {"names": _unified_names()},
            "action": {"names": _unified_names()},
        }
    }

    calibration = resolve_gripper_calibration(
        tmp_path, info, {"left": (None, None), "right": (None, None)}
    )

    np.testing.assert_allclose(calibration["left"], (-0.5, 0.0))
    np.testing.assert_allclose(calibration["right"], (-0.35, 0.0))


def test_metadata_rewrite_maps_selected_visuals_and_drops_unused(tmp_path):
    meta = tmp_path / "meta"
    episodes_dir = meta / "episodes" / "chunk-000"
    episodes_dir.mkdir(parents=True)
    selected = list(CAMERA_KEY_MAP) + list(TACTILE_KEY_MAP)
    unused = "observation.images.unused_rgb"
    columns = {
        "episode_index": pa.array([0], type=pa.int64()),
        "tasks": pa.array([["old"]], type=pa.list_(pa.string())),
        "stats/observation.state/min": pa.array([[0.0]]),
    }
    for key in [*selected, unused]:
        columns[f"videos/{key}/chunk_index"] = pa.array([0], type=pa.int64())
        columns[f"videos/{key}/file_index"] = pa.array([0], type=pa.int64())
        columns[f"videos/{key}/from_timestamp"] = pa.array([0.0])
        columns[f"videos/{key}/to_timestamp"] = pa.array([99.0])
        columns[f"stats/{key}/min"] = pa.array([[[0.0]]])
    episode_path = episodes_dir / "file-000.parquet"
    pq.write_table(pa.table(columns), episode_path)
    (meta / "stats.json").write_text(
        json.dumps({key: {"min": [0]} for key in [*selected, unused, "observation.state"]})
    )
    video_feature = {"dtype": "video"}
    tactile_feature = {"dtype": "tactile"}
    source_info = {
        "features": {
            **{key: video_feature for key in CAMERA_KEY_MAP},
            **{key: tactile_feature for key in TACTILE_KEY_MAP},
            unused: video_feature,
            "observation.state": {"dtype": "float32"},
        }
    }

    rewrite_episode_metadata(tmp_path, "new task", 30, {0: 3}, source_info)
    rewrite_global_stats(tmp_path, source_info)

    table = pq.read_table(episode_path)
    assert table.column("tasks").to_pylist() == [["new task"]]
    assert not any(unused in name for name in table.column_names)
    for key in OUTPUT_VIDEO_KEYS:
        assert f"videos/{key}/file_index" in table.column_names
        assert table.column(f"videos/{key}/to_timestamp").to_pylist() == [0.1]
    stats = json.loads((meta / "stats.json").read_text())
    assert set(stats) == {"observation.state", *CAMERA_KEY_MAP.values()}
