import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

import tools.process_umi_data as process_umi_data
from tools.convert_umi_to_eepose import (
    pose_indices,
    set_gripper_calibration,
    to_absolute_ee_umi,
    to_absolute_quat_umi,
)
from tools.process_umi_data import (
    CAMERA_KEY_MAP,
    resolve_common_episode_lengths,
    resolve_gripper_calibration,
    rewrite_data_task_indices,
    rewrite_episode_metadata,
    rewrite_global_stats,
    rewrite_info,
)


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


def test_auto_gripper_calibration_uses_observed_min_and_zero_closed(tmp_path):
    data_dir = tmp_path / "data" / "chunk-000"
    data_dir.mkdir(parents=True)
    rows = np.array(
        [
            [1, 2, 3, 0, 0, 0, 1, 4, 5, 6, 0, 0, 0, 1, 9930, -0.40, -0.30],
            [1, 2, 3, 0, 0, 0, 1, 4, 5, 6, 0, 0, 0, 1, 9930, -0.15, -0.10],
        ],
        dtype=np.float64,
    )
    table = pa.table({"observation.state": pa.array(rows.tolist())})
    pq.write_table(table, data_dir / "file-000.parquet")
    info = {"features": {"observation.state": {"names": _unified_names()}}}

    calibration = resolve_gripper_calibration(
        tmp_path,
        info,
        {"left": (None, None), "right": (None, None)},
    )

    assert calibration == {"left": (-0.4, 0.0), "right": (-0.3, 0.0)}


def test_common_shortest_alignment_trims_data_rows_and_reindexes(tmp_path, monkeypatch):
    data_dir = tmp_path / "data" / "chunk-000"
    data_dir.mkdir(parents=True)
    table = pa.table(
        {
            "episode_index": [0, 0, 0, 1, 1],
            "frame_index": [0, 1, 2, 0, 1],
            "index": [0, 1, 2, 3, 4],
            "task_index": [9, 9, 9, 9, 9],
        }
    )
    pq.write_table(table, data_dir / "file-000.parquet")

    episode_dir = tmp_path / "meta" / "episodes" / "chunk-000"
    episode_dir.mkdir(parents=True)
    (tmp_path / "meta" / "stats.json").write_text(
        json.dumps({name: {} for name in ("frame_index", "index", "task_index")})
    )
    episode_columns = {
        "episode_index": [0, 1],
        "tasks": [["old"], ["old"]],
        "length": [3, 2],
        "dataset_from_index": [0, 3],
        "dataset_to_index": [3, 5],
        "stats/action/count": [[3], [2]],
    }
    for camera in CAMERA_KEY_MAP:
        episode_columns[f"videos/{camera}/chunk_index"] = [0, 0]
        episode_columns[f"videos/{camera}/file_index"] = [0, 1]
        episode_columns[f"videos/{camera}/from_timestamp"] = [0.0, 0.0]
        episode_columns[f"videos/{camera}/to_timestamp"] = [0.1, 2 / 30]
    stale_video = "observation.images.ego_left"
    stale_tactile = "observation.depth_deformation.tactile_left_left"
    for key in (stale_video, stale_tactile):
        episode_columns[f"videos/{key}/chunk_index"] = [0, 0]
        episode_columns[f"videos/{key}/file_index"] = [0, 1]
        episode_columns[f"videos/{key}/from_timestamp"] = [0.0, 0.0]
        episode_columns[f"videos/{key}/to_timestamp"] = [0.1, 2 / 30]
    pq.write_table(pa.table(episode_columns), episode_dir / "file-000.parquet")

    for camera in CAMERA_KEY_MAP:
        video_dir = tmp_path / "videos" / camera / "chunk-000"
        video_dir.mkdir(parents=True)
        for episode in (0, 1):
            (video_dir / f"file-{episode:03d}.mp4").touch()

    def fake_probe(path):
        if "cam_left_undist" in str(path) and path.name == "file-000.mp4":
            return 2
        return 3

    monkeypatch.setattr(process_umi_data, "_probe_frames", fake_probe)
    source_info = {
        "fps": 30,
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
    }

    target = resolve_common_episode_lengths(tmp_path, source_info)
    lengths = rewrite_data_task_indices(tmp_path, target_lengths=target)
    rewrite_episode_metadata(tmp_path, "new task", 30, lengths)
    rewrite_global_stats(tmp_path)
    trimmed = pq.read_table(data_dir / "file-000.parquet")
    episodes = pq.read_table(episode_dir / "file-000.parquet")
    stats = json.loads((tmp_path / "meta" / "stats.json").read_text())

    assert target == {0: 2, 1: 2}
    assert lengths == target
    assert trimmed.column("frame_index").to_pylist() == [0, 1, 0, 1]
    assert trimmed.column("index").to_pylist() == [0, 1, 2, 3]
    assert trimmed.column("task_index").to_pylist() == [0, 0, 0, 0]
    assert episodes.column("tasks").to_pylist() == [["new task"], ["new task"]]
    assert episodes.column("length").to_pylist() == [2, 2]
    assert episodes.column("dataset_from_index").to_pylist() == [0, 2]
    assert episodes.column("dataset_to_index").to_pylist() == [2, 4]
    assert "stats/action/count" not in episodes.column_names
    assert not any(name.startswith(f"videos/{stale_video}/") for name in episodes.column_names)
    assert not any(name.startswith(f"videos/{stale_tactile}/") for name in episodes.column_names)
    assert stats["index"]["max"] == [3.0]
    assert stats["index"]["count"] == [4]


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
        "features": {
            **{key: json.loads(json.dumps(video_feature)) for key in CAMERA_KEY_MAP},
            "observation.images.ego_left": json.loads(json.dumps(video_feature)),
            "observation.depth_deformation.tactile_left_left": {
                "dtype": "tactile",
                "shape": [3, 96, 128],
                "names": ["channels", "height", "width"],
            },
            "observation.state": {
                "dtype": "float32",
                "shape": [17],
                "names": _unified_names(),
            },
        },
    }

    result = rewrite_info(tmp_path, source_info, size=256)

    assert result["robot_type"] == "umi"
    assert result["ee_num_arms"] == 2
    assert result["ee_arm_sides"] == ["right", "left"]
    assert set(result["features"]) == set(CAMERA_KEY_MAP.values()) | {"observation.state"}
    assert "observation.images.ego_left" not in result["features"]
    assert "observation.depth_deformation.tactile_left_left" not in result["features"]
    for key in CAMERA_KEY_MAP.values():
        feature = result["features"][key]
        assert feature["shape"] == [256, 256, 3]
        assert feature["names"] == ["height", "width", "channels"]
        assert feature["info"]["video.width"] == 256


def test_no_tactile_stats_drop_unemitted_visuals_and_keep_derived_stats(tmp_path):
    meta = tmp_path / "meta"
    meta.mkdir()
    selected_source = "observation.images.cam_left_undist"
    selected_target = CAMERA_KEY_MAP[selected_source]
    stale_video = "observation.images.ego_left"
    stale_tactile = "observation.depth_deformation.tactile_left_left"
    stats = {
        selected_source: {"mean": [0.1]},
        stale_video: {"mean": [0.2]},
        stale_tactile: {"mean": [0.3]},
        "observation.state": {"mean": [0.4]},
        "action_relative_ee": {"mean": [0.5]},
    }
    (meta / "stats.json").write_text(json.dumps(stats))
    source_info = {
        "features": {
            selected_source: {"dtype": "video"},
            stale_video: {"dtype": "video"},
            stale_tactile: {"dtype": "tactile"},
            "observation.state": {"dtype": "float32"},
        }
    }

    rewrite_global_stats(tmp_path, source_info=source_info, tactile_mode="none")

    result = json.loads((meta / "stats.json").read_text())
    assert set(result) == {selected_target, "observation.state", "action_relative_ee"}
