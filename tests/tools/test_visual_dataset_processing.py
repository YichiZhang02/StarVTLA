import json

from tools.downscale_dataset_videos import downscale_videos_in_place
from tools.undistort_dataset_videos import patch_info_json


def test_no_crop_undistort_metadata_keeps_calibration_resolution(tmp_path):
    root = tmp_path / "dataset"
    (root / "meta").mkdir(parents=True)
    key = "observation.images.left_cam_wrist"
    (root / "meta" / "info.json").write_text(
        json.dumps(
            {
                "features": {
                    key: {
                        "dtype": "video",
                        "shape": [1080, 1920, 3],
                        "info": {},
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    calibration = tmp_path / "calibration.json"
    calibration.write_text(
        json.dumps(
            {
                "distortion_model": "equidistant",
                "camera_matrix": [[4, 0, 3.5], [0, 4, 2.5], [0, 0, 1]],
                "distortion_coeffs": [0, 0, 0, 0],
                "resolution": [1920, 1080],
            }
        ),
        encoding="utf-8",
    )

    patch_info_json(root, [key], None, "libx264", {key: str(calibration)})

    info = json.loads((root / "meta" / "info.json").read_text())
    assert info["features"][key]["shape"] == [1080, 1920, 3]
    assert info["undistort"]["crop"] is None


def test_already_resized_dataset_still_gets_visual_contract(tmp_path):
    root = tmp_path / "dataset"
    (root / "meta").mkdir(parents=True)
    (root / "videos").mkdir()
    (root / "meta" / "info.json").write_text(
        json.dumps(
            {
                "features": {
                    "observation.images.left_cam_wrist": {
                        "dtype": "video",
                        "shape": [224, 224, 3],
                        "info": {"video.pix_fmt": "yuv420p"},
                    }
                },
                "undistort": {"crop": None},
            }
        ),
        encoding="utf-8",
    )

    assert downscale_videos_in_place(root, 224, 4, 18, "lanczos", 1) == 0

    info = json.loads((root / "meta" / "info.json").read_text())
    assert info["visual_preprocess"]["wrist_undistort"] is True
    assert info["visual_preprocess"]["resize"]["height"] == 224
