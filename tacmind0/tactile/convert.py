"""Convert a LeRobot v3 root with marker videos to episode-isolated TacDream JSONL."""

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _convert_into(
    root: Path,
    output: Path,
    *,
    left_key="marker_left",
    right_key="marker_right",
    episode_limit: int | None = None,
    allow_tactile_rgb_as_marker: bool = False,
):
    if not allow_tactile_rgb_as_marker and (
        left_key == "tactile_left" or right_key == "tactile_right"
    ):
        raise ValueError(
            "tactile_left/right are policy tactile RGB, not marker videos; "
            "use marker_left/right or explicitly opt into the invalid alias"
        )
    info = json.loads((root / "meta/info.json").read_text())
    fps = float(info["fps"])
    if fps != 20:
        raise ValueError("v6 TacBench history requires 20 Hz data")
    names = ["agentview", "wrist", left_key, right_key]
    for name in names:
        if "observation.images." + name not in info["features"]:
            raise ValueError(
                f"Missing video feature {name}; tactile RGB is not marker input"
            )
    if (root / "meta/episodes.jsonl").exists():
        episodes = read_rows(root / "meta/episodes.jsonl")
    else:
        episodes = [
            row
            for path in sorted((root / "meta/episodes").glob("chunk-*/*.parquet"))
            for row in pq.read_table(path).to_pylist()
        ]
    episodes.sort(key=lambda row: int(row["episode_index"]))
    if not episodes:
        raise ValueError("No episode metadata found")
    task_rows = (
        read_rows(root / "meta/tasks.jsonl")
        if (root / "meta/tasks.jsonl").exists()
        else pq.read_table(root / "meta/tasks.parquet").to_pylist()
    )
    tasks = {
        int(row["task_index"]): row.get("task", row.get("__index_level_0__"))
        for row in task_rows
    }
    if any(not isinstance(text, str) or not text.strip() for text in tasks.values()):
        raise ValueError("Missing task instruction text")
    if episode_limit is not None:
        if (
            isinstance(episode_limit, bool)
            or not isinstance(episode_limit, int)
            or episode_limit <= 0
        ):
            raise ValueError("episode_limit must be a positive integer")
        episodes = episodes[:episode_limit]
    output.mkdir(parents=True, exist_ok=True)
    if list(output.glob("*.jsonl")):
        raise FileExistsError(
            "Choose a new output directory; existing episodes are not overwritten"
        )
    index = {}
    for episode in episodes:
        ep = int(episode["episode_index"])
        path = root / info["data_path"].format(
            chunk_index=int(episode["data/chunk_index"]),
            file_index=int(episode["data/file_index"]),
        )
        rows = pq.read_table(path, filters=[("episode_index", "=", ep)]).to_pylist()
        rows.sort(key=lambda row: row["frame_index"])
        if len(rows) != int(episode["length"]) or len(rows) < 2:
            raise ValueError(f"Invalid episode length: {ep}")
        refs = {}
        for name in names:
            key = "observation.images." + name
            prefix = "videos/" + key
            relative = info["video_path"].format(
                video_key=key,
                chunk_index=int(episode[prefix + "/chunk_index"]),
                file_index=int(episode[prefix + "/file_index"]),
            )
            if not (root / relative).is_file():
                raise FileNotFoundError(root / relative)
            refs[name] = (
                relative,
                round(float(episode[prefix + "/from_timestamp"]) * fps),
            )
        destination = output / f"episode_{ep:06d}.jsonl"
        with destination.open("w") as handle:
            for t, row in enumerate(rows):
                if int(row["frame_index"]) != t:
                    raise ValueError(f"Non-contiguous frame indices: {ep}")
                state, action = (
                    np.asarray(row["observation.state"]),
                    np.asarray(row["action"]),
                )
                if (
                    state.shape != (8,)
                    or action.shape != (8,)
                    or not (np.isfinite(state).all() and np.isfinite(action).all())
                ):
                    raise ValueError("Expected finite 8D state and action")
                record = {
                    "state": state.tolist(),
                    "action": action.tolist(),
                    "prompt": tasks[int(row["task_index"])],
                    "is_robot": True,
                }
                for field, name in zip(
                    ("images_1", "images_2", "marker_left", "marker_right"),
                    names,
                    strict=True,
                ):
                    url, start = refs[name]
                    record[field] = {
                        "type": "video",
                        "url": url,
                        "frame_idx": start + t,
                    }
                handle.write(json.dumps(record) + "\n")
        index[destination.name] = len(rows)
    (output / "index_cache.json").write_text(json.dumps({"data": index}, indent=2))
    (output / "contract.json").write_text(
        json.dumps(
            {
                "source_root": str(root.resolve()),
                "fps": fps,
                "marker_keys": [left_key, right_key],
                "episodes": len(index),
                "frames": sum(index.values()),
            },
            indent=2,
        )
    )
    return index


def convert(
    root: Path,
    output: Path,
    *,
    left_key="marker_left",
    right_key="marker_right",
    episode_limit: int | None = None,
    allow_tactile_rgb_as_marker: bool = False,
):
    """Publish a complete dataset; failed conversions leave no consumable output."""
    root, output = Path(root), Path(output)
    if output.exists():
        raise FileExistsError(
            "Choose a new output directory; existing data is not overwritten"
        )
    if (
        left_key == right_key
        or left_key in ("agentview", "wrist")
        or right_key in ("agentview", "wrist")
    ):
        raise ValueError(
            "Left and right marker keys must be distinct from each other and camera keys"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".tacdream-convert-", dir=output.parent
    ) as temporary:
        staged = Path(temporary) / "episodes"
        index = _convert_into(
            root,
            staged,
            left_key=left_key,
            right_key=right_key,
            episode_limit=episode_limit,
            allow_tactile_rgb_as_marker=allow_tactile_rgb_as_marker,
        )
        if output.exists():
            raise FileExistsError(output)
        staged.rename(output)
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--left-key", default="marker_left")
    parser.add_argument("--right-key", default="marker_right")
    parser.add_argument("--episode-limit", type=int)
    parser.add_argument(
        "--allow-tactile-rgb-as-marker",
        action="store_true",
        help="Explicitly allow the known-invalid tactile RGB to marker alias",
    )
    args = parser.parse_args()
    convert(
        args.dataset_root,
        args.output_dir,
        left_key=args.left_key,
        right_key=args.right_key,
        episode_limit=args.episode_limit,
        allow_tactile_rgb_as_marker=args.allow_tactile_rgb_as_marker,
    )


if __name__ == "__main__":
    main()
