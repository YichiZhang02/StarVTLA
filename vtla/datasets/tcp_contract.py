"""Versioned TCP data semantics, independent of user-selectable frame options."""
from __future__ import annotations

import json
from pathlib import Path

TCP_VERSION = "tcp_rot6d_local_delta_v1"
LEGACY_EE_FEATURES = {
    f"{prefix}_{reference}_quat"
    for prefix in ("observation.state", "action")
    for reference in ("absolute", "episode", "relative")
} | {"action_relative_ee_se3", "action_relative_ee_delta"}


def build_tcp_contract(robot_type: str, horizon: int, action_gap: int) -> dict:
    if horizon <= 0:
        raise ValueError("TCP statistics require horizon > 0")
    calibration = {}
    if robot_type != "umi":
        from deployment.robots import RobotConfig
        config = RobotConfig.get_kinematics_config_class(robot_type)
        calibration = {
            side: [list(x) for x in RobotConfig.get_flange_tcp_calibration(robot_type, side)]
            for side in config.kinematics_sides
        }
    return {
        "version": TCP_VERSION,
        "pose": "tcp",
        "absolute_reference": "base",
        "relative_reference": "current_tcp",
        "rotation": "columns_0_1",
        "relative_rotation": "rot6d(Rs.T@Ra)-[1,0,0,0,1,0]",
        "stats_offsets": list(range(action_gap, action_gap + horizon)),
        "stats_padding": "exclude_out_of_episode",
        "flange_tcp_calibration": calibration,
    }


def validate_tcp_contract(contract: dict | None, *, offsets=None, robot_type=None) -> None:
    if not isinstance(contract, dict) or contract.get("version") != TCP_VERSION:
        raise ValueError("Missing/obsolete TCP data contract; regenerate data with the current "
                         "converter or tools/migrate_tcp_dataset.py and retrain EE checkpoints.")
    stored = contract.get("stats_offsets", [])
    if not stored or stored != list(range(stored[0], stored[0] + len(stored))):
        raise ValueError("Invalid TCP statistics offsets")
    expected = build_tcp_contract(robot_type or "umi", len(stored), stored[0])
    for key in ("pose", "absolute_reference", "relative_reference", "rotation", "relative_rotation", "stats_padding"):
        if contract.get(key) != expected[key]:
            raise ValueError(f"TCP data contract mismatch for {key}")
    if offsets is not None and list(offsets) != stored:
        raise ValueError(f"TCP statistics offsets {stored} do not match training offsets {list(offsets)}; "
                         "rebuild relative statistics with the matching horizon/action-gap.")
    if robot_type is not None and contract.get("flange_tcp_calibration") != expected["flange_tcp_calibration"]:
        raise ValueError("TCP tool calibration changed; regenerate the dataset and retrain.")


def uses_tcp(config) -> bool:
    return (getattr(config, "action_representation", None) == "rot6d"
            or getattr(config, "state_representation", None) == "rot6d")


def drop_legacy_ee_columns(table):
    columns = [name for name in table.column_names
               if name in LEGACY_EE_FEATURES
               or any(name.startswith(f"stats/{key}/") for key in LEGACY_EE_FEATURES)]
    return table.drop(columns) if columns else table


def clean_legacy_ee_metadata(root: Path, info: dict) -> None:
    for key in LEGACY_EE_FEATURES:
        info["features"].pop(key, None)
    stats_path = root / "meta" / "stats.json"
    stats = json.loads(stats_path.read_text())
    for key in LEGACY_EE_FEATURES:
        stats.pop(key, None)
    stats_path.write_text(json.dumps(stats, indent=4) + "\n")
