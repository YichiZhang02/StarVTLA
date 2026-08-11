#!/usr/bin/env python
"""End-to-end test for EE rotation modes: episode_rot6d, absolute_rot6d, episode_quat, absolute_quat.

Tests three phases:
1. Data processing: verify rot6d + quat columns exist with correct dimensions
2. Training: dataset loading → batch routing → processor pipeline (mini forward pass)
3. Inference: EpisodeEEPreprocessorStep → relative/absolute processor round-trip

Usage:
    python tests/test_ee_rot_modes.py [--dataset <dataset_path>]
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import torch

# Add repo root and SDK to path
REPO_ROOT = Path(__file__).resolve().parents[1]
SDK_PATH = REPO_ROOT / "deployment" / "sdk"
for p in [str(REPO_ROOT), str(SDK_PATH)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from vtla.engine.utils.ee_transforms import (
    PER_ARM_DIM_BY_ROT_MODE,
    _unpack,
    per_arm_dim,
)
from vtla.engine.processor.relative_action_processor import route_ee_batch
from vtla.frameworks.ee_processor_utils import make_ee_relative_steps

# All four mode combinations to test
ALL_MODES = [
    ("episode_rot6d", "rot6d"),
    ("absolute_rot6d", "rot6d"),
    ("episode_quat",   "quat"),
    ("absolute_quat",  "quat"),
]

PASS = "✅"
FAIL = "❌"


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1: Data Processing
# ─────────────────────────────────────────────────────────────────────────────
def test_phase1(dataset_root: Path) -> bool:
    """Verify all 10 EE columns exist in stats.json / info.json with correct dims."""
    print("\n=== Phase 1: Data Processing ===")

    stats_path = dataset_root / "meta" / "stats.json"
    info_path  = dataset_root / "meta" / "info.json"

    if not stats_path.exists():
        print(f"  {FAIL} No stats.json in {dataset_root}")
        return False

    stats = json.loads(stats_path.read_text())
    info  = json.loads(info_path.read_text())

    # (column, expected_dim, must_be_in_info)
    expected = [
        ("observation.state_episode_ee",   20, True),
        ("action_episode_ee",               20, True),
        ("observation.state_absolute_ee",   20, True),
        ("action_absolute_ee",              20, True),
        ("action_relative_ee",              20, False),  # stats-only
        ("observation.state_episode_quat",  16, True),
        ("action_episode_quat",             16, True),
        ("observation.state_absolute_quat", 16, True),
        ("action_absolute_quat",            16, True),
        ("action_relative_quat",            16, False),  # stats-only
    ]

    ok = True
    for col, dim, in_info in expected:
        if col not in stats:
            print(f"  {FAIL} {col}: missing in stats.json")
            ok = False
            continue
        got_dim = len(stats[col]["mean"])
        if got_dim != dim:
            print(f"  {FAIL} {col}: dim={got_dim} (expected {dim})")
            ok = False
            continue
        if in_info and col not in info["features"]:
            print(f"  {FAIL} {col}: missing in info.json")
            ok = False
            continue
        if in_info:
            got_shape = info["features"][col]["shape"]
            if got_shape != [dim]:
                print(f"  {FAIL} {col}: info shape={got_shape} (expected [{dim}])")
                ok = False
                continue
        print(f"  {PASS} {col}  dim={dim}")

    # Extra: verify parquet has the columns
    import glob, pyarrow.parquet as pq
    parquet_files = sorted(glob.glob(str(dataset_root / "data" / "**" / "*.parquet"), recursive=True))
    if parquet_files:
        tbl = pq.read_table(parquet_files[0])
        for col, dim, in_info in expected:
            if not in_info:
                continue
            if col not in tbl.column_names:
                print(f"  {FAIL} {col}: missing in parquet")
                ok = False
            else:
                first_val = tbl.column(col)[0].as_py()
                if len(first_val) != dim:
                    print(f"  {FAIL} {col}: parquet row dim={len(first_val)} (expected {dim})")
                    ok = False

    print(f"{'✅ Phase 1 PASS' if ok else '❌ Phase 1 FAIL'}")
    return ok


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: Training Pipeline
# ─────────────────────────────────────────────────────────────────────────────
def _make_mock_stats(state_mode: str, action_mode: str) -> dict:
    """Build minimal dataset stats for the given mode pair."""
    from vtla.frameworks.sensor_routing import (
        OBS_STATE_EPISODE_EE, OBS_STATE_ABSOLUTE_EE,
        OBS_STATE_EPISODE_QUAT, OBS_STATE_ABSOLUTE_QUAT,
        ACTION_RELATIVE_EE, ACTION_RELATIVE_QUAT,
    )
    state_key_map = {
        "episode_rot6d":  OBS_STATE_EPISODE_EE,
        "absolute_rot6d": OBS_STATE_ABSOLUTE_EE,
        "episode_quat":   OBS_STATE_EPISODE_QUAT,
        "absolute_quat":  OBS_STATE_ABSOLUTE_QUAT,
    }
    rot_mode = "quat" if "quat" in action_mode else "rot6d"
    ee_dim = per_arm_dim(rot_mode) * 2  # 2 arms

    stats: dict = {}
    if state_mode in state_key_map:
        stats[state_key_map[state_mode]] = {
            "mean": [0.0] * ee_dim, "std": [1.0] * ee_dim,
            "min": [-1.0]*ee_dim, "max": [1.0]*ee_dim,
        }
    if action_mode in ("rot6d", "quat"):
        rel_key = ACTION_RELATIVE_QUAT if action_mode == "quat" else ACTION_RELATIVE_EE
        stats[rel_key] = {
            "mean": [0.0] * ee_dim, "std": [1.0] * ee_dim,
            "min": [-1.0]*ee_dim, "max": [1.0]*ee_dim,
        }
    return stats


def test_phase2(dataset_root: Path) -> bool:
    """Test training pipeline: dataset load → route_ee_batch → processor pipeline."""
    print("\n=== Phase 2: Training Pipeline ===")

    from vtla.engine.configs.train import DatasetConfig, TrainPipelineConfig
    from vtla.frameworks.diffusion.configuration_diffusion import DiffusionConfig
    from vtla.datasets.factory import make_dataset
    from torch.utils.data import DataLoader
    from vtla.engine.processor import batch_to_transition
    from vtla.frameworks.diffusion.processor_diffusion import make_diffusion_pre_post_processors

    all_ok = True

    for state_mode, action_mode in ALL_MODES:
        label = f"{state_mode} + {action_mode}"
        rot_mode = "quat" if "quat" in state_mode else "rot6d"
        ee_dim = per_arm_dim(rot_mode) * 2
        try:
            # ── Build minimal config ──────────────────────────────────────────
            dataset_cfg = DatasetConfig(
                repo_id=dataset_root.name,
                root=str(dataset_root),
            )
            policy_cfg = DiffusionConfig(
                state_mode=state_mode,
                action_mode=action_mode,
                wrist_only=True,
                n_obs_steps=1,
                horizon=8,
                n_action_steps=8,
            )
            train_cfg = TrainPipelineConfig(
                dataset=dataset_cfg,
                policy=policy_cfg,
                batch_size=2,
                num_workers=0,
                steps=1,
            )

            # ── Load dataset ──────────────────────────────────────────────────
            ds = make_dataset(train_cfg)
            assert len(ds) > 0, "Empty dataset"

            # ── Get a raw batch ───────────────────────────────────────────────
            loader = DataLoader(ds, batch_size=2, shuffle=False, num_workers=0)
            batch = next(iter(loader))

            # ── route_ee_batch ────────────────────────────────────────────────
            batch = route_ee_batch(batch, state_mode, action_mode)

            obs_state = batch.get("observation.state")
            action    = batch.get("action")
            assert obs_state is not None, "observation.state missing after route_ee_batch"
            assert action is not None,    "action missing after route_ee_batch"

            # State shape: (B, n_obs, ee_dim) for diffusion with n_obs_steps=1
            # or (B, ee_dim) — accept both
            state_last_dim = obs_state.shape[-1]
            assert state_last_dim == ee_dim, \
                f"state dim={state_last_dim} expected {ee_dim}"

            action_last_dim = action.shape[-1]
            assert action_last_dim == ee_dim, \
                f"action dim={action_last_dim} expected {ee_dim}"

            print(f"  {PASS} route_ee_batch: state={obs_state.shape}, action={action.shape}")

            # ── Processor configuration ───────────────────────────────────────
            mock_stats = _make_mock_stats(state_mode, action_mode)
            preprocessor, postprocessor = make_diffusion_pre_post_processors(
                policy_cfg, mock_stats
            )

            # Verify the relative step is correctly wired with the right rot_mode
            from vtla.engine.processor import RelativeActionsProcessorStep
            rel_steps = [s for s in preprocessor.steps
                         if isinstance(s, RelativeActionsProcessorStep)]
            assert len(rel_steps) == 1, f"Expected 1 RelativeActionsProcessorStep"
            rel_step = rel_steps[0]
            assert rel_step.mode == "pose"
            assert rel_step.rot_mode == rot_mode, \
                f"rel_step.rot_mode={rel_step.rot_mode} expected {rot_mode}"
            assert rel_step.enabled

            print(f"  {PASS} processor config: {len(preprocessor.steps)} steps, "
                  f"rel_step.rot_mode={rel_step.rot_mode}")
            print(f"  {PASS} [{label}]")

        except Exception as exc:
            print(f"  {FAIL} [{label}]: {exc}")
            traceback.print_exc()
            all_ok = False

    print(f"{'✅ Phase 2 PASS' if all_ok else '❌ Phase 2 FAIL'}")
    return all_ok


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3: Inference Pipeline (no real robot SDK needed; FK mocked)
# ─────────────────────────────────────────────────────────────────────────────
def _make_valid_ee(n: int, rot_mode: str, n_arms: int = 2) -> torch.Tensor:
    """Return a (n, n_arms*per_arm_dim) tensor of valid EE poses."""
    from vtla.engine.utils.ee_transforms import matrix_to_rot6d, matrix_to_quat

    def rand_so3(batch):
        Q, _ = torch.linalg.qr(torch.randn(batch, 3, 3))
        Q[torch.linalg.det(Q) < 0, :, 0] *= -1
        return Q

    parts = []
    for _ in range(n_arms):
        pos  = torch.randn(n, 3)
        R    = rand_so3(n)
        grip = torch.rand(n, 1)
        rot  = matrix_to_rot6d(R) if rot_mode == "rot6d" else matrix_to_quat(R)
        parts.append(torch.cat([pos, rot, grip], dim=-1))
    return torch.cat(parts, dim=-1)


def test_phase3() -> bool:
    """Test inference processor chain without real robot SDK.

    For each (state_mode, action_mode) pair:
    1. Simulate a valid EE state and action (rot6d or quat).
    2. Run RelativeActionsProcessorStep (pose mode).
    3. Run AbsoluteActionsProcessorStep and verify round-trip.
    4. Verify the EpisodeEEPreprocessorStep signature is correct (skip FK call).
    """
    print("\n=== Phase 3: Inference Pipeline ===")

    from vtla.engine.processor import RelativeActionsProcessorStep, AbsoluteActionsProcessorStep
    from vtla.engine.types import TransitionKey

    all_ok = True

    for state_mode, action_mode in ALL_MODES:
        label = f"{state_mode} + {action_mode}"
        rot_mode = "quat" if "quat" in state_mode else "rot6d"
        ee_dim = per_arm_dim(rot_mode) * 2
        try:
            # ── Simulate EE state from "preprocessor" ─────────────────────────
            ref_state  = _make_valid_ee(1, rot_mode)  # (1, ee_dim)
            mock_action = _make_valid_ee(1, rot_mode)  # (1, ee_dim)

            # ── Build processor steps ─────────────────────────────────────────
            import types
            cfg = types.SimpleNamespace(
                state_mode=state_mode,
                action_mode=action_mode,
                ee_num_arms=2,
                use_relative_actions=False,
                relative_exclude_joints=[],
                action_feature_names=None,
            )
            rel_step, abs_step = make_ee_relative_steps(cfg)

            assert rel_step.rot_mode == rot_mode, \
                f"rel_step.rot_mode={rel_step.rot_mode} expected {rot_mode}"
            assert rel_step.mode == "pose"
            assert rel_step.enabled

            # ── relative step ─────────────────────────────────────────────────
            transition = {
                TransitionKey.OBSERVATION: {"observation.state": ref_state},
                TransitionKey.ACTION: mock_action,
            }
            rel_out = rel_step(transition)
            rel_action = rel_out[TransitionKey.ACTION]
            assert rel_action.shape == mock_action.shape, \
                f"rel_action shape {rel_action.shape} != {mock_action.shape}"

            # ── absolute step (round-trip) ─────────────────────────────────────
            abs_out = abs_step(rel_out)
            rec_action = abs_out[TransitionKey.ACTION]

            # Compare in rotation-matrix space (invariant to quat sign flip)
            pa, Ra, ga = _unpack(mock_action, 2, rot_mode)
            pr, Rr, gr = _unpack(rec_action,  2, rot_mode)
            assert torch.allclose(pa, pr, atol=1e-4), f"pos mismatch max={((pa-pr).abs().max()):.2e}"
            assert torch.allclose(Ra, Rr, atol=1e-4), f"rot mismatch max={((Ra-Rr).abs().max()):.2e}"
            assert torch.allclose(ga, gr, atol=1e-4), f"grip mismatch max={((ga-gr).abs().max()):.2e}"

            print(f"  {PASS} relative/absolute round-trip: "
                  f"rot6d={rot_mode=='rot6d'}, dim={ee_dim}")

            # ── Chunk action (B, T, D) round-trip ─────────────────────────────
            mock_chunk = _make_valid_ee(2, rot_mode).unsqueeze(1).expand(-1, 8, -1)  # (2, 8, D)
            ref2 = _make_valid_ee(2, rot_mode)
            t2 = {TransitionKey.OBSERVATION: {"observation.state": ref2},
                  TransitionKey.ACTION: mock_chunk}
            rel2 = rel_step(t2)
            abs2 = abs_step(rel2)
            rec2 = abs2[TransitionKey.ACTION]
            assert torch.allclose(mock_chunk, rec2, atol=1e-4), \
                f"chunk round-trip max_err={(mock_chunk-rec2).abs().max():.2e}"
            print(f"  {PASS} chunk (2,8,{ee_dim}) round-trip OK")

            # ── Verify EpisodeEEPreprocessorStep dataclass fields exist ────────
            from vtla.frameworks.episode_ee_processor import EpisodeEEPreprocessorStep
            import dataclasses
            field_names = {f.name for f in dataclasses.fields(EpisodeEEPreprocessorStep)}
            for required in ("state_feature_names", "relative_to_baseline", "rot_mode", "n_arms"):
                assert required in field_names, f"EpisodeEEPreprocessorStep missing field '{required}'"
            print(f"  {PASS} EpisodeEEPreprocessorStep fields: OK")

            print(f"  {PASS} [{label}]")

        except Exception as exc:
            print(f"  {FAIL} [{label}]: {exc}")
            traceback.print_exc()
            all_ok = False

    print(f"{'✅ Phase 3 PASS' if all_ok else '❌ Phase 3 FAIL'}")
    return all_ok


# ─────────────────────────────────────────────────────────────────────────────
# Extras: numerical consistency checks
# ─────────────────────────────────────────────────────────────────────────────
def test_rot6d_quat_consistency() -> bool:
    """Verify that rot6d and quat represent the same poses (conversion consistency)."""
    print("\n=== Extra: rot6d ↔ quat consistency ===")
    from vtla.engine.utils.ee_transforms import (
        matrix_to_rot6d, matrix_to_quat, rot6d_to_matrix, quat_to_matrix,
    )

    def rand_so3(n):
        Q, _ = torch.linalg.qr(torch.randn(n, 3, 3))
        Q[torch.linalg.det(Q) < 0, :, 0] *= -1
        return Q

    torch.manual_seed(42)
    R = rand_so3(64)  # (64, 3, 3)

    # rot6d round-trip
    r6d = matrix_to_rot6d(R)
    R2 = rot6d_to_matrix(r6d)
    assert torch.allclose(R, R2, atol=1e-5), "rot6d→matrix round-trip FAIL"
    print(f"  {PASS} rot6d → matrix → rot6d round-trip")

    # quat round-trip
    q = matrix_to_quat(R)
    R3 = quat_to_matrix(q)
    assert torch.allclose(R, R3, atol=1e-5), f"quat→matrix max_err={(R-R3).abs().max():.2e}"
    print(f"  {PASS} quat → matrix → quat round-trip")

    # Cross-check: rot6d and quat represent same rotation matrix
    q2 = matrix_to_quat(R2)
    R4 = quat_to_matrix(q2)
    assert torch.allclose(R, R4, atol=1e-5), "cross-check rot6d/quat FAIL"
    print(f"  {PASS} Cross-check rot6d↔quat: same rotation matrix")

    # ee_to_relative consistency: same result in both formats
    from vtla.engine.utils.ee_transforms import ee_to_relative, ee_to_absolute, _pack, _unpack

    ref6 = _make_valid_ee(4, "rot6d")
    actv6 = _make_valid_ee(4, "rot6d")
    rel6 = ee_to_relative(ref6, actv6, n_arms=2, rot_mode="rot6d")

    # Convert same poses to quat
    pr, Rr, gr = _unpack(ref6, 2, "rot6d")
    pa, Ra, ga = _unpack(actv6, 2, "rot6d")
    ref_q  = _pack(pr, Rr, gr, "quat")
    act_q  = _pack(pa, Ra, ga, "quat")
    rel_q  = ee_to_relative(ref_q, act_q, n_arms=2, rot_mode="quat")

    # Check that relative rotation matrix is the same
    _, R_rel6, g6  = _unpack(rel6,  2, "rot6d")
    _, R_relq, gq  = _unpack(rel_q, 2, "quat")
    assert torch.allclose(R_rel6, R_relq, atol=1e-5), \
        f"Relative R mismatch rot6d vs quat max_err={(R_rel6-R_relq).abs().max():.2e}"
    assert torch.allclose(g6, gq, atol=1e-5), "Gripper mismatch"
    print(f"  {PASS} ee_to_relative: same result in rot6d and quat")

    print("✅ Extra PASS")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, help="Path to test dataset (must have quat columns)")
    args = ap.parse_args()

    dataset_root = args.dataset

    # Auto-find smallest dataset with quat columns
    if dataset_root is None:
        data_dir = REPO_ROOT / "playground" / "data"
        for d in sorted(data_dir.iterdir()):
            if not d.is_dir():
                continue
            s = d / "meta" / "stats.json"
            if s.exists():
                stats = json.loads(s.read_text())
                if "observation.state_episode_quat" in stats:
                    dataset_root = d
                    break

    if dataset_root is None:
        print("❌ No dataset with quat columns found. Run first:")
        print("   python tools/convert_joints_to_eepose.py --root playground/data/<dataset>")
        return 1

    print("=" * 70)
    print(f"EE Rotation Modes Test  —  dataset: {dataset_root.name}")
    print("=" * 70)

    p1 = test_phase1(dataset_root)
    p2 = test_phase2(dataset_root)
    p3 = test_phase3()
    pe = test_rot6d_quat_consistency()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Phase 1 — Data Processing:              {'PASS ✅' if p1 else 'FAIL ❌'}")
    print(f"Phase 2 — Training Pipeline (4 modes):  {'PASS ✅' if p2 else 'FAIL ❌'}")
    print(f"Phase 3 — Inference Pipeline (4 modes): {'PASS ✅' if p3 else 'FAIL ❌'}")
    print(f"Extra   — rot6d↔quat consistency:       {'PASS ✅' if pe else 'FAIL ❌'}")
    overall = p1 and p2 and p3 and pe
    print(f"\n{'🎉 ALL TESTS PASSED' if overall else '💥 SOME TESTS FAILED'}")
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
