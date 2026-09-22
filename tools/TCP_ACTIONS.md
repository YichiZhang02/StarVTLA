# TCP rot6d data and action contract

All model EE poses describe the tool center point (TCP). `ee_frame` is removed.
Joint modes keep their existing semantics. EE modes only support rot6d.

| Mode | Position | Rotation | Gripper |
| --- | --- | --- | --- |
| `absolute_rot6d` | Base-frame TCP position | Base-frame TCP rot6d | Absolute |
| `relative_rot6d` | `Rs.T @ (pa - ps)` | `rot6d(Rs.T @ Ra) - [1,0,0,0,1,0]` | Absolute |

`rot6d` concatenates the first two **columns** of a rotation matrix. Relative actions
use one current-observation anchor for the entire chunk. Decode after unnormalization:
`pa = ps + Rs @ dp`, `Ra = Rs @ rot6d_to_matrix(dr + [1,0,0,0,1,0])`.
Zero unnormalized rotation residual means hold the anchor's TCP orientation.
Mean/std normalization is unchanged; zero normalized model output does not necessarily mean hold.
Finite degenerate reconstructed relative rotations hold the anchor orientation;
nonfinite relative commands are rejected.

`state_mode` supports `none`, `absolute_joint`, `episode_joint`, `absolute_rot6d`,
and `episode_rot6d`. Episode TCP states use a geometric `T0^-1 @ Tt` pose (identity
rotation is not zero-centered). Hidden action anchors always use absolute TCP poses.

Joint sources and runtime observations share FK -> flange -> calibrated TCP conversion.
`robot_type` selects FK, arm layout and flange-to-TCP extrinsics. UMI source poses are
already TCP, so only their raw quaternion representation is converted to rot6d. Raw
UMI inputs and SDK pose messages may use quaternions; model features and outputs do not.
The current RealMan adapters consume absolute TCP targets, limit motion in TCP space,
and convert to flange poses for their SDK. Logged sent EE targets remain TCP poses.

## Migrating existing datasets

Old EE checkpoints require retraining. Existing processed joint columns may contain
flange poses and must be regenerated from raw joint data. Never merely relabel their metadata.
The migration tool requires original `observation.state` and `action` source vectors.
It preserves the source and publishes a new destination after successful conversion.

```bash
python tools/migrate_tcp_dataset.py \
  --src playground/data/old_dataset --dst playground/data/tcp_dataset \
  --horizon 32 --action-gap 6 --dry-run

python tools/migrate_tcp_dataset.py \
  --src playground/data/old_dataset --dst playground/data/tcp_dataset \
  --horizon 32 --action-gap 6
```

Migration copies the dataset, including videos; budget destination disk space accordingly.
On failure the `.tcp-migration-partial` directory remains for inspection. Existing
outputs are never overwritten. Migration preserves calibrated grippers from existing absolute EE columns. If only
raw UMI inputs are available and gripper calibration is needed, process them first
using the UMI converter’s open/closed parameters; these are independent of TCP geometry.

Both standard converters emit `tcp_contract` in `meta/info.json`. This records the
encoding version, references, action offsets, padding convention and tool calibration.
Statistics count only valid, unpadded within-episode action pairs. Training verifies
its action offsets match these statistics; model loss masking remains framework-specific.
No quaternion-derived pose features are generated; obsolete derived columns and stats
are removed. Raw source vectors are preserved.

For already migrated TCP datasets, change statistics offsets without rewriting poses/videos:

```bash
python tools/rebuild_relative_ee_stats.py \
  --root playground/data/tcp_dataset --horizon 32 --action-gap 0
```

This updates global and episode relative statistics plus their contract. Gripper
normalization also refreshes relative/global/episode statistics using the same TCP
encoding. Dataset mixtures/merges reject mismatched contracts or missing relative stats.

Train with `action_mode=relative_rot6d` as before, on the newly processed dataset.
Task checkpoints save the contract; old EE checkpoints without it are rejected.
Generic UMI checkpoints still require an explicit physical `--robot.type` at deployment.

Diffusion may train on past actions as well (its first offset is
`1 - n_obs_steps + action_gap`). Rebuild statistics for that exact window, e.g.:

```bash
python tools/rebuild_relative_ee_stats.py \
  --root playground/data/tcp_dataset --horizon 16 --offset-start -1
```

For ACT temporal ensembling, runtime decodes each chunk against its own TCP anchor
before averaging absolute targets. Joint-mode ensembling is unchanged.
