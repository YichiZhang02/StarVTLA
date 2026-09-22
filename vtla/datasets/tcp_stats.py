"""Rebuild TCP relative statistics from absolute columns, using valid episode offsets."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from vtla.engine.utils.ee_transforms import encode_relative_tcp

STATE = 'observation.state_absolute_ee'
ACTION = 'action_absolute_ee'
RELATIVE = 'action_relative_ee'
QUANTILES = {'q01': .01, 'q10': .10, 'q50': .50, 'q90': .90, 'q99': .99}


def feature_stats(values):
    values = np.asarray(values, dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite TCP data cannot be normalized')
    result = {name: getattr(values, name)(axis=0).tolist() for name in ('min', 'max', 'mean', 'std')}
    result['count'] = [len(values)]
    result.update({key: np.quantile(values, q, axis=0).tolist() for key, q in QUANTILES.items()})
    return result


def collect_relative_stats(paths: list[Path], offsets: list[int]):
    """Return global and per-episode stats, never pairing across episode boundaries."""
    if not offsets or any(not isinstance(k, int) for k in offsets):
        raise ValueError('Expected integer action offsets')
    episodes = defaultdict(list)
    for path in paths:
        table = pq.read_table(path, columns=['episode_index', 'frame_index', STATE, ACTION])
        for row in table.to_pylist():
            episodes[row['episode_index']].append(row)
    all_relative, per_episode = [], {}
    for episode, rows in episodes.items():
        rows.sort(key=lambda row: row['frame_index'])
        frames = [row['frame_index'] for row in rows]
        if frames != list(range(len(frames))):
            raise ValueError(f'Episode {episode} must contain unique contiguous frames starting at zero')
        state = torch.tensor([row[STATE] for row in rows], dtype=torch.float32)
        action = torch.tensor([row[ACTION] for row in rows], dtype=torch.float32)
        if state.shape != action.shape or state.shape[-1] not in (10, 20):
            raise ValueError(f'Invalid TCP state/action layout in episode {episode}')
        chunks = []
        for k in offsets:
            start, stop = max(0, -k), min(len(rows), len(rows) - k)
            if start < stop:
                chunks.append(encode_relative_tcp(state[start:stop], action[start+k:stop+k],
                                                   state.shape[-1] // 10).numpy())
        if not chunks:
            per_episode[episode] = None
            continue
        values = np.concatenate(chunks)
        all_relative.append(values)
        per_episode[episode] = feature_stats(values)
    if not all_relative:
        raise ValueError('No valid state/action pairs for requested offsets')
    return feature_stats(np.concatenate(all_relative)), per_episode


def prepare_episode_stats(root: Path, data_paths: list[Path], relative_stats: dict,
                          features: list[str] = ()) -> list[tuple[Path, Path]]:
    """Stage updated episode stats; caller publishes alongside transformed data/global stats.

    Optional numeric features refresh per-episode values after gripper normalization.
    """
    gathered = defaultdict(lambda: defaultdict(list))
    if features:
        for path in data_paths:
            table = pq.read_table(path, columns=['episode_index', *features])
            for row in table.to_pylist():
                for feature in features:
                    gathered[row['episode_index']][feature].append(row[feature])
    prepared = []
    try:
        for path in sorted((root / 'meta/episodes').rglob('*.parquet')):
            table = pq.read_table(path)
            episodes = table['episode_index'].to_pylist()
            feature_map = {RELATIVE: relative_stats}
            for feature in features:
                feature_map[feature] = {ep: feature_stats(gathered[ep][feature]) for ep in episodes}
            for feature, stats in feature_map.items():
                for name in ('min', 'max', 'mean', 'std', 'count', *QUANTILES):
                    key = f'stats/{feature}/{name}'
                    if key in table.column_names:
                        table = table.drop([key])
                    values = [stats[ep][name] if stats.get(ep) is not None else None for ep in episodes]
                    dtype = pa.list_(pa.int64()) if name == 'count' else pa.list_(pa.float64())
                    table = table.append_column(key, pa.array(values, type=dtype))
            temporary = path.with_name('.' + path.name + '.tcp-stats.tmp')
            if temporary.exists():
                raise FileExistsError(temporary)
            prepared.append((temporary, path))
            pq.write_table(table, temporary)
    except Exception:
        for temporary, _ in prepared:
            temporary.unlink(missing_ok=True)
        raise
    return prepared
