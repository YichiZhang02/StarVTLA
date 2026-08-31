"""Parameter-free spatial pooling for downstream tactile tokens."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .backbone import EncodedFeatures, FeatureTokens


def pool_encoded_features(features: EncodedFeatures, pool_size: int = 3) -> FeatureTokens:
    if pool_size <= 0:
        raise ValueError("pool_size must be positive")
    grid = features.spatial_grid
    if grid.ndim != 6:
        raise ValueError(f"spatial_grid must be [B,S,T,H,W,D], got {tuple(grid.shape)}")
    b, sensors, temporal, height, width, dim = grid.shape
    pooled = F.adaptive_avg_pool2d(
        grid.permute(0, 1, 2, 5, 3, 4).reshape(b * sensors * temporal, dim, height, width),
        (pool_size, pool_size),
    )
    pooled = pooled.flatten(2).transpose(1, 2).reshape(b, sensors, temporal, pool_size**2, dim)

    token_groups = []
    sensor_groups = []
    time_groups = []
    globals_ = features.global_tokens
    global_times = features.global_time_ids.to(device=grid.device, dtype=torch.long)
    for sensor in range(sensors):
        if features.interleave_global:
            if globals_.shape[2] != temporal or not torch.equal(
                global_times, torch.arange(temporal, device=grid.device)
            ):
                raise ValueError("interleaved global tokens must have one token per temporal unit")
            for time in range(temporal):
                group = torch.cat([globals_[:, sensor, time : time + 1], pooled[:, sensor, time]], dim=1)
                token_groups.append(group)
                sensor_groups.append(torch.full((group.shape[1],), sensor, device=grid.device, dtype=torch.long))
                time_groups.append(torch.full((group.shape[1],), time, device=grid.device, dtype=torch.long))
        else:
            token_groups.append(globals_[:, sensor])
            sensor_groups.append(
                torch.full((globals_.shape[2],), sensor, device=grid.device, dtype=torch.long)
            )
            time_groups.append(global_times)
            for time in range(temporal):
                token_groups.append(pooled[:, sensor, time])
                sensor_groups.append(
                    torch.full((pool_size**2,), sensor, device=grid.device, dtype=torch.long)
                )
                time_groups.append(
                    torch.full((pool_size**2,), time, device=grid.device, dtype=torch.long)
                )
    tokens = torch.cat(token_groups, dim=1)
    sensor_ids = torch.cat(sensor_groups).unsqueeze(0).expand(b, -1)
    time_ids = torch.cat(time_groups).unsqueeze(0).expand(b, -1)
    token_mask = torch.ones((b, tokens.shape[1]), device=grid.device, dtype=torch.bool)
    return FeatureTokens(tokens=tokens, sensor_ids=sensor_ids, time_ids=time_ids, token_mask=token_mask)
