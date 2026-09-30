"""Greedy farthest-point memory bank, nearest-neighbour scoring and calibration."""

from __future__ import annotations

import torch

from . import config


def build_memory_bank(descriptors: torch.Tensor, max_items: int | None = None) -> torch.Tensor:
    """Deterministic greedy farthest-point selection.

    ``descriptors`` is N x D. The first selected descriptor is index 0. Each
    subsequent point is the one farthest from the already selected set. Only
    distances from one newly added point to the pool are computed per step, so
    no N x N sample distance matrix is ever constructed.
    """
    if descriptors.ndim != 2 or descriptors.shape[0] == 0:
        raise ValueError("记忆库输入必须是非空的 N x D 描述子")
    limit = min(max_items or config.MAX_MEMORY_ITEMS, descriptors.shape[0])
    pool = descriptors.float()
    selected = [0]
    min_distances = torch.cdist(
        pool, pool[0:1]
    ).squeeze(1).clone()  # N, distances to seed
    for _ in range(1, limit):
        next_index = int(torch.argmax(min_distances).item())
        if min_distances[next_index] <= 0:
            break
        selected.append(next_index)
        new_distances = torch.cdist(pool, pool[next_index : next_index + 1]).squeeze(1)
        min_distances = torch.minimum(min_distances, new_distances)
        min_distances[next_index] = -1.0  # never reselect this point
    return pool[selected].contiguous()


def nearest_distances(
    descriptors: torch.Tensor, memory_bank: torch.Tensor, chunk: int = 64
) -> torch.Tensor:
    """Euclidean distance of every local descriptor to its nearest memory item."""
    if memory_bank.shape[0] == 0:
        raise RuntimeError("记忆库为空，无法检测")
    outputs = []
    for start in range(0, descriptors.shape[0], chunk):
        block = descriptors[start : start + chunk].float()
        distances = torch.cdist(block, memory_bank.float())
        outputs.append(distances.min(dim=1).values)
    return torch.cat(outputs)


def linear_quantile(values: torch.Tensor, percentile: float = config.PERCENTILE) -> float:
    """Linear-interpolated quantile (numpy/torch default definition)."""
    ordered, _ = torch.sort(values.float().flatten())
    if ordered.numel() == 1:
        return float(ordered.item())
    position = (percentile / 100.0) * (ordered.numel() - 1)
    lower = int(torch.floor(torch.tensor(position)).item())
    upper = int(torch.ceil(torch.tensor(position)).item())
    fraction = position - lower
    return float(ordered[lower].item() * (1.0 - fraction) + ordered[upper].item() * fraction)


def heatmap_upper_bound(threshold: float, calibration_scores: torch.Tensor) -> float:
    """Fixed color upper bound: twice the calibration threshold.

    When the threshold is zero, the largest observed calibration score is used
    as the explicit bound, with a small floor. This never autoscales per image.
    """
    if threshold > 0:
        return max(2.0 * threshold, config.DISTANCE_FLOOR)
    if calibration_scores.numel() > 0:
        observed = float(calibration_scores.max().item())
        if observed > 0:
            return observed
    return config.DISTANCE_FLOOR
