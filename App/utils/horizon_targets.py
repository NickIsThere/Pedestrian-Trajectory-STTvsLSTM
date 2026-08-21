from __future__ import annotations

import math
from typing import Any


DEFAULT_TARGET_HORIZONS_S = (0.5, 1.0, 2.0)

'''
nick grebe i6377605

These are the horizon settings and utilities for the evaluation in the statistical tests
'''


def available_horizon_seconds(
    *,
    future_steps: int,
    trajectory_stride: int,
    dataset_fps: float,
) -> list[float]:
    if future_steps <= 0:
        raise ValueError("future_steps must be positive")
    if trajectory_stride <= 0:
        raise ValueError("trajectory_stride must be positive")
    if math.isnan(dataset_fps) or dataset_fps <= 0:
        return []
    return [(idx * trajectory_stride) / dataset_fps for idx in range(1, future_steps + 1)]


def nearest_horizon_index(
    *,
    target_horizon_s: float,
    dataset_fps: float,
    trajectory_stride: int,
    future_steps: int,
) -> int:
    if future_steps <= 0:
        raise ValueError("future_steps must be positive")
    if trajectory_stride <= 0:
        raise ValueError("trajectory_stride must be positive")
    if math.isnan(dataset_fps) or dataset_fps <= 0:
        raise ValueError("dataset_fps must be positive")
    raw_index = target_horizon_s * dataset_fps / trajectory_stride
    rounded = int(math.floor(raw_index + 0.5))
    return max(1, min(future_steps, rounded))


def nearest_horizon_targets(
    *,
    target_horizons_s: tuple[float, ...] = DEFAULT_TARGET_HORIZONS_S,
    dataset_fps: float,
    trajectory_stride: int,
    future_steps: int,
) -> list[dict[str, float | int]]:
    mapping: list[dict[str, float | int]] = []
    """
        This is necessary due to the 25.5 fps we are training on
        For 0.5 sec horizon we are rounding up to 0.6 sec because of a frame mismatch
        
    """
    for target in target_horizons_s:
        horizon_idx = nearest_horizon_index(
            target_horizon_s=target,
            dataset_fps=dataset_fps,
            trajectory_stride=trajectory_stride,
            future_steps=future_steps,
        )
        actual = horizon_idx * trajectory_stride / dataset_fps
        mapping.append(
            {
                "target_horizon_s": float(target),
                "horizon_idx": horizon_idx,
                "actual_horizon_s": float(actual),
                "horizon_error_seconds": float(actual - target),
            },
        )
    return mapping


def target_lookup_by_horizon_idx(
    *,
    target_horizons_s: tuple[float, ...] = DEFAULT_TARGET_HORIZONS_S,
    dataset_fps: float,
    trajectory_stride: int,
    future_steps: int,
) -> dict[int, dict[str, float | int]]:
    if math.isnan(dataset_fps) or dataset_fps <= 0:
        return {}
    tolerance = trajectory_stride / dataset_fps / 2
    lookup: dict[int, dict[str, float | int]] = {}
    for item in nearest_horizon_targets(
        target_horizons_s=target_horizons_s,
        dataset_fps=dataset_fps,
        trajectory_stride=trajectory_stride,
        future_steps=future_steps,
    ):
        if abs(float(item["horizon_error_seconds"])) <= tolerance + 1e-12:
            lookup[int(item["horizon_idx"])] = item
    return lookup


def horizon_coverage_report(
    *,
    target_horizons_s: tuple[float, ...] = DEFAULT_TARGET_HORIZONS_S,
    dataset_fps: float,
    trajectory_stride: int,
    future_steps: int,
    tolerance_s: float | None = None,
) -> dict[str, Any]:
    """
    Generates a horizon coverage report for specified target horizon values,
    evaluating the available horizons against given constraints and tolerance.
    """
    available = available_horizon_seconds(
        future_steps=future_steps,
        trajectory_stride=trajectory_stride,
        dataset_fps=dataset_fps,
    )
    if not available:
        return {
            "horizon_s_values": [],
            "max_horizon_s": math.nan,
            "target_mappings": [],
            "target_horizons_s": list(target_horizons_s),
            "covered_targets_s": [],
            "missing_targets_s": list(target_horizons_s),
            "tolerance_s": math.nan if tolerance_s is None else tolerance_s,
            "coverage_pass": False,
        }

    tolerance = trajectory_stride / dataset_fps / 2 if tolerance_s is None else tolerance_s
    mappings = nearest_horizon_targets(
        target_horizons_s=target_horizons_s,
        dataset_fps=dataset_fps,
        trajectory_stride=trajectory_stride,
        future_steps=future_steps,
    )
    covered = [
        float(item["target_horizon_s"])
        for item in mappings
        if abs(float(item["horizon_error_seconds"])) <= tolerance + 1e-12
    ]
    missing = [float(target) for target in target_horizons_s if float(target) not in covered]
    return {
        "horizon_s_values": available,
        "max_horizon_s": max(available),
        "target_mappings": mappings,
        "target_horizons_s": list(target_horizons_s),
        "covered_targets_s": covered,
        "missing_targets_s": missing,
        "tolerance_s": tolerance,
        "coverage_pass": not missing,
    }
