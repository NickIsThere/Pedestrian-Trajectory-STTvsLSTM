from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import torch

try:
    from utils.horizon_targets import target_lookup_by_horizon_idx
except ImportError:
    from utils.horizon_targets import target_lookup_by_horizon_idx


REQUIRED_EVALUATION_UNIT_COLUMNS = [
    "unit_id",
    "run_id",
    "model_name",
    "model_family",
    "model_variant",
    "train_domain",
    "test_domain",
    "synthetic_size_fraction",
    "eval_stage",
    "checkpoint_selection",
    "fold",
    "held_out_sequence",
    "scene_id",
    "sequence_id",
    "track_id",
    "window_id",
    "frame_id",
    "horizon_idx",
    "horizon_s",
    "target_horizon_s",
    "actual_horizon_s",
    "horizon_error_seconds",
    "pred_x",
    "pred_y",
    "target_x",
    "target_y",
    "error_l2",
    "ade_until_horizon",
    "fde_at_horizon",
    "seed",
    "split",
    "checkpoint_path",
    "config_hash",
    "dataset_fps",
    "trajectory_stride",
    "coordinate_space",
    "batch_latency_ms",
    "runtime_windows_per_second",
    "batch_size_windows",
    "batch_size_agents",
    "supervised",
    "notes",
]

CROSS_MODEL_PAIRING_KEY = [
    "sequence_id",
    "track_id",
    "window_id",
    "horizon_idx",
    "seed",
    "split",
    "test_domain",
]

ROW_UNIQUENESS_KEY = [
    "run_id",
    "model_name",
    *CROSS_MODEL_PAIRING_KEY,
]

CRITICAL_PAIRING_KEYS = ROW_UNIQUENESS_KEY


def config_hash(config: Any, extra: dict[str, Any] | None = None) -> str:
    payload = {
        "config": asdict(config) if is_dataclass(config) else dict(config or {}),
        "extra": extra or {},
    }
    encoded = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def write_evaluation_units_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REQUIRED_EVALUATION_UNIT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def validate_evaluation_units_csv(path: Path) -> None:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        missing = [column for column in REQUIRED_EVALUATION_UNIT_COLUMNS if column not in columns]
        if missing:
            raise ValueError(f"evaluation units CSV missing required columns: {missing}")

        seen: set[tuple[str, ...]] = set()
        missing_key_rows: list[int] = []
        duplicate_rows: list[int] = []
        for row_index, row in enumerate(reader, start=2):
            if any(_is_missing(row.get(column)) for column in CRITICAL_PAIRING_KEYS):
                missing_key_rows.append(row_index)
                continue

            key = tuple(str(row[column]) for column in ROW_UNIQUENESS_KEY)
            if key in seen:
                duplicate_rows.append(row_index)
            seen.add(key)

    if missing_key_rows:
        raise ValueError(f"evaluation units CSV has rows with missing critical pairing keys: {missing_key_rows}")
    if duplicate_rows:
        raise ValueError(f"evaluation units CSV has duplicate row uniqueness keys at rows: {duplicate_rows}")


def build_evaluation_unit_rows(
    *,
    samples: list[Any],
    pred_positions: torch.Tensor,
    target_positions: torch.Tensor,
    future_mask: torch.Tensor,
    config: Any,
    context: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    pred = pred_positions.detach().cpu()
    target = target_positions.detach().cpu().to(dtype=pred.dtype)
    valid = future_mask.detach().cpu().to(dtype=torch.bool)

    for batch_index, sample in enumerate(samples):
        metadata = dict(getattr(sample, "metadata", {}) or {})
        sequence_id = str(metadata.get("sequence", metadata.get("sequence_id", "unknown")))
        scene_id = str(metadata.get("scene_id", sequence_id))
        frame_id = metadata.get("frame_id", metadata.get("start_frame", "unknown"))
        dataset_fps = _optional_float(metadata.get("dataset_fps", context.get("dataset_fps", math.nan)))
        trajectory_stride = int(context.get("trajectory_stride", getattr(config, "trajectory_stride", 1)))
        target_lookup = target_lookup_by_horizon_idx(
            dataset_fps=dataset_fps,
            trajectory_stride=trajectory_stride,
            future_steps=pred.size(2),
        )

        track_ids = _track_ids_for_sample(sample)
        max_agents = pred.size(1)
        for agent_index in range(max_agents):
            if batch_index >= valid.size(0) or agent_index >= valid.size(1):
                continue
            if not bool(valid[batch_index, agent_index].any().item()):
                continue

            track_id = track_ids[agent_index] if agent_index < len(track_ids) else "unknown"
            window_id = str(metadata.get("window_id", f"{sequence_id}:{frame_id}"))
            errors = torch.linalg.vector_norm(
                pred[batch_index, agent_index] - target[batch_index, agent_index],
                dim=-1,
            )

            cumulative_errors: list[float] = []
            for horizon_zero_index in range(pred.size(2)):
                if not bool(valid[batch_index, agent_index, horizon_zero_index].item()):
                    continue
                error_l2 = float(errors[horizon_zero_index].item())
                cumulative_errors.append(error_l2)
                horizon_idx = horizon_zero_index + 1
                actual_horizon_s = _horizon_seconds(horizon_idx, trajectory_stride, dataset_fps)
                target_info = target_lookup.get(horizon_idx, {})
                row = {
                    **{column: context.get(column, _default_for_column(column)) for column in REQUIRED_EVALUATION_UNIT_COLUMNS},
                    "scene_id": scene_id,
                    "sequence_id": sequence_id,
                    "track_id": track_id,
                    "window_id": window_id,
                    "frame_id": frame_id,
                    "horizon_idx": horizon_idx,
                    "horizon_s": actual_horizon_s,
                    "target_horizon_s": target_info.get("target_horizon_s", math.nan),
                    "actual_horizon_s": actual_horizon_s,
                    "horizon_error_seconds": target_info.get("horizon_error_seconds", math.nan),
                    "pred_x": float(pred[batch_index, agent_index, horizon_zero_index, 0].item()),
                    "pred_y": float(pred[batch_index, agent_index, horizon_zero_index, 1].item()),
                    "target_x": float(target[batch_index, agent_index, horizon_zero_index, 0].item()),
                    "target_y": float(target[batch_index, agent_index, horizon_zero_index, 1].item()),
                    "error_l2": error_l2,
                    "ade_until_horizon": float(sum(cumulative_errors) / len(cumulative_errors)),
                    "fde_at_horizon": error_l2,
                    "dataset_fps": dataset_fps,
                    "trajectory_stride": trajectory_stride,
                    "supervised": True,
                }
                row["unit_id"] = _unit_id(row)
                rows.append(row)
    return rows


def _track_ids_for_sample(sample: Any) -> list[Any]:
    if hasattr(sample, "track_ids"):
        return list(getattr(sample, "track_ids"))
    metadata = dict(getattr(sample, "metadata", {}) or {})
    if "track_ids" in metadata:
        value = metadata["track_ids"]
        return list(value) if isinstance(value, (list, tuple)) else [value]
    return [metadata.get("track_id", "unknown")]


def _unit_id(row: dict[str, Any]) -> str:
    return "|".join(str(row[column]) for column in ROW_UNIQUENESS_KEY)


def _optional_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _horizon_seconds(horizon_idx: int, trajectory_stride: int, dataset_fps: float) -> float:
    if math.isnan(dataset_fps) or dataset_fps <= 0:
        return math.nan
    return horizon_idx * trajectory_stride / dataset_fps


def _default_for_column(column: str) -> Any:
    if column == "synthetic_size_fraction":
        return math.nan
    if column in {"target_horizon_s", "actual_horizon_s", "horizon_error_seconds"}:
        return math.nan
    if column == "supervised":
        return True
    if column in {"batch_latency_ms", "runtime_windows_per_second"}:
        return math.nan
    if column in {"batch_size_windows", "batch_size_agents", "fold", "seed", "horizon_idx"}:
        return 0
    return "unknown"


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    text = str(value).strip()
    return text == "" or text.lower() in {"nan", "none", "null"}
