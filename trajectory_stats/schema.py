from __future__ import annotations

from typing import Literal

import pandas as pd

from App.evaluation.evaluation_units import REQUIRED_EVALUATION_UNIT_COLUMNS
from App.utils.horizon_targets import horizon_coverage_report

"""
Nick Grebe i6377605

Central defnition of the metrics we are collecting for the research question 


"""

AggregationLevel = Literal["window", "track", "scene"]

COMMON_AGGREGATION_KEYS = [
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
    "sequence_id",
    "horizon_idx",
    "horizon_s",
    "target_horizon_s",
    "actual_horizon_s",
    "horizon_error_seconds",
    "seed",
    "split",
    "coordinate_space",
    "checkpoint_path",
    "config_hash",
]

WINDOW_EXTRA_KEYS = ["track_id", "window_id", "frame_id"]
TRACK_EXTRA_KEYS = ["track_id"]
SCENE_EXTRA_KEYS: list[str] = []

PAIRING_KEYS_BY_LEVEL: dict[AggregationLevel, list[str]] = {
    "window": [
        "sequence_id",
        "track_id",
        "window_id",
        "horizon_idx",
        "seed",
        "split",
        "test_domain",
        "coordinate_space",
    ],
    "track": [
        "sequence_id",
        "track_id",
        "horizon_idx",
        "seed",
        "split",
        "test_domain",
        "coordinate_space",
    ],
    "scene": [
        "sequence_id",
        "horizon_idx",
        "seed",
        "split",
        "test_domain",
        "coordinate_space",
    ],
}

SUPPORTED_COMPARISON_METRICS = [
    "mean_error_l2",
    "mean_ade_until_horizon",
    "mean_fde_at_horizon",
]


def aggregation_group_keys(level: AggregationLevel) -> list[str]:
    if level == "window":
        return [*COMMON_AGGREGATION_KEYS, *WINDOW_EXTRA_KEYS]
    if level == "track":
        return [*COMMON_AGGREGATION_KEYS, *TRACK_EXTRA_KEYS]
    if level == "scene":
        return [*COMMON_AGGREGATION_KEYS, *SCENE_EXTRA_KEYS]
    raise ValueError(f"Unknown aggregation level: {level}")


def validate_required_columns(df: pd.DataFrame, required_columns: list[str]) -> None:
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")


def validate_evaluation_unit_columns(df: pd.DataFrame) -> None:
    validate_required_columns(df, REQUIRED_EVALUATION_UNIT_COLUMNS)


def horizon_coverage(df: pd.DataFrame, research_horizons: tuple[float, ...] = (0.5, 1.0, 2.0)) -> dict[str, object]:
    validate_required_columns(df, ["horizon_s", "dataset_fps", "trajectory_stride", "horizon_idx"])
    horizon_values = sorted(float(value) for value in pd.to_numeric(df["horizon_s"], errors="coerce").dropna().unique())
    dataset_fps_values = pd.to_numeric(df["dataset_fps"], errors="coerce").dropna().unique()
    stride_values = pd.to_numeric(df["trajectory_stride"], errors="coerce").dropna().unique()
    max_horizon_idx = pd.to_numeric(df["horizon_idx"], errors="coerce").max()
    if len(dataset_fps_values) == 1 and len(stride_values) == 1 and pd.notna(max_horizon_idx):
        report = horizon_coverage_report(
            target_horizons_s=research_horizons,
            dataset_fps=float(dataset_fps_values[0]),
            trajectory_stride=int(stride_values[0]),
            future_steps=int(max_horizon_idx),
        )
    else:
        report = {
            "target_horizons_s": list(research_horizons),
            "target_mappings": [],
            "covered_targets_s": [],
            "missing_targets_s": list(research_horizons),
            "tolerance_s": float("nan"),
            "coverage_pass": False,
        }
    return {
        **report,
        "horizon_s_values": horizon_values,
        "max_horizon_s": max(horizon_values) if horizon_values else float("nan"),
        "research_horizons": list(research_horizons),
        "covered_research_horizons": report["covered_targets_s"],
        "missing_research_horizons": report["missing_targets_s"],
    }


def validation_report(df: pd.DataFrame) -> dict[str, object]:
    validate_evaluation_unit_columns(df)
    dataset_fps = pd.to_numeric(df["dataset_fps"], errors="coerce")
    horizon_s = pd.to_numeric(df["horizon_s"], errors="coerce")
    warnings: list[str] = []
    missing_dataset_fps_rows = int(dataset_fps.isna().sum())
    missing_horizon_s_rows = int(horizon_s.isna().sum())
    if missing_dataset_fps_rows:
        warnings.append(
            f"{missing_dataset_fps_rows} row(s) have missing dataset_fps; horizon timing cannot be trusted there.",
        )
    if missing_horizon_s_rows:
        warnings.append(f"{missing_horizon_s_rows} row(s) have missing horizon_s.")
    return {
        "rows": len(df),
        **horizon_coverage(df),
        "missing_dataset_fps_rows": missing_dataset_fps_rows,
        "missing_horizon_s_rows": missing_horizon_s_rows,
        "warnings": warnings,
    }
