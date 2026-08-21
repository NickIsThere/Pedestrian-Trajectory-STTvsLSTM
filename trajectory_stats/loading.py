from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from App.evaluation.evaluation_units import validate_evaluation_units_csv
from schema import validate_evaluation_unit_columns

"""
Nick Grebe i6377605

simply loads the evaluation_units.csv files and converts them into numeric columns with which we can do our tests
"""


NUMERIC_COLUMNS = [
    "synthetic_size_fraction",
    "fold",
    "track_id",
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
    "dataset_fps",
    "trajectory_stride",
    "batch_latency_ms",
    "runtime_windows_per_second",
    "batch_size_windows",
    "batch_size_agents",
]


def load_evaluation_units(inputs: Iterable[str | Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for item in inputs:
        path = Path(item)
        validate_evaluation_units_csv(path)
        frame = pd.read_csv(path)
        validate_evaluation_unit_columns(frame)
        frame["source_file"] = str(path)
        for column in NUMERIC_COLUMNS:
            if column in frame.columns:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frames.append(frame)

    if not frames:
        raise ValueError("At least one evaluation_units.csv input is required")
    return pd.concat(frames, ignore_index=True)
