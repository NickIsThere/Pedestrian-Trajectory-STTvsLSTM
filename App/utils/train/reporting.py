from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from evaluation.evaluation_units import config_hash
from model.config import ModelConfig

from utils.train.constants import DEFAULT_METRIC_FPS, LOSS_CONFIG, SampleMode
from utils.train.types import LeaveOneOutFold, ModelSpec


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_metadata_columns(run_id: str | None, config: ModelConfig) -> dict[str, Any]:
    effective_step_seconds = config.trajectory_stride / DEFAULT_METRIC_FPS
    return {
        "run_id": run_id or "",
        "lookback": config.lookback,
        "future_steps": config.future_steps,
        "trajectory_stride": config.trajectory_stride,
        "window_start_stride": config.window_start_stride,
        "fps": DEFAULT_METRIC_FPS,
        "effective_step_seconds": effective_step_seconds,
        "effective_future_seconds": config.future_steps * effective_step_seconds,
    }


def evaluation_context_columns(
    *,
    run_id: str | None,
    model_spec: ModelSpec,
    config: ModelConfig,
    seed: int,
    model_family: str,
    model_variant: str,
    train_domain: str,
    test_domain: str,
    synthetic_size_fraction: float | None,
) -> dict[str, Any]:
    return {
        "run_id": run_id or "",
        "model_name": model_spec.name,
        "model_family": model_family or "unknown",
        "model_variant": model_variant or "unknown",
        "train_domain": train_domain or "unknown",
        "test_domain": test_domain or "unknown",
        "synthetic_size_fraction": float("nan") if synthetic_size_fraction is None else synthetic_size_fraction,
        "eval_stage": "validation",
        "checkpoint_selection": "best_validation",
        "seed": seed,
        "config_hash": config_hash(
            config,
            {
                "model": model_spec.name,
                "seed": seed,
                "model_family": model_family or "unknown",
                "model_variant": model_variant or "unknown",
                "train_domain": train_domain or "unknown",
                "test_domain": test_domain or "unknown",
                "synthetic_size_fraction": synthetic_size_fraction,
            },
        ),
        "dataset_fps": float("nan"),
        "trajectory_stride": config.trajectory_stride,
        "coordinate_space": "normalized",
        "notes": "",
    }


def write_run_config(
    path: Path,
    *,
    model_spec: ModelSpec,
    config: ModelConfig,
    folds: list[LeaveOneOutFold],
    leave_one_out: bool,
    train_steps: int,
    val_interval: int,
    batch_size: int,
    max_val_samples: int,
    seed: int,
    training_input_mode: SampleMode,
    debug_smoke: bool,
    model_family: str,
    model_variant: str,
    train_domain: str,
    test_domain: str,
    synthetic_size_fraction: float | None,
) -> None:
    payload = {
        "model": model_spec.name,
        "model_family": model_family or "unknown",
        "model_variant": model_variant or "unknown",
        "train_domain": train_domain or "unknown",
        "test_domain": test_domain or "unknown",
        "synthetic_size_fraction": synthetic_size_fraction,
        "seed": seed,
        "training_input_mode": training_input_mode,
        "loss_config": LOSS_CONFIG,
        "leave_one_out": leave_one_out,
        "train_steps": train_steps,
        "val_interval": val_interval,
        "batch_size": batch_size,
        "max_val_samples": max_val_samples,
        "debug_smoke": debug_smoke,
        "metric_selection": "lowest validation ADE, then validation loss",
        "model_config": asdict(config),
        "folds": [
            {
                "fold": fold.index,
                "held_out_sequence": fold.val_sequence,
                "train_sequences": sorted(fold.train_split.sequences),
                "val_sequences": sorted(fold.val_split.sequences),
            }
            for fold in folds
        ],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
