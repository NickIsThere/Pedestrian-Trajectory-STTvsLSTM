from __future__ import annotations

import random
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from data.reader import Dataset
from evaluation.evaluation_units import validate_evaluation_units_csv, write_evaluation_units_csv
from model.config import ModelConfig

from utils.train.config import (
    BATCH_SIZE,
    MAX_VAL_SAMPLES,
    TRAIN_STEPS,
    TRAINING_DEFAULTS,
    VAL_INTERVAL,
)
from utils.train.constants import DOMAIN_VALUES, SAMPLE_MODES, SampleMode
from utils.train.datasets import HTPSceneIterableDataset, HTPIterableDataset
from utils.train.evaluation import evaluate_split_for_model, print_debug_smoke_batch
from utils.train.loss import compute_training_loss
from utils.train.models.registry import ModelSpec
from utils.train.reporting import (
    evaluation_context_columns,
    run_metadata_columns,
    write_csv,
    write_run_config,
)
from utils.train.splits import build_leave_one_out_folds, build_supervised_splits
from utils.train.types import LeaveOneOutFold


def select_training_folds(
    dataset: Dataset,
    *,
    leave_one_out: bool,
    fold_sequence: str | None,
) -> list[LeaveOneOutFold]:
    if leave_one_out:
        folds = build_leave_one_out_folds(dataset)
    else:
        train_split, val_split = build_supervised_splits(dataset)
        val_sequence = next(iter(val_split.sequences))
        folds = [LeaveOneOutFold(index=0, val_sequence=val_sequence, train_split=train_split, val_split=val_split)]

    if fold_sequence is None:
        return folds
    selected = [fold for fold in folds if fold.val_sequence == fold_sequence]
    if not selected:
        available = ", ".join(fold.val_sequence for fold in folds)
        raise KeyError(f"Unknown fold sequence {fold_sequence!r}; available: {available}")
    return selected


def train_one_fold(
    *,
    fold: LeaveOneOutFold,
    model_spec: ModelSpec,
    config: ModelConfig,
    run_dir: Path,
    train_steps: int,
    val_interval: int,
    batch_size: int,
    max_val_samples: int,
    device: torch.device,
    training_input_mode: SampleMode = "gt",
    debug_smoke: bool = False,
    export_evaluation_units: bool = False,
    evaluation_context: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    model = model_spec.build_model(config, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=TRAINING_DEFAULTS.learning_rate)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=TRAINING_DEFAULTS.learning_rate,
        total_steps=train_steps,
        pct_start=0.1,
    )

    train_stream = iter(
        HTPSceneIterableDataset(fold.train_split, config, input_mode=training_input_mode)
        if model_spec.name == "transformer"
        else HTPIterableDataset(fold.train_split, config, input_mode=training_input_mode),
    )
    checkpoint_path = run_dir / "checkpoints" / f"{model_spec.name}__val_{fold.val_sequence}.pt"

    metric_rows: list[dict[str, Any]] = []
    best_row: dict[str, Any] | None = None
    best_evaluation_unit_rows: list[dict[str, Any]] = []
    best_score = (float("inf"), float("inf"))

    for step in tqdm(range(1, train_steps + 1), desc=f"{model_spec.name}:{fold.val_sequence}"):
        samples = [next(train_stream) for _ in range(batch_size)]
        prepared = model_spec.prepare_batch(samples, device=device, config=config)

        model.train()
        optimizer.zero_grad()
        output = model(**prepared.inputs)
        if debug_smoke and step == 1:
            print_debug_smoke_batch(f"train fold={fold.val_sequence}", config, prepared, output)
        pred_deltas, pred_positions = model_spec.read_predictions(output)
        loss_parts = compute_training_loss(
            {"pred_deltas": pred_deltas, "pred_positions": pred_positions},
            prepared.target_deltas,
            prepared.target_positions,
            prepared.future_mask,
        )
        loss = loss_parts["loss"]
        loss.backward()
        optimizer.step()

        scheduler.step()

        if step % val_interval != 0 and step != train_steps:
            continue

        step_evaluation_unit_rows: list[dict[str, Any]] = []
        fold_evaluation_context = {
            **(evaluation_context or {}),
            "fold": fold.index,
            "held_out_sequence": fold.val_sequence,
            "split": fold.val_split.name,
            "checkpoint_path": str(checkpoint_path),
        }
        metrics = evaluate_split_for_model(
            model,
            model_spec,
            fold.val_split,
            config,
            batch_size=batch_size,
            device=device,
            input_mode=training_input_mode,
            max_samples=max_val_samples,
            progress_desc=f"validate:{model_spec.name}:{fold.val_sequence}:step{step}",
            debug_smoke=debug_smoke,
            debug_label=f"val fold={fold.val_sequence}",
            evaluation_unit_rows=step_evaluation_unit_rows if export_evaluation_units else None,
            evaluation_context=fold_evaluation_context,
        )
        base_metric_keys = ("val_loss", "delta_mse", "position_rmse", "ade", "fde")
        row = {
            "model": model_spec.name,
            "fold": fold.index,
            "held_out_sequence": fold.val_sequence,
            "step": step,
            "train_loss": float(loss.item()),
            **{key: float(metrics[key]) for key in base_metric_keys},
            **{
                key: float(value)
                for key, value in metrics.items()
                if key not in {*base_metric_keys, "val_samples"}
            },
            "val_samples": int(metrics["val_samples"]),
            "checkpoint_path": str(checkpoint_path),
        }
        metric_rows.append(row)

        score = (row["ade"], row["val_loss"])
        if score < best_score:
            best_score = score
            best_row = row
            if export_evaluation_units:
                best_evaluation_unit_rows = step_evaluation_unit_rows
            model_spec.save_checkpoint(
                model,
                checkpoint_path,
                {
                    "model": model_spec.name,
                    "model_config": config,
                    "fold": fold.index,
                    "held_out_sequence": fold.val_sequence,
                    "step": step,
                    "training_input_mode": training_input_mode,
                    "metrics": {
                        key: value
                        for key, value in row.items()
                        if key not in {"model", "fold", "held_out_sequence", "step", "train_loss", "checkpoint_path"}
                    },
                },
            )

    if best_row is None:
        raise RuntimeError("No validation step ran; reduce val_interval or increase train_steps.")

    summary = {
        "model": model_spec.name,
        "fold": fold.index,
        "held_out_sequence": fold.val_sequence,
        "best_step": best_row["step"],
        "best_checkpoint": str(checkpoint_path),
    }
    summary.update(
        {
            f"best_{key}": value
            for key, value in best_row.items()
            if key not in {"model", "fold", "held_out_sequence", "step", "train_loss", "checkpoint_path"}
        },
    )
    return metric_rows, summary, best_evaluation_unit_rows


def run_training_folds(
    *,
    dataset: Dataset,
    model_spec: ModelSpec,
    config: ModelConfig,
    output_dir: Path,
    run_id: str | None = None,
    leave_one_out: bool = False,
    fold_sequence: str | None = None,
    train_steps: int = TRAIN_STEPS,
    val_interval: int = VAL_INTERVAL,
    batch_size: int = BATCH_SIZE,
    max_val_samples: int = MAX_VAL_SAMPLES,
    seed: int = 0,
    device: torch.device | None = None,
    training_input_mode: SampleMode = "gt",
    debug_smoke: bool = False,
    export_evaluation_units: bool = False,
    model_family: str = "unknown",
    model_variant: str = "unknown",
    train_domain: str = "unknown",
    test_domain: str = "unknown",
    synthetic_size_fraction: float | None = None,
) -> Path:
    if train_steps <= 0:
        raise ValueError("train_steps must be positive")
    if val_interval <= 0:
        raise ValueError("val_interval must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if max_val_samples < 0:
        raise ValueError("max_val_samples must be non-negative")
    if training_input_mode not in SAMPLE_MODES:
        raise ValueError(f"training_input_mode must be one of {SAMPLE_MODES}, got {training_input_mode!r}")
    if train_domain not in DOMAIN_VALUES:
        raise ValueError(f"train_domain must be one of {sorted(DOMAIN_VALUES)}, got {train_domain!r}")
    if test_domain not in DOMAIN_VALUES:
        raise ValueError(f"test_domain must be one of {sorted(DOMAIN_VALUES)}, got {test_domain!r}")
    if export_evaluation_units and (train_domain == "unknown" or test_domain == "unknown"):
        warnings.warn(
            "evaluation_units.csv export enabled with unknown train_domain/test_domain; "
            "set TRAIN_DOMAIN and TEST_DOMAIN before inferential statistics.",
            UserWarning,
            stacklevel=2,
        )

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    selected_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    folds = select_training_folds(dataset, leave_one_out=leave_one_out, fold_sequence=fold_sequence)
    run_dir = Path(output_dir) / (run_id or time.strftime("%Y%m%d-%H%M%S"))
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    write_run_config(
        run_dir / "config.json",
        model_spec=model_spec,
        config=config,
        folds=folds,
        leave_one_out=leave_one_out,
        train_steps=train_steps,
        val_interval=val_interval,
        batch_size=batch_size,
        max_val_samples=max_val_samples,
        seed=seed,
        training_input_mode=training_input_mode,
        debug_smoke=debug_smoke,
        model_family=model_family,
        model_variant=model_variant,
        train_domain=train_domain,
        test_domain=test_domain,
        synthetic_size_fraction=synthetic_size_fraction,
    )

    metric_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    evaluation_unit_rows: list[dict[str, Any]] = []
    metadata_columns = run_metadata_columns(run_id, config)
    base_evaluation_context = evaluation_context_columns(
        run_id=run_id,
        model_spec=model_spec,
        config=config,
        seed=seed,
        model_family=model_family,
        model_variant=model_variant,
        train_domain=train_domain,
        test_domain=test_domain,
        synthetic_size_fraction=synthetic_size_fraction,
    )
    for fold in folds:
        fold_rows, summary, fold_evaluation_unit_rows = train_one_fold(
            fold=fold,
            model_spec=model_spec,
            config=config,
            run_dir=run_dir,
            train_steps=train_steps,
            val_interval=val_interval,
            batch_size=batch_size,
            max_val_samples=max_val_samples,
            device=selected_device,
            training_input_mode=training_input_mode,
            debug_smoke=debug_smoke,
            export_evaluation_units=export_evaluation_units,
            evaluation_context=base_evaluation_context,
        )
        for row in fold_rows:
            row.update(metadata_columns)
        summary.update(metadata_columns)
        metric_rows.extend(fold_rows)
        summary_rows.append(summary)
        evaluation_unit_rows.extend(fold_evaluation_unit_rows)

    write_csv(
        run_dir / "fold_metrics.csv",
        metric_rows,
        list(metric_rows[0].keys()) if metric_rows else [],
    )
    write_csv(
        run_dir / "summary.csv",
        summary_rows,
        list(summary_rows[0].keys()) if summary_rows else [],
    )
    if export_evaluation_units:
        evaluation_units_path = run_dir / "evaluation_units.csv"
        write_evaluation_units_csv(evaluation_units_path, evaluation_unit_rows)
        validate_evaluation_units_csv(evaluation_units_path)
    return run_dir
