from __future__ import annotations

import time
from typing import Any, Protocol

import torch
from tqdm import tqdm

from data.reader import Split
from evaluation.evaluation_units import build_evaluation_unit_rows


class EvaluationUnitsWriter(Protocol):
    row_count: int

    def write_rows(self, rows: list[dict[str, Any]]) -> None: ...
from model.config import ModelConfig

from utils.train.batches import (
    batch_samples,
    flatten_supervised_agents,
    limit_samples,
    pop_training_targets,
    prepare_batch,
)
from utils.train.datasets import (
    iter_split_samples,
    iter_split_scene_samples,
    samples_to_tuples,
)
from utils.train.loss import compute_training_loss
from utils.train.metrics import (
    constant_velocity_baseline,
    delta_diagnostics,
    horizon_steps_for_config,
    trajectory_metrics,
    zero_motion_baseline,
)
from utils.train.targets import target_reconstruction_error
from utils.train.types import ModelSpec, PreparedBatch, TrainingSample


def format_metrics(prefix: str, metrics: dict[str, float]) -> str:
    loss_part = ""
    if "loss" in metrics:
        loss_part = (
            f"{prefix}_loss={metrics['loss']:.6f}  "
            f"{prefix}_delta_loss={metrics['delta_loss']:.6f}  "
            f"{prefix}_position_loss={metrics['position_loss']:.6f}  "
        )
    return (
        loss_part
        + f"{prefix}_delta_mse={metrics['delta_mse']:.6f}  "
        + f"{prefix}_position_rmse={metrics['position_rmse']:.6f}  "
        + f"{prefix}_ade={metrics['ade']:.6f}  "
        + f"{prefix}_fde={metrics['fde']:.6f}"
    )


def print_debug_smoke_batch(
    stage: str,
    config: ModelConfig,
    prepared: PreparedBatch,
    output: dict[str, torch.Tensor],
) -> None:
    x = prepared.inputs.get("x")
    pred_deltas = output.get("pred_deltas")
    pred_positions = output.get("pred_positions")
    positions = prepared.inputs.get("positions")
    time_mask = prepared.inputs.get("time_mask")
    if positions is not None and time_mask is not None:
        reconstruction_error = target_reconstruction_error(
            positions,
            time_mask,
            prepared.target_deltas,
            prepared.target_positions,
            prepared.future_mask,
        )
    else:
        reconstruction_error = float("nan")

    print(
        "[debug-smoke] "
        f"stage={stage} "
        f"config lookback={config.lookback} future_steps={config.future_steps} "
        f"trajectory_stride={config.trajectory_stride} "
        f"x.shape={tuple(x.shape) if x is not None else None} "
        f"target_positions.shape={tuple(prepared.target_positions.shape)} "
        f"target_deltas.shape={tuple(prepared.target_deltas.shape)} "
        f"future_mask.shape={tuple(prepared.future_mask.shape)} "
        f"pred_deltas.shape={tuple(pred_deltas.shape) if pred_deltas is not None else None} "
        f"pred_positions.shape={tuple(pred_positions.shape) if pred_positions is not None else None} "
        f"target_reconstruction_max_abs_error={reconstruction_error:.8g}",
    )


def evaluate_samples(
    model: torch.nn.Module,
    samples: list[TrainingSample | tuple],
    *,
    device: torch.device,
    config: ModelConfig,
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    batch = prepare_batch(samples_to_tuples(samples), device=device, config=config)
    target_deltas, target_positions, future_mask = pop_training_targets(batch)
    horizons = horizon_steps_for_config(config)
    with torch.no_grad():
        out = model(**batch)

    loss_parts = compute_training_loss(out, target_deltas, target_positions, future_mask)
    model_metrics = trajectory_metrics(
        out["pred_deltas"],
        out["pred_positions"],
        target_deltas,
        target_positions,
        future_mask,
        horizons=horizons,
    )
    model_metrics.update(
        {
            "loss": float(loss_parts["loss"].item()),
            "delta_loss": float(loss_parts["delta_loss"].item()),
            "position_loss": float(loss_parts["position_loss"].item()),
        },
    )

    zero_deltas, zero_positions = zero_motion_baseline(
        batch["positions"],
        batch["time_mask"],
        future_steps=config.future_steps,
    )
    cv_deltas, cv_positions = constant_velocity_baseline(
        batch["positions"],
        batch["time_mask"],
        future_steps=config.future_steps,
    )
    zero_metrics = trajectory_metrics(
        zero_deltas,
        zero_positions,
        target_deltas,
        target_positions,
        future_mask,
        horizons=horizons,
    )
    cv_metrics = trajectory_metrics(
        cv_deltas,
        cv_positions,
        target_deltas,
        target_positions,
        future_mask,
        horizons=horizons,
    )
    return model_metrics, zero_metrics, cv_metrics


def evaluate_split_for_model(
    model: torch.nn.Module,
    model_spec: ModelSpec,
    split: Split,
    config: ModelConfig,
    *,
    batch_size: int,
    device: torch.device,
    input_mode: str = "gt",
    max_samples: int = 0,
    progress_desc: str | None = None,
    debug_smoke: bool = False,
    debug_label: str = "val",
    evaluation_unit_rows: list[dict[str, Any]] | None = None,
    evaluation_units_writer: EvaluationUnitsWriter | None = None,
    evaluation_context: dict[str, Any] | None = None,
    collect_metrics: bool = True,
) -> dict[str, float | int]:
    pred_deltas: list[torch.Tensor] = []
    pred_positions: list[torch.Tensor] = []
    target_deltas: list[torch.Tensor] = []
    target_positions: list[torch.Tensor] = []
    future_masks: list[torch.Tensor] = []
    history_positions: list[torch.Tensor] = []
    history_time_masks: list[torch.Tensor] = []
    had_samples = False

    model.eval()
    with torch.no_grad():
        raw_samples = (
            iter_split_scene_samples(split, config, input_mode=input_mode)
            if model_spec.name == "transformer"
            else iter_split_samples(split, config, input_mode=input_mode)
        )
        sample_iter = limit_samples(raw_samples, max_samples)
        batches = batch_samples(sample_iter, batch_size)
        if progress_desc is not None:
            total = (max_samples + batch_size - 1) // batch_size if max_samples > 0 else None
            batches = tqdm(batches, desc=progress_desc, total=total, leave=False)
        for batch_index, samples in enumerate(batches):
            prepared = model_spec.prepare_batch(samples, device=device, config=config)
            inference_start = time.perf_counter()
            output = model(**prepared.inputs)
            batch_latency_ms = (time.perf_counter() - inference_start) * 1000.0
            if debug_smoke and batch_index == 0:
                print_debug_smoke_batch(debug_label, config, prepared, output)
            batch_pred_deltas, batch_pred_positions = model_spec.read_predictions(output)
            if evaluation_unit_rows is not None or evaluation_units_writer is not None:
                supervised_agents = int(prepared.future_mask.detach().cpu().any(dim=2).sum().item())
                row_context = {
                    **(evaluation_context or {}),
                    "batch_latency_ms": batch_latency_ms,
                    "runtime_windows_per_second": (len(samples) / (batch_latency_ms / 1000.0))
                    if batch_latency_ms > 0
                    else float("inf"),
                    "batch_size_windows": len(samples),
                    "batch_size_agents": supervised_agents,
                }
                batch_rows = build_evaluation_unit_rows(
                    samples=list(samples),
                    pred_positions=batch_pred_positions,
                    target_positions=prepared.target_positions,
                    future_mask=prepared.future_mask,
                    config=config,
                    context=row_context,
                )
                if evaluation_units_writer is not None:
                    evaluation_units_writer.write_rows(batch_rows)
                if evaluation_unit_rows is not None:
                    evaluation_unit_rows.extend(batch_rows)
            if not collect_metrics:
                had_samples = True
                continue
            batch_history_positions = prepared.inputs.get("positions")
            batch_history_time_mask = prepared.inputs.get("time_mask")
            flattened = flatten_supervised_agents(
                pred_deltas=batch_pred_deltas.detach().cpu(),
                pred_positions=batch_pred_positions.detach().cpu(),
                target_deltas=prepared.target_deltas.detach().cpu(),
                target_positions=prepared.target_positions.detach().cpu(),
                future_mask=prepared.future_mask.detach().cpu(),
                history_positions=batch_history_positions.detach().cpu()
                if batch_history_positions is not None
                else None,
                history_time_mask=batch_history_time_mask.detach().cpu()
                if batch_history_time_mask is not None
                else None,
            )
            if not flattened:
                continue
            had_samples = True
            pred_deltas.append(flattened["pred_deltas"])
            pred_positions.append(flattened["pred_positions"])
            target_deltas.append(flattened["target_deltas"])
            target_positions.append(flattened["target_positions"])
            future_masks.append(flattened["future_mask"])
            if batch_history_positions is not None and batch_history_time_mask is not None:
                history_positions.append(flattened["history_positions"])
                history_time_masks.append(flattened["history_time_mask"])

    if not had_samples:
        raise RuntimeError(f"No valid validation samples for split {split.name}.")
    if not collect_metrics:
        val_samples = 0
        if evaluation_units_writer is not None:
            val_samples = int(evaluation_units_writer.row_count)
        elif evaluation_unit_rows is not None:
            val_samples = len(evaluation_unit_rows)
        return {"val_samples": val_samples}

    all_pred_deltas = torch.cat(pred_deltas, dim=0)
    all_pred_positions = torch.cat(pred_positions, dim=0)
    all_target_deltas = torch.cat(target_deltas, dim=0)
    all_target_positions = torch.cat(target_positions, dim=0)
    all_future_masks = torch.cat(future_masks, dim=0)
    horizons = horizon_steps_for_config(config)
    metrics = trajectory_metrics(
        all_pred_deltas,
        all_pred_positions,
        all_target_deltas,
        all_target_positions,
        all_future_masks,
        horizons=horizons,
    )
    cv_metrics: dict[str, float] = {}
    diagnostics: dict[str, float] = {}
    if history_positions and len(history_positions) == len(pred_deltas):
        all_history_positions = torch.cat(history_positions, dim=0)
        all_history_time_masks = torch.cat(history_time_masks, dim=0)
        cv_deltas, cv_positions = constant_velocity_baseline(
            all_history_positions,
            all_history_time_masks,
            future_steps=config.future_steps,
        )
        cv_metrics = trajectory_metrics(
            cv_deltas,
            cv_positions,
            all_target_deltas,
            all_target_positions,
            all_future_masks,
            horizons=horizons,
        )
        diagnostics = delta_diagnostics(
            all_pred_deltas,
            all_target_deltas,
            all_history_positions,
            all_history_time_masks,
            all_future_masks,
        )
    loss_parts = compute_training_loss(
        {"pred_deltas": all_pred_deltas, "pred_positions": all_pred_positions},
        all_target_deltas,
        all_target_positions,
        all_future_masks,
    )
    return {
        "val_loss": float(loss_parts["loss"].item()),
        "delta_loss": float(loss_parts["delta_loss"].item()),
        "position_loss": float(loss_parts["position_loss"].item()),
        "delta_mse": metrics["delta_mse"],
        "position_rmse": metrics["position_rmse"],
        "ade": metrics["ade"],
        "fde": metrics["fde"],
        **{
            key: value
            for key, value in metrics.items()
            if key.startswith("ade_") or key.startswith("fde_")
        },
        **{f"cv_{key}": value for key, value in cv_metrics.items()},
        **diagnostics,
        "val_samples": int(all_pred_deltas.size(0)),
    }
