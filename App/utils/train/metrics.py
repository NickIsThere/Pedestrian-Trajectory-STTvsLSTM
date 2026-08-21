from __future__ import annotations

import torch

from model.config import ModelConfig

from utils.train.constants import DEFAULT_METRIC_FPS
from utils.train.targets import _last_observed_delta, last_observed_position
from utils.horizon_targets import nearest_horizon_index


def horizon_steps_for_config(
    config: ModelConfig,
    *,
    fps: float = DEFAULT_METRIC_FPS,
) -> tuple[tuple[str, int], ...]:
    if config.trajectory_stride <= 0:
        raise ValueError("trajectory_stride must be positive")
    if config.future_steps <= 0:
        raise ValueError("future_steps must be positive")

    def step_for(seconds: float) -> int:
        return nearest_horizon_index(
            target_horizon_s=seconds,
            dataset_fps=fps,
            trajectory_stride=config.trajectory_stride,
            future_steps=config.future_steps,
        )

    return (
        ("0p5s", step_for(0.5)),
        ("1p0s", step_for(1.0)),
        ("2p0s", step_for(2.0)),
        ("full", config.future_steps),
    )


def zero_motion_baseline(
    positions: torch.Tensor,
    time_mask: torch.Tensor,
    *,
    future_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    last_obs = last_observed_position(positions, time_mask)
    pred_deltas = positions.new_zeros(positions.size(0), positions.size(1), future_steps, 2)
    pred_positions = last_obs.expand(-1, -1, future_steps, -1).clone()
    return pred_deltas, pred_positions


def constant_velocity_baseline(
    positions: torch.Tensor,
    time_mask: torch.Tensor,
    *,
    future_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    valid_time = time_mask.to(device=positions.device, dtype=torch.bool)
    time_indices = torch.arange(positions.size(2), device=positions.device).view(1, 1, -1)
    masked_indices = torch.where(valid_time, time_indices, torch.zeros_like(time_indices))
    last_indices = masked_indices.max(dim=2).values

    previous_valid = valid_time & (time_indices < last_indices.unsqueeze(-1))
    previous_indices = torch.where(previous_valid, time_indices, torch.zeros_like(time_indices)).max(dim=2).values

    gather_last = last_indices.view(positions.size(0), positions.size(1), 1, 1).expand(-1, -1, 1, 2)
    gather_previous = previous_indices.view(positions.size(0), positions.size(1), 1, 1).expand(-1, -1, 1, 2)
    last_pos = torch.gather(positions, dim=2, index=gather_last)
    previous_pos = torch.gather(positions, dim=2, index=gather_previous)

    has_two_points = previous_valid.any(dim=2).unsqueeze(-1).unsqueeze(-1)
    step_delta = torch.where(has_two_points, last_pos - previous_pos, torch.zeros_like(last_pos))
    pred_deltas = step_delta.expand(-1, -1, future_steps, -1).clone()
    steps = torch.arange(1, future_steps + 1, device=positions.device, dtype=positions.dtype).view(1, 1, -1, 1)
    pred_positions = last_pos + steps * step_delta
    return pred_deltas, pred_positions


def _ade_fde(
    pred_positions: torch.Tensor,
    target_positions: torch.Tensor,
    future_mask: torch.Tensor,
) -> tuple[float, float]:
    valid = future_mask.to(device=pred_positions.device, dtype=torch.bool)
    if not valid.any():
        return 0.0, 0.0

    target_positions = target_positions.to(device=pred_positions.device, dtype=pred_positions.dtype)
    displacement_error = torch.linalg.vector_norm(pred_positions - target_positions, dim=-1)
    ade = displacement_error[valid].mean()
    horizon_indices = torch.arange(valid.size(2), device=valid.device).view(1, 1, -1)
    last_valid_indices = torch.where(valid, horizon_indices, torch.zeros_like(horizon_indices)).max(dim=2).values
    has_future = valid.any(dim=2)
    fde = displacement_error.gather(2, last_valid_indices.unsqueeze(-1)).squeeze(-1)[has_future].mean()
    return float(ade.item()), float(fde.item())


def horizon_metrics(
    pred_positions: torch.Tensor,
    target_positions: torch.Tensor,
    future_mask: torch.Tensor,
    *,
    horizons: tuple[tuple[str, int], ...] | None = None,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    future_len = pred_positions.size(2)
    if horizons is None:
        horizons = (("full", future_len),)
    for label, one_based_step in horizons:
        step_count = min(one_based_step, future_len)
        if step_count <= 0:
            metrics[f"ade_{label}"] = 0.0
            metrics[f"fde_{label}"] = 0.0
            continue
        ade, fde = _ade_fde(
            pred_positions[:, :, :step_count, :],
            target_positions[:, :, :step_count, :],
            future_mask[:, :, :step_count],
        )
        metrics[f"ade_{label}"] = ade
        metrics[f"fde_{label}"] = fde
    return metrics


def _masked_vector_norm_mean(values: torch.Tensor, mask: torch.Tensor) -> float:
    valid = mask.to(device=values.device, dtype=torch.bool)
    if not valid.any():
        return 0.0
    return float(torch.linalg.vector_norm(values, dim=-1)[valid].mean().item())


def delta_diagnostics(
    pred_deltas: torch.Tensor,
    target_deltas: torch.Tensor,
    positions: torch.Tensor,
    time_mask: torch.Tensor,
    future_mask: torch.Tensor,
) -> dict[str, float]:
    target_deltas = target_deltas.to(device=pred_deltas.device, dtype=pred_deltas.dtype)
    future_mask = future_mask.to(device=pred_deltas.device, dtype=torch.bool)
    last_delta = _last_observed_delta(
        positions.to(device=pred_deltas.device, dtype=pred_deltas.dtype),
        time_mask.to(device=pred_deltas.device),
    )
    repeated_last_delta = last_delta.expand(-1, -1, pred_deltas.size(2), -1)

    diagnostics = {
        "pred_delta_mag_mean": _masked_vector_norm_mean(pred_deltas, future_mask),
        "gt_delta_mag_mean": _masked_vector_norm_mean(target_deltas, future_mask),
        "last_obs_delta_mag_mean": _masked_vector_norm_mean(repeated_last_delta, future_mask),
    }

    accel_mask = future_mask[:, :, 1:] & future_mask[:, :, :-1]
    diagnostics["pred_accel_mag_mean"] = _masked_vector_norm_mean(
        pred_deltas[:, :, 1:, :] - pred_deltas[:, :, :-1, :],
        accel_mask,
    )
    diagnostics["gt_accel_mag_mean"] = _masked_vector_norm_mean(
        target_deltas[:, :, 1:, :] - target_deltas[:, :, :-1, :],
        accel_mask,
    )

    coord_mask = future_mask.unsqueeze(-1).expand_as(pred_deltas)
    if coord_mask.any():
        pred_flat = pred_deltas[coord_mask]
        cv_flat = repeated_last_delta[coord_mask]
        diagnostics["pred_cv_delta_mse"] = float(torch.mean((pred_flat - cv_flat) ** 2).item())
        denom = torch.linalg.vector_norm(pred_flat) * torch.linalg.vector_norm(cv_flat)
        diagnostics["pred_cv_delta_cosine"] = float(
            (torch.dot(pred_flat, cv_flat) / denom).item() if denom > 0 else 0.0,
        )
    else:
        diagnostics["pred_cv_delta_mse"] = 0.0
        diagnostics["pred_cv_delta_cosine"] = 0.0

    return diagnostics


def trajectory_metrics(
    pred_deltas: torch.Tensor,
    pred_positions: torch.Tensor,
    target_deltas: torch.Tensor,
    target_positions: torch.Tensor,
    future_mask: torch.Tensor,
    *,
    horizons: tuple[tuple[str, int], ...] | None = None,
) -> dict[str, float]:
    valid = future_mask.to(device=pred_positions.device, dtype=torch.bool)
    coord_mask = valid.unsqueeze(-1).expand_as(pred_positions)
    if not coord_mask.any():
        return {"delta_mse": 0.0, "position_rmse": 0.0, "ade": 0.0, "fde": 0.0}

    target_deltas = target_deltas.to(device=pred_deltas.device, dtype=pred_deltas.dtype)
    target_positions = target_positions.to(device=pred_positions.device, dtype=pred_positions.dtype)
    delta_error = pred_deltas - target_deltas
    position_error = pred_positions - target_positions

    delta_mse = torch.mean(delta_error[coord_mask] ** 2)
    position_mse = torch.mean(position_error[coord_mask] ** 2)
    displacement_error = torch.linalg.vector_norm(position_error, dim=-1)
    ade = displacement_error[valid].mean()
    horizon_indices = torch.arange(valid.size(2), device=valid.device).view(1, 1, -1)
    last_valid_indices = torch.where(valid, horizon_indices, torch.zeros_like(horizon_indices)).max(dim=2).values
    has_future = valid.any(dim=2)
    fde = displacement_error.gather(2, last_valid_indices.unsqueeze(-1)).squeeze(-1)[has_future].mean()
    return {
        "delta_mse": float(delta_mse.item()),
        "position_rmse": float(torch.sqrt(position_mse).item()),
        "ade": float(ade.item()),
        "fde": float(fde.item()),
        **horizon_metrics(pred_positions, target_positions, future_mask, horizons=horizons),
    }
