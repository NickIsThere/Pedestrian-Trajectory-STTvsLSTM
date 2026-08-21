from __future__ import annotations

import torch


def last_observed_position(
    positions: torch.Tensor,
    time_mask: torch.Tensor,
) -> torch.Tensor:
    """Match TrajectoryTransformer._last_observed_position for agent_mask all-True."""
    valid_time = time_mask.to(device=positions.device, dtype=torch.bool)
    time_indices = torch.arange(positions.size(2), device=positions.device).view(1, 1, -1)
    masked_indices = torch.where(valid_time, time_indices, torch.zeros_like(time_indices))
    last_indices = masked_indices.max(dim=2).values

    gather_index = last_indices.view(positions.size(0), positions.size(1), 1, 1).expand(-1, -1, 1, 2)
    last_pos = torch.gather(positions, dim=2, index=gather_index)
    has_history = valid_time.any(dim=2)
    return last_pos * has_history.unsqueeze(-1).unsqueeze(-1).to(dtype=positions.dtype)


def target_reconstruction_error(
    positions: torch.Tensor,
    time_mask: torch.Tensor,
    target_deltas: torch.Tensor,
    target_positions: torch.Tensor,
    future_mask: torch.Tensor,
) -> float:
    last_obs = last_observed_position(positions, time_mask).to(
        device=target_deltas.device,
        dtype=target_deltas.dtype,
    )
    target_positions = target_positions.to(device=target_deltas.device, dtype=target_deltas.dtype)
    valid = future_mask.to(device=target_deltas.device, dtype=torch.bool)
    reconstructed = last_obs + torch.cumsum(target_deltas, dim=2)
    if not valid.any():
        return 0.0
    error = torch.max(torch.abs(reconstructed - target_positions)[valid.unsqueeze(-1).expand_as(reconstructed)])
    return float(error.item())


def assert_target_reconstruction_matches(
    positions: torch.Tensor,
    time_mask: torch.Tensor,
    target_deltas: torch.Tensor,
    target_positions: torch.Tensor,
    future_mask: torch.Tensor,
    *,
    atol: float = 1e-6,
) -> None:
    error = target_reconstruction_error(positions, time_mask, target_deltas, target_positions, future_mask)
    if error > atol:
        raise ValueError(
            "target_deltas do not reconstruct target_positions from last observed position "
            f"(max_abs_error={error:.8g}, atol={atol:.8g})",
        )


def _last_observed_delta(
    positions: torch.Tensor,
    time_mask: torch.Tensor,
) -> torch.Tensor:
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
    return torch.where(has_two_points, last_pos - previous_pos, torch.zeros_like(last_pos))
