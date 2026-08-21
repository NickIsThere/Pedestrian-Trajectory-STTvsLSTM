from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np
import torch

from model.config import ModelConfig

from utils.train.targets import assert_target_reconstruction_matches, last_observed_position


def prepare_batch(
    samples: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    *,
    device: torch.device,
    config: ModelConfig,
) -> dict[str, Any]:
    feats = np.stack([f for f, _, _ in samples], axis=0)
    masks = np.stack([m for _, m, _ in samples], axis=0)
    answers = np.stack([a for _, _, a in samples], axis=0)

    x = torch.as_tensor(feats, dtype=torch.float32, device=device).unsqueeze(1)
    time_mask = torch.as_tensor(masks > 0, dtype=torch.bool, device=device).unsqueeze(1)
    agent_mask = torch.ones(x.size(0), 1, dtype=torch.bool, device=device)

    positions = x[..., 0:2]
    velocities = x[..., 4:6]
    speed = x[..., 6:7]
    heading_sc = x[..., 7:9]

    answer_t = torch.as_tensor(answers, dtype=torch.float32, device=device).unsqueeze(1)
    last_obs = last_observed_position(positions, time_mask)

    f = config.future_steps
    target_deltas = torch.zeros(x.size(0), 1, f, 2, device=device, dtype=torch.float32)
    target_deltas[:, :, 0, :] = answer_t[:, :, 0, :] - last_obs.squeeze(2)
    if f > 1:
        target_deltas[:, :, 1:, :] = answer_t[:, :, 1:, :] - answer_t[:, :, :-1, :]

    assert_target_reconstruction_matches(
        positions,
        time_mask,
        target_deltas,
        answer_t,
        torch.ones(x.size(0), 1, f, device=device, dtype=torch.bool),
    )

    return {
        "x": x,
        "positions": positions,
        "velocities": velocities,
        "speed": speed,
        "heading_sc": heading_sc,
        "agent_mask": agent_mask,
        "time_mask": time_mask,
        "target_deltas": target_deltas,
        "target_positions": answer_t,
        "future_mask": torch.ones(x.size(0), 1, f, device=device, dtype=torch.bool),
    }


def prepare_scene_batch(
    samples: list[Any],
    *,
    device: torch.device,
    config: ModelConfig,
) -> dict[str, Any]:
    if not samples:
        raise ValueError("samples cannot be empty")

    batch_size = len(samples)
    max_agents = max(int(sample.features.shape[0]) for sample in samples)
    x = torch.zeros(batch_size, max_agents, config.lookback, config.input_dim, dtype=torch.float32, device=device)
    time_mask = torch.zeros(batch_size, max_agents, config.lookback, dtype=torch.bool, device=device)
    target_positions = torch.zeros(batch_size, max_agents, config.future_steps, 2, dtype=torch.float32, device=device)
    future_mask = torch.zeros(batch_size, max_agents, config.future_steps, dtype=torch.bool, device=device)
    agent_mask = torch.zeros(batch_size, max_agents, dtype=torch.bool, device=device)

    for batch_index, sample in enumerate(samples):
        agent_count = int(sample.features.shape[0])
        if agent_count == 0:
            continue
        x[batch_index, :agent_count] = torch.as_tensor(sample.features, dtype=torch.float32, device=device)
        time_mask[batch_index, :agent_count] = torch.as_tensor(sample.mask > 0, dtype=torch.bool, device=device)
        target_positions[batch_index, :agent_count] = torch.as_tensor(sample.answers, dtype=torch.float32, device=device)
        future_mask[batch_index, :agent_count] = torch.as_tensor(sample.future_mask, dtype=torch.bool, device=device)
        agent_mask[batch_index, :agent_count] = time_mask[batch_index, :agent_count].any(dim=1)

    positions = x[..., 0:2]
    velocities = x[..., 4:6]
    speed = x[..., 6:7]
    heading_sc = x[..., 7:9]
    last_obs = last_observed_position(positions, time_mask)

    target_deltas = torch.zeros_like(target_positions)
    target_deltas[:, :, 0, :] = target_positions[:, :, 0, :] - last_obs.squeeze(2)
    if config.future_steps > 1:
        target_deltas[:, :, 1:, :] = target_positions[:, :, 1:, :] - target_positions[:, :, :-1, :]
    target_deltas = target_deltas * future_mask.unsqueeze(-1).to(dtype=target_deltas.dtype)

    assert_target_reconstruction_matches(
        positions,
        time_mask,
        target_deltas,
        target_positions,
        future_mask,
    )

    return {
        "x": x,
        "positions": positions,
        "velocities": velocities,
        "speed": speed,
        "heading_sc": heading_sc,
        "agent_mask": agent_mask,
        "time_mask": time_mask,
        "target_deltas": target_deltas,
        "target_positions": target_positions,
        "future_mask": future_mask,
    }


def pop_training_targets(batch: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    target_deltas = batch.pop("target_deltas")
    target_positions = batch.pop("target_positions")
    future_mask = batch.pop("future_mask")
    return target_deltas, target_positions, future_mask


def batch_samples(samples: Iterator[Any], batch_size: int) -> Iterator[list[Any]]:
    batch: list[Any] = []
    for sample in samples:
        batch.append(sample)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def limit_samples(samples: Iterator[Any], max_samples: int) -> Iterator[Any]:
    if max_samples <= 0:
        yield from samples
        return
    for index, sample in enumerate(samples):
        if index >= max_samples:
            break
        yield sample


def flatten_supervised_agents(
    *,
    pred_deltas: torch.Tensor,
    pred_positions: torch.Tensor,
    target_deltas: torch.Tensor,
    target_positions: torch.Tensor,
    future_mask: torch.Tensor,
    history_positions: torch.Tensor | None = None,
    history_time_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    has_future = future_mask.any(dim=2)
    if not has_future.any():
        return {}

    flattened = {
        "pred_deltas": pred_deltas[has_future].unsqueeze(1),
        "pred_positions": pred_positions[has_future].unsqueeze(1),
        "target_deltas": target_deltas[has_future].unsqueeze(1),
        "target_positions": target_positions[has_future].unsqueeze(1),
        "future_mask": future_mask[has_future].unsqueeze(1),
    }
    if history_positions is not None and history_time_mask is not None:
        flattened["history_positions"] = history_positions[has_future].unsqueeze(1)
        flattened["history_time_mask"] = history_time_mask[has_future].unsqueeze(1)
    return flattened
