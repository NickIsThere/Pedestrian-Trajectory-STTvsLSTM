from __future__ import annotations

from typing import Any

import numpy as np
import torch

from data.feature import extract_features
from model.config import ModelConfig


def prepare_inference_batch(
    features: np.ndarray,
    mask: np.ndarray,
    *,
    device: torch.device,
    config: ModelConfig,
) -> dict[str, torch.Tensor]:
    x = torch.as_tensor(features, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    time_mask = torch.as_tensor(mask > 0, dtype=torch.bool, device=device).unsqueeze(0).unsqueeze(0)
    agent_mask = torch.ones(x.size(0), x.size(1), dtype=torch.bool, device=device)

    return {
        "x": x,
        "positions": x[..., 0:2],
        "velocities": x[..., 4:6],
        "speed": x[..., 6:7],
        "heading_sc": x[..., 7:9],
        "agent_mask": agent_mask,
        "time_mask": time_mask,
    }


def prepare_scene_inference_batch(
    sequence: Any,
    frame_id: int,
    *,
    device: torch.device,
    config: ModelConfig,
    source: str = "det",
) -> tuple[dict[str, torch.Tensor], list[int]]:
    frame = sequence.frames.get(frame_id)
    if frame is None:
        raise KeyError(f"Frame {frame_id} not found in sequence {sequence.name}")

    source_annotations = frame.det if source == "det" else frame.gt
    features: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    track_ids: list[int] = []
    for track_id in sorted(source_annotations):
        feature, mask, _ = extract_features(
            sequence,
            frame_id,
            track_id,
            config.lookback,
            config.future_steps,
            source=source,
            compute_answer=False,
            trajectory_stride=config.trajectory_stride,
        )
        if not mask.any():
            continue
        features.append(feature)
        masks.append(mask)
        track_ids.append(track_id)

    if not features:
        empty = torch.empty(1, 0, config.lookback, config.input_dim, dtype=torch.float32, device=device)
        return (
            {
                "x": empty,
                "positions": empty[..., 0:2],
                "velocities": empty[..., 4:6],
                "speed": empty[..., 6:7],
                "heading_sc": empty[..., 7:9],
                "agent_mask": torch.empty(1, 0, dtype=torch.bool, device=device),
                "time_mask": torch.empty(1, 0, config.lookback, dtype=torch.bool, device=device),
            },
            [],
        )

    x = torch.as_tensor(np.stack(features, axis=0), dtype=torch.float32, device=device).unsqueeze(0)
    time_mask = torch.as_tensor(np.stack(masks, axis=0) > 0, dtype=torch.bool, device=device).unsqueeze(0)
    agent_mask = time_mask.any(dim=2)
    return (
        {
            "x": x,
            "positions": x[..., 0:2],
            "velocities": x[..., 4:6],
            "speed": x[..., 6:7],
            "heading_sc": x[..., 7:9],
            "agent_mask": agent_mask,
            "time_mask": time_mask,
        },
        track_ids,
    )


def prepare_inference_batch_batched(
    samples: list[tuple[np.ndarray, np.ndarray]],
    *,
    device: torch.device,
    config: ModelConfig,
) -> dict[str, torch.Tensor]:
    """Stack independent single-agent trajectories as batch dimension (same layout as training `prepare_batch`)."""
    if not samples:
        raise ValueError("samples cannot be empty")
    feats = np.stack([f for f, _ in samples], axis=0)
    masks = np.stack([m for _, m in samples], axis=0)
    x = torch.as_tensor(feats, dtype=torch.float32, device=device).unsqueeze(1)
    time_mask = torch.as_tensor(masks > 0, dtype=torch.bool, device=device).unsqueeze(1)
    agent_mask = torch.ones(x.size(0), x.size(1), dtype=torch.bool, device=device)

    return {
        "x": x,
        "positions": x[..., 0:2],
        "velocities": x[..., 4:6],
        "speed": x[..., 6:7],
        "heading_sc": x[..., 7:9],
        "agent_mask": agent_mask,
        "time_mask": time_mask,
    }
