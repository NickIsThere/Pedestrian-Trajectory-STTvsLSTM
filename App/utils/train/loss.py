from __future__ import annotations

import torch

from model.loss import trajectory_loss

from utils.train.constants import LOSS_CONFIG


def compute_training_loss(
    predictions: dict[str, torch.Tensor],
    target_deltas: torch.Tensor,
    target_positions: torch.Tensor,
    future_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    return trajectory_loss(
        predictions,
        target_deltas,
        target_positions,
        future_mask,
        **LOSS_CONFIG,
    )
