from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch

from benchmarks.social_lstm.model import SocialLSTM
from model.config import ModelConfig

from utils.train.constants import LOSS_CONFIG, SOCIAL_LSTM_GRID_SIZE
from utils.train.datasets import samples_to_tuples
from utils.train.targets import last_observed_position
from utils.train.types import PreparedBatch, TrainingSample


class SocialLSTMTrainingAdapter(torch.nn.Module):
    """Deprecated compatibility wrapper for the old single-agent LSTM path."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        warnings.warn(
            "SocialLSTMTrainingAdapter is deprecated. Use App.benchmarks.social_lstm.train "
            "for the real multi-agent Social LSTM trainer.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.config = config
        self.lstm = SocialLSTM(
            obs_len=config.lookback,
            pred_len=config.future_steps,
            hidden_size=config.hidden_dim,
            embedding_dim=config.head_hidden_dim,
            grid_size=SOCIAL_LSTM_GRID_SIZE,
        )

    def forward(self, obs_positions: torch.Tensor, obs_masks: torch.Tensor) -> dict[str, torch.Tensor]:
        pred_deltas_by_time = self.lstm(obs_positions, obs_masks)
        pred_deltas = pred_deltas_by_time.permute(0, 2, 1, 3).contiguous()

        history_positions = obs_positions.permute(0, 2, 1, 3).contiguous()
        history_mask = obs_masks.permute(0, 2, 1).contiguous().to(dtype=torch.bool)
        last_obs = last_observed_position(history_positions, history_mask)
        pred_positions = last_obs + torch.cumsum(pred_deltas, dim=2)
        return {"pred_deltas": pred_deltas, "pred_positions": pred_positions}


def social_lstm_prepare_batch(
    samples: list[TrainingSample],
    *,
    device: torch.device,
    config: ModelConfig,
) -> PreparedBatch:
    features, masks, answers = zip(*samples_to_tuples(samples), strict=True)
    obs_positions = torch.as_tensor(
        np.stack([item[:, 0:2] for item in features], axis=0),
        dtype=torch.float32,
        device=device,
    )
    obs_masks = torch.as_tensor(np.stack(masks, axis=0) > 0, dtype=torch.float32, device=device)
    obs_positions = obs_positions.unsqueeze(2)
    obs_masks = obs_masks.unsqueeze(2)

    target_positions = torch.as_tensor(np.stack(answers, axis=0), dtype=torch.float32, device=device).unsqueeze(1)
    history_positions = obs_positions.permute(0, 2, 1, 3).contiguous()
    history_mask = obs_masks.permute(0, 2, 1).contiguous().to(dtype=torch.bool)
    last_obs = last_observed_position(history_positions, history_mask)

    target_deltas = torch.zeros_like(target_positions)
    target_deltas[:, :, 0, :] = target_positions[:, :, 0, :] - last_obs.squeeze(2)
    if config.future_steps > 1:
        target_deltas[:, :, 1:, :] = target_positions[:, :, 1:, :] - target_positions[:, :, :-1, :]

    return PreparedBatch(
        inputs={"obs_positions": obs_positions, "obs_masks": obs_masks},
        target_deltas=target_deltas,
        target_positions=target_positions,
        future_mask=torch.ones(
            target_positions.size(0),
            1,
            config.future_steps,
            dtype=torch.bool,
            device=device,
        ),
    )


def build_social_lstm_model(config: ModelConfig, device: torch.device) -> torch.nn.Module:
    return SocialLSTMTrainingAdapter(config).to(device)


def save_social_lstm_checkpoint(
    model: torch.nn.Module,
    path: Path,
    metadata: dict[str, Any],
) -> None:
    if not isinstance(model, SocialLSTMTrainingAdapter):
        raise TypeError(f"Expected SocialLSTMTrainingAdapter, got {type(model).__name__}")

    config_value = metadata.get("model_config")
    if not isinstance(config_value, ModelConfig):
        config_value = ModelConfig(**dict(config_value or {}))
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.lstm.state_dict(),
            "config": {
                "obs_len": config_value.lookback,
                "pred_len": config_value.future_steps,
                "hidden_size": config_value.hidden_dim,
                "embedding_dim": config_value.head_hidden_dim,
                "grid_size": SOCIAL_LSTM_GRID_SIZE,
            },
            "training_input_mode": str(metadata.get("training_input_mode", "gt")),
            "loss_config": LOSS_CONFIG,
            "metadata": {key: value for key, value in metadata.items() if key != "model_config"},
        },
        path,
    )
