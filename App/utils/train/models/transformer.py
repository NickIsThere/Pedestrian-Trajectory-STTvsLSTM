from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, cast

import torch

from model.config import ModelConfig

from utils.train.batches import pop_training_targets, prepare_batch, prepare_scene_batch
from utils.train.checkpoints import save_model_checkpoint
from utils.train.constants import LOSS_CONFIG
from utils.train.datasets import samples_to_tuples
from utils.train.types import PreparedBatch, SceneTrainingSample, TrainingSample


def transformer_prepare_batch(
    samples: list[TrainingSample | SceneTrainingSample],
    *,
    device: torch.device,
    config: ModelConfig,
) -> PreparedBatch:
    if samples and isinstance(samples[0], SceneTrainingSample):
        batch = prepare_scene_batch(samples, device=device, config=config)
    else:
        batch = prepare_batch(samples_to_tuples(cast(list[TrainingSample], samples)), device=device, config=config)
    target_deltas, target_positions, future_mask = pop_training_targets(batch)
    return PreparedBatch(
        inputs=batch,
        target_deltas=target_deltas,
        target_positions=target_positions,
        future_mask=future_mask,
    )


def read_transformer_predictions(output: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    return output["pred_deltas"], output["pred_positions"]


def save_transformer_checkpoint(
    model: torch.nn.Module,
    path: Path,
    metadata: dict[str, Any],
) -> None:
    config_value = metadata.get("model_config")
    if not isinstance(config_value, ModelConfig):
        config_dict = asdict(config_value) if is_dataclass(config_value) else dict(config_value or {})
        config_value = ModelConfig(**config_dict)
    save_model_checkpoint(
        model,
        path,
        config_value,
        training_input_mode=str(metadata.get("training_input_mode", "gt")),
        loss_config=LOSS_CONFIG,
        metadata={key: value for key, value in metadata.items() if key != "model_config"},
    )
