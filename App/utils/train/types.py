from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from data.reader import Split
from model.config import ModelConfig

from utils.train.constants import DEFAULT_CHECKPOINT_PATH, SampleMode


@dataclass(frozen=True)
class TrainingSample:
    features: np.ndarray
    mask: np.ndarray
    answer: np.ndarray
    metadata: dict[str, Any]


@dataclass(frozen=True)
class SceneTrainingSample:
    features: np.ndarray
    mask: np.ndarray
    answers: np.ndarray
    future_mask: np.ndarray
    track_ids: list[int]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class PreparedBatch:
    inputs: dict[str, torch.Tensor]
    target_deltas: torch.Tensor
    target_positions: torch.Tensor
    future_mask: torch.Tensor


@dataclass(frozen=True)
class LeaveOneOutFold:
    index: int
    val_sequence: str
    train_split: Split
    val_split: Split


@dataclass(frozen=True)
class ModelSpec:
    name: str
    build_model: Callable[[ModelConfig, torch.device], torch.nn.Module]
    prepare_batch: Callable[..., PreparedBatch]
    read_predictions: Callable[[dict[str, torch.Tensor]], tuple[torch.Tensor, torch.Tensor]]
    save_checkpoint: Callable[[torch.nn.Module, Path, dict[str, Any]], None]


@dataclass(frozen=True)
class TrainingDefaults:
    batch_size: int = 1
    train_steps: int = 2000
    val_interval: int = 100
    max_val_samples: int = 4096
    learning_rate: float = 0.001
    seed: int = 0
    model: str = "transformer"
    train_input_mode: SampleMode = "gt"
    output_dir: Path = Path("outputs/loo")
    checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH
    leave_one_out: bool = True
    fold_sequence: str | None = None
    run_id: str | None = None
    debug_smoke: bool = False
    export_evaluation_units: bool = False
    model_family: str = "unknown"
    model_variant: str = "unknown"
    train_domain: str = "unknown"
    test_domain: str = "unknown"
    synthetic_size_fraction: float | None = None


BASE_TRAINING_DEFAULTS = TrainingDefaults()
