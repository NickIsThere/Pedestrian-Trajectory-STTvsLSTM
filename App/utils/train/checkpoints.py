from __future__ import annotations

from dataclasses import asdict
import os
from pathlib import Path
from typing import Any

import torch

from data.feature import MODEL_VERSION
from model.config import ModelConfig
from model.model import HTPModel

from utils.train.constants import DEFAULT_CHECKPOINT_PATH, HTP_OUTPUT_ROOTS


def candidate_htp_checkpoint_paths() -> list[Path]:
    explicit_path = os.environ.get("HTP_CHECKPOINT")
    if explicit_path:
        return [Path(explicit_path).expanduser()]

    loo_checkpoints: list[Path] = []
    for output_root in HTP_OUTPUT_ROOTS:
        if output_root.is_dir():
            loo_checkpoints.extend(output_root.glob("*/checkpoints/htp_model*.pt"))
    loo_checkpoints.sort(key=lambda path: path.stat().st_mtime, reverse=True)

    candidates = [*loo_checkpoints, DEFAULT_CHECKPOINT_PATH]
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in candidates:
        resolved = path.resolve(strict=False)
        if resolved in seen:
            continue
        seen.add(resolved)
        if path.is_file():
            unique.append(path)
    return unique


def save_model_checkpoint(
    model: torch.nn.Module,
    path: Path,
    config: ModelConfig,
    *,
    training_input_mode: str = "gt",
    loss_config: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": asdict(config),
            "MODEL_VERSION": MODEL_VERSION,
            "feature_version": MODEL_VERSION,
            "training_input_mode": training_input_mode,
            "loss_config": loss_config or {},
            "metadata": metadata or {},
        },
        path,
    )


def load_htp_model(
    path: Path | None = None,
    *,
    device: torch.device | None = None,
) -> tuple[HTPModel, ModelConfig]:
    path = path or DEFAULT_CHECKPOINT_PATH
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(path, map_location=dev, weights_only=False)
    checkpoint_version = ckpt.get("MODEL_VERSION", ckpt.get("feature_version"))
    if checkpoint_version != MODEL_VERSION:
        found = checkpoint_version or "missing"
        raise RuntimeError(
            f"Checkpoint MODEL_VERSION={found!r} is incompatible with {MODEL_VERSION!r}; retrain required.",
        )
    cfg = ModelConfig(**ckpt["config"])
    model = HTPModel(cfg).to(dev)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, cfg
