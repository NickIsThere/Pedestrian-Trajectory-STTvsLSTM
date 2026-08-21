from __future__ import annotations

from model.model import HTPModel

from utils.train.models.social_lstm import (
    build_social_lstm_model,
    save_social_lstm_checkpoint,
    social_lstm_prepare_batch,
)
from utils.train.models.transformer import (
    read_transformer_predictions,
    save_transformer_checkpoint,
    transformer_prepare_batch,
)
from utils.train.types import ModelSpec

MODEL_REGISTRY: dict[str, ModelSpec] = {
    "transformer": ModelSpec(
        name="transformer",
        build_model=lambda config, device: HTPModel(config).to(device),
        prepare_batch=transformer_prepare_batch,
        read_predictions=read_transformer_predictions,
        save_checkpoint=save_transformer_checkpoint,
    ),
    "lstm": ModelSpec(
        name="lstm",
        build_model=build_social_lstm_model,
        prepare_batch=social_lstm_prepare_batch,
        read_predictions=read_transformer_predictions,
        save_checkpoint=save_social_lstm_checkpoint,
    ),
}


def register_model_spec(spec: ModelSpec) -> None:
    MODEL_REGISTRY[spec.name] = spec


def get_model_spec(name: str) -> ModelSpec:
    if name == "lstm" and name not in MODEL_REGISTRY:
        raise ValueError("LSTM support requires registering a ModelSpec named 'lstm'.")
    try:
        return MODEL_REGISTRY[name]
    except KeyError as exc:
        raise ValueError(f"Unknown model {name!r}. Registered models: {sorted(MODEL_REGISTRY)}") from exc
