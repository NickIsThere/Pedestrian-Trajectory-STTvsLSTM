from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path
from typing import cast

from model.config import ModelConfig

from utils.train.checkpoints import DEFAULT_CHECKPOINT_PATH
from utils.train.constants import DOMAIN_VALUES, SAMPLE_MODES, SampleMode
from utils.train.types import BASE_TRAINING_DEFAULTS, TrainingDefaults


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_optional_float(name: str, default: float | None = None) -> float | None:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return float(value)


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def training_defaults_from_env(base: TrainingDefaults = BASE_TRAINING_DEFAULTS) -> TrainingDefaults:
    train_domain = os.environ.get("TRAIN_DOMAIN", base.train_domain)
    test_domain = os.environ.get("TEST_DOMAIN", base.test_domain)
    if train_domain not in DOMAIN_VALUES:
        raise ValueError(f"TRAIN_DOMAIN must be one of {sorted(DOMAIN_VALUES)}, got {train_domain!r}")
    if test_domain not in DOMAIN_VALUES:
        raise ValueError(f"TEST_DOMAIN must be one of {sorted(DOMAIN_VALUES)}, got {test_domain!r}")
    return TrainingDefaults(
        model=os.environ.get("MODEL", base.model),
        train_steps=int(os.environ.get("TRAIN_STEPS", str(base.train_steps))),
        val_interval=int(os.environ.get("VAL_INTERVAL", str(base.val_interval))),
        max_val_samples=int(os.environ.get("MAX_VAL_SAMPLES", str(base.max_val_samples))),
        learning_rate=float(os.environ.get("LEARNING_RATE", str(base.learning_rate))),
        leave_one_out=_env_bool("LEAVE_ONE_OUT", base.leave_one_out),
        fold_sequence=os.environ.get("FOLD_SEQUENCE", base.fold_sequence),
        run_id=os.environ.get("RUN_ID", base.run_id),
        debug_smoke=_env_bool("DEBUG_SMOKE_BATCH", base.debug_smoke),
        export_evaluation_units=_env_bool("EXPORT_EVALUATION_UNITS", base.export_evaluation_units),
        model_family=os.environ.get("MODEL_FAMILY", base.model_family),
        model_variant=os.environ.get("MODEL_VARIANT", base.model_variant),
        train_domain=train_domain,
        test_domain=test_domain,
        synthetic_size_fraction=_env_optional_float("SYNTHETIC_SIZE_FRACTION", base.synthetic_size_fraction),
        train_input_mode=cast(SampleMode, os.environ.get("TRAIN_INPUT_MODE", base.train_input_mode)),
        output_dir=Path(os.environ.get("TRAIN_OUTPUT_DIR", str(base.output_dir))),
        checkpoint_path=Path(os.environ.get("HTP_CHECKPOINT", str(DEFAULT_CHECKPOINT_PATH))),
        batch_size=int(os.environ.get("BATCH_SIZE", str(base.batch_size))),
        seed=int(os.environ.get("SEED", str(base.seed))),
    )


TRAINING_DEFAULTS = training_defaults_from_env(BASE_TRAINING_DEFAULTS)
if TRAINING_DEFAULTS.train_input_mode not in SAMPLE_MODES:
    raise ValueError(
        f"TRAIN_INPUT_MODE must be one of {SAMPLE_MODES}, got {TRAINING_DEFAULTS.train_input_mode!r}",
    )

BATCH_SIZE = TRAINING_DEFAULTS.batch_size
TRAIN_STEPS = TRAINING_DEFAULTS.train_steps
VAL_INTERVAL = TRAINING_DEFAULTS.val_interval
MAX_VAL_SAMPLES = TRAINING_DEFAULTS.max_val_samples
CHECKPOINT_PATH = TRAINING_DEFAULTS.checkpoint_path
TRAIN_INPUT_MODE = TRAINING_DEFAULTS.train_input_mode


def build_model_config_from_env(base: ModelConfig | None = None) -> ModelConfig:
    config = base or ModelConfig()
    updated = replace(
        config,
        lookback=_env_int("MODEL_LOOKBACK", config.lookback),
        future_steps=_env_int("MODEL_FUTURE_STEPS", config.future_steps),
        patch_size=_env_int("MODEL_PATCH_SIZE", config.patch_size),
        trajectory_stride=_env_int("TRAJECTORY_STRIDE", config.trajectory_stride),
        window_start_stride=_env_int("WINDOW_START_STRIDE", config.window_start_stride),
    )
    if updated.lookback <= 0:
        raise ValueError("MODEL_LOOKBACK must be positive")
    if updated.future_steps <= 0:
        raise ValueError("MODEL_FUTURE_STEPS must be positive")
    if updated.patch_size <= 0:
        raise ValueError("MODEL_PATCH_SIZE must be positive")
    if updated.trajectory_stride <= 0:
        raise ValueError("TRAJECTORY_STRIDE must be positive")
    if updated.window_start_stride <= 0:
        raise ValueError("WINDOW_START_STRIDE must be positive")
    return updated


model_config = build_model_config_from_env()


def _configure_social_lstm_environment(train_config: TrainingDefaults) -> None:
    """Forward main-trainer settings into the benchmark Social LSTM config."""
    from utils.train.constants import APP_DIR

    os.environ["SOCIAL_LSTM_OUTPUT_DIR"] = str(train_config.output_dir)
    os.environ["SOCIAL_LSTM_CHECKPOINT_DIR"] = str(APP_DIR / "checkpoints" / "social_lstm")
    os.environ["BATCH_SIZE"] = str(train_config.batch_size)
    os.environ["LEARNING_RATE"] = str(train_config.learning_rate)
    os.environ["NUM_EPOCHS"] = os.environ.get("NUM_EPOCHS", str(train_config.train_steps))
    os.environ["OBS_LEN"] = os.environ.get("OBS_LEN", str(model_config.lookback))
    os.environ["PRED_LEN"] = os.environ.get("PRED_LEN", str(model_config.future_steps))
    os.environ["SUBSAMPLE_STEP"] = os.environ.get("SUBSAMPLE_STEP", str(model_config.trajectory_stride))
    os.environ["LEAVE_ONE_OUT"] = "true" if train_config.leave_one_out else "false"
    if train_config.fold_sequence is None:
        os.environ.pop("FOLD_SEQUENCE", None)
    else:
        os.environ["FOLD_SEQUENCE"] = train_config.fold_sequence
    if train_config.run_id is None:
        os.environ.pop("RUN_ID", None)
    else:
        os.environ["RUN_ID"] = train_config.run_id
    os.environ["SEED"] = str(train_config.seed)
