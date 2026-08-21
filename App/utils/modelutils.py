"""Backward-compatible re-exports. Prefer utils.train.*."""

from utils.train.batches import prepare_batch, prepare_scene_batch
from utils.train.checkpoints import (
    DEFAULT_CHECKPOINT_PATH,
    candidate_htp_checkpoint_paths,
    load_htp_model,
    save_model_checkpoint,
)
from utils.train.constants import DEFAULT_METRIC_FPS, HTP_OUTPUT_ROOTS
from utils.train.coordinates import denormalize_foot_xy, norm_offsets_from_seqinfo
from utils.train.inference import (
    prepare_inference_batch,
    prepare_inference_batch_batched,
    prepare_scene_inference_batch,
)
from utils.train.metrics import (
    constant_velocity_baseline,
    delta_diagnostics,
    horizon_steps_for_config,
    trajectory_metrics,
    zero_motion_baseline,
)
from utils.train.targets import (
    assert_target_reconstruction_matches,
    last_observed_position,
    target_reconstruction_error,
)

__all__ = [
    "DEFAULT_CHECKPOINT_PATH",
    "DEFAULT_METRIC_FPS",
    "HTP_OUTPUT_ROOTS",
    "assert_target_reconstruction_matches",
    "candidate_htp_checkpoint_paths",
    "constant_velocity_baseline",
    "delta_diagnostics",
    "denormalize_foot_xy",
    "horizon_steps_for_config",
    "last_observed_position",
    "load_htp_model",
    "norm_offsets_from_seqinfo",
    "prepare_batch",
    "prepare_inference_batch",
    "prepare_inference_batch_batched",
    "prepare_scene_batch",
    "prepare_scene_inference_batch",
    "save_model_checkpoint",
    "target_reconstruction_error",
    "trajectory_metrics",
    "zero_motion_baseline",
]
