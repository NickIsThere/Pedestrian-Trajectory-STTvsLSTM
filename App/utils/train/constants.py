from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

UTILS_TRAIN_DIR = Path(__file__).resolve().parent
APP_DIR = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "htp_model.pt"
HTP_OUTPUT_ROOTS = (
    PROJECT_ROOT / "outputs" / "loo",
    APP_DIR / "outputs" / "loo",
)
DEFAULT_METRIC_FPS = 25.0

MOT20_TRAIN_SEQUENCES = ("MOT20-01", "MOT20-02", "MOT20-03")
MOT20_VAL_SEQUENCES = ("MOT20-05",)
SAMPLE_MODES = ("gt", "det")
SampleMode = Literal["gt", "det"]
DOMAIN_VALUES = {"real", "synthetic", "mixed", "unknown"}
LOSS_CONFIG = {"delta_weight": 0.5, "position_weight": 1.0, "beta": 1.0}
SOCIAL_LSTM_GRID_SIZE = int(os.environ.get("SOCIAL_LSTM_GRID_SIZE", "4"))
