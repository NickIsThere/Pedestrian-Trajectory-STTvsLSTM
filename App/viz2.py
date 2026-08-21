from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import torch
from tqdm import tqdm

from benchmarks.kalman.kalman_filter import Kalman2D
from data.reader import Frame, Sequence, Track, get_dataset
from evaluation.eval import get_denormalize
from model.config import ModelConfig
from model.model import HTPModel
from utils.modelutils import load_htp_model, prepare_scene_inference_batch
from utils.train.constants import PROJECT_ROOT

# ── Which paths to draw ───────────────────────────────────────────────────────
SHOW_TRANSFORMER_MOT20 = True
SHOW_TRANSFORMER_MOTSYNTH = True
SHOW_SOCIAL_LSTM = False
SHOW_KALMAN = True

# ── Annotation source for bboxes + model inputs ───────────────────────────────
BBOX_SOURCE: Literal["det", "gt"] = "det"

# ── Prediction horizon (model timesteps, max 10) ──────────────────────────────
DISPLAY_STEPS = 10

# ── Raw frames advanced per Kalman forecast point (matches Social LSTM subsample)
KALMAN_STEP_STRIDE = 5

# ── Bbox overlay ──────────────────────────────────────────────────────────────
DRAW_BBOXES = True

# ── Sequences to render ───────────────────────────────────────────────────────
SEQUENCES = ("MOT20-01", "MOT20-02", "MOT20-03", "MOT20-05")

# ── Checkpoint roots (relative to project root) ───────────────────────────────
TRANSFORMER_MOT20_CHECKPOINT_DIR = Path("outputs/loo/mot20-50ep/checkpoints")
TRANSFORMER_MOTSYNTH_OUTPUT_ROOT = Path("outputs/loo/motsynth-curriculum-50ep")
SOCIAL_LSTM_OUTPUT_ROOT = Path("outputs/social_lstm")

# ── BGR colors per path ───────────────────────────────────────────────────────
COLOR_TRANSFORMER_MOT20 = (0, 255, 0)
COLOR_TRANSFORMER_MOTSYNTH = (255, 165, 0)
COLOR_SOCIAL_LSTM = (255, 0, 255)
COLOR_KALMAN = (255, 255, 0)
COLOR_BBOX = (0, 0, 255)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_WARNED_MISSING: set[str] = set()
_MOTSYNTH_MODEL: tuple[HTPModel, ModelConfig] | None = None


def _resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _clamp_display_steps(steps: int) -> int:
    return max(1, min(steps, 10))


def _warn_once(key: str, message: str) -> None:
    if key not in _WARNED_MISSING:
        _WARNED_MISSING.add(key)
        print(message)


def _resolve_social_lstm_checkpoint(root: Path, sequence_name: str) -> Path | None:
    """Pick the best available Social LSTM checkpoint for a sequence.

    Prefers the most *complete* run (the one with the most fold checkpoints,
    used as a proxy for "fully trained") over merely the newest run by name, so
    short/aborted runs don't get auto-selected. Ties break on the latest name.
    """
    if not root.is_dir():
        return None

    candidates: list[tuple[int, str, Path]] = []
    for run in (p for p in root.iterdir() if p.is_dir()):
        ckpt_dir = run / "checkpoints"
        ckpt = ckpt_dir / f"social_lstm__val_{sequence_name}.pt"
        if ckpt.is_file():
            fold_count = len(list(ckpt_dir.glob("social_lstm__val_*.pt")))
            candidates.append((fold_count, run.name, ckpt))

    if not candidates:
        return None

    candidates.sort(key=lambda c: (c[0], c[1]), reverse=True)
    return candidates[0][2]


def _sequence_checkpoint(checkpoint_dir: Path, prefix: str, sequence_name: str) -> Path:
    return checkpoint_dir / f"{prefix}__val_{sequence_name}.pt"


def _load_transformer(
    checkpoint_dir: Path,
    sequence_name: str,
    label: str,
) -> tuple[HTPModel, ModelConfig] | None:
    path = _sequence_checkpoint(checkpoint_dir, "transformer", sequence_name)
    if not path.is_file():
        _warn_once(f"{label}:{sequence_name}", f"[{label}] No checkpoint for {sequence_name}: {path}")
        return None
    try:
        model, cfg = load_htp_model(path, device=DEVICE)
        print(f"[{label}] Loaded {path.name}")
        return model, cfg
    except RuntimeError as exc:
        _warn_once(f"{label}:{sequence_name}:err", f"[{label}] Could not load {path}: {exc}")
        return None


def _resolve_motsynth_best_checkpoint(output_root: Path) -> Path | None:
    """Resolve the single shared best MOTSynth checkpoint (not per MOT20 sequence)."""
    summary_path = output_root / "summary.csv"
    if summary_path.is_file():
        with summary_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                raw = (row.get("best_checkpoint") or "").strip()
                if not raw:
                    continue
                path = _resolve_path(Path(raw))
                if path.is_file():
                    return path

    ckpt_dir = output_root / "checkpoints"
    fold_best = sorted(ckpt_dir.glob("transformer__val_MOTSynth-*.pt"))
    if fold_best:
        return fold_best[0]
    return None


def _load_motsynth_transformer(
    output_root: Path,
    label: str,
) -> tuple[HTPModel, ModelConfig] | None:
    global _MOTSYNTH_MODEL

    if _MOTSYNTH_MODEL is not None:
        return _MOTSYNTH_MODEL

    path = _resolve_motsynth_best_checkpoint(output_root)
    if path is None:
        _warn_once(f"{label}:missing", f"[{label}] No best MOTSynth checkpoint found in {output_root}")
        return None
    try:
        model, cfg = load_htp_model(path, device=DEVICE)
        print(f"[{label}] Loaded {path.name}")
        _MOTSYNTH_MODEL = (model, cfg)
        return _MOTSYNTH_MODEL
    except RuntimeError as exc:
        _warn_once(f"{label}:err", f"[{label}] Could not load {path}: {exc}")
        return None


def _load_social_lstm(checkpoint_path: Path, sequence_name: str) -> tuple[torch.nn.Module, dict] | None:
    if not checkpoint_path.is_file():
        _warn_once(f"social_lstm:{sequence_name}", f"[social_lstm] No checkpoint for {sequence_name}: {checkpoint_path}")
        return None

    from benchmarks.social_lstm.model import SocialLSTM

    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    config = ckpt.get("config", {})
    model = SocialLSTM(
        obs_len=int(config.get("obs_len", 15)),
        pred_len=int(config.get("pred_len", 10)),
        hidden_size=int(config.get("hidden_size", 128)),
        embedding_dim=int(config.get("embedding_dim", 64)),
        grid_size=int(config.get("grid_size", 4)),
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"[social_lstm] Loaded {checkpoint_path.name}")
    return model, config


def _frame_annotations(frame: Frame, source: str) -> dict:
    return frame.det if source == "det" else frame.gt


def _source_tracks(sequence: Sequence, source: str) -> dict[int, Track]:
    channel_name = "det" if source == "det" else "gt"
    channel = sequence.channels.get(channel_name)
    if channel is None:
        return {}
    return channel.tracks


def _build_frame_positions(tracks: dict[int, Track]) -> dict[int, dict[int, tuple[float, float]]]:
    frame_positions: dict[int, dict[int, tuple[float, float]]] = {}
    for track_id, track in tracks.items():
        for frame_id, ann in track.gt.items():
            frame_positions.setdefault(frame_id, {})[track_id] = (ann.bbox.foot_x, ann.bbox.foot_y)
    return frame_positions


def _kalman_rollout(kf: Kalman2D, steps: int, stride: int = 1) -> list[tuple[int, int]]:
    """Forecast `steps` future foot points without consuming new measurements.

    Each output point advances `stride` raw frames (so with stride=5 the path
    jumps in 5-frame increments, matching the Social LSTM subsample cadence).
    The filter state is snapshotted and restored so the caller can keep feeding
    real observations afterwards.
    """
    x_save = kf.x.copy()
    p_save = kf.P.copy()
    points: list[tuple[int, int]] = []
    px = py = 0.0
    for _ in range(steps):
        for _ in range(max(1, stride)):
            px, py = kf.predict(dt=1.0)
        points.append((int(px), int(py)))
    kf.x = x_save
    kf.P = p_save
    return points


def _build_kalman_forecasts(
    tracks: dict[int, Track],
    steps: int,
    *,
    stride: int = KALMAN_STEP_STRIDE,
    process_noise: float = 1.0,
    measurement_noise: float = 10.0,
) -> dict[int, dict[int, list[tuple[int, int]]]]:
    """Per-frame constant-velocity Kalman forecasts.

    For every track, the filter is advanced through its observed frames (predict
    over gaps, update on each observation). After incorporating the observation
    at frame f, we roll the state forward `steps` points (each `stride` raw
    frames apart) to obtain the future trajectory predicted *from f using only
    information up to f*.
    """
    forecasts: dict[int, dict[int, list[tuple[int, int]]]] = {}

    for track_id, track in tracks.items():
        if not track.gt:
            continue

        sorted_frames = sorted(track.gt.keys())
        first_ann = track.gt[sorted_frames[0]]
        kf = Kalman2D(
            first_ann.bbox.foot_x,
            first_ann.bbox.foot_y,
            process_noise=process_noise,
            measurement_noise=measurement_noise,
        )

        track_forecasts: dict[int, list[tuple[int, int]]] = {}
        prev_frame = sorted_frames[0]
        track_forecasts[prev_frame] = _kalman_rollout(kf, steps, stride)

        for frame_id in sorted_frames[1:]:
            for _ in range(frame_id - prev_frame):
                kf.predict(dt=1.0)
            ann = track.gt[frame_id]
            kf.update(ann.bbox.foot_x, ann.bbox.foot_y)
            track_forecasts[frame_id] = _kalman_rollout(kf, steps, stride)
            prev_frame = frame_id

        forecasts[track_id] = track_forecasts

    return forecasts


@dataclass
class SequenceModels:
    transformer_mot20: tuple[HTPModel, ModelConfig] | None = None
    transformer_motsynth: tuple[HTPModel, ModelConfig] | None = None
    social_lstm: tuple[torch.nn.Module, dict] | None = None
    kalman_forecasts: dict[int, dict[int, list[tuple[int, int]]]] = field(default_factory=dict)
    frame_positions: dict[int, dict[int, tuple[float, float]]] = field(default_factory=dict)
    subsampled_frames: list[int] = field(default_factory=list)
    img_width: float = 1.0
    img_height: float = 1.0


def _load_sequence_models(sequence: Sequence) -> SequenceModels:
    models = SequenceModels()
    name = sequence.name

    if SHOW_TRANSFORMER_MOT20:
        models.transformer_mot20 = _load_transformer(
            _resolve_path(TRANSFORMER_MOT20_CHECKPOINT_DIR),
            name,
            "transformer_mot20",
        )

    if SHOW_TRANSFORMER_MOTSYNTH:
        models.transformer_motsynth = _load_motsynth_transformer(
            _resolve_path(TRANSFORMER_MOTSYNTH_OUTPUT_ROOT),
            "transformer_motsynth",
        )

    if SHOW_SOCIAL_LSTM:
        ckpt_path = _resolve_social_lstm_checkpoint(_resolve_path(SOCIAL_LSTM_OUTPUT_ROOT), name)
        if ckpt_path is None:
            _warn_once("social_lstm:run", f"[social_lstm] No checkpoint found for {name}.")
        else:
            models.social_lstm = _load_social_lstm(ckpt_path, name)

    seq_info = sequence.info["Sequence"]
    models.img_width = float(seq_info.get("imWidth", 1920))
    models.img_height = float(seq_info.get("imHeight", 1080))

    source_tracks = _source_tracks(sequence, BBOX_SOURCE)
    models.frame_positions = _build_frame_positions(source_tracks)

    if SHOW_SOCIAL_LSTM and models.frame_positions:
        subsample_step = 5
        if models.social_lstm is not None:
            subsample_step = int(models.social_lstm[1].get("subsample_step", 5))
        all_frame_ids = sorted(sequence.frames.keys())
        models.subsampled_frames = all_frame_ids[::subsample_step]

    if SHOW_KALMAN and source_tracks:
        models.kalman_forecasts = _build_kalman_forecasts(
            source_tracks, _clamp_display_steps(DISPLAY_STEPS)
        )

    return models


def _draw_path(image: np.ndarray, anchor: tuple[int, int], points: list[tuple[int, int]], color: tuple[int, int, int]) -> None:
    if not points:
        return
    path = [anchor, *points]
    cv2.polylines(image, [np.array(path, dtype=np.int32)], False, color, 2)
    for point in points:
        cv2.circle(image, point, 3, color, -1)


def _draw_bboxes(image: np.ndarray, frame: Frame) -> None:
    for annotation in _frame_annotations(frame, BBOX_SOURCE).values():
        x = int(annotation.bbox.x)
        y = int(annotation.bbox.y)
        x2 = int(annotation.bbox.x + annotation.bbox.width)
        y2 = int(annotation.bbox.y + annotation.bbox.height)
        cv2.rectangle(image, (x, y), (x2, y2), COLOR_BBOX, 1)
        cv2.putText(
            image,
            str(annotation.track_id),
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            COLOR_BBOX,
            1,
        )


def _predict_transformer_paths(
    sequence: Sequence,
    frame: Frame,
    model: HTPModel,
    cfg: ModelConfig,
    denormalize,
    display_steps: int,
) -> dict[int, list[tuple[int, int]]]:
    prepared, track_ids = prepare_scene_inference_batch(
        sequence,
        frame.id,
        device=DEVICE,
        config=cfg,
        source=BBOX_SOURCE,
    )
    if not track_ids:
        return {}

    with torch.no_grad():
        pred = model(**prepared)

    steps = min(display_steps, cfg.future_steps)
    annotations = _frame_annotations(frame, BBOX_SOURCE)
    paths: dict[int, list[tuple[int, int]]] = {}

    for agent_index, track_id in enumerate(track_ids):
        if track_id not in annotations:
            continue
        points: list[tuple[int, int]] = []
        for i in range(steps):
            px_n = float(pred["pred_positions"][0, agent_index, i, 0])
            py_n = float(pred["pred_positions"][0, agent_index, i, 1])
            px, py = denormalize(px_n, py_n)
            points.append((int(px), int(py)))
        paths[track_id] = points

    return paths


def _predict_social_lstm_paths(
    frame: Frame,
    models: SequenceModels,
    display_steps: int,
) -> dict[int, list[tuple[int, int]]]:
    if models.social_lstm is None or not models.subsampled_frames:
        return {}

    model, config = models.social_lstm
    obs_len = int(config.get("obs_len", 15))
    pred_len = int(config.get("pred_len", 10))
    steps = min(display_steps, pred_len)

    eligible = [i for i, fid in enumerate(models.subsampled_frames) if fid <= frame.id]
    if len(eligible) < obs_len:
        return {}

    idx = eligible[-1]
    start = idx - obs_len + 1
    obs_frames = models.subsampled_frames[start : idx + 1]
    last_obs_frame = obs_frames[-1]

    annotations = _frame_annotations(frame, BBOX_SOURCE)
    track_ids = sorted(
        tid for tid in annotations if tid in models.frame_positions.get(last_obs_frame, {})
    )
    if not track_ids:
        return {}

    img_width = models.img_width
    img_height = models.img_height
    max_peds = len(track_ids)
    tid_to_slot = {tid: slot for slot, tid in enumerate(track_ids)}

    pos_arr = np.zeros((obs_len, max_peds, 2), dtype=np.float32)
    mask_arr = np.zeros((obs_len, max_peds), dtype=np.float32)

    for t, fid in enumerate(obs_frames):
        fp = models.frame_positions.get(fid, {})
        for tid, slot in tid_to_slot.items():
            if tid in fp:
                pos_arr[t, slot, 0] = fp[tid][0] / img_width
                pos_arr[t, slot, 1] = fp[tid][1] / img_height
                mask_arr[t, slot] = 1.0

    obs_pos = torch.from_numpy(pos_arr).unsqueeze(0).to(DEVICE)
    obs_mask = torch.from_numpy(mask_arr).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred_disps = model(obs_pos, obs_mask, pred_len=steps)

    pred_disps = pred_disps.squeeze(0).cpu().numpy()
    # Cumulative normalized displacements over the subsampled prediction horizon.
    cum_disps = np.cumsum(pred_disps, axis=0)  # (steps, max_peds, 2)

    paths: dict[int, list[tuple[int, int]]] = {}
    for tid, slot in tid_to_slot.items():
        if mask_arr[-1, slot] == 0:
            continue
        ann = annotations.get(tid)
        if ann is None:
            continue
        # Re-root the relative forecast at the live foot position rather than the
        # last subsampled observation. last_obs_frame only advances every
        # subsample_step frames, so anchoring there makes the path lag behind the
        # pedestrian between subsamples; anchoring at the current foot tracks it.
        cur_x = ann.bbox.foot_x / img_width
        cur_y = ann.bbox.foot_y / img_height
        points: list[tuple[int, int]] = []
        for step in range(steps):
            px = (cur_x + float(cum_disps[step, slot, 0])) * img_width
            py = (cur_y + float(cum_disps[step, slot, 1])) * img_height
            points.append((int(px), int(py)))
        paths[tid] = points

    return paths


def _predict_kalman_paths(
    frame: Frame,
    models: SequenceModels,
    display_steps: int,
) -> dict[int, list[tuple[int, int]]]:
    if not models.kalman_forecasts:
        return {}

    annotations = _frame_annotations(frame, BBOX_SOURCE)
    paths: dict[int, list[tuple[int, int]]] = {}

    for track_id in annotations:
        track_forecasts = models.kalman_forecasts.get(track_id)
        if not track_forecasts:
            continue
        points = track_forecasts.get(frame.id)
        if points:
            paths[track_id] = points[:display_steps]

    return paths


def _draw_model_paths(
    image: np.ndarray,
    frame: Frame,
    paths: dict[int, list[tuple[int, int]]],
    color: tuple[int, int, int],
) -> None:
    annotations = _frame_annotations(frame, BBOX_SOURCE)
    for track_id, points in paths.items():
        ann = annotations.get(track_id)
        if ann is None:
            continue
        anchor = (int(ann.bbox.foot_x), int(ann.bbox.foot_y))
        _draw_path(image, anchor, points, color)


def draw_frame(image: np.ndarray, frame: Frame, sequence: Sequence, models: SequenceModels) -> np.ndarray:
    display_steps = _clamp_display_steps(DISPLAY_STEPS)
    denormalize = get_denormalize(sequence)

    if DRAW_BBOXES:
        _draw_bboxes(image, frame)

    if SHOW_TRANSFORMER_MOT20 and models.transformer_mot20 is not None:
        model, cfg = models.transformer_mot20
        paths = _predict_transformer_paths(sequence, frame, model, cfg, denormalize, display_steps)
        _draw_model_paths(image, frame, paths, COLOR_TRANSFORMER_MOT20)

    if SHOW_TRANSFORMER_MOTSYNTH and models.transformer_motsynth is not None:
        model, cfg = models.transformer_motsynth
        paths = _predict_transformer_paths(sequence, frame, model, cfg, denormalize, display_steps)
        _draw_model_paths(image, frame, paths, COLOR_TRANSFORMER_MOTSYNTH)

    if SHOW_SOCIAL_LSTM and models.social_lstm is not None:
        paths = _predict_social_lstm_paths(frame, models, display_steps)
        _draw_model_paths(image, frame, paths, COLOR_SOCIAL_LSTM)

    if SHOW_KALMAN and models.kalman_forecasts:
        paths = _predict_kalman_paths(frame, models, display_steps)
        _draw_model_paths(image, frame, paths, COLOR_KALMAN)

    return image


def generate_visualizations() -> None:
    dataset = get_dataset("MOT20")
    sequence_set = set(SEQUENCES)

    for split in dataset.splits.values():
        for sequence in split.sequences.values():
            if sequence.name not in sequence_set:
                continue

            print(f"Rendering {sequence.name}...")
            models = _load_sequence_models(sequence)

            seq_info = sequence.info["Sequence"]
            fps = int(seq_info.get("frameRate", 25))
            width = int(seq_info.get("imWidth", 1920))
            height = int(seq_info.get("imHeight", 1080))
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(f"{sequence.name}.mp4", fourcc, fps, (width, height))

            for frame in tqdm(
                sorted(sequence.frames.values(), key=lambda f: f.id),
                desc=sequence.name,
                unit="frame",
                leave=False,
            ):
                image = cv2.imread(str(frame.path))
                if image is None:
                    continue
                image = draw_frame(image, frame, sequence, models)
                video_writer.write(image)

            video_writer.release()


if __name__ == "__main__":
    generate_visualizations()
