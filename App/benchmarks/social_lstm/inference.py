"""
Social LSTM inference — real multi-agent batched rollout.

Replaces the placeholder (hold-last-position) with a proper forward pass
through the trained SocialLSTM decoder, matching the training regime exactly:
  - obs_len / pred_len / subsample_step read from checkpoint config
  - All tracks batched into a single forward call (no per-track loop)
  - Normalization derived from the observed window (same as training)
  - Model cached after first load (no repeated disk I/O)

Usage (same API as before):
    from App.benchmarks.social_lstm.inference import predict_tracks_social_lstm
    predicted = predict_tracks_social_lstm(tracks, future_steps=10)
"""

from __future__ import annotations

import torch
import numpy as np
from pathlib import Path

from App.data.reader import Track, Annotation, BBox

SOCIAL_LSTM_CHECKPOINT = (
    Path(__file__).parent.parent.parent / "checkpoints" / "social_lstm" / "best_model.pt"
)
# Default: 10 steps × 0.2 s/step = 2.0 s  (aligned to STT pred_len)
_DEFAULT_FUTURE_STEPS = 10

# Module-level singleton — loaded once, reused across calls
_CACHE: dict = {}


def _load_model(checkpoint_path: Path, device: str):
    """
    Main writer: Keez Cuijpers
    Reviewer: 
    Contributors: Ciprian Driscu

    Load (or return cached) SocialLSTM model + config.
    """
    key = str(checkpoint_path)
    if key in _CACHE:
        return _CACHE[key]

    from .model import SocialLSTM

    ckpt   = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt.get("config", {})
    model  = SocialLSTM(
        obs_len=int(config.get("obs_len", 15)),
        pred_len=int(config.get("pred_len", 10)),
        hidden_size=int(config.get("hidden_size", 128)),
        embedding_dim=int(config.get("embedding_dim", 64)),
        grid_size=int(config.get("grid_size", 4)),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    _CACHE[key] = (model, config)
    return model, config


def _build_obs_tensor(
    tracks: dict[int, Track],
    obs_len: int,
    device: str,
) -> tuple[
    torch.Tensor,   # obs_positions  (1, obs_len, max_peds, 2)  normalised
    torch.Tensor,   # obs_masks      (1, obs_len, max_peds)
    list[int],      # ordered track_ids
    float, float,   # x_min, x_range
    float, float,   # y_min, y_range
]:
    """
    Main writer: Keez Cuijpers
    Reviewer: 
    Contributors: Ciprian Driscu

    Convert the last `obs_len` frames of each track into a normalised
    observation tensor suitable for SocialLSTM.forward().
    """
    track_ids = sorted(tracks.keys())
    max_peds  = len(track_ids)

    # Collect all frame ids that appear in any track; take the last obs_len
    all_frames: set[int] = set()
    for t in tracks.values():
        all_frames.update(t.gt.keys())
    frames = sorted(all_frames)[-obs_len:]

    # Gather raw foot-point coordinates for normalisation
    xs, ys = [], []
    for tid in track_ids:
        for f in frames:
            ann = tracks[tid].gt.get(f)
            if ann is not None:
                xs.append(ann.bbox.foot_x)
                ys.append(ann.bbox.foot_y)

    x_min, x_range = (min(xs), max(max(xs) - min(xs), 1e-6)) if xs else (0.0, 1.0)
    y_min, y_range = (min(ys), max(max(ys) - min(ys), 1e-6)) if ys else (0.0, 1.0)

    # Fill normalised arrays
    pos_arr  = np.zeros((obs_len, max_peds, 2), dtype=np.float32)
    mask_arr = np.zeros((obs_len, max_peds),    dtype=np.float32)

    for f_idx, f in enumerate(frames):
        for t_idx, tid in enumerate(track_ids):
            ann = tracks[tid].gt.get(f)
            if ann is not None:
                pos_arr[f_idx, t_idx, 0] = (ann.bbox.foot_x - x_min) / x_range
                pos_arr[f_idx, t_idx, 1] = (ann.bbox.foot_y - y_min) / y_range
                mask_arr[f_idx, t_idx]   = 1.0

    obs_pos  = torch.from_numpy(pos_arr).unsqueeze(0).to(device)   # (1, T, P, 2)
    obs_mask = torch.from_numpy(mask_arr).unsqueeze(0).to(device)  # (1, T, P)
    return obs_pos, obs_mask, track_ids, x_min, x_range, y_min, y_range


def predict_tracks_social_lstm(
    tracks: dict[int, Track],
    *,
    future_steps: int = _DEFAULT_FUTURE_STEPS,
    checkpoint_path: Path = SOCIAL_LSTM_CHECKPOINT,
) -> dict[int, Track]:
    """
    Main writer: Keez Cuijpers
    Reviewer: 
    Contributors: Ciprian Driscu

    Generate Social LSTM predictions for all tracks in one batched forward pass.

    Args:
        tracks:          Dict of track_id → Track with ground-truth observations.
        future_steps:    Number of future frames to predict (default 10 = 2.0 s).
        checkpoint_path: Path to trained checkpoint.

    Returns:
        Dict of track_id → Track — all GT frames preserved, predicted future
        frames appended.
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Social LSTM checkpoint not found: {checkpoint_path}\n"
            "Train first:  python -m App.benchmarks.social_lstm.train"
        )

    device        = "cuda" if torch.cuda.is_available() else "cpu"
    model, config = _load_model(checkpoint_path, device)
    obs_len       = int(config.get("obs_len", 15))

    # ── build input tensors ───────────────────────────────────────────────────
    obs_pos, obs_mask, track_ids, x_min, x_range, y_min, y_range = (
        _build_obs_tensor(tracks, obs_len, device)
    )

    # ── single batched forward pass ───────────────────────────────────────────
    with torch.no_grad():
        # Free-running rollout (no teacher forcing at inference)
        pred_disps = model(obs_pos, obs_mask, pred_len=future_steps)
        # pred_disps: (1, future_steps, max_peds, 2)  normalised displacements

    pred_disps = pred_disps.squeeze(0).cpu().numpy()   # (T, P, 2)

    # Cumulative sum → absolute normalised positions
    last_norm   = obs_pos[0, -1].cpu().numpy()          # (P, 2)
    pred_norm   = np.cumsum(pred_disps, axis=0) + last_norm[np.newaxis]  # (T, P, 2)

    # Denormalise
    pred_abs        = pred_norm.copy()
    pred_abs[..., 0] = pred_norm[..., 0] * x_range + x_min
    pred_abs[..., 1] = pred_norm[..., 1] * y_range + y_min

    # ── assemble Track objects ────────────────────────────────────────────────
    predicted_tracks: dict[int, Track] = {}

    for t_idx, tid in enumerate(track_ids):
        track = tracks.get(tid)
        if track is None or not track.gt:
            continue

        predicted_gt = dict(track.gt)          # copy GT frames
        last_frame   = max(track.gt.keys())
        last_ann     = track.gt[last_frame]

        for step in range(future_steps):
            future_frame = last_frame + step + 1
            px = float(pred_abs[step, t_idx, 0])
            py = float(pred_abs[step, t_idx, 1])

            orig = last_ann.bbox
            new_bbox = BBox(
                x=px - orig.width  / 2.0,
                y=py - orig.height,
                width=orig.width,
                height=orig.height,
            )

            predicted_gt[future_frame] = Annotation(
                frame_id=future_frame,
                track_id=tid,
                bbox=new_bbox,
                score=last_ann.score,
                class_id=last_ann.class_id,
                visibility=last_ann.visibility,
                x=px,
                y=py,
                z=last_ann.z,
            )

        predicted_tracks[tid] = Track(id=tid, gt=predicted_gt)

    return predicted_tracks
