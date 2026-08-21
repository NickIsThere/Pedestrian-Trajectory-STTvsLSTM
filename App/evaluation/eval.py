"""Channel providers for dynamic sequence visualization."""

from __future__ import annotations

import os
import copy
import numpy as np
from pathlib import Path
from typing import Callable
import torch
from tqdm import tqdm

from App.data.reader import Annotation, BBox, Channel, Forecast, ForecastChannel, ForecastPoint, ForecastTrack, Sequence, Track
from App.benchmarks.kalman.kalman_filter import predict_tracks
from App.model.config import ModelConfig
from App.model.model import HTPModel
from App.utils.modelutils import prepare_scene_inference_batch


_ACTIVE_MODEL_ID = None
_ACTIVE_MODEL = None
_ACTIVE_CFG = None
_ACTIVE_TYPE = None
_ACTIVE_DEVICE = None

def _ensure_dynamic_model_loaded(model_id: str):
    """Dynamically finds and loads any checkpoint file from the root checkpoints dir."""
    global _ACTIVE_MODEL_ID, _ACTIVE_MODEL, _ACTIVE_CFG, _ACTIVE_TYPE, _ACTIVE_DEVICE

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    if _ACTIVE_MODEL_ID == model_id and _ACTIVE_MODEL is not None:
        return _ACTIVE_MODEL, _ACTIVE_CFG, _ACTIVE_TYPE, dev

    project_root = Path(__file__).resolve().parents[2]
    checkpoints_dir = project_root / "checkpoints"

    found_paths = list(checkpoints_dir.rglob(f"{model_id}.pt"))
    if not found_paths:
        print(f"[Error] Checkpoint {model_id}.pt not found in {checkpoints_dir}")
        return None, None, None, dev

    path = found_paths[0]
    print(f"\n--- Loading {model_id} from {path} ---")

    ckpt = torch.load(path, map_location=dev, weights_only=False)
    state = ckpt.get("model_state_dict") or ckpt
    raw_config = ckpt.get("config", {})

    is_lstm = "obs_len" in raw_config or "social_lstm" in str(path)

    if is_lstm:
        from App.benchmarks.social_lstm.model import SocialLSTM
        cfg = {
            "obs_len": int(raw_config.get("obs_len", 15)),
            "pred_len": int(raw_config.get("pred_len", 10)),
            "hidden_size": int(raw_config.get("hidden_size", 128)),
            "embedding_dim": int(raw_config.get("embedding_dim", 64)),
            "grid_size": int(raw_config.get("grid_size", 4)),
        }
        model = SocialLSTM(**cfg).to(dev)
        if hasattr(model, "lstm") and not any(k.startswith("lstm.") for k in state.keys()):
            model.lstm.load_state_dict(state)
        else:
            model.load_state_dict(state)
        _ACTIVE_TYPE = "lstm"
        _ACTIVE_CFG = cfg
    else:
        from train import model_config as default_model_config
        cfg_obj = copy.deepcopy(default_model_config)


        if isinstance(raw_config, dict):
            key_mapping = {"obs_len": "lookback", "pred_len": "future_steps", "hidden_size": "hidden_dim"}
            for k, v in raw_config.items():
                target_k = key_mapping.get(k, k)
                if hasattr(cfg_obj, target_k):
                    setattr(cfg_obj, target_k, v)

        model = HTPModel(cfg_obj).to(dev)
        model.load_state_dict(state)
        _ACTIVE_TYPE = "transformer"
        _ACTIVE_CFG = cfg_obj

    model.eval()
    _ACTIVE_MODEL = model
    _ACTIVE_MODEL_ID = model_id
    _ACTIVE_DEVICE = dev

    return _ACTIVE_MODEL, _ACTIVE_CFG, _ACTIVE_TYPE, _ACTIVE_DEVICE

def _annotation_from_det(ann: Annotation, frame_id: int, track_id: int) -> Annotation:
    return Annotation(
        frame_id=frame_id, track_id=track_id,
        bbox=BBox(x=ann.bbox.x, y=ann.bbox.y, width=ann.bbox.width, height=ann.bbox.height),
        score=ann.score, class_id=ann.class_id, visibility=ann.visibility,
        x=ann.x, y=ann.y, z=ann.z,
    )

def get_denormalize(sequence: Sequence) -> Callable[[float, float], tuple[float, float]]:
    width = int(sequence.info["Sequence"]["imWidth"])
    height = int(sequence.info["Sequence"]["imHeight"])
    max_dim = max(width, height)
    offset_x = (1 - width / max_dim) / 2
    offset_y = (1 - height / max_dim) / 2

    def denormalize(x: float, y: float) -> tuple[float, float]:
        return (x - offset_x) * max_dim, (y - offset_y) * max_dim
    return denormalize

def build_forecast_for_agent(*, anchor_frame_id: int, anchor_annotation: Annotation, pred_positions: torch.Tensor, agent_index: int, denormalize: Callable[[float, float], tuple[float, float]], trajectory_stride: int = 1) -> Forecast:
    points: list[ForecastPoint] = []
    future_steps = pred_positions.size(2)
    for k in range(future_steps):
        px_n = float(pred_positions[0, agent_index, k, 0])
        py_n = float(pred_positions[0, agent_index, k, 1])
        px, py = denormalize(px_n, py_n)
        points.append(ForecastPoint(frame_id=anchor_frame_id + (k + 1) * trajectory_stride, horizon=k + 1, x=px, y=py))

    return Forecast(anchor_frame_id=anchor_frame_id, x=anchor_annotation.bbox.foot_x, y=anchor_annotation.bbox.foot_y, points=points)

def _predict_forecasts_transformer(sequence: Sequence, model, cfg, device) -> ForecastChannel:
    denormalize = get_denormalize(sequence)
    forecast_tracks: dict[int, ForecastTrack] = {}

    for frame_id in tqdm(sorted(sequence.frames.keys()), desc="Building Forecasts"):
        frame = sequence.frames[frame_id]
        prepared, track_ids = prepare_scene_inference_batch(sequence, frame_id, device=device, config=cfg, source="det")
        if not track_ids:
            continue
        with torch.no_grad():
            pred = model(**prepared)

        for agent_index, track_id in enumerate(track_ids):
            ann = frame.det[track_id]
            if track_id not in forecast_tracks:
                forecast_tracks[track_id] = ForecastTrack(id=track_id, forecasts=[])
            forecast_tracks[track_id].forecasts.append(
                build_forecast_for_agent(
                    anchor_frame_id=frame_id, anchor_annotation=ann, pred_positions=pred["pred_positions"],
                    agent_index=agent_index, denormalize=denormalize, trajectory_stride=cfg.trajectory_stride,
                ),
            )
    return ForecastChannel(name="transformer", tracks=forecast_tracks)

def _predict_tracks_transformer(sequence: Sequence, model, cfg, device) -> dict[int, Track]:
    denormalize = get_denormalize(sequence)
    predicted_tracks: dict[int, Track] = {}

    for frame_id in tqdm(sorted(sequence.frames.keys()), desc="Running Transformer Inference"):
        frame = sequence.frames[frame_id]
        for track_id, ann in frame.det.items():
            if track_id not in predicted_tracks:
                predicted_tracks[track_id] = Track(id=track_id, gt={})

        prepared, track_ids = prepare_scene_inference_batch(sequence, frame_id, device=device, config=cfg, source="det")
        if not track_ids:
            for track_id, ann in frame.det.items():
                predicted_tracks[track_id].gt[frame_id] = _annotation_from_det(ann, frame_id, track_id)
            continue

        with torch.no_grad():
            pred = model(**prepared)

        predicted_track_ids = set(track_ids)
        for track_id, ann in frame.det.items():
            if track_id not in predicted_track_ids:
                predicted_tracks[track_id].gt[frame_id] = _annotation_from_det(ann, frame_id, track_id)
                continue
            agent_index = track_ids.index(track_id)
            px_n = float(pred["pred_positions"][0, agent_index, 0, 0])
            py_n = float(pred["pred_positions"][0, agent_index, 0, 1])
            px, py = denormalize(px_n, py_n)
            target_frame_id = frame_id + cfg.trajectory_stride

            w, h = ann.bbox.width, ann.bbox.height
            predicted_tracks[track_id].gt[target_frame_id] = Annotation(
                frame_id=target_frame_id, track_id=track_id,
                bbox=BBox(x=px - w / 2, y=py - h, width=w, height=h),
                score=ann.score, class_id=ann.class_id, visibility=ann.visibility,
                x=px, y=py, z=ann.z,
            )
    return predicted_tracks

def _predict_tracks_lstm(sequence: Sequence, model, config, device) -> dict[int, Track]:
    obs_len = int(config.get("obs_len", 15))
    pred_len = int(config.get("pred_len", 10))
    subsample_step = int(config.get("subsample_step", 5))

    gt_channel = sequence.channels.get("gt")
    if gt_channel is None:
        return {}

    all_frame_ids = sorted(sequence.frames.keys())
    if len(all_frame_ids) < obs_len:
        return {}

    frame_positions: dict[int, dict[int, tuple[float, float]]] = {}
    for track_id, track in gt_channel.tracks.items():
        for frame_id, ann in track.gt.items():
            if frame_id not in frame_positions:
                frame_positions[frame_id] = {}
            frame_positions[frame_id][track_id] = (ann.bbox.foot_x, ann.bbox.foot_y)

    all_x = [pos[0] for fp in frame_positions.values() for pos in fp.values()]
    all_y = [pos[1] for fp in frame_positions.values() for pos in fp.values()]
    x_min, x_range = min(all_x), max(max(all_x) - min(all_x), 1e-6)
    y_min, y_range = min(all_y), max(max(all_y) - min(all_y), 1e-6)

    all_track_ids = sorted(gt_channel.tracks.keys())
    max_peds = len(all_track_ids)
    tid_to_slot = {tid: i for i, tid in enumerate(all_track_ids)}
    track_predictions: dict[int, dict[int, tuple[float, float]]] = {tid: {} for tid in all_track_ids}
    subsampled = all_frame_ids[::subsample_step]
    window_size = obs_len + pred_len

    for start in tqdm(range(len(subsampled) - window_size + 1), desc="Running LSTM Inference"):
        obs_frames  = subsampled[start : start + obs_len]
        pred_frames = subsampled[start + obs_len : start + window_size]

        pos_arr  = np.zeros((obs_len, max_peds, 2), dtype=np.float32)
        mask_arr = np.zeros((obs_len, max_peds),    dtype=np.float32)

        for t, fid in enumerate(obs_frames):
            fp = frame_positions.get(fid, {})
            for tid, slot in tid_to_slot.items():
                if tid in fp:
                    fx, fy = fp[tid]
                    pos_arr[t, slot, 0] = (fx - x_min) / x_range
                    pos_arr[t, slot, 1] = (fy - y_min) / y_range
                    mask_arr[t, slot]   = 1.0

        obs_pos  = torch.from_numpy(pos_arr).unsqueeze(0).to(device)
        obs_mask = torch.from_numpy(mask_arr).unsqueeze(0).to(device)

        with torch.no_grad():
            pred_disps = model(obs_pos, obs_mask, pred_len=pred_len)

        pred_disps = pred_disps.squeeze(0).cpu().numpy()
        last_norm  = pos_arr[-1]
        pred_norm  = np.cumsum(pred_disps, axis=0) + last_norm[np.newaxis]

        for step, fid in enumerate(pred_frames):
            for tid, slot in tid_to_slot.items():
                if mask_arr[-1, slot] == 0: continue
                if fid in track_predictions[tid]: continue
                px = float(pred_norm[step, slot, 0]) * x_range + x_min
                py = float(pred_norm[step, slot, 1]) * y_range + y_min
                track_predictions[tid][fid] = (px, py)

    predicted_tracks: dict[int, Track] = {}
    for tid, track in gt_channel.tracks.items():
        predicted_gt: dict[int, Annotation] = dict(track.gt)
        last_frame = max(track.gt.keys()) if track.gt else 0
        last_ann   = track.gt.get(last_frame)
        if last_ann is None:
            predicted_tracks[tid] = Track(id=tid, gt=predicted_gt)
            continue

        for fid, (px, py) in track_predictions[tid].items():
            if fid in predicted_gt: continue
            w, h = last_ann.bbox.width, last_ann.bbox.height
            predicted_gt[fid] = Annotation(
                frame_id=fid, track_id=tid, bbox=BBox(x=px - w / 2, y=py - h, width=w, height=h),
                score=last_ann.score, class_id=last_ann.class_id, visibility=last_ann.visibility,
                x=px, y=py, z=last_ann.z,
            )
        predicted_tracks[tid] = Track(id=tid, gt=predicted_gt)
    return predicted_tracks



def get_channel(sequence: Sequence, channel: str) -> Channel:
    if channel in sequence.channels:
        return sequence.channels[channel]

    if channel == "kalman":
        det_tracks = sequence.channels["det"].tracks
        predicted_tracks = predict_tracks(det_tracks)
        sequence.channels[channel] = Channel(name=channel, tracks=predicted_tracks)
        return sequence.channels[channel]

    model, cfg, m_type, device = _ensure_dynamic_model_loaded(channel)
    if model is None:
        raise ValueError(f"Could not find or load model: {channel}")

    if m_type == "lstm":
        predicted_tracks = _predict_tracks_lstm(sequence, model, cfg, device)
    else:
        predicted_tracks = _predict_tracks_transformer(sequence, model, cfg, device)

    sequence.channels[channel] = Channel(name=channel, tracks=predicted_tracks)
    return sequence.channels[channel]

def get_forecast_channel(sequence: Sequence, channel: str) -> ForecastChannel:
    model, cfg, m_type, device = _ensure_dynamic_model_loaded(channel)
    if model is None or m_type == "lstm":

        return ForecastChannel(name=channel, tracks={})

    return _predict_forecasts_transformer(sequence, model, cfg, device)