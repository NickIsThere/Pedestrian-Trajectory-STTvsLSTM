from __future__ import annotations

from typing import Literal

from data.reader import Annotation, Frame, Sequence
import numpy as np

MODEL_VERSION = "past_window_stride_v2"
FEATURE_VERSION = MODEL_VERSION


def observed_frame_id(anchor_frame_id: int, sample_index: int, lookback: int, trajectory_stride: int) -> int:
    """
    Main writer: Nick Grebe
    Reviewer: 
    Contributors:
    """
    return anchor_frame_id - (lookback - 1 - sample_index) * trajectory_stride


def future_frame_id(anchor_frame_id: int, future_index: int, trajectory_stride: int) -> int:
    """
    Main writer: Nick Grebe
    Reviewer: 
    Contributors:
    """
    return anchor_frame_id + (future_index + 1) * trajectory_stride


def padded_lookup(sequence: Sequence, frame_id: int, offset: int) -> Frame:
    """
    Main writer: Noah Nuelandt
    Reviewer: 
    Contributors:
    """
    if frame_id + offset < 1:
        return sequence.frames[1]
    elif frame_id + offset > len(sequence.frames):
        return sequence.frames[len(sequence.frames)]
    else:
        return sequence.frames[frame_id + offset]


def _ann(frame: Frame, source: Literal["gt", "det"]) -> dict[int, Annotation]:
    """
    Main writer: Noah Nuelandt
    Reviewer: 
    Contributors:
    """
    return frame.det if source == "det" else frame.gt


def extract_features(
    sequence: Sequence,
    frame_id: int,
    track_id: int,
    lookback: int,
    future_steps: int = 10,
    *,
    source: Literal["gt", "det"] = "gt",
    answer_source: Literal["gt", "det"] | None = None,
    answer_track_id: int | None = None,
    compute_answer: bool = False,
    trajectory_stride: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """
    Main writer: Noah Nuelandt
    Reviewer: 
    Contributors: Nick Grebe
    """
    if trajectory_stride <= 0:
        raise ValueError("trajectory_stride must be positive")

    features = np.zeros((lookback, 9))
    mask = np.zeros(lookback)
    frame_rate = float(sequence.info["Sequence"]["frameRate"])
    sampled_dt_seconds = trajectory_stride / frame_rate
    width = int(sequence.info["Sequence"]["imWidth"])
    height = int(sequence.info["Sequence"]["imHeight"])
    max_dim = max(width, height)
    offset_x = (1 - width / max_dim) / 2
    offset_y = (1 - height / max_dim) / 2

    for i in range(lookback):
        current_frame_id = observed_frame_id(frame_id, i, lookback, trajectory_stride)
        previous_frame_id = current_frame_id - trajectory_stride
        frame = sequence.frames.get(current_frame_id)
        if frame is None:
            continue
        prev_frame = sequence.frames.get(previous_frame_id)
        cur = _ann(frame, source)
        if track_id not in cur:
            continue

        pos_x = cur[track_id].bbox.foot_x / max_dim + offset_x
        pos_y = cur[track_id].bbox.foot_y / max_dim + offset_y
        prev = _ann(prev_frame, source) if prev_frame is not None else {}
        if prev_frame is not None and track_id in prev:
            prev_x = prev[track_id].bbox.foot_x / max_dim + offset_x
            prev_y = prev[track_id].bbox.foot_y / max_dim + offset_y
            dx = pos_x - prev_x
            dy = pos_y - prev_y
        else:
            dx = 0.0
            dy = 0.0
        r = float(np.hypot(dx, dy))

        features[i, 0] = pos_x
        features[i, 1] = pos_y
        features[i, 2] = dx
        features[i, 3] = dy
        features[i, 4] = dx / sampled_dt_seconds
        features[i, 5] = dy / sampled_dt_seconds
        features[i, 6] = r / sampled_dt_seconds
        if r > 0.0:
            features[i, 7] = dy / r
            features[i, 8] = dx / r
        else:
            features[i, 7] = 0.0
            features[i, 8] = 0.0

        mask[i] = 1

    if not compute_answer:
        return features, mask, None

    target_source = answer_source or source
    target_track_id = track_id if answer_track_id is None else answer_track_id
    answer = np.zeros((future_steps, 2), dtype=np.float64)
    for k in range(future_steps):
        fut = sequence.frames.get(future_frame_id(frame_id, k, trajectory_stride))
        if fut is None:
            return features, mask, None
        fut_ann = _ann(fut, target_source)
        if target_track_id not in fut_ann:
            return features, mask, None
        bbox = fut_ann[target_track_id].bbox
        answer[k, 0] = bbox.foot_x / max_dim + offset_x
        answer[k, 1] = bbox.foot_y / max_dim + offset_y

    return features, mask, answer
