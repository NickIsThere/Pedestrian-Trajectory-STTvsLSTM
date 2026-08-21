from __future__ import annotations

import numpy as np

from data.feature import extract_features
from data.reader import Annotation, Frame, Sequence
from model.config import ModelConfig

from utils.train.constants import SampleMode
from utils.train.types import SceneTrainingSample


def match_gt_track_for_detection(
    frame: Frame,
    detection: Annotation,
    *,
    iou_threshold: float = 0.3,
) -> int | None:
    best_iou = iou_threshold
    best_track_id: int | None = None
    for track_id, gt_ann in frame.gt.items():
        overlap = detection.bbox.iou(gt_ann.bbox)
        if overlap > best_iou:
            best_iou = overlap
            best_track_id = track_id
    return best_track_id


def build_supervised_sample(
    sequence: Sequence,
    frame_id: int,
    track_id: int,
    config: ModelConfig,
    *,
    input_mode: SampleMode,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    frame = sequence.frames.get(frame_id)
    if frame is None:
        return None

    if input_mode == "gt":
        if track_id not in frame.gt:
            return None
        features, mask, answer = extract_features(
            sequence,
            frame_id,
            track_id,
            config.lookback,
            config.future_steps,
            source="gt",
            trajectory_stride=config.trajectory_stride,
        )
    else:
        det_ann = frame.det.get(track_id)
        if det_ann is None:
            return None
        gt_track_id = match_gt_track_for_detection(frame, det_ann)
        if gt_track_id is None:
            return None
        features, mask, answer = extract_features(
            sequence,
            frame_id,
            track_id,
            config.lookback,
            config.future_steps,
            source="det",
            answer_source="gt",
            answer_track_id=gt_track_id,
            trajectory_stride=config.trajectory_stride,
        )

    if answer is None or not np.any(mask):
        return None
    return features, mask, answer


def _sequence_dataset_fps(sequence: Sequence) -> float:
    try:
        value = sequence.info["Sequence"].get("frameRate")
        return float(value) if value is not None else float("nan")
    except Exception:
        return float("nan")


def _future_answer_for_track(
    sequence: Sequence,
    frame_id: int,
    track_id: int,
    config: ModelConfig,
    *,
    source: SampleMode,
) -> np.ndarray | None:
    width = int(sequence.info["Sequence"]["imWidth"])
    height = int(sequence.info["Sequence"]["imHeight"])
    max_dim = max(width, height)
    offset_x = (1 - width / max_dim) / 2
    offset_y = (1 - height / max_dim) / 2
    answer = np.zeros((config.future_steps, 2), dtype=np.float64)

    for step in range(config.future_steps):
        future_frame = sequence.frames.get(frame_id + (step + 1) * config.trajectory_stride)
        if future_frame is None:
            return None
        annotations = future_frame.gt if source == "gt" else future_frame.det
        ann = annotations.get(track_id)
        if ann is None:
            return None
        answer[step, 0] = ann.bbox.foot_x / max_dim + offset_x
        answer[step, 1] = ann.bbox.foot_y / max_dim + offset_y
    return answer


def build_supervised_scene_sample(
    sequence: Sequence,
    frame_id: int,
    config: ModelConfig,
    *,
    input_mode: SampleMode,
) -> SceneTrainingSample | None:
    frame = sequence.frames.get(frame_id)
    if frame is None:
        return None

    source_annotations = frame.gt if input_mode == "gt" else frame.det
    if not source_annotations:
        return None

    features_by_agent: list[np.ndarray] = []
    masks_by_agent: list[np.ndarray] = []
    answers_by_agent: list[np.ndarray] = []
    future_masks_by_agent: list[np.ndarray] = []
    track_ids: list[int] = []

    for track_id in sorted(source_annotations):
        features, mask, _ = extract_features(
            sequence,
            frame_id,
            track_id,
            config.lookback,
            config.future_steps,
            source=input_mode,
            compute_answer=False,
            trajectory_stride=config.trajectory_stride,
        )
        if not np.any(mask):
            continue

        target_track_id = track_id
        target_source: SampleMode = input_mode
        if input_mode == "det":
            gt_track_id = match_gt_track_for_detection(frame, source_annotations[track_id])
            if gt_track_id is None:
                answer = None
            else:
                target_track_id = gt_track_id
                target_source = "gt"
                answer = _future_answer_for_track(
                    sequence,
                    frame_id,
                    target_track_id,
                    config,
                    source=target_source,
                )
        else:
            answer = _future_answer_for_track(
                sequence,
                frame_id,
                target_track_id,
                config,
                source=target_source,
            )

        if answer is None:
            answer = np.zeros((config.future_steps, 2), dtype=np.float64)
            future_mask = np.zeros(config.future_steps, dtype=bool)
        else:
            future_mask = np.ones(config.future_steps, dtype=bool)

        features_by_agent.append(features)
        masks_by_agent.append(mask)
        answers_by_agent.append(answer)
        future_masks_by_agent.append(future_mask)
        track_ids.append(track_id)

    if not features_by_agent:
        return None

    future_mask_arr = np.stack(future_masks_by_agent, axis=0)
    if not future_mask_arr.any():
        return None

    return SceneTrainingSample(
        features=np.stack(features_by_agent, axis=0),
        mask=np.stack(masks_by_agent, axis=0),
        answers=np.stack(answers_by_agent, axis=0),
        future_mask=future_mask_arr,
        track_ids=track_ids,
        metadata={
            "sequence": sequence.name,
            "frame_id": frame_id,
            "input_mode": input_mode,
            "track_ids": track_ids,
            "dataset_fps": _sequence_dataset_fps(sequence),
        },
    )
