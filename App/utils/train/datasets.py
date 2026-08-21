from __future__ import annotations

import random
from collections.abc import Iterator

import numpy as np

from data.reader import Split
from model.config import ModelConfig

from utils.train.constants import SampleMode
from utils.train.samples import (
    _sequence_dataset_fps,
    build_supervised_sample,
    build_supervised_scene_sample,
)
from utils.train.types import SceneTrainingSample, TrainingSample


class HTPIterableDataset:
    """Infinite stream of supervised (features, mask, answer) tuples for one split."""

    def __init__(self, split: Split, config: ModelConfig, *, input_mode: SampleMode = "gt"):
        self.split = split
        self.config = config
        self.input_mode = input_mode

        self.sequence_weights: dict[str, int] = {}
        for sequence in self.split.sequences.values():
            for frame in sequence.frames.values():
                if sequence.name not in self.sequence_weights:
                    self.sequence_weights[sequence.name] = 0
                source_annotations = frame.gt if input_mode == "gt" else frame.det
                if frame.gt:
                    self.sequence_weights[sequence.name] += len(source_annotations)

    def __iter__(self) -> Iterator[TrainingSample]:
        while True:
            names = list(self.sequence_weights.keys())
            weights = list(self.sequence_weights.values())
            if not names:
                raise ValueError("Split has no sequences.")
            if sum(weights) <= 0:
                chosen_sequence = random.choice(names)
            else:
                chosen_sequence = random.choices(names, weights=weights, k=1)[0]
            sequence = self.split.sequences.get(chosen_sequence)
            if sequence is None:
                continue

            frame_ids = sorted(sequence.frames)
            eligible_frame_ids = frame_ids[:: self.config.window_start_stride]
            chosen_frame = random.choice(eligible_frame_ids)
            frame = sequence.frames.get(chosen_frame)
            if frame is None:
                continue

            source_annotations = frame.gt if self.input_mode == "gt" else frame.det
            if not source_annotations or not frame.gt:
                continue

            chosen_track = random.choice(list(source_annotations.keys()))
            sample = build_supervised_sample(
                sequence,
                chosen_frame,
                chosen_track,
                self.config,
                input_mode=self.input_mode,
            )
            if sample is None:
                continue

            features, mask, answer = sample
            yield TrainingSample(
                features=features,
                mask=mask,
                answer=answer,
                metadata={
                    "sequence": sequence.name,
                    "frame_id": chosen_frame,
                    "track_id": chosen_track,
                    "input_mode": self.input_mode,
                    "dataset_fps": _sequence_dataset_fps(sequence),
                },
            )


class HTPSceneIterableDataset:
    """Infinite stream of supervised frame-level scene samples for one split."""

    def __init__(self, split: Split, config: ModelConfig, *, input_mode: SampleMode = "gt"):
        self.split = split
        self.config = config
        self.input_mode = input_mode
        self.sequence_weights: dict[str, int] = {}
        for sequence in self.split.sequences.values():
            count = 0
            for frame in sequence.frames.values():
                source_annotations = frame.gt if input_mode == "gt" else frame.det
                if source_annotations and frame.gt:
                    count += 1
            self.sequence_weights[sequence.name] = count

    def __iter__(self) -> Iterator[SceneTrainingSample]:
        while True:
            names = list(self.sequence_weights.keys())
            weights = list(self.sequence_weights.values())
            if not names:
                raise ValueError("Split has no sequences.")
            chosen_sequence = random.choices(names, weights=weights, k=1)[0] if sum(weights) > 0 else random.choice(
                names,
            )
            sequence = self.split.sequences.get(chosen_sequence)
            if sequence is None:
                continue
            frame_ids = sorted(sequence.frames)
            eligible_frame_ids = frame_ids[:: self.config.window_start_stride]
            chosen_frame = random.choice(eligible_frame_ids)
            sample = build_supervised_scene_sample(
                sequence,
                chosen_frame,
                self.config,
                input_mode=self.input_mode,
            )
            if sample is None:
                continue
            yield sample


def sample_to_tuple(sample: TrainingSample | tuple[np.ndarray, np.ndarray, np.ndarray]) -> tuple[
    np.ndarray, np.ndarray, np.ndarray,
]:
    if isinstance(sample, TrainingSample):
        return sample.features, sample.mask, sample.answer
    return sample


def samples_to_tuples(
    samples: list[TrainingSample | tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    return [sample_to_tuple(sample) for sample in samples]


def iter_split_samples(
    split: Split,
    config: ModelConfig,
    *,
    input_mode: SampleMode = "gt",
) -> Iterator[TrainingSample]:
    for sequence in split.sequences.values():
        for frame_id in sorted(sequence.frames)[:: config.window_start_stride]:
            frame = sequence.frames[frame_id]
            source_annotations = frame.gt if input_mode == "gt" else frame.det
            if not source_annotations or not frame.gt:
                continue
            for track_id in sorted(source_annotations):
                sample = build_supervised_sample(
                    sequence,
                    frame_id,
                    track_id,
                    config,
                    input_mode=input_mode,
                )
                if sample is None:
                    continue
                features, mask, answer = sample
                yield TrainingSample(
                    features=features,
                    mask=mask,
                    answer=answer,
                    metadata={
                        "sequence": sequence.name,
                        "frame_id": frame_id,
                        "track_id": track_id,
                        "input_mode": input_mode,
                        "dataset_fps": _sequence_dataset_fps(sequence),
                    },
                )


def iter_split_scene_samples(
    split: Split,
    config: ModelConfig,
    *,
    input_mode: SampleMode = "gt",
) -> Iterator[SceneTrainingSample]:
    for sequence in split.sequences.values():
        for frame_id in sorted(sequence.frames)[:: config.window_start_stride]:
            sample = build_supervised_scene_sample(
                sequence,
                frame_id,
                config,
                input_mode=input_mode,
            )
            if sample is None:
                continue
            yield sample
