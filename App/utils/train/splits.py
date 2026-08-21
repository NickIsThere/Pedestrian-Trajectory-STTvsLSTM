from __future__ import annotations

from typing import Iterable

from data.reader import Dataset, Sequence, Split

from utils.train.constants import MOT20_TRAIN_SEQUENCES, MOT20_VAL_SEQUENCES
from utils.train.types import LeaveOneOutFold


def split_has_gt(split: Split) -> bool:
    for sequence in split.sequences.values():
        for frame in sequence.frames.values():
            if frame.gt:
                return True
    return False


def sequence_has_gt(sequence: Sequence) -> bool:
    return any(bool(frame.gt) for frame in sequence.frames.values())


def all_dataset_sequences(dataset: Dataset) -> dict[str, Sequence]:
    sequences: dict[str, Sequence] = {}
    for split in dataset.splits.values():
        for name, sequence in split.sequences.items():
            sequences[name] = sequence
    return sequences


def split_from_sequences(dataset: Dataset, name: str, sequences: dict[str, Sequence]) -> Split:
    return Split(name=name, path=dataset.path / name, sequences=dict(sequences))


def filter_split_by_sequences(split: Split, name: str, sequence_names: Iterable[str]) -> Split:
    names = tuple(sequence_names)
    missing = [sequence_name for sequence_name in names if sequence_name not in split.sequences]
    if missing:
        raise KeyError(
            f"Missing required {name} sequence(s): {missing}. Available: {sorted(split.sequences)}",
        )
    return Split(
        name=name,
        path=split.path,
        sequences={sequence_name: split.sequences[sequence_name] for sequence_name in names},
    )


def build_supervised_splits(dataset: Dataset) -> tuple[Split, Split]:
    if "train" not in dataset.splits:
        raise KeyError(f"No MOT20 train split in dataset; available: {list(dataset.splits)}")
    if "val" not in dataset.splits:
        raise KeyError(f"No MOT20 val split in dataset; available: {list(dataset.splits)}")

    train_source = dataset.splits["train"]
    val_source = dataset.splits["val"]

    train_split = filter_split_by_sequences(train_source, "train", MOT20_TRAIN_SEQUENCES)
    val_split = filter_split_by_sequences(val_source, "validation", MOT20_VAL_SEQUENCES)

    overlap = set(train_split.sequences).intersection(val_split.sequences)
    if overlap:
        raise ValueError(f"Train/validation sequence overlap is not allowed: {sorted(overlap)}")
    if not split_has_gt(train_split):
        raise ValueError("Training split has no ground truth annotations.")
    if not split_has_gt(val_split):
        raise ValueError("validation split has no ground truth annotations.")
    return train_split, val_split


def build_leave_one_out_folds(dataset: Dataset) -> list[LeaveOneOutFold]:
    supervised = {
        name: sequence
        for name, sequence in sorted(all_dataset_sequences(dataset).items())
        if sequence_has_gt(sequence)
    }
    if len(supervised) < 2:
        raise ValueError("leave-one-out training requires at least two supervised sequences")

    folds: list[LeaveOneOutFold] = []
    for index, val_sequence in enumerate(supervised):
        train_sequences = {
            name: sequence
            for name, sequence in supervised.items()
            if name != val_sequence
        }
        folds.append(
            LeaveOneOutFold(
                index=index,
                val_sequence=val_sequence,
                train_split=split_from_sequences(dataset, "train", train_sequences),
                val_split=split_from_sequences(dataset, "val", {val_sequence: supervised[val_sequence]}),
            ),
        )
    return folds
