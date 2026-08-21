import pandas as pd
from dataclasses import dataclass
from pathlib import Path
from configparser import ConfigParser
from tqdm import tqdm
from functools import lru_cache

from data.loader import load, DATA_ROOT
from utils.utils import cached


@dataclass
class BBox:
    x: float
    y: float
    width: float
    height: float

    @property
    def foot_x(self) -> float:
        return self.x + self.width / 2

    @property
    def foot_y(self) -> float:
        return self.y + self.height

    def iou(self, other: "BBox") -> float:
        x1 = max(self.x, other.x)
        y1 = max(self.y, other.y)
        x2 = min(self.x + self.width, other.x + other.width)
        y2 = min(self.y + self.height, other.y + other.height)
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        union = self.width * self.height + other.width * other.height - intersection
        return intersection / union


@dataclass
class Annotation:
    frame_id: int
    track_id: int
    bbox: BBox
    score: float
    class_id: int
    visibility: float
    x: float
    y: float
    z: float


@dataclass
class Track:
    id: int
    gt: dict[int, Annotation]


@dataclass
class ForecastPoint:
    frame_id: int
    horizon: int
    x: float
    y: float


@dataclass
class Forecast:
    anchor_frame_id: int
    x: float
    y: float
    points: list[ForecastPoint]


@dataclass
class ForecastTrack:
    id: int
    forecasts: list[Forecast]


@dataclass
class ForecastChannel:
    name: str
    tracks: dict[int, ForecastTrack]


@dataclass
class Channel:
    name: str
    tracks: dict[int, Track]


@dataclass
class Frame:
    id: int
    path: Path
    det: dict[int, Annotation]
    gt: dict[int, Annotation]


@dataclass
class Sequence:
    name: str
    path: Path
    info: ConfigParser
    channels: dict[str, Channel]
    frames: dict[int, Frame]


@dataclass
class Split:
    name: str
    path: Path
    sequences: dict[str, Sequence]


@dataclass
class Dataset:
    splits: dict[str, Split]
    name: str
    path: Path


def get_value(df: pd.DataFrame, column: str) -> str:
    return df[column].unique()[0]

def get_path(df: pd.DataFrame, level: int) -> Path:
    return Path(get_value(df, "path")).parents[level]

#region Loaders
def load_annotation(annotation: pd.Series) -> Annotation:
    """
    Main writer: Noah Nuelandt
    Reviewer: 
    Contributors: Neo Deward
    """
    return Annotation(
        track_id=annotation["track"],
        frame_id=annotation["frame"],
        bbox=BBox(
            x=annotation["bb_left"],
            y=annotation["bb_top"],
            width=annotation["bb_width"],
            height=annotation["bb_height"],
        ),
        score=annotation["conf"],
        class_id=annotation["class"] if "class" in annotation else None,
        visibility=annotation["visibility"] if "visibility" in annotation else None,
        x=annotation["x"] if "x" in annotation else None,
        y=annotation["y"] if "y" in annotation else None,
        z=annotation["z"] if "z" in annotation else None,
    )

def load_frame(frame: int, gt_df: pd.DataFrame, det_df: pd.DataFrame) -> Frame:
    """
    Main writer: Noah Nuelandt
    Reviewer: 
    Contributors: Neo Deward
    """
    det_df = det_df[det_df["frame"] == frame]
    gt_df = gt_df[gt_df["frame"] == frame]

    frame = Frame(id=get_value(det_df, "frame"), path=Path(get_value(det_df, "path")), det={}, gt={})
    for i, annotation in det_df.iterrows():
        frame.det[i] = load_annotation(annotation)
    for _, annotation in gt_df.iterrows():
        frame.gt[annotation["track"]] = load_annotation(annotation)
    return frame

def load_sequence(sequence, gt_df: pd.DataFrame, det_df: pd.DataFrame) -> Sequence:
    """
    Main writer: Noah Nuelandt
    Reviewer: 
    Contributors: Neo Deward
    """
    gt_df = gt_df[gt_df["sequence"] == sequence]
    det_df = det_df[det_df["sequence"] == sequence]
    sequence = Sequence(
        name=sequence,
        path=get_path(det_df, 1),
        info=ConfigParser(),
        channels={"gt": Channel(name="gt", tracks={})},
        frames={},
    )
    sequence.info.read(sequence.path / "seqinfo.ini")

    frames = det_df["frame"].unique()
    for frame in tqdm(frames, desc=f"{sequence.name} frames", unit="frame", leave=False):
        sequence.frames[frame] = load_frame(frame, gt_df, det_df)

    for frame in sequence.frames.values():
        for track_id, track in frame.gt.items():
            if track_id not in sequence.channels["gt"].tracks:
                sequence.channels["gt"].tracks[track_id] = Track(id=track_id, gt={})
            sequence.channels["gt"].tracks[track_id].gt[frame.id] = track

    return sequence

def load_split(split: str, gt_df: pd.DataFrame, det_df: pd.DataFrame) -> Split:
    """
    Main writer: Noah Nuelandt
    Reviewer: 
    Contributors: Neo Deward
    """
    gt_df = gt_df[gt_df["split"] == split]
    det_df = det_df[det_df["split"] == split]
    split = Split(
        name=split, path=get_path(det_df, 2), sequences={}
    )

    for sequence in det_df["sequence"].unique():
        split.sequences[sequence] = load_sequence(sequence, gt_df,det_df)

    return split

def load_dataset(gt_df: pd.DataFrame, det_df: pd.DataFrame) -> Dataset:
    """
    Main writer: Noah Nuelandt
    Reviewer: 
    Contributors: Neo Deward
    """
    dataset = Dataset(
        splits={}, name=get_value(det_df, "dataset"), path=get_path(det_df, 3)
    )
    print(f"Loading dataset {dataset.name} from {dataset.path}")

    for split in det_df["split"].unique():
        dataset.splits[split] = load_split(split, gt_df, det_df)

    return dataset
#endregion

def generate_det_tracks(sequence: Sequence, iou_threshold: float = 0.3) -> Sequence:
    """
    Main writer: Noah Nuelandt
    Reviewer: 
    Contributors: Neo Deward
    """
    prev_frame = None
    next_track_id = 1
    for frame in sequence.frames.values():
        new_det: dict[int, Annotation] = {}
        used_prev: set[int] = set()
        detections = sorted(frame.det.values(), key=lambda a: a.score, reverse=True)

        for annotation in detections:
            best_iou = iou_threshold
            best_track_id: int | None = None

            if prev_frame is not None:
                for tid, prev_annotation in prev_frame.det.items():
                    if tid < 0 or tid in used_prev:
                        continue
                    overlap = annotation.bbox.iou(prev_annotation.bbox)
                    if overlap > best_iou:
                        best_iou = overlap
                        best_track_id = tid

            if best_track_id is None:
                chosen_track_id = next_track_id
                next_track_id += 1
            else:
                chosen_track_id = best_track_id
                used_prev.add(chosen_track_id)

            annotation.track_id = chosen_track_id
            new_det[chosen_track_id] = annotation

        frame.det = new_det
        prev_frame = frame

    return sequence

def ensure_det_channel(sequence: Sequence) -> Sequence:
    """
    Main writer: Noah Nuelandt
    Reviewer: 
    Contributors:
    """
    if "det" not in sequence.channels:
        sequence.channels["det"] = Channel(name="det", tracks={})
        for frame in sequence.frames.values():
            for track_id, track in frame.det.items():
                if track_id not in sequence.channels["det"].tracks:
                    sequence.channels["det"].tracks[track_id] = Track(id=track_id, gt={})
                sequence.channels["det"].tracks[track_id].gt[frame.id] = track
    return sequence

DATASET_CACHE_VERSION = 5
@lru_cache(maxsize=1)
def get_dataset(name: str) -> Dataset:
    """
    Main writer: Noah Nuelandt
    Reviewer: 
    Contributors:
    """
    def fn():
        from evaluation.eval import get_channel

        ds = load_dataset(*load(name))
        for split in ds.splits.values():
            for sequence in tqdm(split.sequences.values(), desc="Preloading channels", leave=False):
                generate_det_tracks(sequence)
                ensure_det_channel(sequence)
                get_channel(sequence, "kalman")
                #get_channel(sequence, "transformer")
                #get_channel(sequence, "social_lstm")
        return ds
    return cached(DATA_ROOT / f"{name}.pkl", fn, DATASET_CACHE_VERSION)

def get_sequence(sequence_name: str) -> Sequence:
    """
    Main writer: Noah Nuelandt
    Reviewer: 
    Contributors:
    """
    dataset = sequence_name.split("-")[0]
    ds = get_dataset(dataset)
    for split in ds.splits.values():
        for sequence in split.sequences.values():
            if sequence.name == sequence_name:
                return sequence
    raise ValueError(f"Sequence {sequence_name} not found in dataset {dataset}")
