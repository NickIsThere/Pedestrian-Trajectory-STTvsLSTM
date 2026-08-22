from configparser import ConfigParser
import requests
from dataclasses import dataclass
from tqdm import tqdm
import tarfile
from pathlib import Path
from typing import Any, Dict, List
import pandas as pd


def _parse_seqinfo(seqinfo_path: Path) -> Dict[str, Any]:
    """Parse a seqinfo.ini file and return a dict mapping frame number to annotations."""
    parser = ConfigParser()
    parser.read(seqinfo_path, encoding="utf-8")
    if "Sequence" not in parser:
        return {}

    seq = parser["Sequence"]
    return {
        "name":seq.get("name", seqinfo_path.parent.name),
        "imDir":seq.get("imDir","img1"),
        "frameRate":seq.getint("frameRate",fallback=0),
        "seqLength":seq.getint("seqLength",fallback=0),
        "imWidth":seq.getint("imWidth",fallback=0),
        "imHeight":seq.getint("imHeight",fallback=0),
        "imExt":seq.get("imExt", ".jpg"),
    }

def _parse_mot_annotations(file_path: Path, source: str) -> Dict[int, List[Dict[str, Any]]]:
    """Parse MOT annotation file (det.txt or gt.txt) and return a dict mapping frame number to list of annotations."""
    by_frame: Dict[int, List[Dict[str, Any]]] = {}
    if not file_path.exists():
        return by_frame

    with file_path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            line=line.split(",")

            frame=int(float(line[0]))
            track_id=int(float(line[1]))
            x=float(line[2])
            y=float(line[3])
            w=float(line[4])
            h=float(line[5])
            score=float(line[6])

            annotation={
                'source':source,
                'track_id':track_id,
                'bbox':[x,y,w,h],
                'score_or_marked':score
            }
            if len(line) > 7:
                annotation["class_id"] = int(float(line[7]))
            if len(line) > 8:
                annotation["visibility"] = float(line[8])

            if frame not in by_frame:
                by_frame[frame] = []
            by_frame[frame].append(annotation)
    return by_frame


def discover_sequences(mot_source_dir: str | Path) -> List[Path]:
    """
    find sequences in the MOT source directory. Looks for "seqinfo.ini" files under MOT20 and MOTSynth datasets.
    """
    root = Path(mot_source_dir)
    sequences: List[Path] = []

    for dataset_name in ("MOT20", "MOTSynth"):
        dataset_root = root / dataset_name
        if not dataset_root.exists():
            continue
        for seqinfo in dataset_root.rglob("seqinfo.ini"):
            sequences.append(seqinfo.parent)

    return sorted(sequences)


def build_frame_samples(mot_source_dir: str | Path, include_empty_frames: bool = True) -> List[Dict[str, Any]]:
    """
    Build a list of frame samples from the MOT dataset.
    """
    root = Path(mot_source_dir)
    samples: List[Dict[str, Any]] = []

    for seq_dir in discover_sequences(root):
        info = _parse_seqinfo(seq_dir / "seqinfo.ini")
        if not info:
            continue

        image_dir = seq_dir / info["imDir"]

        # Get Dict from det and gt files
        det_by_frame= _parse_mot_annotations(seq_dir / "det" / "det.txt", source="det")
        gt_by_frame= _parse_mot_annotations(seq_dir / "gt" / "gt.txt", source="gt")

        seq_len = int(info["seqLength"])
        if seq_len <= 0:
            frame_ids = sorted(set(det_by_frame) | set(gt_by_frame))
        else:
            frame_ids = list(range(1, seq_len + 1))

        #get image path
        for frame_id in frame_ids:
            image_path = image_dir / f"{frame_id:06d}{info['imExt']}"

            det_items = det_by_frame.get(frame_id, [])
            gt_items = gt_by_frame.get(frame_id, [])

            # safe check to see if img has data
            if not include_empty_frames and not det_items and not gt_items:
                continue

            # Determine dataset name and split
            if seq_dir.parent.name in {"train", "test"}:
                split_name = seq_dir.parent.name
                ds_name = seq_dir.parent.parent.name
            else:
                split_name = "unknown"
                ds_name = seq_dir.parent.name

            samples.append(
                {
                    "dataset": ds_name,
                    "split": split_name,
                    "sequence": info["name"],
                    "frame_id": frame_id,
                    "image_path": str(image_path),
                    "image_size": [info["imWidth"], info["imHeight"]],
                    "frame_rate": info["frameRate"],
                    "det": det_items,
                    "gt": gt_items,
                }
            )

    return samples


## stuff to load MOT into the dataframes


def build_frame_table(
    mot_source_dir: str | Path,
    include_empty_frames: bool = True,
) -> pd.DataFrame:
    """
    Build a frame-level table:
    one row per frame.

    Nick Grebe - i6377605
    """
    rows = build_frame_samples(mot_source_dir, include_empty_frames=include_empty_frames)

    frame_rows: List[Dict[str, Any]] = []
    for row in rows:
        image_width, image_height = row["image_size"]

        frame_rows.append(
            {
                "dataset": row["dataset"],
                "split": row["split"],
                "sequence": row["sequence"],
                "frame_id": row["frame_id"],
                "image_path": row["image_path"],
                "image_width": image_width,
                "image_height": image_height,
                "frame_rate": row["frame_rate"],
                "det_count": len(row["det"]),
                "gt_count": len(row["gt"]),
            }
        )

    frame_df = pd.DataFrame(frame_rows)

    if not frame_df.empty:
        frame_df = frame_df.sort_values(
            by=["dataset", "split", "sequence", "frame_id"]
        ).reset_index(drop=True)

    return frame_df

def load_sequence_samples(mot_source_dir: str | Path,
    dataset: str,
    split: str,
    sequence: str,
    include_empty_frames: bool = True,
) -> List[Dict[str, Any]]:

    root = Path(mot_source_dir)
    seq_dir = root / dataset / split / sequence
    seqinfo_path = seq_dir / "seqinfo.ini"

    if not seqinfo_path.exists():
        raise FileNotFoundError(f"not found: {seqinfo_path}")

    info = _parse_seqinfo(seqinfo_path)
    if not info:
        return []

    image_dir = seq_dir / info["imDir"]
    det_by_frame = _parse_mot_annotations(seq_dir / "det" / "det.txt", source="det")
    gt_by_frame = _parse_mot_annotations(seq_dir / "gt" / "gt.txt", source="gt")

    seq_len = int(info["seqLength"])
    if seq_len <= 0:
        frame_ids = sorted(set(det_by_frame) | set(gt_by_frame))
    else:
        frame_ids = list(range(1, seq_len + 1))

    samples: List[Dict[str, Any]] = []
    for frame_id in frame_ids:
        det_items = det_by_frame.get(frame_id, [])
        gt_items = gt_by_frame.get(frame_id, [])

        if not include_empty_frames and not det_items and not gt_items:
            continue

        image_path = image_dir / f"{frame_id:06d}{info['imExt']}"
        samples.append(
            {
                "dataset": dataset,
                "sequence": info["name"],
                "frame_id": frame_id,
                "image_path": str(image_path),
                "image_size": [info["imWidth"], info["imHeight"]],
                "frame_rate": info["frameRate"],
                "det": det_items,
                "gt": gt_items,
            }
        )

    return samples

def build_annotation_table(
    mot_source_dir: str | Path,
    include_empty_frames: bool = True,
    include_gt: bool = True,
    include_det: bool = True,
) -> pd.DataFrame:
    """
    Build an annotation-level table:
    one row per annotation.

    Includes GT and/or DET annotations depending on flags.

    Nick Grebe - i6377506
    """
    rows = build_frame_samples(mot_source_dir, include_empty_frames=include_empty_frames)

    ann_rows: List[Dict[str, Any]] = []

    for row in rows:
        image_width, image_height = row["image_size"]

        annotation_groups: List[List[Dict[str, Any]]] = []
        if include_gt:
            annotation_groups.append(row["gt"])
        if include_det:
            annotation_groups.append(row["det"])

        for annotations in annotation_groups:
            for ann in annotations:
                x, y, w, h = ann["bbox"]

                ann_rows.append(
                    {
                        "dataset": row["dataset"],
                        "split": row["split"],
                        "sequence": row["sequence"],
                        "frame_id": row["frame_id"],
                        "image_path": row["image_path"],
                        "image_width": image_width,
                        "image_height": image_height,
                        "frame_rate": row["frame_rate"],
                        "source": ann.get("source"),
                        "track_id": ann.get("track_id"),
                        "x": x,
                        "y": y,
                        "w": w,
                        "h": h,
                        "x2": x + w,
                        "y2": y + h,
                        "cx": x + w / 2.0,
                        "cy": y + h / 2.0,
                        "foot_x": x + w / 2.0,
                        "foot_y": y + h,
                        "bbox_area": w * h,
                        "score_or_marked": ann.get("score_or_marked"),
                        "class_id": ann.get("class_id"),
                        "visibility": ann.get("visibility"),
                    }
                )

    ann_df = pd.DataFrame(ann_rows)

    if not ann_df.empty:
        ann_df = ann_df.sort_values(
            by=["dataset", "split", "sequence", "frame_id", "source", "track_id"]
        ).reset_index(drop=True)

    return ann_df


def build_mot_tables(
    mot_source_dir: str | Path,
    include_empty_frames: bool = True,
    include_gt: bool = True,
    include_det: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convenience function:
    returns (frame_df, ann_df)

    nick grebe - i6377605
    """
    frame_df = build_frame_table(
        mot_source_dir=mot_source_dir,
        include_empty_frames=include_empty_frames,
    )

    ann_df = build_annotation_table(
        mot_source_dir=mot_source_dir,
        include_empty_frames=include_empty_frames,
        include_gt=include_gt,
        include_det=include_det,
    )

    return frame_df, ann_df


def split_mot_tables(
    frame_df: pd.DataFrame,
    ann_df: pd.DataFrame,
) -> Dict[str, pd.DataFrame]:
    """
    Split the already-built tables into explicit MOT20 train / val / test sets.

    Split definition:
    - Train: MOT20-01, MOT20-02, MOT20-03
    - Val:   MOT20-05
    - Test:  MOT20-04, MOT20-06, MOT20-07, MOT20-08

    All rows belonging to a full sequence stay together.

    nick grebe - i6377605

    """
    required_cols = {"dataset", "split", "sequence"}
    if not required_cols.issubset(frame_df.columns):
        raise ValueError("frame_df must contain dataset, split, and sequence columns.")
    if not required_cols.issubset(ann_df.columns):
        raise ValueError("ann_df must contain dataset, split, and sequence columns.")

    train_sequences = {"MOT20-01", "MOT20-02", "MOT20-03"}
    val_sequences = {"MOT20-05"}
    test_sequences = {"MOT20-04", "MOT20-06", "MOT20-07", "MOT20-08"}

    mot20_frame = frame_df[frame_df["dataset"] == "MOT20"].copy()
    mot20_ann = ann_df[ann_df["dataset"] == "MOT20"].copy()

    train_frame = mot20_frame[mot20_frame["sequence"].isin(train_sequences)].copy()
    val_frame = mot20_frame[mot20_frame["sequence"].isin(val_sequences)].copy()
    test_frame = mot20_frame[mot20_frame["sequence"].isin(test_sequences)].copy()

    train_ann = mot20_ann[mot20_ann["sequence"].isin(train_sequences)].copy()
    val_ann = mot20_ann[mot20_ann["sequence"].isin(val_sequences)].copy()
    test_ann = mot20_ann[mot20_ann["sequence"].isin(test_sequences)].copy()

    return {
        "train_frame": train_frame.reset_index(drop=True),
        "val_frame": val_frame.reset_index(drop=True),
        "test_frame": test_frame.reset_index(drop=True),
        "train_ann": train_ann.reset_index(drop=True),
        "val_ann": val_ann.reset_index(drop=True),
        "test_ann": test_ann.reset_index(drop=True),
    }


def save_mot_tables(
    mot_source_dir: str | Path,
    output_dir: str | Path,
    include_empty_frames: bool = True,
    include_gt: bool = True,
    include_det: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build and save both tables as CSV files.
    nick grebe - i6377605
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_df, ann_df = build_mot_tables(
        mot_source_dir=mot_source_dir,
        include_empty_frames=include_empty_frames,
        include_gt=include_gt,
        include_det=include_det,
    )

    frame_df.to_csv(output_dir / "mot_frame_table.csv", index=False)
    ann_df.to_csv(output_dir / "mot_annotation_table.csv", index=False)

    return frame_df, ann_df

def ensure_downloaded(name: str, cache_dir: str | Path = Path("MOTsource")) -> Path:
    """
    Ensures the dataset is downloaded and cached.
        """

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    dataset_dir = cache_dir / name
    if dataset_dir.exists():
        return dataset_dir

    url = f"https://pub-8198d2983e174428a4f55e00f0addb18.r2.dev/datasets/{name}.tar.gz"
    with requests.get(url, stream=True) as response:
        response.raise_for_status()
        size = int(response.headers.get("Content-Length", 0))
        with tqdm(total=size, unit="B", unit_scale=True, desc=name) as pbar:

            class FileProgress:
                def read(self, n: int = -1) -> bytes:
                    data = response.raw.read(n)
                    pbar.update(len(data))
                    return data

            with tarfile.open(fileobj=FileProgress(), mode="r|gz") as tar:
                tar.extractall(path=dataset_dir)

    return dataset_dir


@dataclass
class Person:
    id: int
    split: str
    sequence: str
    detections: List[pd.DataFrame]
    ground_truth: List[pd.DataFrame]

def load_people(
    frame_df: pd.DataFrame,
    ann_df: pd.DataFrame,
    sequence: str | None = None,
) -> Dict[int, Person]:
    """
    Build a mapping from track_id to Person.

    Each Person stores annotation rows grouped per frame for both detections (`source='det'`)
    and ground truth (`source='gt'`), sorted by `frame_id`.
    If `sequence` is provided, only rows from that sequence are included.
    """
    required_frame_cols = {"split", "sequence"}
    required_ann_cols = {"track_id", "source", "split", "sequence", "frame_id"}

    if not required_frame_cols.issubset(frame_df.columns):
        raise ValueError(
            f"frame_df must contain columns: {sorted(required_frame_cols)}"
        )
    if not required_ann_cols.issubset(ann_df.columns):
        raise ValueError(
            f"ann_df must contain columns: {sorted(required_ann_cols)}"
        )

    if ann_df.empty:
        return {}

    filtered_ann_df = ann_df
    if sequence is not None:
        filtered_ann_df = ann_df[ann_df["sequence"] == sequence]
        if filtered_ann_df.empty:
            return {}

    people: Dict[int, Person] = {}

    for track_id, track_df in filtered_ann_df.groupby("track_id", sort=True):
        if pd.isna(track_id):
            continue

        split_values = track_df["split"].dropna().unique()
        sequence_values = track_df["sequence"].dropna().unique()

        if len(split_values) != 1 or len(sequence_values) != 1:
            raise ValueError(
                "Each track_id must belong to exactly one split and sequence in ann_df. "
                f"track_id={track_id} has split={list(split_values)}, "
                f"sequence={list(sequence_values)}"
            )

        split = str(split_values[0])
        person_sequence = str(sequence_values[0])

        det_df = track_df[track_df["source"] == "det"].sort_values("frame_id")
        gt_df = track_df[track_df["source"] == "gt"].sort_values("frame_id")

        detections = [g.copy() for _, g in det_df.groupby("frame_id", sort=True)]
        ground_truth = [g.copy() for _, g in gt_df.groupby("frame_id", sort=True)]

        people[int(track_id)] = Person(
            id=int(track_id),
            split=split,
            sequence=person_sequence,
            detections=detections,
            ground_truth=ground_truth,
        )

    return people


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    mot_source = project_root / "Dataprep/MOTsource"

    ensure_downloaded("MOT20", cache_dir=mot_source)

    # So we have the pd.df loaded in the project
    output_dir = project_root / "data/tables"

    frame_df, ann_df = save_mot_tables(
        mot_source_dir=mot_source,
        output_dir=output_dir,
        include_empty_frames=False,
        include_gt=True,
        include_det=True,
    )

    splits = split_mot_tables(
        frame_df=frame_df,
        ann_df=ann_df,
    )

    print("train_frame shape:", splits["train_frame"].shape)
    print("val_frame shape:", splits["val_frame"].shape)
    print("test_frame shape:", splits["test_frame"].shape)
    print("train_ann shape:", splits["train_ann"].shape)
    print("val_ann shape:", splits["val_ann"].shape)
    print("test_ann shape:", splits["test_ann"].shape)

    # loading the loader
    rows=build_frame_samples(mot_source, include_empty_frames=False)

    for row in rows[:100]:
        print(
            f"Dataset: {row['dataset']} | "
            f"Split: {row['split']} | "
            f"Seq: {row['sequence']} | "
            f"Frame: {row['frame_id']} | "
            f"Det count: {len(row['det'])} | "
            f"GT count: {len(row['gt'])} | "
            f"Path: {row['image_path']}"
        )

    # loading the people
    people = load_people(
        frame_df=frame_df,
        ann_df=ann_df,
        sequence="MOT20-01",
    )

    for person in people.values():
        print(f"Split: {person.split} | Sequence: {person.sequence} | ID: {person.id} | Detections: {len(person.detections)} | Ground Truth: {len(person.ground_truth)}")
