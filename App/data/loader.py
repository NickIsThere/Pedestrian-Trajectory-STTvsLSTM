from configparser import ConfigParser
from os import listdir, sep
from pathlib import Path
import requests
from tqdm import tqdm
import tarfile
import pandas as pd

DATA_ROOT = Path(__file__).resolve().parents[1] / "MOTsource"
VALID_DATASETS = ["MOT20", "MOTSynth"]

def ensure_downloaded(name: str, cache_dir: str | Path = DATA_ROOT) -> Path:
    """Ensures the dataset is downloaded and cached.

    This function verifies whether the dataset is already present and ready to be loaded, and downloads it otherwise.

    Args:
        cache_dir (Path): The directory to cache the dataset in.
        name (str): The name of the dataset to download.

    Returns:
        Path: The path to the cached dataset.

    Main writer: Noah Nuelandt
    Reviewer: 
    Contributors: Neo Deward
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

base_columns = ["frame", "track", "bb_left", "bb_top", "bb_width", "bb_height", "conf"]
columns_gt = base_columns + ["class", "visibility"]
columns_det = base_columns + ["x", "y", "z"]
def load_ann(ann: Path, seqinfo: ConfigParser) -> pd.DataFrame:
    """
    Main writer: Noah Nuelandt
    Reviewer: 
    Contributors: Neo Deward
    """
    if not ann.exists():
        return pd.DataFrame()

    typ = ann.stem.split(".")[0]
    if typ == "gt":
        columns = columns_gt
    elif typ == "det":
        columns = columns_det
    else:
        raise ValueError(f"Invalid annotation type: {typ}")
    
    df = pd.read_csv(ann, header=None, names=columns)

    seq_dir = ann.parent.parent
    sequence = seq_dir.name
    split = seq_dir.parent.name
    dataset = seq_dir.parent.parent.name

    path_prefix = str(seq_dir / seqinfo["Sequence"]["imDir"]) + sep
    path_suffix = seqinfo["Sequence"]["imExt"]

    df["dataset"] = dataset
    df["split"] = split
    df["sequence"] = sequence
    df["path"] = df["frame"].map(
        lambda f: f"{path_prefix}{int(f):06d}{path_suffix}"
    )

    return df

def load(name: str, cache_dir: str | Path = DATA_ROOT) -> (pd.DataFrame, pd.DataFrame):
    """Loads the dataset from the cache directory.

    Args:
        name (str): The name of the dataset to load.
        cache_dir (Path): The directory to cache the dataset in.

    Returns:
        Path: The path to the loaded dataset.

    Main writer: Noah Nuelandt
    Reviewer: 
    Contributors: Neo Deward
    """
    if name not in VALID_DATASETS:
        raise ValueError(f"Invalid dataset name: {name}")
    dataset_dir = ensure_downloaded(name, cache_dir)

    gt_df = pd.DataFrame()
    det_df = pd.DataFrame()

    for split in ["train", "val", "test"]:
        split_dir = dataset_dir / split
        if not split_dir.exists():
            continue
        for sequence in listdir(split_dir):
            seq_dir = split_dir / sequence
            seqinfo = ConfigParser()
            seqinfo.read(seq_dir / "seqinfo.ini")
            
            gt = load_ann(seq_dir / "gt" / "gt.txt", seqinfo)
            det = load_ann(seq_dir / "det" / "det.txt", seqinfo)

            gt_df = pd.concat([gt_df, gt])            
            det_df = pd.concat([det_df, det])

    return gt_df, det_df

if __name__ == "__main__":
    gt_df, det_df = load("MOT20")
    print(gt_df.head())
    print(det_df.head())