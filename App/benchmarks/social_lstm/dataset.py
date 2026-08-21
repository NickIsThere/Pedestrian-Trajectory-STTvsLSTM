"""
Dataset adapter for Social LSTM: reads prepared baseline trajectories,
groups by sequence, and yields temporal windows with variable pedestrian counts.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
import torch
from torch.utils.data import Dataset

from App.model.config import ModelConfig

_STT_DEFAULTS = ModelConfig()


class SocialLSTMDataset(Dataset):
    """
    Reads per-frame trajectory data from prepared_baseline CSVs.
    Groups by sequence and generates sliding observation/prediction windows.
    Each sample yields variable-pedestrian sets per frame, padded to max_pedestrians.
    """

    def __init__(
        self,
        baseline_dir: Path,
        split: str = "train",
        obs_len: int = _STT_DEFAULTS.lookback,
        pred_len: int = _STT_DEFAULTS.future_steps,
        subsample_step: int = _STT_DEFAULTS.trajectory_stride,
        max_pedestrians: Optional[int] = None,
    ):
        """
        Main writer: Keez Cuijpers
        Reviewer: Ciprian Driscu
        Contributors: 
        
        Args:
            baseline_dir: Path to data/prepared_baseline/
            split: "train" or "val"
            obs_len: Number of observation frames
            pred_len: Number of prediction frames
            subsample_step: Raw-frame stride between sampled timesteps (matches STT trajectory_stride)
            max_pedestrians: Pad to this many peds per frame; if None, auto-detect from data
        """
        self.baseline_dir = Path(baseline_dir)
        self.split = split
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.subsample_step = subsample_step
        self.sequence_data: dict[str, pd.DataFrame] = {}
        self.windows: list[dict] = []

        # Load baseline CSV for this split
        csv_path = self.baseline_dir / split / "baseline_trajectory.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Baseline CSV not found: {csv_path}")

        df = pd.read_csv(csv_path)
        if df.empty:
            raise ValueError(f"Baseline CSV is empty: {csv_path}")

        # Group by sequence
        for seq_name, seq_df in df.groupby("sequence"):
            seq_df = seq_df.sort_values(["track_id", "frame_id"]).reset_index(drop=True)
            self.sequence_data[seq_name] = seq_df

        # Auto-detect max pedestrians if not provided
        if max_pedestrians is None:
            max_peds = 0
            for seq_df in self.sequence_data.values():
                frame_counts = seq_df.groupby("frame_id").size()
                max_peds = max(max_peds, frame_counts.max())
            self.max_pedestrians = max_peds
        else:
            self.max_pedestrians = max_pedestrians

        # Precompute per-frame track arrays to avoid pandas ops during sampling.
        # Slots are assigned per-window (not globally) to prevent track-id bias.
        self.precomputed: dict[str, dict] = {}
        for seq_name, seq_df in self.sequence_data.items():
            frames = sorted(seq_df["frame_id"].unique())
            frame_data = {}

            for frame_id in frames:
                frame_df = seq_df[seq_df["frame_id"] == frame_id].sort_values("track_id")
                frame_data[int(frame_id)] = {
                    "track_ids": frame_df["track_id"].astype(np.int64).to_numpy(),
                    "positions": frame_df[["foot_x_norm", "foot_y_norm"]].to_numpy(dtype=np.float32),
                    "displacements": frame_df[["dx_norm", "dy_norm"]].to_numpy(dtype=np.float32),
                }

            self.precomputed[seq_name] = {"frames": frames, "frame_data": frame_data}

        # Build sliding windows per sequence
        self._build_windows()

    def _build_windows(self):
        """
        Main writer: Keez Cuijpers
        Reviewer: Ciprian Driscu
        Contributors: 
        
        Build sliding observation→prediction windows from sequences.
        """
        min_window_len = self.obs_len + self.pred_len
        subsample_window_len = (min_window_len - 1) * self.subsample_step + 1

        for seq_name in self.sequence_data:
            frames = self.precomputed[seq_name]["frames"]

            for start_idx in range(len(frames) - subsample_window_len + 1):
                end_idx = start_idx + subsample_window_len
                frame_ids = frames[start_idx:end_idx:self.subsample_step]
                self.windows.append({
                    "sequence": seq_name,
                    "start_frame_id": frames[start_idx],
                    "end_frame_id": frames[end_idx - 1],
                    "frame_ids": frame_ids,
                })

    def _build_window_slot_map(self, seq_name: str, frame_ids: list[int]) -> dict[int, int]:
        """
        Main writer: Keez Cuijpers
        Reviewer: Ciprian Driscu
        Contributors: 

        Build a deterministic slot map for the current window.

        If union(track_ids) exceeds max_pedestrians, keep the most frequently observed
        track IDs in this window to minimize information loss.
        """
        frame_data = self.precomputed[seq_name]["frame_data"]
        frequencies: dict[int, int] = {}
        for frame_id in frame_ids:
            for track_id in frame_data[int(frame_id)]["track_ids"]:
                tid = int(track_id)
                frequencies[tid] = frequencies.get(tid, 0) + 1

        if not frequencies:
            return {}

        ranked_track_ids = sorted(
            frequencies,
            key=lambda tid: (-frequencies[tid], tid),
        )
        selected = ranked_track_ids[: self.max_pedestrians]
        return {track_id: slot for slot, track_id in enumerate(selected)}

    def _frame_to_padded(
        self,
        *,
        seq_name: str,
        frame_id: int,
        slot_map: dict[int, int],
        key: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Main writer: Keez Cuijpers
        Reviewer: Ciprian Driscu
        Contributors: 

        Convert a frame's values into a padded pedestrian tensor and mask.
        """
        data = self.precomputed[seq_name]["frame_data"][int(frame_id)]
        values = data[key]
        track_ids = data["track_ids"]

        padded = np.zeros((self.max_pedestrians, 2), dtype=np.float32)
        mask = np.zeros(self.max_pedestrians, dtype=np.float32)

        for row_index, track_id in enumerate(track_ids):
            slot = slot_map.get(int(track_id))
            if slot is None:
                continue
            padded[slot, :] = values[row_index, :]
            mask[slot] = 1.0

        return padded, mask

    def __len__(self) -> int:
        """
        Main writer: Keez Cuijpers
        Reviewer: Ciprian Driscu
        Contributors: 
        """
        return len(self.windows)

    def __getitem__(self, idx: int) -> dict:
        """
        Main writer: Keez Cuijpers
        Reviewer: Ciprian Driscu
        Contributors: 
        
        Returns:
            {
                "obs_positions": (obs_len, max_peds, 2),  # normalized [x, y]
                "obs_masks": (obs_len, max_peds),  # 1 if ped present, 0 if padded
                "pred_displacements": (pred_len, max_peds, 2),  # normalized [dx, dy]
                "pred_masks": (pred_len, max_peds),  # 1 if ped present, 0 if padded
                "sequence": str,
                "start_frame": int,
            }
        """
        window = self.windows[idx]
        seq_name = window["sequence"]
        start_frame_id = window["start_frame_id"]
        subsampled_frames = window["frame_ids"]

        obs_frames = subsampled_frames[:self.obs_len]
        pred_frames = subsampled_frames[self.obs_len : self.obs_len + self.pred_len]
        slot_map = self._build_window_slot_map(seq_name, subsampled_frames)

        # Extract observation windows
        obs_positions = []
        obs_masks = []
        for frame_id in obs_frames:
            padded_positions, mask = self._frame_to_padded(
                seq_name=seq_name,
                frame_id=int(frame_id),
                slot_map=slot_map,
                key="positions",
            )
            obs_positions.append(padded_positions)
            obs_masks.append(mask)

        obs_positions = np.array(obs_positions, dtype=np.float32)  # (obs_len, max_peds, 2)
        obs_masks = np.array(obs_masks, dtype=np.float32)  # (obs_len, max_peds)

        # Extract prediction windows
        pred_displacements = []
        pred_masks = []
        for frame_id in pred_frames:
            padded_displacements, mask = self._frame_to_padded(
                seq_name=seq_name,
                frame_id=int(frame_id),
                slot_map=slot_map,
                key="displacements",
            )
            pred_displacements.append(padded_displacements)
            pred_masks.append(mask)

        pred_displacements = np.array(pred_displacements, dtype=np.float32)  # (pred_len, max_peds, 2)
        pred_masks = np.array(pred_masks, dtype=np.float32)  # (pred_len, max_peds)

        return {
            "obs_positions": torch.from_numpy(obs_positions),
            "obs_masks": torch.from_numpy(obs_masks),
            "pred_displacements": torch.from_numpy(pred_displacements),
            "pred_masks": torch.from_numpy(pred_masks),
            "sequence": seq_name,
            "start_frame": int(start_frame_id),
        }




def collate_batch(batch: list[dict]) -> dict:
    """
    Main writer: Keez Cuijpers
    Reviewer: Ciprian Driscu
    Contributors: 
    
    Stack batch dicts into tensors.
    """
    return {
        "obs_positions": torch.stack([item["obs_positions"] for item in batch]),
        "obs_masks": torch.stack([item["obs_masks"] for item in batch]),
        "pred_displacements": torch.stack([item["pred_displacements"] for item in batch]),
        "pred_masks": torch.stack([item["pred_masks"] for item in batch]),
        "sequences": [item["sequence"] for item in batch],
        "start_frames": [item["start_frame"] for item in batch],
    }
