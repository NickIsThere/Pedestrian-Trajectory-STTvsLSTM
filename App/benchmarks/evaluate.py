"""
Generic evaluator for trajectory prediction models.
Computes ADE/FDE on held-out validation sequences.
Reusable for any model that implements the PredictionModel interface.
"""

from __future__ import annotations

import numpy as np
from typing import Protocol
from abc import abstractmethod
import pandas as pd

from data.reader import Sequence, Channel


class PredictionModel(Protocol):
    """Interface that any prediction model must implement."""
    
    @abstractmethod
    def predict(
        self,
        sequence: Sequence,
        obs_len: int,
        pred_len: int,
    ) -> Channel:
        """
        Main writer: Keez Cuijpers
        Reviewer: Ciprian Driscu
        Contributors: 

        Generate predictions for a sequence.
        
        Args:
            sequence: Sequence object with ground truth tracks
            obs_len: Number of observation frames
            pred_len: Number of prediction frames
        
        Returns:
            Channel object with predicted tracks (same structure as GT)
        """
        ...


def compute_ade_fde(
    gt_channel: Channel,
    pred_channel: Channel,
) -> dict:
    """
    Main writer: Keez Cuijpers
    Reviewer: Ciprian Driscu
    Contributors: 
    
    Compute ADE (average displacement error) and FDE (final displacement error)
    in normalized pixel coordinates.
    
    Args:
        gt_channel: Ground truth Channel
        pred_channel: Predicted Channel (same track IDs as GT)
    
    Returns:
        {
            "ade": float,
            "fde": float,
            "num_tracks": int,
            "num_errors": int,
        }
    """
    observation_errors = []
    final_errors = []
    
    for track_id, gt_track in gt_channel.tracks.items():
        if track_id not in pred_channel.tracks:
            continue
        
        pred_track = pred_channel.tracks[track_id]
        
        # Find common frames
        common_frames = sorted(
            set(gt_track.gt.keys()).intersection(pred_track.gt.keys())
        )
        
        if len(common_frames) < 2:
            continue
        
        # Compute displacements
        track_errors = []
        for frame_id in common_frames[1:]:
            gt_ann = gt_track.gt[frame_id]
            pred_ann = pred_track.gt[frame_id]
            
            # Use normalized foot points
            gx = gt_ann.bbox.foot_x
            gy = gt_ann.bbox.foot_y
            px = pred_ann.bbox.foot_x
            py = pred_ann.bbox.foot_y
            
            error = ((px - gx) ** 2 + (py - gy) ** 2) ** 0.5
            track_errors.append(error)
        
        if track_errors:
            observation_errors.extend(track_errors)
            final_errors.append(track_errors[-1])
    
    ade = float(np.mean(observation_errors)) if observation_errors else 0.0
    fde = float(np.mean(final_errors)) if final_errors else 0.0
    
    return {
        "ade": ade,
        "fde": fde,
        "num_tracks": len(gt_channel.tracks),
        "num_errors": len(observation_errors),
    }


def evaluate_model(
    model: PredictionModel,
    sequences: dict[str, Sequence],
    obs_len: int,
    pred_len: int,
) -> pd.DataFrame:
    """
    Main writer: Keez Cuijpers
    Reviewer: Ciprian Driscu
    Contributors: 
    
    Evaluate model on multiple sequences.
    
    Args:
        model: Any object implementing PredictionModel protocol
        sequences: Dict of sequence_name → Sequence
        obs_len: Observation window length
        pred_len: Prediction window length
    
    Returns:
        DataFrame with one row per sequence: [sequence, ade, fde, num_tracks]
    """
    results = []
    
    for seq_name, sequence in sequences.items():
        try:
            pred_channel = model.predict(sequence, obs_len, pred_len)
            gt_channel = sequence.channels["gt"]
            
            metrics = compute_ade_fde(gt_channel, pred_channel)
            metrics["sequence"] = seq_name
            results.append(metrics)
        except Exception as e:
            print(f"Error evaluating {seq_name}: {e}")
            continue
    
    df = pd.DataFrame(results)
    
    # Summary statistics
    if not df.empty:
        print("\n" + "=" * 60)
        print(f"Evaluation Results (obs_len={obs_len}, pred_len={pred_len})")
        print("=" * 60)
        print(f"{'Sequence':<20} {'ADE':<12} {'FDE':<12} {'Tracks':<10}")
        print("-" * 60)
        
        for _, row in df.iterrows():
            print(f"{row['sequence']:<20} {row['ade']:<12.4f} {row['fde']:<12.4f} {row['num_tracks']:<10}")
        
        print("-" * 60)
        print(f"{'Average':<20} {df['ade'].mean():<12.4f} {df['fde'].mean():<12.4f}")
        print("=" * 60)
    
    return df
