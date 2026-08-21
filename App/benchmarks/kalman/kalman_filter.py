"""Minimal 2D constant-velocity Kalman filter for MOT-style sequences."""

from __future__ import annotations

import numpy as np

from data.reader import Track, Annotation, BBox
from utils.utils import KALMAN_PRECOMPUTED_FUTURE_STEPS


class Kalman2D:
    """
    Main writer: Neo Deward
    Reviewer: Noah Nuelandt
    Contributors: Noah Nuelandt
    """

    """State [x, y, vx, vy]; observe position only."""

    def __init__(
        self,
        x: float,
        y: float,
        *,
        process_noise: float = 1.0,
        measurement_noise: float = 1.0,
    ) -> None:
        self.x = np.array([[x], [y], [0.0], [0.0]], dtype=float)
        self.P = np.eye(4, dtype=float) * 500.0
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
        self.R = np.eye(2, dtype=float) * measurement_noise
        self.process_noise = process_noise

    def _transition(self, dt: float) -> np.ndarray:
        return np.array(
            [
                [1, 0, dt, 0],
                [0, 1, 0, dt],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            dtype=float,
        )

    def _process_cov(self, dt: float) -> np.ndarray:
        q = self.process_noise
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt2 * dt2
        return q * np.array(
            [
                [dt4 / 4, 0, dt3 / 2, 0],
                [0, dt4 / 4, 0, dt3 / 2],
                [dt3 / 2, 0, dt2, 0],
                [0, dt3 / 2, 0, dt2],
            ],
            dtype=float,
        )

    def predict(self, dt: float = 1.0) -> tuple[float, float]:
        f = self._transition(dt)
        self.x = f @ self.x
        q = self._process_cov(dt)
        self.P = f @ self.P @ f.T + q
        return float(self.x[0, 0]), float(self.x[1, 0])

    def update(self, mx: float, my: float) -> tuple[float, float]:
        z = np.array([[mx], [my]], dtype=float)
        y = z - (self.H @ self.x)
        s = self.H @ self.P @ self.H.T + self.R
        k = self.P @ self.H.T @ np.linalg.inv(s)
        self.x = self.x + k @ y
        self.P = (np.eye(4) - k @ self.H) @ self.P
        return float(self.x[0, 0]), float(self.x[1, 0])


def predict_tracks(
    tracks: dict[int, Track],
    *,
    future_steps: int = KALMAN_PRECOMPUTED_FUTURE_STEPS,
    process_noise: float = 1.0,
    measurement_noise: float = 10.0,
) -> dict[int, Track]:
    """
    Main writer: Noah Nuelandt
    Reviewer:
    Contributors: Neo Deward
    """
    """Predict future trajectories and fill gaps using a Kalman filter."""
    predicted_tracks: dict[int, Track] = {}
    
    for track_id, track in tracks.items():
        if not track.gt:
            continue
            
        sorted_frames = sorted(track.gt.keys())
        first_frame = sorted_frames[0]
        first_ann = track.gt[first_frame]
        
        kf = Kalman2D(
            first_ann.bbox.foot_x,
            first_ann.bbox.foot_y,
            process_noise=process_noise,
            measurement_noise=measurement_noise,
        )
        
        predicted_gt: dict[int, Annotation] = {}
        
        predicted_gt[first_frame] = Annotation(
            frame_id=first_frame,
            track_id=track_id,
            bbox=BBox(
                x=first_ann.bbox.x,
                y=first_ann.bbox.y,
                width=first_ann.bbox.width,
                height=first_ann.bbox.height,
            ),
            score=first_ann.score,
            class_id=first_ann.class_id,
            visibility=first_ann.visibility,
            x=first_ann.x,
            y=first_ann.y,
            z=first_ann.z,
        )
        
        prev_frame = first_frame
        last_ann = first_ann
        
        for frame_id in sorted_frames[1:]:
            ann = track.gt[frame_id]
            gap = frame_id - prev_frame
            
            for step in range(1, gap + 1):
                px, py = kf.predict(dt=1.0)
                target_frame = prev_frame + step
                
                if target_frame == frame_id:
                    fx, fy = ann.bbox.foot_x, ann.bbox.foot_y
                    px, py = kf.update(fx, fy)
                    last_ann = ann
                
                new_bbox = BBox(
                    x=px - last_ann.bbox.width / 2,
                    y=py - last_ann.bbox.height,
                    width=last_ann.bbox.width,
                    height=last_ann.bbox.height,
                )
                
                predicted_gt[target_frame] = Annotation(
                    frame_id=target_frame,
                    track_id=track_id,
                    bbox=new_bbox,
                    score=last_ann.score,
                    class_id=last_ann.class_id,
                    visibility=last_ann.visibility,
                    x=px,
                    y=py,
                    z=last_ann.z,
                )
                
            prev_frame = frame_id
            
        for step in range(1, future_steps + 1):
            px, py = kf.predict(dt=1.0)
            target_frame = prev_frame + step
            
            new_bbox = BBox(
                x=px - last_ann.bbox.width / 2,
                y=py - last_ann.bbox.height,
                width=last_ann.bbox.width,
                height=last_ann.bbox.height,
            )
            
            predicted_gt[target_frame] = Annotation(
                frame_id=target_frame,
                track_id=track_id,
                bbox=new_bbox,
                score=last_ann.score,
                class_id=last_ann.class_id,
                visibility=last_ann.visibility,
                x=px,
                y=py,
                z=last_ann.z,
            )
            
        predicted_tracks[track_id] = Track(id=track_id, gt=predicted_gt)
        
    return predicted_tracks
