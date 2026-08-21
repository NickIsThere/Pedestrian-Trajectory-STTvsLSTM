"""Shared utilities: pickle disk cache and Kalman forecast defaults."""
import sys
import pickle
from pathlib import Path
from typing import Any, Callable

project_root = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

KALMAN_PRECOMPUTED_FUTURE_STEPS = 20

def cached(cache_file: Path, fn: Callable[[], Any], min_version: int | None = 0) -> Any:
    """
    Main writer: Keez Cuijpers
    Reviewer:
    Contributors:
    """
    if cache_file.exists():
        with open(cache_file, "rb") as f:
            data = pickle.load(f)
        if isinstance(data, dict) and "data" in data:
            ver = data.get("version")
            if min_version is None or (ver is not None and ver >= min_version):
                return data["data"]
        else:
            return data
    result = fn()
    with open(cache_file, "wb") as f:
        pickle.dump({"version": min_version, "data": result}, f)
    return result

def get_channel(sequence: 'Sequence', channel: str) -> 'Channel':
    """
    Main writer: Keez Cuijpers
    Reviewer:
    Contributors:
    """
    if channel in sequence.channels:
        return sequence.channels[channel]

    from benchmarks.kalman.kalman_filter import predict_tracks
    from App.data.reader import Channel

    if channel == "kalman":
        predicted_tracks = predict_tracks(sequence.channels["gt"].tracks)
        sequence.channels[channel] = Channel(name=channel, tracks=predicted_tracks)
        return sequence.channels[channel]

    elif channel == "social_lstm":
        from benchmarks.social_lstm.inference import predict_tracks_social_lstm
        predicted_tracks = predict_tracks_social_lstm(sequence.channels["gt"].tracks)
        sequence.channels[channel] = Channel(name=channel, tracks=predicted_tracks)
        return sequence.channels[channel]

    else:
        raise ValueError(f"Invalid channel: {channel}")