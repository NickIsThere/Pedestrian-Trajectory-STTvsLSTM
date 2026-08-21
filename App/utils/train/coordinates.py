from __future__ import annotations

from configparser import ConfigParser


def norm_offsets_from_seqinfo(info: ConfigParser) -> tuple[float, float, float]:
    width = int(info["Sequence"]["imWidth"])
    height = int(info["Sequence"]["imHeight"])
    max_dim = max(width, height)
    offset_x = (1 - width / max_dim) / 2
    offset_y = (1 - height / max_dim) / 2
    return float(max_dim), float(offset_x), float(offset_y)


def denormalize_foot_xy(
    norm_x: float,
    norm_y: float,
    *,
    max_dim: float,
    offset_x: float,
    offset_y: float,
) -> tuple[float, float]:
    return (norm_x - offset_x) * max_dim, (norm_y - offset_y) * max_dim
