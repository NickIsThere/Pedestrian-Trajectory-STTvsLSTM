from __future__ import annotations

from typing import Sequence

import pandas as pd

"""
Nick Grebe i6377605

Implmentations of Holm, Bonferroni and BH FDR Corrections

"""

def holm_correction(p_values: Sequence[float], *, alpha: float = 0.05) -> pd.DataFrame:
    rows = pd.DataFrame({"original_index": range(len(p_values)), "p_value": [float(value) for value in p_values]})
    if rows.empty:
        return rows.assign(corrected_p_value=[], significant=[])

    ordered = rows.sort_values("p_value", kind="mergesort").reset_index(drop=True)
    m = len(ordered)
    raw_adjusted = [(m - rank) * p_value for rank, p_value in enumerate(ordered["p_value"])]
    monotonic_adjusted: list[float] = []
    running_max = 0.0
    for value in raw_adjusted:
        running_max = max(running_max, min(1.0, float(value)))
        monotonic_adjusted.append(running_max)
    ordered["corrected_p_value"] = monotonic_adjusted
    ordered["significant"] = ordered["corrected_p_value"] <= alpha
    return ordered.sort_values("original_index", kind="mergesort").reset_index(drop=True)


def bonferroni_correction(p_values: Sequence[float], *, alpha: float = 0.05) -> pd.DataFrame:
    rows = pd.DataFrame({"original_index": range(len(p_values)), "p_value": [float(value) for value in p_values]})
    if rows.empty:
        return rows.assign(corrected_p_value=[], significant=[])
    rows["corrected_p_value"] = (rows["p_value"] * len(rows)).clip(upper=1.0)
    rows["significant"] = rows["corrected_p_value"] <= alpha
    return rows


def bh_fdr_correction(p_values: Sequence[float], *, alpha: float = 0.05) -> pd.DataFrame:
    rows = pd.DataFrame({"original_index": range(len(p_values)), "p_value": [float(value) for value in p_values]})
    if rows.empty:
        return rows.assign(corrected_p_value=[], significant=[])

    ordered = rows.sort_values("p_value", ascending=False, kind="mergesort").reset_index(drop=True)
    m = len(ordered)
    running_min = 1.0
    adjusted_desc: list[float] = []
    for rank_from_largest, p_value in enumerate(ordered["p_value"]):
        rank = m - rank_from_largest
        running_min = min(running_min, min(1.0, float(p_value) * m / rank))
        adjusted_desc.append(running_min)
    ordered["corrected_p_value"] = adjusted_desc
    ordered["significant"] = ordered["corrected_p_value"] <= alpha
    return ordered.sort_values("original_index", kind="mergesort").reset_index(drop=True)
