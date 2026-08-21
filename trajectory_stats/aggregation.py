from __future__ import annotations

import pandas as pd

from schema import AggregationLevel, aggregation_group_keys, validate_required_columns

"""
Nick Grebe i6377605

Aggregates raw per unit prediction errors into 3 different summaries 
window, track and scene level

Also checks the real timey-ness with a simple pass or fail segment

"""

AGGREGATION_REQUIRED_COLUMNS = [
    "error_l2",
    "ade_until_horizon",
    "fde_at_horizon",
    "batch_latency_ms",
    "runtime_windows_per_second",
]


def aggregate_units(df: pd.DataFrame, *, level: AggregationLevel = "track") -> pd.DataFrame:
    keys = aggregation_group_keys(level)
    validate_required_columns(df, [*keys, *AGGREGATION_REQUIRED_COLUMNS])
    grouped = df.groupby(keys, dropna=False, sort=True)
    aggregated = grouped.agg(
        mean_error_l2=("error_l2", "mean"),
        mean_ade_until_horizon=("ade_until_horizon", "mean"),
        mean_fde_at_horizon=("fde_at_horizon", "mean"),
        n_rows=("error_l2", "size"),
        mean_batch_latency_ms=("batch_latency_ms", "mean"),
        p95_batch_latency_ms=("batch_latency_ms", lambda values: values.quantile(0.95)),
        mean_runtime_windows_per_second=("runtime_windows_per_second", "mean"),
    ).reset_index()
    return annotate_real_time_budget(aggregated)


def annotate_real_time_budget(
    df: pd.DataFrame,
    *,
    latency_ms_max: float = 100.0,
    runtime_windows_per_second_min: float = 15.0,
) -> pd.DataFrame:
    validate_required_columns(df, ["mean_batch_latency_ms", "mean_runtime_windows_per_second"])
    annotated = df.copy()
    annotated["budget_pass"] = (
        (annotated["mean_batch_latency_ms"] <= latency_ms_max)
        & (annotated["mean_runtime_windows_per_second"] >= runtime_windows_per_second_min)
    )
    return annotated
