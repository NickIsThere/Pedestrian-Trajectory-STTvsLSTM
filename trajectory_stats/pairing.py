from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from schema import (
    AggregationLevel,
    PAIRING_KEYS_BY_LEVEL,
    SUPPORTED_COMPARISON_METRICS,
    validate_required_columns,
)

"""
Nick Grebe i6377605

Matches two models on the same evaluation units
(same sequence, track, horizon, seed, split, test domain)


"""

@dataclass(frozen=True)
class PairingResult:
    paired: pd.DataFrame
    report: dict[str, Any]
    pairing_keys: list[str]


def pair_aggregated(
    df: pd.DataFrame,
    *,
    baseline_model: str,
    treatment_model: str,
    level: AggregationLevel = "track",
) -> PairingResult:
    pairing_keys = PAIRING_KEYS_BY_LEVEL[level]
    validate_required_columns(df, ["model_name", *pairing_keys])

    left = df[df["model_name"] == baseline_model].copy()
    right = df[df["model_name"] == treatment_model].copy()
    missing_keys = {
        "baseline": _rows_with_missing_keys(left, pairing_keys),
        "treatment": _rows_with_missing_keys(right, pairing_keys),
    }

    left_valid = left.dropna(subset=pairing_keys)
    right_valid = right.dropna(subset=pairing_keys)
    duplicated = {
        "baseline": _duplicated_key_group_count(left_valid, pairing_keys),
        "treatment": _duplicated_key_group_count(right_valid, pairing_keys),
    }

    if duplicated["baseline"] or duplicated["treatment"]:
        report = _report(
            left=left,
            right=right,
            paired_count=0,
            unmatched_left=len(left_valid),
            unmatched_right=len(right_valid),
            duplicated=duplicated,
            missing_keys=missing_keys,
        )
        return PairingResult(pd.DataFrame(), report, pairing_keys)

    paired = left_valid.merge(
        right_valid,
        on=pairing_keys,
        how="inner",
        suffixes=("_baseline", "_treatment"),
    )
    matched_left_keys = paired[pairing_keys].drop_duplicates()
    left_unmatched = _anti_join_count(left_valid, matched_left_keys, pairing_keys)
    right_unmatched = _anti_join_count(right_valid, matched_left_keys, pairing_keys)
    report = _report(
        left=left,
        right=right,
        paired_count=len(paired),
        unmatched_left=left_unmatched,
        unmatched_right=right_unmatched,
        duplicated=duplicated,
        missing_keys=missing_keys,
    )
    return PairingResult(paired, report, pairing_keys)


def prepare_paired_differences(
    df: pd.DataFrame,
    *,
    baseline_model: str,
    treatment_model: str,
    metric: str,
    level: AggregationLevel = "track",
) -> pd.DataFrame:
    if metric not in SUPPORTED_COMPARISON_METRICS:
        raise ValueError(f"Unsupported metric {metric!r}; expected one of {SUPPORTED_COMPARISON_METRICS}")

    result = pair_aggregated(
        df,
        baseline_model=baseline_model,
        treatment_model=treatment_model,
        level=level,
    )
    if result.report["n_pairs"] == 0:
        return pd.DataFrame(
            columns=[
                *result.pairing_keys,
                "baseline_model",
                "treatment_model",
                "baseline_metric_value",
                "treatment_metric_value",
                "difference",
                "percent_improvement",
            ],
        )

    output = result.paired[result.pairing_keys].copy()
    baseline_values = result.paired[f"{metric}_baseline"]
    treatment_values = result.paired[f"{metric}_treatment"]
    output["baseline_model"] = baseline_model
    output["treatment_model"] = treatment_model
    output["baseline_metric_value"] = baseline_values
    output["treatment_metric_value"] = treatment_values
    output["difference"] = baseline_values - treatment_values
    output["percent_improvement"] = output["difference"] / baseline_values * 100.0
    return output


def _rows_with_missing_keys(df: pd.DataFrame, keys: list[str]) -> int:
    if df.empty:
        return 0
    return int(df[keys].isna().any(axis=1).sum())


def _duplicated_key_group_count(df: pd.DataFrame, keys: list[str]) -> int:
    if df.empty:
        return 0
    duplicated_rows = df[df.duplicated(keys, keep=False)]
    if duplicated_rows.empty:
        return 0
    return int(len(duplicated_rows[keys].drop_duplicates()))


def _anti_join_count(df: pd.DataFrame, matched_keys: pd.DataFrame, keys: list[str]) -> int:
    if df.empty:
        return 0
    if matched_keys.empty:
        return len(df)
    merged = df[keys].drop_duplicates().merge(matched_keys, on=keys, how="left", indicator=True)
    return int((merged["_merge"] == "left_only").sum())


def _report(
    *,
    left: pd.DataFrame,
    right: pd.DataFrame,
    paired_count: int,
    unmatched_left: int,
    unmatched_right: int,
    duplicated: dict[str, int],
    missing_keys: dict[str, int],
) -> dict[str, Any]:
    n_left = len(left)
    n_right = len(right)
    return {
        "n_left": n_left,
        "n_right": n_right,
        "n_pairs": paired_count,
        "n_unmatched_left": unmatched_left,
        "n_unmatched_right": unmatched_right,
        "match_rate_left": paired_count / n_left if n_left else 0.0,
        "match_rate_right": paired_count / n_right if n_right else 0.0,
        "duplicated_pairing_keys": duplicated,
        "missing_pairing_keys": missing_keys,
        "pairing_possible": paired_count > 0 and not any(duplicated.values()) and not any(missing_keys.values()),
    }
