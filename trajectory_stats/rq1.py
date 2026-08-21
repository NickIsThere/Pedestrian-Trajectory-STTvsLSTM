from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from aggregation import aggregate_units
from bootstrap import paired_bootstrap_ci
from loading import load_evaluation_units
from multiple_testing import holm_correction
from pairing import pair_aggregated
from permutation import Alternative, paired_permutation_test
from schema import AggregationLevel, SUPPORTED_COMPARISON_METRICS, validate_required_columns


"""
RQ1

"""

RQ1_RESULT_COLUMNS = [
    "baseline_model",
    "treatment_model",
    "metric",
    "aggregation_level",
    "target_horizon_s",
    "actual_horizon_s",
    "horizon_error_seconds",
    "n_pairs",
    "mean_baseline",
    "mean_treatment",
    "mean_difference",
    "median_difference",
    "percent_improvement",
    "bootstrap_ci_lower",
    "bootstrap_ci_upper",
    "bootstrap_confidence_level",
    "permutation_p_value",
    "permutation_alternative",
    "n_permutations",
    "n_bootstrap",
    "budget_mode",
    "baseline_budget_pass",
    "treatment_budget_pass",
    "baseline_mean_batch_latency_ms",
    "treatment_mean_batch_latency_ms",
    "baseline_p95_batch_latency_ms",
    "treatment_p95_batch_latency_ms",
    "baseline_mean_runtime_windows_per_second",
    "treatment_mean_runtime_windows_per_second",
    "correction_method",
    "corrected_p_value",
    "significant",
    "notes",
]

RQ1_ABLATION_RESULT_COLUMNS = [
    *RQ1_RESULT_COLUMNS,
    "ablation_step",
    "ablation_baseline_rank",
    "ablation_treatment_rank",
]


@dataclass(frozen=True)
class RQ1Result:
    results: pd.DataFrame
    output_path: Path | None = None


def run_rq1_pairwise(
    data: pd.DataFrame | Iterable[str | Path],
    *,
    baseline_model: str,
    treatment_model: str,
    metrics: Sequence[str] = ("mean_ade_until_horizon", "mean_fde_at_horizon"),
    target_horizons: Sequence[float] | None = None,
    aggregation_level: AggregationLevel = "track",
    outdir: str | Path | None = None,
    n_permutations: int = 10000,
    n_bootstrap: int = 10000,
    random_seed: int = 42,
    alternative: Alternative = "greater",
    budget_mode: str = "report",
    correction_method: str = "holm",
) -> RQ1Result:
    specs = [{"baseline_model": baseline_model, "treatment_model": treatment_model}]
    return _run_rq1(
        data,
        comparison_specs=specs,
        metrics=metrics,
        target_horizons=target_horizons,
        aggregation_level=aggregation_level,
        outdir=outdir,
        output_filename="stats_rq1_pairwise_results.csv",
        n_permutations=n_permutations,
        n_bootstrap=n_bootstrap,
        random_seed=random_seed,
        alternative=alternative,
        budget_mode=budget_mode,
        correction_method=correction_method,
        result_columns=RQ1_RESULT_COLUMNS,
    )


def run_rq1_ablation(
    data: pd.DataFrame | Iterable[str | Path],
    *,
    ablation_chain: Sequence[str],
    metrics: Sequence[str] = ("mean_ade_until_horizon", "mean_fde_at_horizon"),
    target_horizons: Sequence[float] | None = None,
    aggregation_level: AggregationLevel = "track",
    outdir: str | Path | None = None,
    n_permutations: int = 10000,
    n_bootstrap: int = 10000,
    random_seed: int = 42,
    alternative: Alternative = "greater",
    budget_mode: str = "report",
    correction_method: str = "holm",
) -> RQ1Result:
    if len(ablation_chain) < 2:
        raise ValueError("ablation_chain must contain at least two models")
    specs = [
        {
            "baseline_model": ablation_chain[index],
            "treatment_model": ablation_chain[index + 1],
            "ablation_step": index + 1,
            "ablation_baseline_rank": index + 1,
            "ablation_treatment_rank": index + 2,
        }
        for index in range(len(ablation_chain) - 1)
    ]
    return _run_rq1(
        data,
        comparison_specs=specs,
        metrics=metrics,
        target_horizons=target_horizons,
        aggregation_level=aggregation_level,
        outdir=outdir,
        output_filename="stats_rq1_incremental_ablation_results.csv",
        n_permutations=n_permutations,
        n_bootstrap=n_bootstrap,
        random_seed=random_seed,
        alternative=alternative,
        budget_mode=budget_mode,
        correction_method=correction_method,
        result_columns=RQ1_ABLATION_RESULT_COLUMNS,
    )


def _run_rq1(
    data: pd.DataFrame | Iterable[str | Path],
    *,
    comparison_specs: Sequence[dict[str, object]],
    metrics: Sequence[str],
    target_horizons: Sequence[float] | None,
    aggregation_level: AggregationLevel,
    outdir: str | Path | None,
    output_filename: str,
    n_permutations: int,
    n_bootstrap: int,
    random_seed: int,
    alternative: Alternative,
    budget_mode: str,
    correction_method: str,
    result_columns: list[str],
) -> RQ1Result:
    _validate_metrics(metrics)
    if budget_mode not in ("report", "strict"):
        raise ValueError("budget_mode must be 'report' or 'strict'")
    if correction_method not in ("holm", "none"):
        raise ValueError("correction_method must be 'holm' or 'none'")

    aggregated = _load_or_aggregate(data, level=aggregation_level)
    rows: list[dict[str, object]] = []
    for horizon in _horizon_slices(aggregated, target_horizons):
        horizon_frame = horizon.frame
        for spec in comparison_specs:
            filtered, budget_notes = _apply_budget_mode(
                horizon_frame,
                baseline_model=str(spec["baseline_model"]),
                treatment_model=str(spec["treatment_model"]),
                budget_mode=budget_mode,
            )
            for metric in metrics:
                row = _comparison_row(
                    filtered,
                    baseline_model=str(spec["baseline_model"]),
                    treatment_model=str(spec["treatment_model"]),
                    metric=metric,
                    aggregation_level=aggregation_level,
                    target_horizon_s=horizon.target_horizon_s,
                    actual_horizon_s=horizon.actual_horizon_s,
                    horizon_error_seconds=horizon.horizon_error_seconds,
                    n_permutations=n_permutations,
                    n_bootstrap=n_bootstrap,
                    random_seed=random_seed,
                    alternative=alternative,
                    budget_mode=budget_mode,
                    notes=[*horizon.notes, *budget_notes],
                )
                for key, value in spec.items():
                    if key not in ("baseline_model", "treatment_model"):
                        row[key] = value
                rows.append(row)

    results = pd.DataFrame(rows, columns=result_columns)
    results = _apply_p_value_correction(results, correction_method=correction_method)
    results = results[result_columns]
    for column in ("baseline_budget_pass", "treatment_budget_pass", "significant"):
        if column in results.columns:
            results[column] = results[column].astype(object)

    output_path = None
    if outdir is not None:
        output_path = Path(outdir) / output_filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(output_path, index=False)
    return RQ1Result(results=results, output_path=output_path)


def _load_or_aggregate(data: pd.DataFrame | Iterable[str | Path], *, level: AggregationLevel) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        frame = data.copy()
    else:
        frame = load_evaluation_units(data)
    if "mean_error_l2" in frame.columns:
        return frame
    return aggregate_units(frame, level=level)


def _validate_metrics(metrics: Sequence[str]) -> None:
    unsupported = [metric for metric in metrics if metric not in SUPPORTED_COMPARISON_METRICS]
    if unsupported:
        raise ValueError(f"Unsupported RQ1 metric(s): {unsupported}")


@dataclass(frozen=True)
class _HorizonSlice:
    target_horizon_s: float
    actual_horizon_s: float
    horizon_error_seconds: float
    frame: pd.DataFrame
    notes: list[str]


def _horizon_slices(df: pd.DataFrame, target_horizons: Sequence[float] | None) -> list[_HorizonSlice]:
    validate_required_columns(df, ["horizon_s"])
    if not target_horizons:
        return [
            _slice_for_actual_horizon(df, float(horizon_s), target_horizon_s=float("nan"), notes=[])
            for horizon_s in sorted(pd.to_numeric(df["horizon_s"], errors="coerce").dropna().unique())
        ]

    if "target_horizon_s" in df.columns and pd.to_numeric(df["target_horizon_s"], errors="coerce").notna().any():
        return [_slice_for_target_metadata(df, float(target)) for target in target_horizons]
    return [_slice_for_nearest_actual(df, float(target)) for target in target_horizons]


def _slice_for_target_metadata(df: pd.DataFrame, target: float) -> _HorizonSlice:
    target_values = pd.to_numeric(df["target_horizon_s"], errors="coerce")
    frame = df[np.isclose(target_values, target, equal_nan=False)].copy()
    actual = _single_or_nan(frame.get("actual_horizon_s", frame.get("horizon_s")))
    if np.isnan(actual):
        actual = _single_or_nan(frame.get("horizon_s"))
    return _HorizonSlice(
        target_horizon_s=target,
        actual_horizon_s=actual,
        horizon_error_seconds=actual - target if np.isfinite(actual) else float("nan"),
        frame=frame,
        notes=[] if len(frame) else [f"no_rows_for_target_horizon_s={target}"],
    )


def _slice_for_nearest_actual(df: pd.DataFrame, target: float) -> _HorizonSlice:
    horizons = sorted(float(value) for value in pd.to_numeric(df["horizon_s"], errors="coerce").dropna().unique())
    if not horizons:
        return _HorizonSlice(target, float("nan"), float("nan"), df.iloc[0:0].copy(), ["missing_horizon_s"])
    actual = min(horizons, key=lambda value: abs(value - target))
    return _slice_for_actual_horizon(
        df,
        actual,
        target_horizon_s=target,
        notes=[f"target_horizon_metadata_missing; nearest_actual_horizon_s={actual}"],
    )


def _slice_for_actual_horizon(
    df: pd.DataFrame,
    actual: float,
    *,
    target_horizon_s: float,
    notes: list[str],
) -> _HorizonSlice:
    frame = df[np.isclose(pd.to_numeric(df["horizon_s"], errors="coerce"), actual)].copy()
    error = actual - target_horizon_s if np.isfinite(target_horizon_s) else float("nan")
    return _HorizonSlice(target_horizon_s, actual, error, frame, notes)


def _single_or_nan(values: object) -> float:
    if not isinstance(values, pd.Series):
        return float("nan")
    uniques = pd.to_numeric(values, errors="coerce").dropna().unique()
    return float(uniques[0]) if len(uniques) == 1 else float("nan")


def _apply_budget_mode(
    df: pd.DataFrame,
    *,
    baseline_model: str,
    treatment_model: str,
    budget_mode: str,
) -> tuple[pd.DataFrame, list[str]]:
    if budget_mode != "strict":
        return df, []
    validate_required_columns(df, ["model_name", "budget_pass"])
    baseline_excluded = int(((df["model_name"] == baseline_model) & (~df["budget_pass"].astype(bool))).sum())
    treatment_excluded = int(((df["model_name"] == treatment_model) & (~df["budget_pass"].astype(bool))).sum())
    notes = [
        f"budget_excluded_baseline_rows={baseline_excluded}",
        f"budget_excluded_treatment_rows={treatment_excluded}",
    ]
    return df[df["budget_pass"].astype(bool)].copy(), notes


def _comparison_row(
    df: pd.DataFrame,
    *,
    baseline_model: str,
    treatment_model: str,
    metric: str,
    aggregation_level: AggregationLevel,
    target_horizon_s: float,
    actual_horizon_s: float,
    horizon_error_seconds: float,
    n_permutations: int,
    n_bootstrap: int,
    random_seed: int,
    alternative: Alternative,
    budget_mode: str,
    notes: list[str],
) -> dict[str, object]:
    result = pair_aggregated(
        df,
        baseline_model=baseline_model,
        treatment_model=treatment_model,
        level=aggregation_level,
    )
    base = _empty_row(
        baseline_model=baseline_model,
        treatment_model=treatment_model,
        metric=metric,
        aggregation_level=aggregation_level,
        target_horizon_s=target_horizon_s,
        actual_horizon_s=actual_horizon_s,
        horizon_error_seconds=horizon_error_seconds,
        n_permutations=n_permutations,
        n_bootstrap=n_bootstrap,
        alternative=alternative,
        budget_mode=budget_mode,
        notes=notes,
    )
    if result.report["n_pairs"] == 0:
        base["notes"] = _join_notes([*notes, "no_paired_units", f"pairing_report={result.report}"])
        return base

    paired = result.paired
    baseline_values = pd.to_numeric(paired[f"{metric}_baseline"], errors="coerce")
    treatment_values = pd.to_numeric(paired[f"{metric}_treatment"], errors="coerce")
    differences = baseline_values - treatment_values
    finite_mask = differences.notna() & baseline_values.notna() & treatment_values.notna()
    if not finite_mask.all():
        paired = paired[finite_mask].copy()
        baseline_values = baseline_values[finite_mask]
        treatment_values = treatment_values[finite_mask]
        differences = differences[finite_mask]
        notes = [*notes, f"dropped_nonfinite_pairs={int((~finite_mask).sum())}"]
    if differences.empty:
        base["notes"] = _join_notes([*notes, "no_finite_paired_differences"])
        return base

    permutation = paired_permutation_test(
        differences.to_numpy(),
        n_permutations=n_permutations,
        random_seed=random_seed,
        alternative=alternative,
    )
    bootstrap = paired_bootstrap_ci(
        differences.to_numpy(),
        n_bootstrap=n_bootstrap,
        random_seed=random_seed,
    )
    mean_baseline = float(baseline_values.mean())
    mean_difference = float(differences.mean())
    if mean_baseline == 0.0 or not np.isfinite(mean_baseline):
        percent_improvement = float("nan")
        notes = [*notes, "percent_improvement_undefined"]
    else:
        percent_improvement = 100.0 * mean_difference / mean_baseline

    return {
        **base,
        "n_pairs": int(len(differences)),
        "mean_baseline": mean_baseline,
        "mean_treatment": float(treatment_values.mean()),
        "mean_difference": mean_difference,
        "median_difference": float(differences.median()),
        "percent_improvement": percent_improvement,
        "bootstrap_ci_lower": bootstrap.ci_lower,
        "bootstrap_ci_upper": bootstrap.ci_upper,
        "bootstrap_confidence_level": bootstrap.confidence_level,
        "permutation_p_value": permutation.p_value,
        "baseline_budget_pass": _budget_pass(paired, "baseline"),
        "treatment_budget_pass": _budget_pass(paired, "treatment"),
        "baseline_mean_batch_latency_ms": _paired_mean(paired, "mean_batch_latency_ms_baseline"),
        "treatment_mean_batch_latency_ms": _paired_mean(paired, "mean_batch_latency_ms_treatment"),
        "baseline_p95_batch_latency_ms": _paired_mean(paired, "p95_batch_latency_ms_baseline"),
        "treatment_p95_batch_latency_ms": _paired_mean(paired, "p95_batch_latency_ms_treatment"),
        "baseline_mean_runtime_windows_per_second": _paired_mean(
            paired,
            "mean_runtime_windows_per_second_baseline",
        ),
        "treatment_mean_runtime_windows_per_second": _paired_mean(
            paired,
            "mean_runtime_windows_per_second_treatment",
        ),
        "notes": _join_notes(notes),
    }


def _empty_row(
    *,
    baseline_model: str,
    treatment_model: str,
    metric: str,
    aggregation_level: AggregationLevel,
    target_horizon_s: float,
    actual_horizon_s: float,
    horizon_error_seconds: float,
    n_permutations: int,
    n_bootstrap: int,
    alternative: Alternative,
    budget_mode: str,
    notes: list[str],
) -> dict[str, object]:
    return {
        "baseline_model": baseline_model,
        "treatment_model": treatment_model,
        "metric": metric,
        "aggregation_level": aggregation_level,
        "target_horizon_s": target_horizon_s,
        "actual_horizon_s": actual_horizon_s,
        "horizon_error_seconds": horizon_error_seconds,
        "n_pairs": 0,
        "mean_baseline": float("nan"),
        "mean_treatment": float("nan"),
        "mean_difference": float("nan"),
        "median_difference": float("nan"),
        "percent_improvement": float("nan"),
        "bootstrap_ci_lower": float("nan"),
        "bootstrap_ci_upper": float("nan"),
        "bootstrap_confidence_level": 0.95,
        "permutation_p_value": float("nan"),
        "permutation_alternative": alternative,
        "n_permutations": n_permutations,
        "n_bootstrap": n_bootstrap,
        "budget_mode": budget_mode,
        "baseline_budget_pass": False,
        "treatment_budget_pass": False,
        "baseline_mean_batch_latency_ms": float("nan"),
        "treatment_mean_batch_latency_ms": float("nan"),
        "baseline_p95_batch_latency_ms": float("nan"),
        "treatment_p95_batch_latency_ms": float("nan"),
        "baseline_mean_runtime_windows_per_second": float("nan"),
        "treatment_mean_runtime_windows_per_second": float("nan"),
        "correction_method": "",
        "corrected_p_value": float("nan"),
        "significant": False,
        "notes": _join_notes(notes),
    }


def _budget_pass(paired: pd.DataFrame, suffix: str) -> bool:
    column = f"budget_pass_{suffix}"
    if column not in paired.columns or paired.empty:
        return False
    return bool(paired[column].astype(bool).all())


def _paired_mean(paired: pd.DataFrame, column: str) -> float:
    if column not in paired.columns:
        return float("nan")
    return float(pd.to_numeric(paired[column], errors="coerce").mean())


def _join_notes(notes: Sequence[str]) -> str:
    return "; ".join(note for note in notes if note)


def _apply_p_value_correction(results: pd.DataFrame, *, correction_method: str) -> pd.DataFrame:
    corrected = results.copy()
    corrected["correction_method"] = correction_method
    corrected["corrected_p_value"] = np.nan
    corrected["significant"] = False
    valid = pd.to_numeric(corrected["permutation_p_value"], errors="coerce").notna()
    if not valid.any():
        return corrected
    if correction_method == "none":
        corrected.loc[valid, "corrected_p_value"] = corrected.loc[valid, "permutation_p_value"]
        corrected.loc[valid, "significant"] = corrected.loc[valid, "corrected_p_value"] <= 0.05
        return corrected

    holm = holm_correction(corrected.loc[valid, "permutation_p_value"].tolist())
    valid_indices = corrected.index[valid].tolist()
    for row_position, original_index in enumerate(valid_indices):
        corrected.loc[original_index, "corrected_p_value"] = holm.loc[row_position, "corrected_p_value"]
        corrected.loc[original_index, "significant"] = bool(holm.loc[row_position, "significant"])
    return corrected
