from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, Sequence

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from aggregation import aggregate_units
from loading import load_evaluation_units
from multiple_testing import bh_fdr_correction, bonferroni_correction, holm_correction
from pairing import pair_aggregated
from rq1 import _horizon_slices, _join_notes
from schema import AggregationLevel, PAIRING_KEYS_BY_LEVEL, SUPPORTED_COMPARISON_METRICS, validate_required_columns

"""
RQ2
"""

MarginType = Literal["absolute", "relative_to_real", "relative_to_baseline"]
CorrectionMethod = Literal["hierarchical", "holm", "bonferroni", "bh_fdr", "none"]


RQ2_RESULT_COLUMNS = [
    "real_model",
    "synthetic_model",
    "synthetic_size_fraction",
    "checkpoint_selection",
    "synthetic_checkpoint_path",
    "checkpoint_epoch",
    "checkpoint_step",
    "checkpoint_rule",
    "metric",
    "aggregation_level",
    "target_horizon_s",
    "actual_horizon_s",
    "horizon_error_seconds",
    "n_pairs",
    "mean_real",
    "mean_synthetic",
    "mean_diff_synthetic_minus_real",
    "median_diff_synthetic_minus_real",
    "relative_diff_percent",
    "margin_type",
    "margin_value",
    "delta",
    "alpha",
    "ci_level",
    "ci_lower",
    "ci_upper",
    "p_lower",
    "p_upper",
    "p_tost",
    "equivalent",
    "ci_equivalent",
    "correction_method",
    "corrected_p_value",
    "equivalent_adjusted",
    "gatekeeping_status",
    "budget_mode",
    "real_budget_pass",
    "synthetic_budget_pass",
    "real_mean_batch_latency_ms",
    "synthetic_mean_batch_latency_ms",
    "real_p95_batch_latency_ms",
    "synthetic_p95_batch_latency_ms",
    "real_mean_runtime_windows_per_second",
    "synthetic_mean_runtime_windows_per_second",
    "notes",
]

RQ2_THRESHOLD_SUMMARY_COLUMNS = [
    "metric",
    "aggregation_level",
    "target_horizon_s",
    "actual_horizon_s",
    "margin_type",
    "margin_value",
    "delta_reference",
    "correction_method",
    "alpha",
    "smallest_equivalent_fraction",
    "largest_tested_fraction",
    "gatekeeping_stopped_at_fraction",
    "n_tested_fractions",
    "n_equivalent_fractions",
    "notes",
]


@dataclass(frozen=True)
class TOSTResult:
    mean_diff: float
    median_diff: float
    std_diff: float
    se_diff: float
    df: int
    delta: float
    alpha: float
    ci_lower: float
    ci_upper: float
    ci_level: float
    p_lower: float
    p_upper: float
    p_tost: float
    equivalent: bool
    ci_equivalent: bool


@dataclass(frozen=True)
class RQ2Result:
    results: pd.DataFrame
    threshold_summary: pd.DataFrame
    results_output_path: Path | None = None
    threshold_output_path: Path | None = None


RQ2_REAL_PAIRING_LABEL = "rq2_real"
RQ2_SYNTHETIC_PAIRING_LABEL = "rq2_synthetic"
RQ2_CHECKPOINT_RULE = "last_curriculum_step_per_fraction"
CURRICULUM_CHECKPOINT_SELECTION = "curriculum_step"
CHECKPOINT_PATTERN = re.compile(r"pct\d+_epoch(?P<epoch>\d+)_step(?P<step>\d+)")


def paired_tost(differences: Sequence[float], *, delta: float, alpha: float = 0.05) -> TOSTResult:
    values = np.asarray(differences, dtype=float)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("at least one paired difference is required")
    if not np.isfinite(values).all():
        raise ValueError("paired differences must be finite")
    if delta <= 0 or not np.isfinite(delta):
        raise ValueError("delta must be a positive finite value")
    if not 0.0 < alpha < 0.5:
        raise ValueError("alpha must be between 0 and 0.5")

    n = len(values)
    df = n - 1
    mean_diff = float(np.mean(values))
    median_diff = float(np.median(values))
    std_diff = float(np.std(values, ddof=1)) if n > 1 else 0.0
    se_diff = std_diff / float(np.sqrt(n)) if n > 1 else 0.0
    ci_level = 1.0 - 2.0 * alpha

    if se_diff == 0.0:
        p_lower = 0.0 if mean_diff > -delta else 1.0
        p_upper = 0.0 if mean_diff < delta else 1.0
        ci_lower = mean_diff
        ci_upper = mean_diff
    else:
        t_lower = (mean_diff + delta) / se_diff
        t_upper = (mean_diff - delta) / se_diff
        p_lower = float(scipy_stats.t.sf(t_lower, df))
        p_upper = float(scipy_stats.t.cdf(t_upper, df))
        t_crit = float(scipy_stats.t.ppf(1.0 - alpha, df))
        ci_lower = mean_diff - t_crit * se_diff
        ci_upper = mean_diff + t_crit * se_diff

    p_tost = max(p_lower, p_upper)
    return TOSTResult(
        mean_diff=mean_diff,
        median_diff=median_diff,
        std_diff=std_diff,
        se_diff=se_diff,
        df=df,
        delta=delta,
        alpha=alpha,
        ci_lower=float(ci_lower),
        ci_upper=float(ci_upper),
        ci_level=ci_level,
        p_lower=p_lower,
        p_upper=p_upper,
        p_tost=p_tost,
        equivalent=p_lower < alpha and p_upper < alpha,
        ci_equivalent=ci_lower > -delta and ci_upper < delta,
    )


def compute_equivalence_margin(margin_type: str, margin_value: float, *, mean_real: float) -> float:
    if margin_type == "absolute":
        delta = float(margin_value)
    elif margin_type in ("relative_to_real", "relative_to_baseline"):
        delta = float(margin_value) * float(mean_real)
    else:
        raise ValueError("margin_type must be one of: absolute, relative_to_real, relative_to_baseline")
    if delta <= 0.0 or not np.isfinite(delta):
        raise ValueError("equivalence margin delta must be positive and finite")
    return delta


def run_rq2(
    data: pd.DataFrame | Iterable[str | Path],
    *,
    real_model: str,
    synthetic_models: Sequence[str],
    margin_type: str | None,
    margin_value: float | None,
    metrics: Sequence[str] = ("mean_ade_until_horizon", "mean_fde_at_horizon"),
    target_horizons: Sequence[float] | None = None,
    aggregation_level: AggregationLevel = "track",
    alpha: float = 0.05,
    correction_method: CorrectionMethod = "hierarchical",
    outdir: str | Path | None = None,
    budget_mode: str = "report",
) -> RQ2Result:
    if margin_type is None or margin_value is None:
        raise ValueError("margin_type and margin_value are required for RQ2 equivalence testing")
    _validate_inputs(metrics, correction_method, budget_mode)

    aggregated = _load_or_aggregate(data, level=aggregation_level)
    _validate_domain_metadata(
        aggregated,
        real_model=real_model,
        synthetic_models=synthetic_models,
    )
    rows: list[dict[str, object]] = []
    for horizon in _horizon_slices(aggregated, target_horizons):
        for synthetic_model in synthetic_models:
            for fraction in _synthetic_fractions(horizon.frame, synthetic_model):
                frame = _comparison_frame(
                    horizon.frame,
                    real_model=real_model,
                    synthetic_model=synthetic_model,
                    synthetic_size_fraction=fraction,
                    level=aggregation_level,
                )
                filtered, budget_notes = _apply_budget_mode(
                    frame,
                    real_model=real_model,
                    synthetic_model=synthetic_model,
                    budget_mode=budget_mode,
                )
                for metric in metrics:
                    rows.append(
                        _comparison_row(
                            filtered,
                            real_model=real_model,
                            synthetic_model=synthetic_model,
                            synthetic_size_fraction=fraction,
                            metric=metric,
                            aggregation_level=aggregation_level,
                            target_horizon_s=horizon.target_horizon_s,
                            actual_horizon_s=horizon.actual_horizon_s,
                            horizon_error_seconds=horizon.horizon_error_seconds,
                            margin_type=margin_type,
                            margin_value=float(margin_value),
                            alpha=alpha,
                            budget_mode=budget_mode,
                            notes=[*horizon.notes, *budget_notes],
                        ),
                    )

    results = pd.DataFrame(rows, columns=RQ2_RESULT_COLUMNS)
    results = _apply_correction(results, correction_method=correction_method, alpha=alpha)
    threshold_summary = _threshold_summary(
        results,
        aggregation_level=aggregation_level,
        margin_type=margin_type,
        margin_value=float(margin_value),
        correction_method=correction_method,
        alpha=alpha,
    )
    results = results[RQ2_RESULT_COLUMNS]
    threshold_summary = threshold_summary[RQ2_THRESHOLD_SUMMARY_COLUMNS]
    for column in ("equivalent", "ci_equivalent", "equivalent_adjusted", "real_budget_pass", "synthetic_budget_pass"):
        if column in results.columns:
            results[column] = results[column].astype(object)

    results_output_path = None
    threshold_output_path = None
    if outdir is not None:
        outdir_path = Path(outdir)
        outdir_path.mkdir(parents=True, exist_ok=True)
        results_output_path = outdir_path / "stats_rq2_tost_equivalence_results.csv"
        threshold_output_path = outdir_path / "stats_rq2_equivalence_threshold_summary.csv"
        results.to_csv(results_output_path, index=False)
        threshold_summary.to_csv(threshold_output_path, index=False)

    return RQ2Result(
        results=results,
        threshold_summary=threshold_summary,
        results_output_path=results_output_path,
        threshold_output_path=threshold_output_path,
    )


def _validate_inputs(metrics: Sequence[str], correction_method: str, budget_mode: str) -> None:
    unsupported = [metric for metric in metrics if metric not in SUPPORTED_COMPARISON_METRICS]
    if unsupported:
        raise ValueError(f"Unsupported RQ2 metric(s): {unsupported}")
    if correction_method not in ("hierarchical", "holm", "bonferroni", "bh_fdr", "none"):
        raise ValueError("correction_method must be hierarchical, holm, bonferroni, bh_fdr, or none")
    if budget_mode not in ("report", "strict"):
        raise ValueError("budget_mode must be 'report' or 'strict'")


def _load_or_aggregate(data: pd.DataFrame | Iterable[str | Path], *, level: AggregationLevel) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        frame = data.copy()
    else:
        frame = load_evaluation_units(data)
    if "mean_error_l2" in frame.columns:
        return frame
    return aggregate_units(frame, level=level)


def _validate_domain_metadata(
    df: pd.DataFrame,
    *,
    real_model: str,
    synthetic_models: Sequence[str],
) -> None:
    validate_required_columns(df, ["model_name", "train_domain", "test_domain"])
    model_mask = df["model_name"].eq(real_model)
    for synthetic_model in synthetic_models:
        model_mask |= df["model_name"].eq(synthetic_model)
    relevant = df[model_mask]
    if relevant.empty:
        return
    train_unknown = relevant["train_domain"].fillna("unknown").astype(str).eq("unknown")
    test_unknown = relevant["test_domain"].fillna("unknown").astype(str).eq("unknown")
    if bool((train_unknown | test_unknown).any()):
        raise ValueError(
            "RQ2 refuses inferential analysis with unknown train_domain/test_domain metadata; "
            "rerun exports with explicit TRAIN_DOMAIN and TEST_DOMAIN.",
        )


def _synthetic_fractions(df: pd.DataFrame, synthetic_model: str) -> list[float]:
    validate_required_columns(df, ["model_name", "train_domain", "test_domain", "synthetic_size_fraction"])
    mask = (
        df["model_name"].eq(synthetic_model)
        & df["train_domain"].eq("synthetic")
        & df["test_domain"].eq("real")
    )
    if "checkpoint_selection" in df.columns:
        mask &= df["checkpoint_selection"].eq(CURRICULUM_CHECKPOINT_SELECTION)
    values = pd.to_numeric(df.loc[mask, "synthetic_size_fraction"], errors="coerce")
    fractions = sorted(float(value) for value in values.dropna().unique())
    return fractions if fractions else [float("nan")]


def _comparison_frame(
    df: pd.DataFrame,
    *,
    real_model: str,
    synthetic_model: str,
    synthetic_size_fraction: float,
    level: AggregationLevel,
) -> pd.DataFrame:
    validate_required_columns(
        df,
        [
            "model_name",
            "train_domain",
            "test_domain",
            "synthetic_size_fraction",
            "checkpoint_selection",
            "checkpoint_path",
            "run_id",
            "fold",
            "sequence_id",
        ],
    )
    synthetic_fraction = pd.to_numeric(df["synthetic_size_fraction"], errors="coerce")
    synthetic_mask = (
        df["model_name"].eq(synthetic_model)
        & df["train_domain"].eq("synthetic")
        & df["test_domain"].eq("real")
        & df["checkpoint_selection"].eq(CURRICULUM_CHECKPOINT_SELECTION)
    )
    if np.isfinite(synthetic_size_fraction):
        synthetic_mask &= np.isclose(synthetic_fraction, synthetic_size_fraction, equal_nan=False)
    real_mask = (
        df["model_name"].eq(real_model)
        & df["train_domain"].eq("real")
        & df["test_domain"].eq("real")
    )

    real = df[real_mask].copy()
    synthetic = _last_curriculum_checkpoint_rows(df[synthetic_mask].copy())
    _assert_unique_synthetic_pairing_keys(synthetic, level=level)

    real["original_model_name"] = real["model_name"]
    synthetic["original_model_name"] = synthetic["model_name"]
    real["model_name"] = RQ2_REAL_PAIRING_LABEL
    synthetic["model_name"] = RQ2_SYNTHETIC_PAIRING_LABEL
    return pd.concat([real, synthetic], ignore_index=True, sort=False)


def _last_curriculum_checkpoint_rows(synthetic: pd.DataFrame) -> pd.DataFrame:
    output = synthetic.copy()
    if output.empty:
        output["checkpoint_epoch"] = pd.Series(dtype="float64")
        output["checkpoint_step"] = pd.Series(dtype="float64")
        return output

    parsed = output.apply(_parse_checkpoint_epoch_step, axis=1, result_type="expand")
    output["checkpoint_epoch"] = pd.to_numeric(parsed[0], errors="raise")
    output["checkpoint_step"] = pd.to_numeric(parsed[1], errors="raise")

    group_keys = ["fold", "sequence_id", "synthetic_size_fraction"]
    max_step = output.groupby(group_keys, dropna=False)["checkpoint_step"].transform("max")
    selected = output[output["checkpoint_step"].eq(max_step)].copy()

    checkpoint_counts = selected.groupby(group_keys, dropna=False)["checkpoint_path"].nunique(dropna=False)
    ambiguous = checkpoint_counts[checkpoint_counts > 1]
    if not ambiguous.empty:
        raise ValueError(
            "RQ2 final curriculum checkpoint selection is ambiguous; multiple checkpoint_path values share "
            f"the largest checkpoint_step for group(s): {_group_diagnostic(ambiguous)}",
        )

    return selected


def _parse_checkpoint_epoch_step(row: pd.Series) -> tuple[int, int]:
    existing_epoch = pd.to_numeric(pd.Series([row.get("checkpoint_epoch")]), errors="coerce").iloc[0]
    existing_step = pd.to_numeric(pd.Series([row.get("checkpoint_step")]), errors="coerce").iloc[0]
    if pd.notna(existing_epoch) and pd.notna(existing_step):
        return int(existing_epoch), int(existing_step)

    for column in ("run_id", "checkpoint_path"):
        value = row.get(column)
        if pd.isna(value):
            continue
        match = CHECKPOINT_PATTERN.search(str(value))
        if match:
            return int(match.group("epoch")), int(match.group("step"))

    raise ValueError(
        "RQ2 curriculum checkpoint rows must include parseable checkpoint epoch/step metadata "
        "in run_id or checkpoint_path, e.g. pct005_epoch003_step009543.",
    )


def _assert_unique_synthetic_pairing_keys(synthetic: pd.DataFrame, *, level: AggregationLevel) -> None:
    if synthetic.empty:
        return
    keys = PAIRING_KEYS_BY_LEVEL[level]
    duplicated = synthetic[synthetic.duplicated(keys, keep=False)]
    if duplicated.empty:
        return
    diagnostic_columns = [
        *keys,
        "synthetic_size_fraction",
        "checkpoint_path",
        "checkpoint_epoch",
        "checkpoint_step",
    ]
    sample = duplicated[diagnostic_columns].head(5).to_dict(orient="records")
    raise ValueError(
        "RQ2 final curriculum checkpoint filtering left duplicate synthetic pairing keys; "
        f"duplicate_key_groups={len(duplicated[keys].drop_duplicates())}; sample={sample}",
    )


def _group_diagnostic(values: pd.Series) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for key, count in values.head(5).items():
        if not isinstance(key, tuple):
            key = (key,)
        output.append(
            {
                "fold": key[0],
                "sequence_id": key[1],
                "synthetic_size_fraction": key[2],
                "checkpoint_count": int(count),
            },
        )
    return output


def _apply_budget_mode(
    df: pd.DataFrame,
    *,
    real_model: str,
    synthetic_model: str,
    budget_mode: str,
) -> tuple[pd.DataFrame, list[str]]:
    if budget_mode != "strict":
        return df, []
    validate_required_columns(df, ["model_name", "budget_pass"])
    budget = df["budget_pass"].astype(bool)
    real_excluded = int((df["model_name"].eq(RQ2_REAL_PAIRING_LABEL) & ~budget).sum())
    synthetic_excluded = int((df["model_name"].eq(RQ2_SYNTHETIC_PAIRING_LABEL) & ~budget).sum())
    return df[budget].copy(), [
        f"budget_excluded_real_rows={real_excluded}",
        f"budget_excluded_synthetic_rows={synthetic_excluded}",
    ]


def _comparison_row(
    df: pd.DataFrame,
    *,
    real_model: str,
    synthetic_model: str,
    synthetic_size_fraction: float,
    metric: str,
    aggregation_level: AggregationLevel,
    target_horizon_s: float,
    actual_horizon_s: float,
    horizon_error_seconds: float,
    margin_type: str,
    margin_value: float,
    alpha: float,
    budget_mode: str,
    notes: list[str],
) -> dict[str, object]:
    base = _empty_row(
        real_model=real_model,
        synthetic_model=synthetic_model,
        synthetic_size_fraction=synthetic_size_fraction,
        metric=metric,
        aggregation_level=aggregation_level,
        target_horizon_s=target_horizon_s,
        actual_horizon_s=actual_horizon_s,
        horizon_error_seconds=horizon_error_seconds,
        margin_type=margin_type,
        margin_value=margin_value,
        alpha=alpha,
        budget_mode=budget_mode,
        notes=notes,
    )
    paired_result = pair_aggregated(
        df,
        baseline_model=RQ2_REAL_PAIRING_LABEL,
        treatment_model=RQ2_SYNTHETIC_PAIRING_LABEL,
        level=aggregation_level,
    )
    if paired_result.report["n_pairs"] == 0:
        base["notes"] = _join_notes([*notes, "no_paired_units", f"pairing_report={paired_result.report}"])
        return base

    paired = paired_result.paired
    real_values = pd.to_numeric(paired[f"{metric}_baseline"], errors="coerce")
    synthetic_values = pd.to_numeric(paired[f"{metric}_treatment"], errors="coerce")
    differences = synthetic_values - real_values
    finite_mask = real_values.notna() & synthetic_values.notna() & differences.notna()
    if not finite_mask.all():
        paired = paired[finite_mask].copy()
        real_values = real_values[finite_mask]
        synthetic_values = synthetic_values[finite_mask]
        differences = differences[finite_mask]
        notes = [*notes, f"dropped_nonfinite_pairs={int((~finite_mask).sum())}"]
    if differences.empty:
        base["notes"] = _join_notes([*notes, "no_finite_paired_differences"])
        return base

    mean_real = float(real_values.mean())
    mean_synthetic = float(synthetic_values.mean())
    delta = compute_equivalence_margin(margin_type, margin_value, mean_real=mean_real)
    tost = paired_tost(differences.to_numpy(), delta=delta, alpha=alpha)
    relative_diff = float("nan") if mean_real == 0.0 else 100.0 * tost.mean_diff / mean_real

    return {
        **base,
        "n_pairs": int(len(differences)),
        "mean_real": mean_real,
        "mean_synthetic": mean_synthetic,
        "mean_diff_synthetic_minus_real": tost.mean_diff,
        "median_diff_synthetic_minus_real": tost.median_diff,
        "relative_diff_percent": relative_diff,
        "delta": delta,
        "ci_level": tost.ci_level,
        "ci_lower": tost.ci_lower,
        "ci_upper": tost.ci_upper,
        "p_lower": tost.p_lower,
        "p_upper": tost.p_upper,
        "p_tost": tost.p_tost,
        "equivalent": tost.equivalent,
        "ci_equivalent": tost.ci_equivalent,
        "real_budget_pass": _budget_pass(paired, "baseline"),
        "synthetic_budget_pass": _budget_pass(paired, "treatment"),
        "real_mean_batch_latency_ms": _paired_mean(paired, "mean_batch_latency_ms_baseline"),
        "synthetic_mean_batch_latency_ms": _paired_mean(paired, "mean_batch_latency_ms_treatment"),
        "real_p95_batch_latency_ms": _paired_mean(paired, "p95_batch_latency_ms_baseline"),
        "synthetic_p95_batch_latency_ms": _paired_mean(paired, "p95_batch_latency_ms_treatment"),
        "real_mean_runtime_windows_per_second": _paired_mean(paired, "mean_runtime_windows_per_second_baseline"),
        "synthetic_mean_runtime_windows_per_second": _paired_mean(
            paired,
            "mean_runtime_windows_per_second_treatment",
        ),
        "checkpoint_selection": _paired_unique_text(paired, "checkpoint_selection_treatment"),
        "synthetic_checkpoint_path": _paired_unique_text(paired, "checkpoint_path_treatment"),
        "checkpoint_epoch": _paired_unique_number(paired, "checkpoint_epoch_treatment"),
        "checkpoint_step": _paired_unique_number(paired, "checkpoint_step_treatment"),
        "checkpoint_rule": RQ2_CHECKPOINT_RULE,
        "notes": _join_notes(notes),
    }


def _empty_row(
    *,
    real_model: str,
    synthetic_model: str,
    synthetic_size_fraction: float,
    metric: str,
    aggregation_level: AggregationLevel,
    target_horizon_s: float,
    actual_horizon_s: float,
    horizon_error_seconds: float,
    margin_type: str,
    margin_value: float,
    alpha: float,
    budget_mode: str,
    notes: list[str],
) -> dict[str, object]:
    return {
        "real_model": real_model,
        "synthetic_model": synthetic_model,
        "synthetic_size_fraction": synthetic_size_fraction,
        "checkpoint_selection": "",
        "synthetic_checkpoint_path": "",
        "checkpoint_epoch": float("nan"),
        "checkpoint_step": float("nan"),
        "checkpoint_rule": RQ2_CHECKPOINT_RULE,
        "metric": metric,
        "aggregation_level": aggregation_level,
        "target_horizon_s": target_horizon_s,
        "actual_horizon_s": actual_horizon_s,
        "horizon_error_seconds": horizon_error_seconds,
        "n_pairs": 0,
        "mean_real": float("nan"),
        "mean_synthetic": float("nan"),
        "mean_diff_synthetic_minus_real": float("nan"),
        "median_diff_synthetic_minus_real": float("nan"),
        "relative_diff_percent": float("nan"),
        "margin_type": margin_type,
        "margin_value": margin_value,
        "delta": float("nan"),
        "alpha": alpha,
        "ci_level": 1.0 - 2.0 * alpha,
        "ci_lower": float("nan"),
        "ci_upper": float("nan"),
        "p_lower": float("nan"),
        "p_upper": float("nan"),
        "p_tost": float("nan"),
        "equivalent": False,
        "ci_equivalent": False,
        "correction_method": "",
        "corrected_p_value": float("nan"),
        "equivalent_adjusted": False,
        "gatekeeping_status": "",
        "budget_mode": budget_mode,
        "real_budget_pass": False,
        "synthetic_budget_pass": False,
        "real_mean_batch_latency_ms": float("nan"),
        "synthetic_mean_batch_latency_ms": float("nan"),
        "real_p95_batch_latency_ms": float("nan"),
        "synthetic_p95_batch_latency_ms": float("nan"),
        "real_mean_runtime_windows_per_second": float("nan"),
        "synthetic_mean_runtime_windows_per_second": float("nan"),
        "notes": _join_notes(notes),
    }


def _apply_correction(results: pd.DataFrame, *, correction_method: str, alpha: float) -> pd.DataFrame:
    corrected = results.copy()
    corrected["correction_method"] = correction_method
    corrected["corrected_p_value"] = np.nan
    corrected["equivalent_adjusted"] = False
    corrected["gatekeeping_status"] = "not_applicable"
    if corrected.empty:
        return corrected
    if correction_method == "hierarchical":
        return _apply_hierarchical(corrected)
    valid = pd.to_numeric(corrected["p_tost"], errors="coerce").notna()
    if correction_method == "none":
        corrected.loc[valid, "corrected_p_value"] = corrected.loc[valid, "p_tost"]
        corrected.loc[valid, "equivalent_adjusted"] = corrected.loc[valid, "equivalent"]
        return corrected

    p_values = corrected.loc[valid, "p_tost"].tolist()
    if correction_method == "holm":
        adjusted = holm_correction(p_values, alpha=alpha)
    elif correction_method == "bonferroni":
        adjusted = bonferroni_correction(p_values, alpha=alpha)
    else:
        adjusted = bh_fdr_correction(p_values, alpha=alpha)
        corrected.loc[:, "notes"] = corrected["notes"].apply(lambda note: _join_notes([note, "exploratory_bh_fdr"]))
    valid_indices = corrected.index[valid].tolist()
    for row_position, index in enumerate(valid_indices):
        corrected.loc[index, "corrected_p_value"] = adjusted.loc[row_position, "corrected_p_value"]
        corrected.loc[index, "equivalent_adjusted"] = bool(adjusted.loc[row_position, "significant"])
    return corrected


def _apply_hierarchical(results: pd.DataFrame) -> pd.DataFrame:
    corrected = results.copy()
    group_keys = ["synthetic_model", "metric", "target_horizon_s", "actual_horizon_s"]
    for _, group in corrected.groupby(group_keys, dropna=False, sort=True):
        group = group.sort_values("synthetic_size_fraction", ascending=False, kind="mergesort")
        gate_open = True
        for index, row in group.iterrows():
            corrected.loc[index, "corrected_p_value"] = row["p_tost"]
            if not gate_open:
                corrected.loc[index, "gatekeeping_status"] = "not_tested_due_to_gatekeeping"
                corrected.loc[index, "equivalent_adjusted"] = False
                continue
            if bool(row["equivalent"]):
                corrected.loc[index, "gatekeeping_status"] = "tested_equivalent"
                corrected.loc[index, "equivalent_adjusted"] = True
            else:
                corrected.loc[index, "gatekeeping_status"] = "tested_not_equivalent_stop"
                corrected.loc[index, "equivalent_adjusted"] = False
                gate_open = False
    return corrected


def _threshold_summary(
    results: pd.DataFrame,
    *,
    aggregation_level: AggregationLevel,
    margin_type: str,
    margin_value: float,
    correction_method: str,
    alpha: float,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if results.empty:
        return pd.DataFrame(rows, columns=RQ2_THRESHOLD_SUMMARY_COLUMNS)
    group_keys = ["metric", "target_horizon_s", "actual_horizon_s"]
    for (metric, target, actual), group in results.groupby(group_keys, dropna=False, sort=True):
        tested = group[~group["gatekeeping_status"].eq("not_tested_due_to_gatekeeping")]
        equivalent = group[group["equivalent_adjusted"].astype(bool)]
        stopped = group[group["gatekeeping_status"].eq("tested_not_equivalent_stop")]
        rows.append(
            {
                "metric": metric,
                "aggregation_level": aggregation_level,
                "target_horizon_s": target,
                "actual_horizon_s": actual,
                "margin_type": margin_type,
                "margin_value": margin_value,
                "delta_reference": "mean_real" if margin_type in ("relative_to_real", "relative_to_baseline") else "absolute",
                "correction_method": correction_method,
                "alpha": alpha,
                "smallest_equivalent_fraction": (
                    float(equivalent["synthetic_size_fraction"].min()) if not equivalent.empty else float("nan")
                ),
                "largest_tested_fraction": (
                    float(tested["synthetic_size_fraction"].max()) if not tested.empty else float("nan")
                ),
                "gatekeeping_stopped_at_fraction": (
                    float(stopped["synthetic_size_fraction"].iloc[0]) if not stopped.empty else float("nan")
                ),
                "n_tested_fractions": int(len(tested)),
                "n_equivalent_fractions": int(len(equivalent)),
                "notes": "",
            },
        )
    return pd.DataFrame(rows, columns=RQ2_THRESHOLD_SUMMARY_COLUMNS)


def _budget_pass(paired: pd.DataFrame, suffix: str) -> bool:
    column = f"budget_pass_{suffix}"
    if column not in paired.columns or paired.empty:
        return False
    return bool(paired[column].astype(bool).all())


def _paired_mean(paired: pd.DataFrame, column: str) -> float:
    if column not in paired.columns:
        return float("nan")
    return float(pd.to_numeric(paired[column], errors="coerce").mean())


def _paired_unique_text(paired: pd.DataFrame, column: str) -> str:
    if column not in paired.columns:
        return ""
    values = [str(value) for value in paired[column].dropna().unique().tolist() if str(value).strip()]
    if not values:
        return ""
    return values[0] if len(values) == 1 else ";".join(sorted(values))


def _paired_unique_number(paired: pd.DataFrame, column: str) -> float | int:
    if column not in paired.columns:
        return float("nan")
    values = pd.to_numeric(paired[column], errors="coerce").dropna().unique()
    if len(values) != 1:
        return float("nan")
    value = float(values[0])
    return int(value) if value.is_integer() else value
