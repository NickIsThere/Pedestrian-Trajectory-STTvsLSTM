from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from aggregation import aggregate_units
from loading import load_evaluation_units
from multiple_testing import holm_correction
from rq1 import _horizon_slices, _join_notes
from schema import AggregationLevel, SUPPORTED_COMPARISON_METRICS, validate_required_columns

"""
RQ3
"""

REQUIRED_CELLS = (("real", "real"), ("real", "synthetic"), ("synthetic", "real"), ("synthetic", "synthetic"))
OPTIONAL_CELLS = (("mixed", "real"), ("mixed", "synthetic"))
_UNPAIRED_LIMITATION = (
    "The current implementation permutes train_domain/test_domain labels without blocking by "
    "track, sequence, or fold; it is not a repeated-measures design."
)

RQ3_READINESS_COLUMNS = [
    "train_domain",
    "test_domain",
    "required_cell",
    "n_rows",
    "n_units",
    "model_names",
    "model_families",
    "model_variants",
    "metrics_available",
    "target_horizons_available",
    "actual_horizons_available",
    "status",
    "notes",
]

RQ3_DOMAIN_SUMMARY_COLUMNS = [
    "metric",
    "aggregation_level",
    "target_horizon_s",
    "actual_horizon_s",
    "horizon_error_seconds",
    "train_domain",
    "test_domain",
    "n_units",
    "mean_error",
    "median_error",
    "std_error",
    "se_error",
    "ci_lower",
    "ci_upper",
    "ci_method",
    "mean_batch_latency_ms",
    "p95_batch_latency_ms",
    "mean_runtime_windows_per_second",
    "budget_pass_rate",
]

RQ3_FACTORIAL_COLUMNS = [
    "metric",
    "aggregation_level",
    "target_horizon_s",
    "actual_horizon_s",
    "horizon_error_seconds",
    "effect",
    "statistic",
    "p_value",
    "correction_method",
    "corrected_p_value",
    "significant",
    "n_permutations",
    "n_units",
    "method",
    "permutation_scheme",
    "budget_mode",
    "notes",
]

RQ3_POSTHOC_COLUMNS = [
    "metric",
    "aggregation_level",
    "target_horizon_s",
    "actual_horizon_s",
    "horizon_error_seconds",
    "comparison",
    "group_a",
    "group_b",
    "mean_a",
    "mean_b",
    "mean_difference",
    "percent_difference_relative_to_a",
    "ci_lower",
    "ci_upper",
    "ci_method",
    "comparison_type",
    "n_a",
    "n_b",
    "n_pairs",
    "notes",
]


@dataclass(frozen=True)
class RQ3Result:
    readiness_report: pd.DataFrame
    readiness_summary: dict[str, object]
    domain_summary: pd.DataFrame
    factorial_results: pd.DataFrame
    posthoc_results: pd.DataFrame
    factorial_input: pd.DataFrame


@dataclass(frozen=True)
class RQ3ReportResult:
    rq3_result: RQ3Result
    summary: dict[str, object]
    aggregated_input_path: Path
    summary_json_path: Path
    summary_markdown_path: Path


@dataclass(frozen=True)
class RQ3FinalSetup:
    main_horizons: tuple[float, ...] = (1.0, 2.0)
    synthetic_size_fraction: float = 1.0
    synthetic_checkpoint_selection: str = "curriculum_step"
    synthetic_checkpoint_rule: str = "last_curriculum_step_per_fraction"


def run_rq3(
    data: pd.DataFrame | Iterable[str | Path],
    *,
    metrics: Sequence[str] = ("mean_ade_until_horizon", "mean_fde_at_horizon"),
    target_horizons: Sequence[float] | None = None,
    aggregation_level: AggregationLevel = "track",
    outdir: str | Path | None = None,
    n_permutations: int = 10000,
    random_seed: int = 42,
    budget_mode: str = "report",
    config: RQ3FinalSetup = RQ3FinalSetup(),
) -> RQ3Result:
    _validate_inputs(metrics, budget_mode)
    aggregated = _load_or_aggregate(data, level=aggregation_level)
    aggregated = prepare_final_rq3_input(aggregated, config=config)
    target_horizons = config.main_horizons
    readiness_report, readiness_summary = build_readiness_report(aggregated)
    final_checks = build_final_rq3_readiness_checks(aggregated, config=config)
    readiness_summary["final_setup_checks"] = final_checks
    readiness_summary["method_limitation"] = _UNPAIRED_LIMITATION
    if readiness_summary["rq3_ready"] and not final_checks["ready"]:
        readiness_summary["rq3_ready"] = False
        readiness_summary["reason"] = "final RQ3 setup checks failed: " + "; ".join(final_checks["issues"])
    filtered, budget_notes = _apply_budget_mode(aggregated, budget_mode=budget_mode)
    domain_summary = build_domain_summary(
        filtered,
        metrics=metrics,
        target_horizons=target_horizons,
        aggregation_level=aggregation_level,
    )
    if readiness_summary["rq3_ready"]:
        factorial_results, factorial_input = build_factorial_results(
            filtered,
            metrics=metrics,
            target_horizons=target_horizons,
            aggregation_level=aggregation_level,
            n_permutations=n_permutations,
            random_seed=random_seed,
            budget_mode=budget_mode,
            notes=budget_notes,
        )
        factorial_results = _apply_holm_correction(factorial_results)
        posthoc_results = build_posthoc_results(
            filtered,
            metrics=metrics,
            target_horizons=target_horizons,
            aggregation_level=aggregation_level,
        )
    else:
        factorial_results = pd.DataFrame(columns=RQ3_FACTORIAL_COLUMNS)
        posthoc_results = pd.DataFrame(columns=RQ3_POSTHOC_COLUMNS)
        factorial_input = pd.DataFrame()
    result = RQ3Result(
        readiness_report=readiness_report[RQ3_READINESS_COLUMNS],
        readiness_summary=readiness_summary,
        domain_summary=domain_summary[RQ3_DOMAIN_SUMMARY_COLUMNS],
        factorial_results=factorial_results[RQ3_FACTORIAL_COLUMNS],
        posthoc_results=posthoc_results[RQ3_POSTHOC_COLUMNS],
        factorial_input=factorial_input,
    )
    if outdir is not None:
        _write_outputs(result, Path(outdir))
    return result


def run_rq3_report(
    data: pd.DataFrame | Iterable[str | Path],
    *,
    outdir: str | Path,
    metrics: Sequence[str] = ("mean_ade_until_horizon", "mean_fde_at_horizon"),
    aggregation_level: AggregationLevel = "track",
    n_permutations: int = 10000,
    random_seed: int = 42,
    budget_mode: str = "report",
    config: RQ3FinalSetup = RQ3FinalSetup(),
) -> RQ3ReportResult:
    input_paths: list[str] = []
    if isinstance(data, pd.DataFrame):
        source = data.copy()
    else:
        materialized = [Path(path) for path in data]
        input_paths = [str(path) for path in materialized]
        source = _load_report_input_csvs(materialized)

    aggregated = _load_or_aggregate(source, level=aggregation_level)
    prepared = prepare_final_rq3_input(aggregated, config=config)
    output_dir = Path(outdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    aggregated_path = output_dir / f"aggregated_{aggregation_level}.csv"
    prepared.to_csv(aggregated_path, index=False)

    rq3_result = run_rq3(
        aggregated,
        metrics=metrics,
        target_horizons=config.main_horizons,
        aggregation_level=aggregation_level,
        outdir=output_dir / "rq3",
        n_permutations=n_permutations,
        random_seed=random_seed,
        budget_mode=budget_mode,
        config=config,
    )
    summary = _final_report_summary(
        input_paths=input_paths,
        aggregated_input_path=aggregated_path,
        prepared=prepared,
        config=config,
        metrics=metrics,
        aggregation_level=aggregation_level,
        rq3_result=rq3_result,
    )
    summary_json_path = output_dir / "stats_pipeline_summary.json"
    summary_markdown_path = output_dir / "stats_pipeline_summary.md"
    summary_json_path.write_text(json.dumps(summary, indent=2, default=_json_default), encoding="utf-8")
    summary_markdown_path.write_text(_final_report_markdown(summary), encoding="utf-8")
    return RQ3ReportResult(
        rq3_result=rq3_result,
        summary=summary,
        aggregated_input_path=aggregated_path,
        summary_json_path=summary_json_path,
        summary_markdown_path=summary_markdown_path,
    )


def prepare_final_rq3_input(df: pd.DataFrame, *, config: RQ3FinalSetup = RQ3FinalSetup()) -> pd.DataFrame:
    frame = df.copy()
    validate_required_columns(frame, ["train_domain", "test_domain", "target_horizon_s", "synthetic_size_fraction", "checkpoint_selection"])
    frame = frame[
        frame["train_domain"].isin(["real", "synthetic"])
        & frame["test_domain"].isin(["real", "synthetic"])
    ].copy()
    targets = pd.to_numeric(frame["target_horizon_s"], errors="coerce")
    horizon_mask = np.zeros(len(frame), dtype=bool)
    for target in config.main_horizons:
        horizon_mask |= np.isclose(targets, float(target), equal_nan=False)
    frame = frame[horizon_mask].copy()

    synthetic_mask = frame["train_domain"] == "synthetic"
    synthetic = frame[synthetic_mask].copy()
    if not synthetic.empty:
        fractions = pd.to_numeric(synthetic["synthetic_size_fraction"], errors="coerce")
        synthetic = synthetic[
            np.isclose(fractions, config.synthetic_size_fraction, equal_nan=False)
            & (synthetic["checkpoint_selection"].astype(str) == config.synthetic_checkpoint_selection)
        ].copy()
        synthetic = _apply_synthetic_checkpoint_rule(synthetic, config)
        synthetic["checkpoint_rule"] = config.synthetic_checkpoint_rule

    real = frame[~synthetic_mask].copy()
    if "checkpoint_rule" not in real.columns:
        real["checkpoint_rule"] = ""
    elif "checkpoint_rule" not in synthetic.columns:
        synthetic["checkpoint_rule"] = config.synthetic_checkpoint_rule
    return pd.concat([real, synthetic], ignore_index=True)


def _load_report_input_csvs(paths: Sequence[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in paths:
        frame = pd.read_csv(path)
        frame["source_file"] = str(path)
        for column in [
            "synthetic_size_fraction",
            "fold",
            "track_id",
            "horizon_idx",
            "horizon_s",
            "target_horizon_s",
            "actual_horizon_s",
            "horizon_error_seconds",
            "seed",
            "mean_error_l2",
            "mean_ade_until_horizon",
            "mean_fde_at_horizon",
            "n_rows",
            "mean_batch_latency_ms",
            "p95_batch_latency_ms",
            "mean_runtime_windows_per_second",
            "budget_pass",
        ]:
            if column in frame.columns and column != "budget_pass":
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if "budget_pass" in frame.columns:
            frame["budget_pass"] = frame["budget_pass"].astype(str).str.lower().isin({"true", "1", "yes"})
        frames.append(frame)
    if not frames:
        raise ValueError("At least one RQ3 input CSV is required")
    return pd.concat(frames, ignore_index=True)


def build_final_rq3_readiness_checks(
    df: pd.DataFrame,
    *,
    config: RQ3FinalSetup = RQ3FinalSetup(),
) -> dict[str, object]:
    checks: dict[str, object] = {}
    issues: list[str] = []
    checks["main_horizons"] = _final_main_horizon_check(df, config, issues)
    checks["required_cells"] = _final_required_cells_check(df, issues)
    checks["synthetic_checkpoint"] = _final_synthetic_checkpoint_check(df, config, issues)
    checks["horizon_alignment"] = _final_horizon_alignment_check(df, config, issues)
    return {"ready": not issues, "issues": issues, "checks": checks}


def _apply_synthetic_checkpoint_rule(df: pd.DataFrame, final_setup: RQ3FinalSetup) -> pd.DataFrame:
    if df.empty:
        return df
    if final_setup.synthetic_checkpoint_rule != "last_curriculum_step_per_fraction":
        return df.copy()
    step = df.apply(_curriculum_step_number, axis=1)
    if step.notna().any():
        max_step = step.max()
        return df[step == max_step].copy()
    return df.copy()


def _curriculum_step_number(row: pd.Series) -> float:
    for column in ("run_id", "checkpoint_path"):
        value = str(row.get(column, ""))
        match = re.search(r"step(\d+)", value)
        if match:
            return float(match.group(1))
    return float("nan")


def _final_main_horizon_check(df: pd.DataFrame, final_setup: RQ3FinalSetup, issues: list[str]) -> dict[str, object]:
    validate_required_columns(df, ["target_horizon_s"])
    targets = sorted(float(value) for value in pd.to_numeric(df["target_horizon_s"], errors="coerce").dropna().unique())
    expected = sorted(float(value) for value in final_setup.main_horizons)
    includes_half_second = any(np.isclose(targets, 0.5))
    missing = [target for target in expected if not any(np.isclose(target, value) for value in targets)]
    extra = [value for value in targets if not any(np.isclose(value, target) for target in expected)]
    passed = not includes_half_second and not missing and not extra
    if includes_half_second:
        issues.append("0.5s target horizon is included in final RQ3 input")
    if missing:
        issues.append(f"missing final RQ3 main target horizons: {missing}")
    if extra:
        issues.append(f"unexpected target horizons in final RQ3 input: {extra}")
    return {
        "status": "pass" if passed else "fail",
        "expected_target_horizons": expected,
        "observed_target_horizons": targets,
        "excluded_0p5": not includes_half_second,
        "missing_target_horizons": missing,
        "unexpected_target_horizons": extra,
    }


def _final_required_cells_check(df: pd.DataFrame, issues: list[str]) -> dict[str, object]:
    validate_required_columns(df, ["train_domain", "test_domain"])
    n_units: dict[str, int] = {}
    missing: list[str] = []
    for train_domain, test_domain in REQUIRED_CELLS:
        key = _cell_key(train_domain, test_domain)
        count = int(((df["train_domain"] == train_domain) & (df["test_domain"] == test_domain)).sum())
        n_units[key] = count
        if count == 0:
            missing.append(key)
    if missing:
        issues.append(f"missing required train_domain/test_domain cells: {missing}")
    return {
        "status": "pass" if not missing else "fail",
        "n_units_per_cell": n_units,
        "missing_cells": missing,
    }


def _final_synthetic_checkpoint_check(
    df: pd.DataFrame,
    final_setup: RQ3FinalSetup,
    issues: list[str],
) -> dict[str, object]:
    validate_required_columns(df, ["train_domain", "synthetic_size_fraction", "checkpoint_selection"])
    synthetic = df[df["train_domain"] == "synthetic"].copy()
    if synthetic.empty:
        issues.append("no synthetic train-domain rows remain after final RQ3 filtering")
        return {
            "status": "fail",
            "n_units": 0,
            "synthetic_size_fraction_values": [],
            "checkpoint_selection_values": [],
            "checkpoint_rule_values": [],
            "run_ids": [],
            "checkpoint_ids": [],
            "raw_checkpoint_paths": [],
            "per_fold_best_rows": 0,
        }

    fractions = sorted(float(value) for value in pd.to_numeric(synthetic["synthetic_size_fraction"], errors="coerce").dropna().unique())
    selections = sorted(str(value) for value in synthetic["checkpoint_selection"].dropna().unique())
    rules = sorted(str(value) for value in synthetic.get("checkpoint_rule", pd.Series(dtype=str)).dropna().unique())
    run_ids = sorted(str(value) for value in synthetic.get("run_id", pd.Series(dtype=str)).dropna().unique())
    raw_paths = sorted(str(value) for value in synthetic.get("checkpoint_path", pd.Series(dtype=str)).dropna().unique())
    checkpoint_ids = sorted({_checkpoint_identity(path) for path in raw_paths if path})
    per_fold_best_rows = int((synthetic["checkpoint_selection"].astype(str) == "per_fold_best").sum())

    passed = True
    if fractions != [float(final_setup.synthetic_size_fraction)]:
        passed = False
        issues.append(f"synthetic train-domain rows do not contain only fraction {final_setup.synthetic_size_fraction}: {fractions}")
    if selections != [final_setup.synthetic_checkpoint_selection]:
        passed = False
        issues.append(
            "synthetic train-domain rows do not contain only checkpoint_selection="
            f"{final_setup.synthetic_checkpoint_selection}: {selections}",
        )
    if per_fold_best_rows:
        passed = False
        issues.append(f"synthetic train-domain rows include per_fold_best rows: {per_fold_best_rows}")
    if rules and rules != [final_setup.synthetic_checkpoint_rule]:
        passed = False
        issues.append(
            "synthetic train-domain rows do not contain only checkpoint_rule="
            f"{final_setup.synthetic_checkpoint_rule}: {rules}",
        )
    if len(run_ids) != 1:
        passed = False
        issues.append(f"synthetic train-domain rows do not identify exactly one run_id: {run_ids}")
    if len(checkpoint_ids) != 1:
        passed = False
        issues.append(f"synthetic train-domain rows do not identify exactly one checkpoint: {checkpoint_ids}")

    return {
        "status": "pass" if passed else "fail",
        "n_units": int(len(synthetic)),
        "synthetic_size_fraction_values": fractions,
        "checkpoint_selection_values": selections,
        "checkpoint_rule_values": rules,
        "run_ids": run_ids,
        "checkpoint_ids": checkpoint_ids,
        "raw_checkpoint_paths": raw_paths,
        "per_fold_best_rows": per_fold_best_rows,
    }


def _final_horizon_alignment_check(
    df: pd.DataFrame,
    final_setup: RQ3FinalSetup,
    issues: list[str],
) -> dict[str, object]:
    validate_required_columns(df, ["train_domain", "test_domain", "target_horizon_s", "actual_horizon_s", "horizon_idx"])
    targets = pd.to_numeric(df["target_horizon_s"], errors="coerce")
    alignment: dict[str, object] = {}
    passed = True
    for target in final_setup.main_horizons:
        target_frame = df[np.isclose(targets, float(target), equal_nan=False)]
        cells: dict[str, object] = {}
        actual_union: set[float] = set()
        for train_domain, test_domain in REQUIRED_CELLS:
            cell = target_frame[(target_frame["train_domain"] == train_domain) & (target_frame["test_domain"] == test_domain)]
            actuals = sorted(float(value) for value in pd.to_numeric(cell["actual_horizon_s"], errors="coerce").dropna().unique())
            horizon_indices = sorted(int(value) for value in pd.to_numeric(cell["horizon_idx"], errors="coerce").dropna().unique())
            errors = sorted(float(value) for value in pd.to_numeric(cell["horizon_error_seconds"], errors="coerce").dropna().unique()) if "horizon_error_seconds" in cell else []
            actual_union.update(actuals)
            cells[_cell_key(train_domain, test_domain)] = {
                "n_units": int(len(cell)),
                "actual_horizon_s_values": actuals,
                "horizon_idx_values": horizon_indices,
                "horizon_error_seconds_values": errors,
            }
        target_passed = len(actual_union) == 1 and all(
            len(cells[_cell_key(train_domain, test_domain)]["actual_horizon_s_values"]) == 1
            for train_domain, test_domain in REQUIRED_CELLS
        )
        if not target_passed:
            passed = False
            issues.append(f"actual_horizon_s is not aligned across cells for target_horizon_s={float(target)}")
        alignment[str(float(target))] = {
            "status": "pass" if target_passed else "fail",
            "cells": cells,
            "actual_horizon_s_values_across_cells": sorted(actual_union),
        }
    return {"status": "pass" if passed else "fail", "targets": alignment}


def _checkpoint_identity(value: str) -> str:
    normalized = str(value).replace("\\", "/")
    return normalized.rsplit("/", 1)[-1]


def _cell_key(train_domain: str, test_domain: str) -> str:
    return f"{train_domain}->{test_domain}"


def _final_report_summary(
    *,
    input_paths: list[str],
    aggregated_input_path: Path,
    prepared: pd.DataFrame,
    config: RQ3FinalSetup,
    metrics: Sequence[str],
    aggregation_level: AggregationLevel,
    rq3_result: RQ3Result,
) -> dict[str, object]:
    readiness = rq3_result.readiness_summary
    final_checks = readiness.get("final_setup_checks", {})
    return {
        "inputs": {
            "input_paths": input_paths,
            "aggregated_input_output_path": str(aggregated_input_path),
            "n_aggregated_rows_after_final_filter": int(len(prepared)),
        },
        "final_setup": {
            "design": "2x2 train_domain/test_domain factorial for RQ3",
            "train_domains": ["real", "synthetic"],
            "test_domains": ["real", "synthetic"],
            "metrics": list(metrics),
            "target_horizons": list(config.main_horizons),
            "excluded_target_horizons": [0.5],
            "synthetic_size_fraction": config.synthetic_size_fraction,
            "synthetic_checkpoint_selection": config.synthetic_checkpoint_selection,
            "synthetic_checkpoint_rule": config.synthetic_checkpoint_rule,
        },
        "aggregation": {
            "level": aggregation_level,
            "n_units_per_cell": _summary_n_units_per_cell(prepared),
        },
        "rq3": {
            "status": _final_report_status(readiness),
            "reason": readiness.get("reason", ""),
            "method": "unpaired permutation-based 2x2 factorial analysis at aggregated track level",
            "permutation_limitation": _UNPAIRED_LIMITATION,
            "readiness": readiness,
            "final_setup_checks": final_checks,
            "readiness_output_path": "rq3/stats_rq3_readiness_report.csv",
            "domain_summary_output_path": "rq3/stats_rq3_domain_summary.csv",
            "factorial_output_path": "rq3/stats_rq3_factorial_results.csv",
            "posthoc_output_path": "rq3/stats_rq3_posthoc_results.csv",
            "n_factorial_tests": int(len(rq3_result.factorial_results)),
            "factorial_correction_method": "holm",
        },
    }


def _summary_n_units_per_cell(df: pd.DataFrame) -> dict[str, int]:
    if df.empty or not {"train_domain", "test_domain"}.issubset(df.columns):
        return {_cell_key(train_domain, test_domain): 0 for train_domain, test_domain in REQUIRED_CELLS}
    return {
        _cell_key(train_domain, test_domain): int(
            ((df["train_domain"] == train_domain) & (df["test_domain"] == test_domain)).sum(),
        )
        for train_domain, test_domain in REQUIRED_CELLS
    }


def _final_report_status(readiness: dict[str, object]) -> str:
    if readiness.get("rq3_ready"):
        return "ran"
    missing = readiness.get("missing_required_cells")
    if missing:
        return "skipped_missing_domain_cells"
    return "skipped_final_setup_checks"


def _final_report_markdown(summary: dict[str, object]) -> str:
    rq3 = summary["rq3"]
    final_setup = summary["final_setup"]
    aggregation = summary["aggregation"]
    lines = [
        "# RQ3 Final Trajectory Stats Pipeline",
        "",
        f"- Status: {rq3['status']} {rq3.get('reason', '')}",
        f"- Aggregation level: {summary['aggregation']['level']}",
        f"- Metrics: {final_setup['metrics']}",
        f"- Main horizons: {final_setup['target_horizons']}",
        f"- Excluded horizons: {final_setup['excluded_target_horizons']}",
        f"- Synthetic fraction: {final_setup['synthetic_size_fraction']}",
        f"- Synthetic checkpoint selection: {final_setup['synthetic_checkpoint_selection']}",
        f"- Synthetic checkpoint rule: {final_setup['synthetic_checkpoint_rule']}",
        f"- N units per cell: {aggregation['n_units_per_cell']}",
        f"- Factorial tests: {rq3['n_factorial_tests']}",
        f"- Multiple testing correction: {rq3['factorial_correction_method']}",
        f"- Method limitation: {rq3['permutation_limitation']}",
    ]
    return "\n".join(lines) + "\n"


def build_readiness_report(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    validate_required_columns(
        df,
        [
            "train_domain",
            "test_domain",
            "model_name",
            "model_family",
            "model_variant",
            "target_horizon_s",
            "actual_horizon_s",
            *SUPPORTED_COMPARISON_METRICS,
        ],
    )
    rows: list[dict[str, object]] = []
    all_cells = [*REQUIRED_CELLS, *OPTIONAL_CELLS]
    for train_domain, test_domain in all_cells:
        cell = df[(df["train_domain"] == train_domain) & (df["test_domain"] == test_domain)]
        required = (train_domain, test_domain) in REQUIRED_CELLS
        rows.append(
            {
                "train_domain": train_domain,
                "test_domain": test_domain,
                "required_cell": required,
                "n_rows": int(cell["n_rows"].sum()) if "n_rows" in cell.columns and not cell.empty else int(len(cell)),
                "n_units": int(len(cell)),
                "model_names": _joined_unique(cell.get("model_name")),
                "model_families": _joined_unique(cell.get("model_family")),
                "model_variants": _joined_unique(cell.get("model_variant")),
                "metrics_available": _metrics_available(cell),
                "target_horizons_available": _joined_numeric(cell.get("target_horizon_s")),
                "actual_horizons_available": _joined_numeric(cell.get("actual_horizon_s", cell.get("horizon_s"))),
                "status": "available" if not cell.empty else "missing",
                "notes": "" if not cell.empty else "missing required cell" if required else "optional cell not present",
            },
        )
    report = pd.DataFrame(rows, columns=RQ3_READINESS_COLUMNS)
    missing_required = [
        (row.train_domain, row.test_domain)
        for row in report.itertuples(index=False)
        if row.required_cell and row.status != "available"
    ]
    available_required = [
        (row.train_domain, row.test_domain)
        for row in report.itertuples(index=False)
        if row.required_cell and row.status == "available"
    ]
    optional_available = [
        (row.train_domain, row.test_domain)
        for row in report.itertuples(index=False)
        if not row.required_cell and row.status == "available"
    ]
    summary = {
        "rq3_ready": not missing_required,
        "missing_required_cells": missing_required,
        "available_required_cells": available_required,
        "optional_cells_available": optional_available,
        "reason": "all required domain cells available"
        if not missing_required
        else f"missing required domain cells: {missing_required}",
    }
    return report, summary


def build_domain_summary(
    df: pd.DataFrame,
    *,
    metrics: Sequence[str],
    target_horizons: Sequence[float] | None,
    aggregation_level: AggregationLevel,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for horizon in _horizon_slices(df, target_horizons):
        for metric in metrics:
            validate_required_columns(horizon.frame, [metric])
            for (train_domain, test_domain), group in horizon.frame.groupby(["train_domain", "test_domain"], dropna=False, sort=True):
                values = pd.to_numeric(group[metric], errors="coerce").dropna()
                if values.empty:
                    continue
                ci_lower, ci_upper = _t_ci(values)
                rows.append(
                    {
                        "metric": metric,
                        "aggregation_level": aggregation_level,
                        "target_horizon_s": horizon.target_horizon_s,
                        "actual_horizon_s": horizon.actual_horizon_s,
                        "horizon_error_seconds": horizon.horizon_error_seconds,
                        "train_domain": train_domain,
                        "test_domain": test_domain,
                        "n_units": int(len(values)),
                        "mean_error": float(values.mean()),
                        "median_error": float(values.median()),
                        "std_error": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                        "se_error": _se(values),
                        "ci_lower": ci_lower,
                        "ci_upper": ci_upper,
                        "ci_method": "t_95",
                        "mean_batch_latency_ms": float(pd.to_numeric(group["mean_batch_latency_ms"], errors="coerce").mean()),
                        "p95_batch_latency_ms": float(pd.to_numeric(group["p95_batch_latency_ms"], errors="coerce").mean()),
                        "mean_runtime_windows_per_second": float(
                            pd.to_numeric(group["mean_runtime_windows_per_second"], errors="coerce").mean(),
                        ),
                        "budget_pass_rate": float(group["budget_pass"].astype(bool).mean()),
                    },
                )
    return pd.DataFrame(rows, columns=RQ3_DOMAIN_SUMMARY_COLUMNS)


def build_factorial_results(
    df: pd.DataFrame,
    *,
    metrics: Sequence[str],
    target_horizons: Sequence[float] | None,
    aggregation_level: AggregationLevel,
    n_permutations: int,
    random_seed: int,
    budget_mode: str,
    notes: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    input_frames: list[pd.DataFrame] = []
    rng = np.random.default_rng(random_seed)
    for horizon in _horizon_slices(df, target_horizons):
        primary = horizon.frame[
            horizon.frame["train_domain"].isin(["real", "synthetic"])
            & horizon.frame["test_domain"].isin(["real", "synthetic"])
        ].copy()
        input_frames.append(primary)
        for metric in metrics:
            metric_frame = primary.dropna(subset=[metric]).copy()
            observed = _effect_statistics(metric_frame, metric)
            permuted_counts = {effect: 0 for effect in observed}
            labels = metric_frame[["train_domain", "test_domain"]].to_numpy(copy=True)
            for _ in range(n_permutations):
                permuted = metric_frame.copy()
                permuted_labels = labels[rng.permutation(len(labels))]
                permuted["train_domain"] = permuted_labels[:, 0]
                permuted["test_domain"] = permuted_labels[:, 1]
                permuted_stats = _effect_statistics(permuted, metric)
                for effect, statistic in observed.items():
                    if permuted_stats.get(effect, 0.0) >= statistic:
                        permuted_counts[effect] += 1
            for effect, statistic in observed.items():
                rows.append(
                    {
                        "metric": metric,
                        "aggregation_level": aggregation_level,
                        "target_horizon_s": horizon.target_horizon_s,
                        "actual_horizon_s": horizon.actual_horizon_s,
                        "horizon_error_seconds": horizon.horizon_error_seconds,
                        "effect": effect,
                        "statistic": statistic,
                        "p_value": (permuted_counts[effect] + 1) / (n_permutations + 1),
                        "n_permutations": n_permutations,
                        "n_units": int(len(metric_frame)),
                        "method": "permutation_factorial_2x2",
                        "permutation_scheme": "joint train_domain/test_domain label permutation preserving response values",
                        "budget_mode": budget_mode,
                        "notes": _join_notes(notes),
                    },
                )
    factorial_input = pd.concat(input_frames, ignore_index=True) if input_frames else pd.DataFrame()
    return pd.DataFrame(rows, columns=RQ3_FACTORIAL_COLUMNS), factorial_input


def _apply_holm_correction(results: pd.DataFrame, *, alpha: float = 0.05) -> pd.DataFrame:
    corrected = results.copy()
    corrected["correction_method"] = "holm"
    corrected["corrected_p_value"] = np.nan
    corrected["significant"] = False
    if corrected.empty:
        return corrected

    p_values = pd.to_numeric(corrected["p_value"], errors="coerce")
    valid = p_values.notna() & np.isfinite(p_values)
    if not valid.any():
        return corrected

    adjusted = holm_correction(p_values[valid].tolist(), alpha=alpha)
    valid_indices = corrected.index[valid].tolist()
    for row_position, index in enumerate(valid_indices):
        corrected.loc[index, "corrected_p_value"] = adjusted.loc[row_position, "corrected_p_value"]
        corrected.loc[index, "significant"] = bool(adjusted.loc[row_position, "significant"])
    return corrected


def build_posthoc_results(
    df: pd.DataFrame,
    *,
    metrics: Sequence[str],
    target_horizons: Sequence[float] | None,
    aggregation_level: AggregationLevel,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    specs = [
        ("real_vs_synthetic_train_on_real_test", {"test_domain": "real"}, ("train_domain", "real", "synthetic")),
        ("real_vs_synthetic_train_on_synthetic_test", {"test_domain": "synthetic"}, ("train_domain", "real", "synthetic")),
        ("real_vs_synthetic_test_for_real_train", {"train_domain": "real"}, ("test_domain", "real", "synthetic")),
        ("real_vs_synthetic_test_for_synthetic_train", {"train_domain": "synthetic"}, ("test_domain", "real", "synthetic")),
    ]
    for horizon in _horizon_slices(df, target_horizons):
        for metric in metrics:
            for comparison, fixed, varying in specs:
                frame = horizon.frame.copy()
                for key, value in fixed.items():
                    frame = frame[frame[key] == value]
                column, a_value, b_value = varying
                a = pd.to_numeric(frame.loc[frame[column] == a_value, metric], errors="coerce").dropna()
                b = pd.to_numeric(frame.loc[frame[column] == b_value, metric], errors="coerce").dropna()
                if a.empty or b.empty:
                    continue
                diff = float(a.mean() - b.mean())
                ci_lower, ci_upper = _bootstrap_unpaired_difference(a, b)
                mean_a = float(a.mean())
                rows.append(
                    {
                        "metric": metric,
                        "aggregation_level": aggregation_level,
                        "target_horizon_s": horizon.target_horizon_s,
                        "actual_horizon_s": horizon.actual_horizon_s,
                        "horizon_error_seconds": horizon.horizon_error_seconds,
                        "comparison": comparison,
                        "group_a": f"{column}={a_value}",
                        "group_b": f"{column}={b_value}",
                        "mean_a": mean_a,
                        "mean_b": float(b.mean()),
                        "mean_difference": diff,
                        "percent_difference_relative_to_a": float("nan") if mean_a == 0 else 100.0 * diff / mean_a,
                        "ci_lower": ci_lower,
                        "ci_upper": ci_upper,
                        "ci_method": "bootstrap_95_mean_difference",
                        "comparison_type": "unpaired",
                        "n_a": int(len(a)),
                        "n_b": int(len(b)),
                        "n_pairs": 0,
                        "notes": "CI only; no posthoc p-value computed",
                    },
                )
    return pd.DataFrame(rows, columns=RQ3_POSTHOC_COLUMNS)


def _load_or_aggregate(data: pd.DataFrame | Iterable[str | Path], *, level: AggregationLevel) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        frame = data.copy()
    else:
        frame = load_evaluation_units(data)
    if "mean_error_l2" in frame.columns:
        return frame
    return aggregate_units(frame, level=level)


def _validate_inputs(metrics: Sequence[str], budget_mode: str) -> None:
    unsupported = [metric for metric in metrics if metric not in SUPPORTED_COMPARISON_METRICS]
    if unsupported:
        raise ValueError(f"Unsupported RQ3 metric(s): {unsupported}")
    if budget_mode not in ("report", "strict"):
        raise ValueError("budget_mode must be 'report' or 'strict'")


def _apply_budget_mode(df: pd.DataFrame, *, budget_mode: str) -> tuple[pd.DataFrame, list[str]]:
    if budget_mode == "report":
        return df.copy(), []
    validate_required_columns(df, ["budget_pass"])
    budget = df["budget_pass"].astype(bool)
    excluded = int((~budget).sum())
    return df[budget].copy(), [f"budget_excluded_rows={excluded}"]


def _effect_statistics(df: pd.DataFrame, metric: str) -> dict[str, float]:
    means = df.groupby(["train_domain", "test_domain"], sort=True)[metric].mean().to_dict()
    train_real = float(df.loc[df["train_domain"] == "real", metric].mean())
    train_synthetic = float(df.loc[df["train_domain"] == "synthetic", metric].mean())
    test_real = float(df.loc[df["test_domain"] == "real", metric].mean())
    test_synthetic = float(df.loc[df["test_domain"] == "synthetic", metric].mean())
    rr = means.get(("real", "real"), math.nan)
    rs = means.get(("real", "synthetic"), math.nan)
    sr = means.get(("synthetic", "real"), math.nan)
    ss = means.get(("synthetic", "synthetic"), math.nan)
    return {
        "train_domain": abs(train_real - train_synthetic),
        "test_domain": abs(test_real - test_synthetic),
        "train_domain:test_domain": abs((rr - rs) - (sr - ss)),
    }


def _t_ci(values: pd.Series, confidence: float = 0.95) -> tuple[float, float]:
    mean = float(values.mean())
    if len(values) <= 1:
        return mean, mean
    se = _se(values)
    # 1.96 is sufficient here; the method is reported as an approximate t-style CI.
    margin = 1.96 * se
    return mean - margin, mean + margin


def _se(values: pd.Series) -> float:
    return float(values.std(ddof=1) / math.sqrt(len(values))) if len(values) > 1 else 0.0


def _bootstrap_unpaired_difference(a: pd.Series, b: pd.Series, *, n_bootstrap: int = 1000) -> tuple[float, float]:
    rng = np.random.default_rng(42)
    a_values = a.to_numpy(dtype=float)
    b_values = b.to_numpy(dtype=float)
    diffs = []
    for _ in range(n_bootstrap):
        diffs.append(
            float(
                rng.choice(a_values, size=len(a_values), replace=True).mean()
                - rng.choice(b_values, size=len(b_values), replace=True).mean(),
            ),
        )
    return float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))


def _metrics_available(df: pd.DataFrame) -> str:
    available = [metric for metric in SUPPORTED_COMPARISON_METRICS if metric in df.columns and df[metric].notna().any()]
    return ";".join(available)


def _joined_unique(values: pd.Series | None) -> str:
    if values is None:
        return ""
    return ";".join(sorted(str(value) for value in values.dropna().unique()))


def _joined_numeric(values: pd.Series | None) -> str:
    if values is None:
        return ""
    numeric = pd.to_numeric(values, errors="coerce").dropna().unique()
    return ";".join(str(float(value)) for value in sorted(numeric))


def _write_outputs(result: RQ3Result, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    result.readiness_report.to_csv(outdir / "stats_rq3_readiness_report.csv", index=False)
    (outdir / "stats_rq3_readiness_report.json").write_text(
        json.dumps(
            {"summary": _json_ready_summary(result.readiness_summary), "cells": result.readiness_report.to_dict("records")},
            indent=2,
            default=_json_default,
        ),
        encoding="utf-8",
    )
    result.domain_summary.to_csv(outdir / "stats_rq3_domain_summary.csv", index=False)
    result.factorial_results.to_csv(outdir / "stats_rq3_factorial_results.csv", index=False)
    result.posthoc_results.to_csv(outdir / "stats_rq3_posthoc_results.csv", index=False)


def _json_ready_summary(summary: dict[str, object]) -> dict[str, object]:
    output = dict(summary)
    for key in ("missing_required_cells", "available_required_cells", "optional_cells_available"):
        output[key] = [list(value) for value in output.get(key, [])]
    return output


def _json_default(value: object) -> object:
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")
