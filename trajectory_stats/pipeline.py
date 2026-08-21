from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

from aggregation import aggregate_units
from loading import load_evaluation_units
from pairing import pair_aggregated
from rq1 import run_rq1_pairwise
from rq2 import run_rq2
from rq3 import run_rq3
from schema import AggregationLevel, SUPPORTED_COMPARISON_METRICS, validation_report


@dataclass(frozen=True)
class PipelineResult:
    summary: dict[str, object]
    combined_units_path: Path
    aggregated_path: Path
    summary_markdown_path: Path
    summary_json_path: Path




def _run_or_skip_rq1(
    *,
    selected: bool,
    units: pd.DataFrame,
    aggregated: pd.DataFrame,
    outdir: Path,
    baseline: str,
    treatment: str,
    metrics: Sequence[str],
    target_horizons: Sequence[float],
    aggregation_level: AggregationLevel,
    budget_mode: str,
    n_permutations: int,
    n_bootstrap: int,
    random_seed: int,
) -> dict[str, object]:
    if not selected:
        return {"status": "not_selected", "reason": ""}
    models = set(aggregated["model_name"].astype(str))
    missing = sorted({baseline, treatment} - models)
    if missing:
        return {"status": "skipped_missing_models", "reason": f"Missing model(s): {', '.join(missing)}"}
    pair = pair_aggregated(aggregated, baseline_model=baseline, treatment_model=treatment, level=aggregation_level)
    if pair.report["n_pairs"] == 0:
        return {
            "status": "skipped_no_valid_pairs",
            "reason": f"No matching {aggregation_level}-level pairing keys for {baseline} vs {treatment}.",
        }
    try:
        result = run_rq1_pairwise(
            units,
            baseline_model=baseline,
            treatment_model=treatment,
            metrics=metrics,
            target_horizons=target_horizons,
            aggregation_level=aggregation_level,
            outdir=outdir,
            n_permutations=n_permutations,
            n_bootstrap=n_bootstrap,
            random_seed=random_seed,
            budget_mode=budget_mode,
        )
    except Exception as exc:
        return {"status": "failed", "reason": str(exc)}
    return {"status": "ran", "reason": "", "output_path": str(result.output_path)}


def _run_or_skip_rq2(
    *,
    selected: bool,
    units: pd.DataFrame,
    aggregated: pd.DataFrame,
    outdir: Path,
    margin_type: str | None,
    margin_value: float | None,
    metrics: Sequence[str],
    target_horizons: Sequence[float],
    aggregation_level: AggregationLevel,
    budget_mode: str,
) -> dict[str, object]:
    if not selected:
        return {"status": "not_selected", "reason": ""}
    if margin_type is None or margin_value is None:
        return {"status": "skipped_missing_margin", "reason": "RQ2 requires margin_type and margin_value."}
    real_models = sorted(aggregated.loc[aggregated["train_domain"] == "real", "model_name"].astype(str).unique())
    synthetic_models = sorted(aggregated.loc[aggregated["train_domain"] == "synthetic", "model_name"].astype(str).unique())
    if not real_models or not synthetic_models:
        return {"status": "skipped_missing_domain_cells", "reason": "RQ2 requires real and synthetic train_domain rows."}
    try:
        result = run_rq2(
            units,
            real_model=real_models[0],
            synthetic_models=synthetic_models,
            metrics=metrics,
            target_horizons=target_horizons,
            aggregation_level=aggregation_level,
            margin_type=margin_type,
            margin_value=margin_value,
            outdir=outdir,
            budget_mode=budget_mode,
        )
    except Exception as exc:
        return {"status": "failed", "reason": str(exc)}
    return {
        "status": "ran",
        "reason": "",
        "results_output_path": str(result.results_output_path),
        "threshold_output_path": str(result.threshold_output_path),
    }


def _run_or_skip_rq3(
    *,
    selected: bool,
    units: pd.DataFrame,
    outdir: Path,
    metrics: Sequence[str],
    target_horizons: Sequence[float],
    aggregation_level: AggregationLevel,
    budget_mode: str,
    n_permutations: int,
    random_seed: int,
) -> dict[str, object]:
    if not selected:
        return {"status": "not_selected", "reason": ""}
    try:
        result = run_rq3(
            units,
            metrics=metrics,
            target_horizons=target_horizons,
            aggregation_level=aggregation_level,
            outdir=outdir,
            n_permutations=n_permutations,
            random_seed=random_seed,
            budget_mode=budget_mode,
        )
    except Exception as exc:
        return {"status": "failed", "reason": str(exc)}
    status = "ran" if result.readiness_summary["rq3_ready"] else "skipped_missing_domain_cells"
    return {
        "status": status,
        "reason": result.readiness_summary["reason"],
        "readiness_output_path": str(outdir / "stats_rq3_readiness_report.csv"),
        "factorial_output_path": str(outdir / "stats_rq3_factorial_results.csv") if result.readiness_summary["rq3_ready"] else "",
    }


def _summary(
    *,
    units: pd.DataFrame,
    aggregated: pd.DataFrame,
    target_horizons: Sequence[float],
    aggregation_level: str,
    combined_path: Path,
    aggregated_path: Path,
    checkpoint_paths: Mapping[str, str | Path] | None,
    rq1: dict[str, object],
    rq2: dict[str, object],
    rq3: dict[str, object],
) -> dict[str, object]:
    coverage = validation_report(units)
    return {
        "inputs": {
            "n_evaluation_rows": len(units),
            "combined_units_output_path": str(combined_path),
            "aggregated_output_path": str(aggregated_path),
        },
        "checkpoints": {key: str(value) for key, value in (checkpoint_paths or {}).items()},
        "metadata": {
            "model_names": _sorted_unique(units["model_name"]),
            "model_variants": _sorted_unique(units["model_variant"]),
            "train_domains": _sorted_unique(units["train_domain"]),
            "test_domains": _sorted_unique(units["test_domain"]),
            "synthetic_size_fractions": _sorted_unique(units["synthetic_size_fraction"]),
        },
        "horizons": {
            "requested_target_horizons": list(target_horizons),
            "target_mappings": coverage.get("target_mappings", []),
            "max_horizon_s": coverage.get("max_horizon_s"),
            "coverage_pass": coverage.get("coverage_pass"),
        },
        "aggregation": {
            "level": aggregation_level,
            "n_rows": len(aggregated),
            "metrics_available": [metric for metric in SUPPORTED_COMPARISON_METRICS if metric in aggregated.columns],
        },
        "rq1": rq1,
        "rq2": rq2,
        "rq3": rq3,
    }


def _summary_markdown(summary: dict[str, object]) -> str:
    lines = [
        "# Trajectory Stats Pipeline",
        "",
        f"- Evaluation rows: {summary['inputs']['n_evaluation_rows']}",
        f"- Aggregated rows: {summary['aggregation']['n_rows']}",
        f"- Aggregation level: {summary['aggregation']['level']}",
        f"- Models: {summary['metadata']['model_names']}",
        f"- Train domains: {summary['metadata']['train_domains']}",
        f"- Test domains: {summary['metadata']['test_domains']}",
        f"- Requested horizons: {summary['horizons']['requested_target_horizons']}",
        f"- Max horizon: {summary['horizons']['max_horizon_s']}",
        f"- Horizon coverage passes: {summary['horizons']['coverage_pass']}",
        f"- RQ1: {summary['rq1']['status']} {summary['rq1'].get('reason', '')}",
        f"- RQ2: {summary['rq2']['status']} {summary['rq2'].get('reason', '')}",
        f"- RQ3: {summary['rq3']['status']} {summary['rq3'].get('reason', '')}",
    ]
    return "\n".join(lines) + "\n"


def _sorted_unique(values: pd.Series) -> list[object]:
    output = []
    for value in values.drop_duplicates().tolist():
        if isinstance(value, float) and math.isnan(value):
            output.append("NaN")
        else:
            output.append(value)
    return sorted(output, key=str)


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and math.isnan(value):
        return None
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")



def main(
        *,
        evaluation_unit_paths: Sequence[str | Path],
        outdir: str | Path,
        run_rq1: bool = False,
        run_rq2: bool = False,
        run_rq3: bool = False,
        rq1_baseline: str = "temporal_only_transformer",
        rq1_treatment: str = "spatio_temporal_transformer",
        rq2_margin_type: str | None = None,
        rq2_margin_value: float | None = None,
        checkpoint_paths: Mapping[str, str | Path] | None = None,
        target_horizons: Sequence[float] = (0.5, 1.0, 2.0),
        aggregation_level: AggregationLevel = "track",
        metrics: Sequence[str] = ("mean_ade_until_horizon", "mean_fde_at_horizon"),
        budget_mode: str = "report",
        n_permutations: int = 10000,
        n_bootstrap: int = 10000,
        random_seed: int = 42,
) -> PipelineResult:
    output_dir = Path(outdir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Here")

    units = load_evaluation_units(evaluation_unit_paths)
    combined_path = output_dir / "combined_evaluation_units.csv"
    units.to_csv(combined_path, index=False)

    aggregated = aggregate_units(units, level=aggregation_level)
    aggregated_path = output_dir / f"aggregated_{aggregation_level}.csv"
    aggregated.to_csv(aggregated_path, index=False)

    rq1 = _run_or_skip_rq1(
        selected=run_rq1,
        units=units,
        aggregated=aggregated,
        outdir=output_dir / "rq1",
        baseline=rq1_baseline,
        treatment=rq1_treatment,
        metrics=metrics,
        target_horizons=target_horizons,
        aggregation_level=aggregation_level,
        budget_mode=budget_mode,
        n_permutations=n_permutations,
        n_bootstrap=n_bootstrap,
        random_seed=random_seed,
    )
    rq2 = _run_or_skip_rq2(
        selected=run_rq2,
        units=units,
        aggregated=aggregated,
        outdir=output_dir / "rq2",
        margin_type=rq2_margin_type,
        margin_value=rq2_margin_value,
        metrics=metrics,
        target_horizons=target_horizons,
        aggregation_level=aggregation_level,
        budget_mode=budget_mode,
    )
    rq3 = _run_or_skip_rq3(
        selected=run_rq3,
        units=units,
        outdir=output_dir / "rq3",
        metrics=metrics,
        target_horizons=target_horizons,
        aggregation_level=aggregation_level,
        budget_mode=budget_mode,
        n_permutations=n_permutations,
        random_seed=random_seed,
    )
    summary = _summary(
        units=units,
        aggregated=aggregated,
        target_horizons=target_horizons,
        aggregation_level=aggregation_level,
        combined_path=combined_path,
        aggregated_path=aggregated_path,
        checkpoint_paths=checkpoint_paths,
        rq1=rq1,
        rq2=rq2,
        rq3=rq3,
    )
    markdown_path = output_dir / "stats_pipeline_summary.md"
    json_path = output_dir / "stats_pipeline_summary.json"
    markdown_path.write_text(_summary_markdown(summary), encoding="utf-8")
    json_path.write_text(json.dumps(summary, indent=2, default=_json_default), encoding="utf-8")
    return PipelineResult(
        summary=summary,
        combined_units_path=combined_path,
        aggregated_path=aggregated_path,
        summary_markdown_path=markdown_path,
        summary_json_path=json_path,
    )


if __name__ == "__main__":
    # 1. Dynamically resolve your project root
    project_root = Path(__file__).resolve().parents[1]

    # 2. Define the path to your raw evaluation units CSV generated by App/test.py
    csv_paths = [
        project_root / "outputs" / "evaluation_units.csv",
    ]

    # 3. Define where the statistical reports should be saved
    out_dir = project_root / "outputs" / "statistical_analysis"

    print("Starting Statistical Pipeline...")

    result = main(
        evaluation_unit_paths=csv_paths,
        outdir=out_dir,

        # Enable ALL Research Questions
        run_rq1=True,
        rq1_baseline="social_lstm",
        rq1_treatment="transformer",

        run_rq2=True,
        rq2_margin_type="relative",
        rq2_margin_value=0.05,

        run_rq3=True,

        target_horizons=(0.5, 1.0, 2.0),
        aggregation_level="track",
        n_permutations=10000,
        n_bootstrap=10000,
        random_seed=42
    )

    print(f"Pipeline finished! Results saved to: {out_dir}")