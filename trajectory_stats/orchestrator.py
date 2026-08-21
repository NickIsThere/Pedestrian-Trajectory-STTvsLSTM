from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from .discovery import DiscoveredRun, discover_runs, write_discovery_reports
from .loading import load_evaluation_units

"""
The Orchestration unit 
Tom Pepples would be proud of me with my software engineering skills
"""


@dataclass(frozen=True)
class PrepareUnitsResult:
    runs: list[DiscoveredRun]
    discovery_csv_path: Path
    discovery_json_path: Path
    combined_units_path: Path | None


def prepare_units(
    runs_root: str | Path,
    outdir: str | Path,
    *,
    export_missing: bool = False,
) -> PrepareUnitsResult:
    output_dir = Path(outdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    discovered = discover_runs(runs_root)
    runs = _annotate_export_missing(discovered) if export_missing else discovered
    csv_path, json_path = write_discovery_reports(runs, output_dir)
    ready_paths = [Path(run.run_dir) / "evaluation_units.csv" for run in runs if run.state == "ready"]
    if not ready_paths:
        raise ValueError("No ready evaluation_units.csv files were discovered")
    combined = load_evaluation_units(ready_paths)
    combined_path = output_dir / "evaluation_units.csv"
    combined.to_csv(combined_path, index=False)
    return PrepareUnitsResult(
        runs=runs,
        discovery_csv_path=csv_path,
        discovery_json_path=json_path,
        combined_units_path=combined_path,
    )


def _annotate_export_missing(runs: list[DiscoveredRun]) -> list[DiscoveredRun]:
    annotated: list[DiscoveredRun] = []
    for run in runs:
        if run.state != "needs_evaluation":
            annotated.append(run)
            continue
        note = "export_missing requested, but checkpoint-to-evaluation export is not implemented"
        notes = "; ".join(part for part in [run.notes, note] if part)
        annotated.append(replace(run, notes=notes))
    return annotated
