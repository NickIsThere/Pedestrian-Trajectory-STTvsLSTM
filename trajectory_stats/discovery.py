from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

from App.evaluation.evaluation_units import validate_evaluation_units_csv


RunState = Literal["ready", "needs_evaluation", "aggregate_only", "invalid"]


@dataclass(frozen=True)
class DiscoveredRun:
    run_dir: str
    run_id: str
    model_name: str
    model_family: str
    model_variant: str
    train_domain: str
    test_domain: str
    synthetic_size_fraction: float
    seed: int | float
    checkpoint_path: str
    has_config: bool
    has_summary: bool
    has_fold_metrics: bool
    has_evaluation_units: bool
    has_checkpoint: bool
    metadata_complete: bool
    evaluation_units_valid: bool
    state: RunState
    notes: str = ""


def discover_runs(runs_root: str | Path) -> list[DiscoveredRun]:
    root = Path(runs_root)
    if not root.exists():
        raise FileNotFoundError(f"runs root does not exist: {root}")
    candidates = _candidate_run_dirs(root)
    return sorted((_inspect_run(path) for path in candidates), key=lambda run: run.run_dir)


def write_discovery_reports(runs: list[DiscoveredRun], outdir: str | Path) -> tuple[Path, Path]:
    output_dir = Path(outdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [asdict(run) for run in runs]
    csv_path = output_dir / "run_discovery_report.csv"
    json_path = output_dir / "run_discovery_report.json"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(rows, indent=2, default=_json_default), encoding="utf-8")
    return csv_path, json_path


def _candidate_run_dirs(root: Path) -> list[Path]:
    known_names = {"config.json", "summary.csv", "fold_metrics.csv", "evaluation_units.csv"}
    candidates: set[Path] = set()
    for path in root.rglob("*"):
        if path.is_file() and path.name in known_names:
            candidates.add(path.parent)
        elif path.is_file() and path.suffix == ".pt":
            candidates.add(path.parent.parent if path.parent.name == "checkpoints" else path.parent)
    if any((root / name).exists() for name in known_names) or list(root.glob("*.pt")) or list((root / "checkpoints").glob("*.pt")):
        candidates.add(root)
    return [path for path in candidates if path.name != "checkpoints"]


def _inspect_run(run_dir: Path) -> DiscoveredRun:
    config_path = run_dir / "config.json"
    summary_path = run_dir / "summary.csv"
    fold_metrics_path = run_dir / "fold_metrics.csv"
    units_path = run_dir / "evaluation_units.csv"
    checkpoints = sorted([*run_dir.glob("*.pt"), *run_dir.glob("checkpoints/*.pt")])
    config = _read_config(config_path)
    checkpoint_path = str(checkpoints[0]) if checkpoints else ""
    evaluation_units_valid, validation_note = _validate_units(units_path)
    metadata_complete = _metadata_complete(config)
    state = _classify(
        has_config=config_path.exists(),
        has_summary=summary_path.exists(),
        has_fold_metrics=fold_metrics_path.exists(),
        has_units=units_path.exists(),
        has_checkpoint=bool(checkpoints),
        units_valid=evaluation_units_valid,
        metadata_complete=metadata_complete,
    )
    notes = "; ".join(note for note in [_metadata_note(config), validation_note] if note)
    return DiscoveredRun(
        run_dir=str(run_dir),
        run_id=str(config.get("run_id") or run_dir.name),
        model_name=str(config.get("model") or config.get("model_name") or run_dir.name),
        model_family=str(config.get("model_family") or "unknown"),
        model_variant=str(config.get("model_variant") or "unknown"),
        train_domain=str(config.get("train_domain") or "unknown"),
        test_domain=str(config.get("test_domain") or "unknown"),
        synthetic_size_fraction=_optional_float(config.get("synthetic_size_fraction")),
        seed=_optional_int(config.get("seed")),
        checkpoint_path=checkpoint_path,
        has_config=config_path.exists(),
        has_summary=summary_path.exists(),
        has_fold_metrics=fold_metrics_path.exists(),
        has_evaluation_units=units_path.exists(),
        has_checkpoint=bool(checkpoints),
        metadata_complete=metadata_complete,
        evaluation_units_valid=evaluation_units_valid,
        state=state,
        notes=notes,
    )


def _read_config(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _validate_units(path: Path) -> tuple[bool, str]:
    if not path.exists():
        return False, ""
    try:
        validate_evaluation_units_csv(path)
    except Exception as exc:
        return False, f"invalid_evaluation_units={exc}"
    return True, ""


def _metadata_complete(config: dict[str, object]) -> bool:
    required = ("model_family", "model_variant", "train_domain", "test_domain")
    return all(str(config.get(key) or "unknown") != "unknown" for key in required)


def _metadata_note(config: dict[str, object]) -> str:
    missing = [
        key
        for key in ("model_family", "model_variant", "train_domain", "test_domain")
        if str(config.get(key) or "unknown") == "unknown"
    ]
    return f"incomplete_metadata={','.join(missing)}" if missing else ""


def _classify(
    *,
    has_config: bool,
    has_summary: bool,
    has_fold_metrics: bool,
    has_units: bool,
    has_checkpoint: bool,
    units_valid: bool,
    metadata_complete: bool,
) -> RunState:
    if has_units and units_valid and metadata_complete:
        return "ready"
    if has_config and has_checkpoint and metadata_complete:
        return "needs_evaluation"
    if (has_summary or has_fold_metrics) and not has_units and not has_checkpoint:
        return "aggregate_only"
    return "invalid"


def _optional_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _optional_int(value: object) -> int | float:
    try:
        return int(value)
    except (TypeError, ValueError):
        return math.nan


def _json_default(value: object) -> object:
    if isinstance(value, float) and math.isnan(value):
        return None
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")
