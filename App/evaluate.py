"""
Main Writer: Noah Nuelandt
Reviewer: 
Contributors:
"""

from __future__ import annotations

import sys
from pathlib import Path

_APP_ROOT = Path(__file__).resolve().parent
if str(_APP_ROOT) not in sys.path:
    sys.path.insert(0, str(_APP_ROOT))

import argparse
import csv
import json
import math
import re
import warnings
from dataclasses import asdict
from typing import Any, Literal

import pandas as pd
import torch

from data.loader import VALID_DATASETS
from data.reader import Dataset, Split, get_dataset
from evaluation.evaluation_units import (
    EvaluationUnitsCsvWriter,
    config_hash,
    validate_evaluation_units_csv,
)
from model.config import ModelConfig
from model.model import HTPModel
from train import (
    DOMAIN_VALUES,
    SampleMode,
    all_dataset_sequences,
    evaluate_split_for_model,
    get_model_spec,
    sequence_has_gt,
    split_from_sequences,
)
from utils.modelutils import load_htp_model

TEST_DOMAIN_TO_DATASET = {
    "real": "MOT20",
    "synthetic": "MOTSynth",
}
CHECKPOINT_VAL_PATTERN = re.compile(r"__val_(?P<sequence>[^.]+)")
CHECKPOINT_PCT_PATTERN = re.compile(r"pct(?P<digits>\d{3})", re.IGNORECASE)
SOCIAL_LSTM_CONFIG_KEY_MAP = {
    "obs_len": "lookback",
    "pred_len": "future_steps",
    "hidden_size": "hidden_dim",
    "embedding_dim": "head_hidden_dim",
    "subsample_step": "trajectory_stride",
}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate training checkpoints and export evaluation_units.csv.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "RQ2 cross-domain usage:\n"
            "  Evaluate real and synthetic runs on the same MOT20 sequences:\n"
            "    python App/evaluate.py <real_run> --test-domain real --train-domain real "
            "--model-name transformer_real\n"
            "    python App/evaluate.py <synth_run> --test-domain real --train-domain synthetic "
            "--model-name transformer_synth --checkpoints all\n"
            "  Then pass both evaluation_units.csv files to trajectory_stats/rq2.py.\n"
        ),
    )
    parser.add_argument(
        "results_dir",
        type=Path,
        help="Training results folder containing config.json and checkpoints/",
    )
    parser.add_argument(
        "--test-domain",
        required=True,
        choices=sorted(DOMAIN_VALUES),
        help="Value written to test_domain and used to pick the default test dataset.",
    )
    parser.add_argument(
        "--test-dataset",
        choices=VALID_DATASETS,
        default=None,
        help="Dataset to load for evaluation (default: MOT20 for real, MOTSynth for synthetic).",
    )
    parser.add_argument(
        "--sequences",
        default=None,
        help="Comma-separated test sequence names (default: val-split sequences with GT). Ignored with --held-out.",
    )
    parser.add_argument(
        "--held-out",
        action="store_true",
        help=(
            "Evaluate each checkpoint on its own LOO held-out sequence (from checkpoint filename "
            "or config.json folds). Use for in-domain MOT20 LOO runs instead of a common test set."
        ),
    )
    parser.add_argument(
        "--checkpoints",
        choices=("best", "all"),
        default="best",
        help="best = checkpoints/*.pt only; all = checkpoints/**/*.pt including curriculum steps.",
    )
    parser.add_argument(
        "--checkpoint-glob",
        default=None,
        help="Optional glob relative to checkpoints/ (overrides --checkpoints).",
    )
    parser.add_argument("--model-name", default=None, help="Override model_name column (default: config model).")
    parser.add_argument("--model-family", default=None, help="Override model_family (default: config).")
    parser.add_argument("--model-variant", default=None, help="Override model_variant (default: config).")
    parser.add_argument(
        "--train-domain",
        default=None,
        choices=sorted(DOMAIN_VALUES),
        help="Override train_domain column (default: config, or unknown).",
    )
    parser.add_argument("--seed", type=int, default=None, help="Override seed (default: config).")
    parser.add_argument(
        "--synthetic-size-fraction",
        type=float,
        default=None,
        help="Fallback synthetic_size_fraction when not parseable from checkpoint/summary.",
    )
    parser.add_argument(
        "--input-mode",
        choices=("gt", "det"),
        default="gt",
        help="Evaluation input mode (default: gt).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Cap windows per checkpoint (0 = all; deterministic order).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Evaluation batch size (default: config batch_size or 1).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device (default: cuda if available else cpu).",
    )
    parser.add_argument(
        "--lstm",
        action="store_true",
        help="Load checkpoints as Social LSTM models (benchmark social_lstm format).",
    )
    parser.add_argument(
        "--ignore-version",
        action="store_true",
        help="Load checkpoints even when MODEL_VERSION differs from current code.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output CSV path (default: <results_dir>/evaluation_units.csv).",
    )
    return parser.parse_args(argv)


def _read_run_config(results_dir: Path) -> dict[str, Any]:
    config_path = results_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.json in {results_dir}")
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"config.json must be a JSON object: {config_path}")
    return payload


def _resolve_test_dataset(test_domain: str, test_dataset: str | None) -> str:
    if test_dataset is not None:
        return test_dataset
    if test_domain in TEST_DOMAIN_TO_DATASET:
        return TEST_DOMAIN_TO_DATASET[test_domain]
    raise ValueError(
        f"--test-domain {test_domain!r} has no default dataset; pass --test-dataset explicitly.",
    )


def _default_test_sequence_names(dataset: Dataset) -> list[str]:
    if "val" in dataset.splits:
        candidates = [
            name
            for name, sequence in dataset.splits["val"].sequences.items()
            if sequence_has_gt(sequence)
        ]
        if candidates:
            return sorted(candidates)
    supervised = {
        name: sequence
        for name, sequence in all_dataset_sequences(dataset).items()
        if sequence_has_gt(sequence)
    }
    if not supervised:
        raise ValueError(f"No supervised sequences with ground truth in dataset {dataset.name!r}.")
    return sorted(supervised)


def _build_test_split(dataset: Dataset, sequence_names: list[str]) -> Split:
    all_sequences = all_dataset_sequences(dataset)
    missing = [name for name in sequence_names if name not in all_sequences]
    if missing:
        raise KeyError(
            f"Test sequence(s) not found in {dataset.name}: {missing}. "
            f"Available: {sorted(all_sequences)}",
        )
    selected = {
        name: all_sequences[name]
        for name in sequence_names
        if not sequence_has_gt(all_sequences[name])
    }
    if selected:
        raise ValueError(f"Test sequence(s) lack ground truth: {sorted(selected)}")
    test_sequences = {name: all_sequences[name] for name in sequence_names}
    return split_from_sequences(dataset, "test", test_sequences)


def _warn_train_test_overlap(run_config: dict[str, Any], test_sequences: list[str], train_domain: str, test_domain: str) -> None:
    if train_domain == "unknown" or test_domain == "unknown" or train_domain != test_domain:
        return
    train_sequences: set[str] = set()
    for fold in run_config.get("folds", []):
        if isinstance(fold, dict):
            train_sequences.update(str(name) for name in fold.get("train_sequences", []))
    overlap = sorted(set(test_sequences).intersection(train_sequences))
    if overlap:
        warnings.warn(
            f"Test sequences {overlap} overlap training sequences from this run; "
            "metrics may be optimistically biased.",
            UserWarning,
            stacklevel=2,
        )


def _fold_lookup(run_config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    folds = run_config.get("folds", [])
    if not isinstance(folds, list):
        return lookup
    for fold in folds:
        if not isinstance(fold, dict):
            continue
        held_out = str(fold.get("held_out_sequence", ""))
        if held_out:
            lookup[held_out] = fold
    return lookup


def _checkpoint_fold_info(
    checkpoint_path: Path,
    *,
    fold_lookup: dict[str, dict[str, Any]],
    default_fold: dict[str, Any] | None,
) -> tuple[int, str]:
    match = CHECKPOINT_VAL_PATTERN.search(checkpoint_path.stem)
    if match is not None:
        held_out = match.group("sequence")
        fold_info = fold_lookup.get(held_out, default_fold or {})
        fold_index = int(fold_info.get("fold", 0))
        return fold_index, held_out
    if default_fold is not None:
        return int(default_fold.get("fold", 0)), str(default_fold.get("held_out_sequence", "unknown"))
    if len(fold_lookup) == 1:
        only = next(iter(fold_lookup.values()))
        return int(only.get("fold", 0)), str(only.get("held_out_sequence", "unknown"))
    return 0, "unknown"


def _discover_checkpoints(
    results_dir: Path,
    *,
    mode: Literal["best", "all"],
    checkpoint_glob: str | None,
) -> list[Path]:
    checkpoints_dir = results_dir / "checkpoints"
    if not checkpoints_dir.is_dir():
        raise FileNotFoundError(f"Missing checkpoints directory: {checkpoints_dir}")

    if checkpoint_glob is not None:
        paths = sorted(checkpoints_dir.glob(checkpoint_glob))
    elif mode == "best":
        paths = sorted(checkpoints_dir.glob("*.pt"))
    else:
        paths = sorted(checkpoints_dir.rglob("*.pt"))

    if not paths:
        raise FileNotFoundError(f"No checkpoint .pt files found under {checkpoints_dir}")
    return paths


def _read_summary_by_fold(results_dir: Path) -> dict[int, dict[str, Any]]:
    summary_path = results_dir / "summary.csv"
    if not summary_path.exists():
        return {}
    frame = pd.read_csv(summary_path)
    by_fold: dict[int, dict[str, Any]] = {}
    if "fold" not in frame.columns:
        return by_fold
    for _, row in frame.iterrows():
        try:
            fold_index = int(row["fold"])
        except (TypeError, ValueError):
            continue
        by_fold[fold_index] = row.to_dict()
    return by_fold


def _parse_fraction_from_filename(path: Path) -> float | None:
    match = CHECKPOINT_PCT_PATTERN.search(path.stem)
    if match is None:
        return None
    return int(match.group("digits")) / 100.0


def _resolve_synthetic_size_fraction(
    checkpoint_path: Path,
    *,
    fold_index: int,
    summary_by_fold: dict[int, dict[str, Any]],
    cli_fallback: float | None,
    config_fallback: float | None,
) -> float:
    from_name = _parse_fraction_from_filename(checkpoint_path)
    if from_name is not None:
        return from_name

    summary_row = summary_by_fold.get(fold_index, {})
    for column in ("best_subset_fraction", "best_synthetic_size_fraction", "synthetic_size_fraction"):
        if column in summary_row:
            value = summary_row[column]
            if value is not None and not (isinstance(value, float) and math.isnan(value)):
                try:
                    return float(value)
                except (TypeError, ValueError):
                    pass

    if cli_fallback is not None:
        return cli_fallback
    if config_fallback is not None and not math.isnan(config_fallback):
        return config_fallback
    return float("nan")


def _checkpoint_selection(checkpoint_path: Path) -> str:
    if "steps" in checkpoint_path.parts or CHECKPOINT_PCT_PATTERN.search(checkpoint_path.stem):
        return "curriculum_step"
    return "per_fold_best"


def _model_config_from_social_lstm_checkpoint(
    loaded: dict[str, Any],
    *,
    fallback: ModelConfig | None = None,
) -> ModelConfig:
    base = fallback if fallback is not None else ModelConfig()
    fields = asdict(base)
    raw_cfg = loaded.get("config")
    if isinstance(raw_cfg, dict):
        for key, value in raw_cfg.items():
            target_key = SOCIAL_LSTM_CONFIG_KEY_MAP.get(key, key)
            if target_key in fields:
                fields[target_key] = value
    return ModelConfig(**fields)


def _load_social_lstm_checkpoint(
    checkpoint_path: Path,
    *,
    device: torch.device,
    fallback_config: ModelConfig | None = None,
) -> tuple[torch.nn.Module, ModelConfig]:
    loaded = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(loaded, dict):
        raise TypeError(f"Unexpected checkpoint payload in {checkpoint_path}")

    state = loaded.get("model_state_dict") or loaded
    if not isinstance(state, dict):
        raise TypeError(f"Unexpected model state in {checkpoint_path}")

    cfg = _model_config_from_social_lstm_checkpoint(loaded, fallback=fallback_config)
    model_spec = get_model_spec("lstm")
    model = model_spec.build_model(cfg, device)
    if hasattr(model, "lstm") and not any(key.startswith("lstm.") for key in state):
        model.lstm.load_state_dict(state)
    else:
        model.load_state_dict(state)
    model.eval()
    return model, cfg


def _load_transformer_checkpoint(
    checkpoint_path: Path,
    *,
    device: torch.device,
    ignore_version: bool,
) -> tuple[HTPModel, ModelConfig]:
    if not ignore_version:
        return load_htp_model(checkpoint_path, device=device)

    from data.feature import MODEL_VERSION

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    checkpoint_version = ckpt.get("MODEL_VERSION", ckpt.get("feature_version"))
    if checkpoint_version != MODEL_VERSION:
        warnings.warn(
            f"Loading {checkpoint_path.name} with MODEL_VERSION={checkpoint_version!r} "
            f"(current={MODEL_VERSION!r}) because --ignore-version was set.",
            UserWarning,
            stacklevel=2,
        )
    cfg = ModelConfig(**ckpt["config"])
    model = HTPModel(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, cfg


def _model_config_from_run(run_config: dict[str, Any]) -> ModelConfig:
    raw = run_config.get("model_config", {})
    if not isinstance(raw, dict):
        return ModelConfig()
    return ModelConfig(**raw)


def _build_evaluation_context(
    *,
    run_id: str,
    model_name: str,
    model_family: str,
    model_variant: str,
    train_domain: str,
    test_domain: str,
    synthetic_size_fraction: float,
    seed: int,
    config: ModelConfig,
    fold: int,
    held_out_sequence: str,
    checkpoint_path: Path,
    checkpoint_selection: str,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "model_name": model_name,
        "model_family": model_family,
        "model_variant": model_variant,
        "train_domain": train_domain,
        "test_domain": test_domain,
        "synthetic_size_fraction": synthetic_size_fraction,
        "eval_stage": "evaluation",
        "checkpoint_selection": checkpoint_selection,
        "fold": fold,
        "held_out_sequence": held_out_sequence,
        "seed": seed,
        "split": "test",
        "checkpoint_path": str(checkpoint_path),
        "config_hash": config_hash(
            config,
            {
                "model": model_name,
                "seed": seed,
                "model_family": model_family,
                "model_variant": model_variant,
                "train_domain": train_domain,
                "test_domain": test_domain,
                "synthetic_size_fraction": synthetic_size_fraction,
                "checkpoint_path": str(checkpoint_path),
            },
        ),
        "dataset_fps": float("nan"),
        "trajectory_stride": config.trajectory_stride,
        "coordinate_space": "normalized",
        "notes": "",
    }


def _write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "checkpoint_path",
        "run_id",
        "fold",
        "held_out_sequence",
        "test_sequences",
        "synthetic_size_fraction",
        "n_rows",
        "checkpoint_selection",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_run(args: argparse.Namespace) -> Path:
    results_dir = args.results_dir.resolve()
    run_config = _read_run_config(results_dir)

    if args.lstm:
        architecture = "lstm"
    else:
        architecture = str(run_config.get("model") or run_config.get("model_name") or "transformer")
    if architecture not in ("transformer", "lstm"):
        raise NotImplementedError(
            f"Checkpoint evaluation for architecture {architecture!r} is not implemented yet; "
            "use --lstm for Social LSTM checkpoints or omit it for transformer.",
        )
    model_name = args.model_name or ("social_lstm" if architecture == "lstm" else architecture)

    model_family = args.model_family or str(run_config.get("model_family") or "unknown")
    model_variant = args.model_variant or str(run_config.get("model_variant") or "unknown")
    train_domain = args.train_domain or str(run_config.get("train_domain") or "unknown")
    test_domain = args.test_domain
    seed = args.seed if args.seed is not None else int(run_config.get("seed", 0))
    config_fallback = run_config.get("synthetic_size_fraction")
    if config_fallback is not None:
        try:
            config_fallback = float(config_fallback)
        except (TypeError, ValueError):
            config_fallback = None

    if train_domain == "unknown" or test_domain == "unknown":
        warnings.warn(
            "train_domain or test_domain is 'unknown'; trajectory_stats RQ2/RQ3 will refuse inferential analysis.",
            UserWarning,
            stacklevel=2,
        )

    if args.held_out and args.sequences:
        warnings.warn(
            "--sequences is ignored when --held-out is set; each checkpoint uses its own held-out sequence.",
            UserWarning,
            stacklevel=2,
        )

    dataset_name = _resolve_test_dataset(test_domain, args.test_dataset)
    dataset = get_dataset(dataset_name)
    common_sequence_names: list[str] | None = None
    common_test_split: Split | None = None
    if not args.held_out:
        common_sequence_names = (
            [name.strip() for name in args.sequences.split(",") if name.strip()]
            if args.sequences
            else _default_test_sequence_names(dataset)
        )
        common_test_split = _build_test_split(dataset, common_sequence_names)
        _warn_train_test_overlap(run_config, common_sequence_names, train_domain, test_domain)

    checkpoint_paths = _discover_checkpoints(
        results_dir,
        mode=args.checkpoints,
        checkpoint_glob=args.checkpoint_glob,
    )
    summary_by_fold = _read_summary_by_fold(results_dir)
    fold_lookup = _fold_lookup(run_config)
    default_fold = next(iter(fold_lookup.values()), None)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model_spec = get_model_spec(architecture)
    batch_size = args.batch_size or int(run_config.get("batch_size", 1))
    input_mode = args.input_mode
    base_run_id = str(run_config.get("run_id") or results_dir.name)

    manifest_rows: list[dict[str, Any]] = []
    output_path = (args.output or (results_dir / "evaluation_units.csv")).resolve()

    print(f"Results dir: {results_dir}")
    print(f"Test dataset: {dataset_name}")
    if args.held_out:
        print("Evaluation mode: held-out (per-checkpoint LOO sequence)")
    else:
        print(f"Evaluation mode: common test set  sequences: {common_sequence_names}")
    print(f"Checkpoints: {len(checkpoint_paths)}  device: {device}")
    print(f"Output: {output_path}")

    total_rows = 0
    with EvaluationUnitsCsvWriter(output_path) as csv_writer:
        for index, checkpoint_path in enumerate(checkpoint_paths, start=1):
            checkpoint_path = checkpoint_path.resolve()
            fold_index, held_out_sequence = _checkpoint_fold_info(
                checkpoint_path,
                fold_lookup=fold_lookup,
                default_fold=default_fold,
            )
            if args.held_out:
                if held_out_sequence == "unknown":
                    raise ValueError(
                        f"Cannot resolve held-out sequence for {checkpoint_path.name}. "
                        "Use fold checkpoints named *__val_<sequence>.pt or ensure config.json defines folds.",
                    )
                test_sequence_names = [held_out_sequence]
                test_split = _build_test_split(dataset, test_sequence_names)
            else:
                test_sequence_names = common_sequence_names or []
                test_split = common_test_split
                if test_split is None:
                    raise RuntimeError("Common test split was not built.")

            synthetic_size_fraction = _resolve_synthetic_size_fraction(
                checkpoint_path,
                fold_index=fold_index,
                summary_by_fold=summary_by_fold,
                cli_fallback=args.synthetic_size_fraction,
                config_fallback=config_fallback,
            )
            checkpoint_selection = _checkpoint_selection(checkpoint_path)
            unique_run_id = f"{base_run_id}::{checkpoint_path.stem}"

            sequence_note = f" seq={held_out_sequence}" if args.held_out else ""
            print(
                f"[{index}/{len(checkpoint_paths)}] {checkpoint_path.name} "
                f"fold={fold_index} fraction={synthetic_size_fraction}{sequence_note}",
            )

            if architecture == "lstm":
                model, eval_config = _load_social_lstm_checkpoint(
                    checkpoint_path,
                    device=device,
                    fallback_config=_model_config_from_run(run_config),
                )
            else:
                model, eval_config = _load_transformer_checkpoint(
                    checkpoint_path,
                    device=device,
                    ignore_version=args.ignore_version,
                )
            context = _build_evaluation_context(
                run_id=unique_run_id,
                model_name=model_name,
                model_family=model_family,
                model_variant=model_variant,
                train_domain=train_domain,
                test_domain=test_domain,
                synthetic_size_fraction=synthetic_size_fraction,
                seed=seed,
                config=eval_config,
                fold=fold_index,
                held_out_sequence=held_out_sequence,
                checkpoint_path=checkpoint_path,
                checkpoint_selection=checkpoint_selection,
            )
            rows_before = csv_writer.row_count
            evaluate_split_for_model(
                model,
                model_spec,
                test_split,
                eval_config,
                batch_size=batch_size,
                device=device,
                input_mode=input_mode,
                max_samples=args.max_samples,
                progress_desc=f"eval {checkpoint_path.name}",
                evaluation_units_writer=csv_writer,
                evaluation_context=context,
                collect_metrics=False,
            )
            manifest_rows.append(
                {
                    "checkpoint_path": str(checkpoint_path),
                    "run_id": unique_run_id,
                    "fold": fold_index,
                    "held_out_sequence": held_out_sequence,
                    "test_sequences": ",".join(test_sequence_names),
                    "synthetic_size_fraction": synthetic_size_fraction,
                    "n_rows": csv_writer.row_count - rows_before,
                    "checkpoint_selection": checkpoint_selection,
                },
            )
            del model
        total_rows = csv_writer.row_count

    if total_rows == 0:
        raise RuntimeError("No evaluation unit rows were produced.")

    validate_evaluation_units_csv(output_path)

    manifest_path = output_path.with_name("evaluation_units_manifest.csv")
    _write_manifest(manifest_path, manifest_rows)

    print(f"Wrote {total_rows} rows across {len(checkpoint_paths)} checkpoint(s) -> {output_path}")
    print(f"Manifest -> {manifest_path}")
    return output_path


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    evaluate_run(args)


if __name__ == "__main__":
    main()
