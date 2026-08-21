from __future__ import annotations

import json
import math
import os
import re
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

'''
A messy collection of utils that i needed in all 3 notebooks for the reporting, mostly exporting of data formats to pandas

Nick Grebe i6377605
'''

METRIC_LABELS = {
    "mean_ade_until_horizon": "ADE",
    "mean_fde_at_horizon": "FDE",
}
METRIC_LABEL_ORDER = ["ADE", "FDE"]
MODEL_ORDER = ["LSTM", "STT"]
MODEL_COLORS = {"LSTM": "#4C78A8", "STT": "#F58518"}
FOLD_ORDER = ["MOT20-01", "MOT20-02", "MOT20-03", "MOT20-05"]
TRAIN_ORDER = ["real", "synthetic"]
TEST_ORDER = ["real", "synthetic"]
RQ3_METRIC_ORDER = ["mean_ade_until_horizon", "mean_fde_at_horizon"]
HORIZON_ORDER = [1.0, 2.0]
EFFECT_ORDER = ["train_domain", "test_domain", "train_domain:test_domain"]
TRAIN_LABELS = {"real": "real-trained", "synthetic": "synthetic-trained"}
DOMAIN_LABELS = {"real": "Real", "synthetic": "Synthetic"}
DOMAIN_COLORS = {"real": "#2f6f9f", "synthetic": "#c85a37"}


def configure_notebook(rq: str, *, font_size: int = 10, display_width: int = 160):
    mpl_config_dir = Path(tempfile.gettempdir()) / f"{rq}_matplotlib_cache"
    mpl_config_dir.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir.resolve()))

    import matplotlib.pyplot as plt

    pd.set_option("display.max_columns", 120)
    pd.set_option("display.width", display_width)
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.size": font_size,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    base_dir = reporting_base_dir(rq)
    figures_dir = base_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    print(f"Base directory: {base_dir.resolve()}")
    print(f"Figures directory: {figures_dir.resolve()}")
    return base_dir, figures_dir, plt


def reporting_base_dir(rq: str) -> Path:
    cwd = Path.cwd()
    if any(cwd.glob(f"{rq}_mot20-*")):
        return cwd

    for candidate in [cwd, *cwd.parents]:
        rq_dir = candidate / "Reporting" / rq
        if rq_dir.exists() and any(rq_dir.glob(f"{rq}_mot20-*")):
            return rq_dir

    raise FileNotFoundError(
        f"Could not find {rq}_mot20-* folders. Run from Reporting/{rq} "
        f"or from the project root containing Reporting/{rq}."
    )


def save_figure(fig, figures_dir: Path, base_dir: Path, stem: str) -> None:
    png_path = figures_dir / f"{stem}.png"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    print(f"Saved {png_path.relative_to(base_dir)}")


def metric_label(metric: str) -> str:
    return METRIC_LABELS.get(metric, metric)


def horizon_label(horizon) -> str:
    return f"{float(horizon):.1f}s"


def horizon_file_label(horizon) -> str:
    horizon = float(horizon)
    return f"{int(horizon)}s" if horizon.is_integer() else f"{horizon:g}s"


def sorted_unique(series: pd.Series) -> list:
    return sorted(series.dropna().unique().tolist())


def ordered_unique(values, preferred_order: list) -> list:
    present = list(pd.Series(values).dropna().unique())
    ordered = [item for item in preferred_order if item in present]
    return ordered + sorted([item for item in present if item not in ordered])


def require_columns(dataframe: pd.DataFrame, required: list[str] | set[str], context: str) -> None:
    missing = [column for column in required if column not in dataframe.columns]
    if missing:
        raise ValueError(
            f"Missing required column(s) in {context}: {missing}. "
            f"Available columns: {list(dataframe.columns)}"
        )


def first_present(dataframe: pd.DataFrame, candidates: list[str]) -> str | None:
    for candidate in candidates:
        if candidate in dataframe.columns:
            return candidate
    return None


def rename_first_available(dataframe: pd.DataFrame, target: str, candidates: list[str]) -> None:
    if target in dataframe.columns:
        return
    source = first_present(dataframe, candidates)
    if source is not None:
        dataframe.rename(columns={source: target}, inplace=True)


def split_label_from_folder(rq: str, path: Path) -> str:
    match = re.search(rf"{rq}_(mot20-.+)$", path.name.lower())
    return match.group(1).upper() if match else path.name


def split_sort_key(label: str):
    if "POOLED" in label:
        return (1, 10_000, label)
    match = re.search(r"MOT20-(\d+)", label)
    return (0, int(match.group(1)) if match else 9_999, label)


def read_json_if_present(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def find_numeric_budget_values(payload: dict) -> dict:
    found = {}

    def walk(value, prefix=""):
        if isinstance(value, dict):
            for key, child in value.items():
                walk(child, f"{prefix}.{key}" if prefix else str(key))
        elif isinstance(value, (int, float)) and "budget" in prefix.lower() and "pass" not in prefix.lower():
            found[prefix] = value

    walk(payload)
    return found


def find_rq1_results_file(split_dir: Path) -> Path:
    rq1_dir = split_dir / "rq1"
    candidates = [
        rq1_dir / "stats_rq1_results.csv",
        rq1_dir / "stats_rq1_pairwise_results.csv",
        *sorted(rq1_dir.glob("*rq1*results*.csv")),
        *sorted(rq1_dir.glob("*.csv")),
    ]
    for candidate in dict.fromkeys(path for path in candidates if path.exists()):
        return candidate
    raise FileNotFoundError(f"No RQ1 result CSV found in {rq1_dir}")


def normalize_rq1_columns(dataframe: pd.DataFrame, split_label: str, result_file: Path, base_dir: Path) -> pd.DataFrame:
    data = dataframe.copy()
    rename_first_available(data, "mean_lstm_error", ["mean_baseline_error", "mean_baseline", "baseline_mean_error"])
    rename_first_available(data, "mean_stt_error", ["mean_treatment_error", "mean_treatment", "treatment_mean_error"])
    rename_first_available(data, "relative_improvement_pct", ["percent_improvement", "improvement_pct"])
    rename_first_available(data, "p_value", ["permutation_p_value", "raw_p_value"])
    rename_first_available(data, "holm_p_value", ["adjusted_p_value", "corrected_p_value"])

    data["split"] = split_label
    data["split_folder"] = result_file.parents[1].name
    data["source_file"] = str(result_file.relative_to(base_dir))
    data["is_pooled"] = data["split"].str.contains("POOLED", case=False, na=False)
    if "significant_raw" not in data.columns and "p_value" in data.columns:
        data["significant_raw"] = pd.to_numeric(data["p_value"], errors="coerce") < 0.05
    if "significant_adjusted" not in data.columns:
        if "significant" in data.columns and "holm_p_value" in data.columns:
            data["significant_adjusted"] = data["significant"]
        elif "holm_p_value" in data.columns:
            data["significant_adjusted"] = pd.to_numeric(data["holm_p_value"], errors="coerce") < 0.05
    return data


def load_rq1_outputs(base_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_dirs = sorted(
        [path for path in base_dir.glob("rq1_mot20-*") if path.is_dir()],
        key=lambda path: split_sort_key(split_label_from_folder("rq1", path)),
    )
    if not split_dirs:
        raise FileNotFoundError(f"No rq1_mot20-* folders found in {base_dir}")

    result_frames = []
    summary_records = []
    for split_dir in split_dirs:
        split_label = split_label_from_folder("rq1", split_dir)
        result_file = find_rq1_results_file(split_dir)
        result_frames.append(normalize_rq1_columns(pd.read_csv(result_file), split_label, result_file, base_dir))

        summary_path = split_dir / "stats_pipeline_summary.json"
        summary = read_json_if_present(summary_path)
        summary_records.append(
            {
                "split": split_label,
                "summary_file": str(summary_path.relative_to(base_dir)) if summary_path.exists() else None,
                "aggregation_level": summary.get("aggregation", {}).get("level"),
                "aggregation_rows": summary.get("aggregation", {}).get("n_rows"),
                "max_horizon_s": summary.get("horizons", {}).get("max_horizon_s"),
                "budget_values": find_numeric_budget_values(summary),
            }
        )

    df = pd.concat(result_frames, ignore_index=True)
    require_columns(
        df,
        [
            "split",
            "aggregation_level",
            "baseline_model",
            "treatment_model",
            "metric",
            "n_pairs",
            "mean_lstm_error",
            "mean_stt_error",
            "p_value",
            "holm_p_value",
            "correction_method",
        ],
        "combined RQ1 result files",
    )
    if "actual_horizon_s" not in df.columns and "target_horizon_s" not in df.columns:
        raise ValueError("Expected actual_horizon_s or target_horizon_s in combined RQ1 result files.")

    summary_df = pd.DataFrame(summary_records).sort_values("split", key=lambda s: s.map(split_sort_key)).reset_index(drop=True)
    print(f"Loaded {len(split_dirs)} RQ1 split folder(s): {[split_label_from_folder('rq1', path) for path in split_dirs]}")
    print(f"Combined RQ1 result shape: {df.shape}")
    return df, summary_df


def derive_rq1_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rq1_df = df.copy()
    rq1_df["metric_label"] = rq1_df["metric"].map(metric_label).fillna(rq1_df["metric"])
    rq1_df["horizon_for_plot"] = rq1_df["actual_horizon_s"] if "actual_horizon_s" in rq1_df.columns else rq1_df["target_horizon_s"]
    rq1_df["horizon_label"] = rq1_df["horizon_for_plot"].map(lambda value: f"{value:.1f}s" if pd.notna(value) else "unknown")

    if "relative_improvement_pct" in rq1_df.columns and rq1_df["relative_improvement_pct"].notna().any():
        rq1_df["improvement_pct"] = rq1_df["relative_improvement_pct"]
    else:
        rq1_df["improvement_pct"] = 100 * (rq1_df["mean_lstm_error"] - rq1_df["mean_stt_error"]) / rq1_df["mean_lstm_error"]

    rq1_df["significance_label"] = np.select(
        [rq1_df["holm_p_value"] < 0.001, rq1_df["holm_p_value"] < 0.01, rq1_df["holm_p_value"] < 0.05],
        ["Holm p < 0.001", "Holm p < 0.01", "Holm p < 0.05"],
        default="n.s.",
    )
    split_level_df = rq1_df.loc[~rq1_df["is_pooled"]].copy()
    pooled_df = rq1_df.loc[rq1_df["is_pooled"]].copy()
    return rq1_df, split_level_df if not split_level_df.empty else rq1_df.copy(), pooled_df


def results_to_long(dataframe: pd.DataFrame) -> pd.DataFrame:
    base_columns = ["split", "metric_label", "horizon_for_plot", "horizon_label", "is_pooled"]
    lstm = dataframe[base_columns + ["mean_lstm_error"]].rename(columns={"mean_lstm_error": "mean_error"})
    stt = dataframe[base_columns + ["mean_stt_error"]].rename(columns={"mean_stt_error": "mean_error"})
    lstm["model_label"] = "LSTM"
    stt["model_label"] = "STT"
    return pd.concat([lstm, stt], ignore_index=True)


def sem(values: pd.Series) -> float:
    clean = values.dropna()
    return np.nan if len(clean) <= 1 else float(clean.std(ddof=1) / math.sqrt(len(clean)))


def diverging_norm(values: np.ndarray):
    from matplotlib.colors import TwoSlopeNorm

    finite = values[np.isfinite(values)]
    if finite.size == 0 or finite.min() >= 0 or finite.max() <= 0:
        return None
    return TwoSlopeNorm(vmin=float(finite.min()), vcenter=0.0, vmax=float(finite.max()))


def budget_line_from_summary(summary_df: pd.DataFrame) -> float | None:
    budget_values = []
    for values in summary_df.get("budget_values", []):
        if isinstance(values, dict):
            budget_values.extend(values.values())
    unique_values = sorted(set(float(value) for value in budget_values if pd.notna(value)))
    return unique_values[0] if len(unique_values) == 1 else None


def normalize_rq2_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    df = dataframe.copy()
    candidate_names = {
        "mean_real_error": ["mean_real", "mean_baseline_error", "baseline_mean_error", "real_mean_error"],
        "mean_synthetic_error": ["mean_synthetic", "mean_treatment_error", "treatment_mean_error", "synthetic_mean_error"],
        "mean_difference": ["mean_diff_synthetic_minus_real", "mean_diff"],
        "equivalence_margin": ["delta", "absolute_equivalence_margin"],
    }
    for canonical, candidates in candidate_names.items():
        rename_first_available(df, canonical, candidates)

    if "mean_difference" not in df.columns and {"mean_real_error", "mean_synthetic_error"}.issubset(df.columns):
        df["mean_difference"] = df["mean_synthetic_error"] - df["mean_real_error"]
    if "equivalence_margin" not in df.columns and {"margin_value", "mean_real_error"}.issubset(df.columns):
        df["equivalence_margin"] = df["margin_value"] * df["mean_real_error"]
    if "actual_horizon_s" not in df.columns and "target_horizon_s" in df.columns:
        df["actual_horizon_s"] = df["target_horizon_s"]

    require_columns(
        df,
        [
            "split",
            "synthetic_size_fraction",
            "metric",
            "actual_horizon_s",
            "n_pairs",
            "mean_real_error",
            "mean_synthetic_error",
            "mean_difference",
            "equivalence_margin",
            "equivalent",
        ],
        "combined RQ2 TOST results",
    )
    return df


def load_rq2_outputs(base_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict]]:
    result_frames = []
    threshold_frames = []
    summaries = {}

    split_dirs = sorted(base_dir.glob("rq2_mot20-*"), key=lambda p: split_sort_key(split_label_from_folder("rq2", p)))
    if not split_dirs:
        raise FileNotFoundError(f"No split folders matching rq2_mot20-* found in {base_dir}")

    for split_dir in split_dirs:
        split = split_label_from_folder("rq2", split_dir)
        tost_path = split_dir / "rq2" / "stats_rq2_tost_equivalence_results.csv"
        threshold_path = split_dir / "rq2" / "stats_rq2_equivalence_threshold_summary.csv"
        summary_path = split_dir / "stats_pipeline_summary.json"
        if not tost_path.exists():
            raise FileNotFoundError(f"Missing TOST results for {split}: {tost_path}")

        result_df = pd.read_csv(tost_path)
        result_df["split"] = split
        result_frames.append(result_df)
        if threshold_path.exists():
            threshold_df = pd.read_csv(threshold_path)
            threshold_df["split"] = split
            threshold_frames.append(threshold_df)
        if summary_path.exists():
            summaries[split] = read_json_if_present(summary_path)

    combined = normalize_rq2_columns(pd.concat(result_frames, ignore_index=True))
    thresholds = pd.concat(threshold_frames, ignore_index=True) if threshold_frames else pd.DataFrame()
    print(f"Loaded {len(combined):,} RQ2 result rows from {combined['split'].nunique()} splits.")
    print(f"Loaded {len(thresholds):,} threshold summary rows.")
    return combined, thresholds, summaries


def derive_rq2_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    numeric_columns = [
        "synthetic_size_fraction",
        "actual_horizon_s",
        "n_pairs",
        "mean_real_error",
        "mean_synthetic_error",
        "mean_difference",
        "equivalence_margin",
    ]
    for numeric_column in numeric_columns:
        df[numeric_column] = pd.to_numeric(df[numeric_column], errors="coerce")

    bad = df.loc[df["mean_real_error"].isna() | (df["mean_real_error"] == 0)]
    if not bad.empty:
        columns = ["split", "metric", "actual_horizon_s", "synthetic_size_fraction", "mean_real_error"]
        raise ValueError("mean_real_error must be non-missing and nonzero:\n" + bad[columns].to_string(index=False))

    df["relative_gap_pct"] = 100 * (df["mean_synthetic_error"] - df["mean_real_error"]) / df["mean_real_error"]
    df["equivalence_margin_pct"] = 100 * df["equivalence_margin"] / df["mean_real_error"]
    df["metric_label"] = df["metric"].map(metric_label).fillna(df["metric"])
    df["horizon_label"] = df["actual_horizon_s"].map(lambda value: f"{value:.1f}s" if pd.notna(value) else "unknown")
    return df


def data_at_horizon(dataframe: pd.DataFrame, horizon_s: float) -> pd.DataFrame:
    subset = dataframe[np.isclose(dataframe["actual_horizon_s"], horizon_s)].copy()
    if subset.empty:
        raise ValueError(f"No rows found for actual_horizon_s={horizon_s}. Available: {sorted_unique(dataframe['actual_horizon_s'])}")
    return subset


def fraction_labels(values) -> list[str]:
    return [f"{value:.2g}" for value in values]


RQ3_TABLE_FILES = {
    "factorial": "stats_rq3_factorial_results.csv",
    "domain": "stats_rq3_domain_summary.csv",
    "posthoc": "stats_rq3_posthoc_results.csv",
    "readiness": "stats_rq3_readiness_report.csv",
}
RQ3_REQUIRED_COLUMNS = {
    "factorial": {
        "metric",
        "target_horizon_s",
        "actual_horizon_s",
        "effect",
        "statistic",
        "p_value",
        "corrected_p_value",
        "significant",
        "correction_method",
        "n_units",
    },
    "domain": {
        "metric",
        "target_horizon_s",
        "actual_horizon_s",
        "train_domain",
        "test_domain",
        "n_units",
        "mean_error",
        "ci_lower",
        "ci_upper",
        "se_error",
    },
    "posthoc": {
        "metric",
        "target_horizon_s",
        "actual_horizon_s",
        "comparison",
        "mean_a",
        "mean_b",
        "mean_difference",
        "percent_difference_relative_to_a",
        "ci_lower",
        "ci_upper",
        "comparison_type",
    },
    "readiness": {"train_domain", "test_domain", "required_cell", "n_units"},
}


def fold_from_folder(folder: Path) -> str:
    return folder.name.replace("rq3_", "").upper()


def load_rq3_outputs(base_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    fold_dirs = sorted(base_dir.glob("rq3_mot20-*"), key=fold_from_folder)
    if not fold_dirs:
        raise FileNotFoundError(f"No rq3_mot20-* folders found in {base_dir}")

    missing_files = [folder / "rq3" / filename for folder in fold_dirs for filename in RQ3_TABLE_FILES.values() if not (folder / "rq3" / filename).exists()]
    if missing_files:
        raise FileNotFoundError("Missing required RQ3 CSV files:\n" + "\n".join(str(path) for path in missing_files))

    loaded = {}
    for table_name, filename in RQ3_TABLE_FILES.items():
        frames = []
        for folder in fold_dirs:
            path = folder / "rq3" / filename
            frame = pd.read_csv(path)
            missing_columns = RQ3_REQUIRED_COLUMNS[table_name] - set(frame.columns)
            if missing_columns:
                raise ValueError(f"{path} is missing columns: {sorted(missing_columns)}")
            frame.insert(0, "fold", fold_from_folder(folder))
            frames.append(frame)
        loaded[table_name] = pd.concat(frames, ignore_index=True)

    factorial, domain, posthoc, readiness = (loaded[name].copy() for name in ["factorial", "domain", "posthoc", "readiness"])
    for frame in [factorial, domain, posthoc]:
        frame["metric_label"] = frame["metric"].map(metric_label)
        frame["horizon_label"] = frame["target_horizon_s"].map(horizon_label)
        frame["target_horizon_s"] = frame["target_horizon_s"].astype(float)
        frame["actual_horizon_s"] = frame["actual_horizon_s"].astype(float)
    for frame in [domain, readiness]:
        frame["condition_label"] = frame["train_domain"] + " train / " + frame["test_domain"] + " test"
        frame["cell_label"] = frame["train_domain"] + "->" + frame["test_domain"]
    readiness["required_cell"] = readiness["required_cell"].astype(bool)

    folds = ordered_unique(factorial["fold"], FOLD_ORDER)
    print(f"Loaded RQ3 tables from {base_dir}")
    return factorial, domain, posthoc, readiness, folds


def percent_ci(row: pd.Series) -> tuple[float, float]:
    if pd.isna(row["mean_a"]) or row["mean_a"] == 0:
        return np.nan, np.nan
    return row["ci_lower"] / row["mean_a"] * 100, row["ci_upper"] / row["mean_a"] * 100
