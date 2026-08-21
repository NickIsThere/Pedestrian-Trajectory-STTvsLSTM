"""Evaluate Social LSTM fold checkpoints with the native benchmark pipeline.

This evaluator intentionally avoids the generic App/train compatibility adapter so
checkpoints are measured with the same model and data regime used during benchmark
training.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Subset

try:
    from .dataset import SocialLSTMDataset, collate_batch
    from .model import SocialLSTM
except Exception:
    import sys

    repo_root = Path(__file__).resolve().parents[3]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)

    from App.benchmarks.social_lstm.dataset import SocialLSTMDataset, collate_batch
    from App.benchmarks.social_lstm.model import SocialLSTM

from App.model.config import ModelConfig

_STT_DEFAULTS = ModelConfig()


def _masked_sums(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Main writer: Keez Cuijpers
    Reviewer: Ciprian Driscu
    Contributors: 

    Compute masked squared error sums and the number of valid coordinates.
    """
    mask_expanded = mask.unsqueeze(-1).to(dtype=pred.dtype)
    squared_error = (pred - target).pow(2) * mask_expanded
    sum_squared_error = squared_error.sum()
    valid_coords = mask_expanded.sum()
    return sum_squared_error, valid_coords


def _masked_ade_fde(
    pred_pos: torch.Tensor,
    target_pos: torch.Tensor,
    mask: torch.Tensor,
    *,
    step_count: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Main writer: Keez Cuijpers
    Reviewer: Ciprian Driscu
    Contributors: 

    Compute masked ADE/FDE totals and counts for a rollout.
    """
    if step_count is not None:
        pred_pos = pred_pos[:, :step_count, :, :]
        target_pos = target_pos[:, :step_count, :, :]
        mask = mask[:, :step_count, :]

    valid = mask.to(dtype=torch.bool)
    displacement = torch.linalg.vector_norm(pred_pos - target_pos, dim=-1)

    ade_sum = displacement[valid].sum() if valid.any() else displacement.new_tensor(0.0)
    ade_count = valid.sum().to(dtype=pred_pos.dtype)

    # FDE: final valid horizon per (batch, pedestrian)
    valid_by_agent = valid.any(dim=1)
    if valid_by_agent.any():
        horizon_idx = torch.arange(valid.size(1), device=valid.device).view(1, -1, 1)
        last_idx = torch.where(valid, horizon_idx, torch.zeros_like(horizon_idx)).max(dim=1).values
        final_errors = displacement.gather(1, last_idx.unsqueeze(1)).squeeze(1)
        fde_sum = final_errors[valid_by_agent].sum()
        fde_count = valid_by_agent.sum().to(dtype=pred_pos.dtype)
    else:
        fde_sum = displacement.new_tensor(0.0)
        fde_count = displacement.new_tensor(0.0)

    return ade_sum, ade_count, fde_sum, fde_count


def _constant_velocity_rollout(
    obs_pos: torch.Tensor,
    obs_mask: torch.Tensor,
    pred_len: int,
) -> torch.Tensor:
    """
    Main writer: Keez Cuijpers
    Reviewer: Ciprian Driscu
    Contributors: 

    Extrapolate future positions with a constant-velocity baseline.
    """
    # Shapes: obs_pos (B, obs_len, P, 2), obs_mask (B, obs_len, P)
    B, obs_len, P, _ = obs_pos.shape
    valid = obs_mask.to(dtype=torch.bool)
    time_idx = torch.arange(obs_len, device=obs_pos.device).view(1, obs_len, 1)
    masked_idx = torch.where(valid, time_idx, torch.zeros_like(time_idx))
    last_idx = masked_idx.max(dim=1).values

    prev_valid = valid & (time_idx < last_idx.unsqueeze(1))
    prev_idx = torch.where(prev_valid, time_idx, torch.zeros_like(time_idx)).max(dim=1).values

    gather_last = last_idx.unsqueeze(1).unsqueeze(-1).expand(B, 1, P, 2)
    gather_prev = prev_idx.unsqueeze(1).unsqueeze(-1).expand(B, 1, P, 2)

    last_pos = obs_pos.gather(dim=1, index=gather_last).squeeze(1)
    prev_pos = obs_pos.gather(dim=1, index=gather_prev).squeeze(1)

    has_two = prev_valid.any(dim=1).unsqueeze(-1)
    step_delta = torch.where(has_two, last_pos - prev_pos, torch.zeros_like(last_pos))

    steps = torch.arange(1, pred_len + 1, device=obs_pos.device, dtype=obs_pos.dtype).view(1, pred_len, 1, 1)
    return last_pos.unsqueeze(1) + steps * step_delta.unsqueeze(1)


def _extract_held_out_sequence(checkpoint: dict, checkpoint_path: Path) -> str:
    """
    Main writer: Keez Cuijpers
    Reviewer: Ciprian Driscu
    Contributors: 

    Recover the held-out sequence name from checkpoint metadata.
    """
    held_out = checkpoint.get("held_out_sequence")
    if isinstance(held_out, str) and held_out:
        return held_out

    metadata = checkpoint.get("metadata")
    if isinstance(metadata, dict):
        held_out = metadata.get("held_out_sequence")
        if isinstance(held_out, str) and held_out:
            return held_out

    name = checkpoint_path.stem
    marker = "__val_"
    if marker in name:
        return name.split(marker, 1)[1]

    raise KeyError(f"Could not infer held-out sequence for checkpoint {checkpoint_path}")


def _step_for_seconds(*, seconds: float, fps: float, subsample_step: int, pred_len: int) -> int:
    """
    Main writer: Keez Cuijpers
    Reviewer: Ciprian Driscu
    Contributors: 

    Convert a time horizon in seconds to a sampled prediction step count.
    """
    sampled_fps = fps / max(1, subsample_step)
    return max(1, min(pred_len, int(round(seconds * sampled_fps))))


def evaluate_checkpoint(
    checkpoint_path: Path,
    *,
    baseline_dir: Path,
    device: torch.device,
    batch_size: int,
    max_samples: int,
) -> dict[str, float | int | str]:
    """
    Main writer: Keez Cuijpers
    Reviewer: Ciprian Driscu
    Contributors: 

    Evaluate one checkpoint against the held-out sequence windows.
    """
    loaded = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = loaded.get("model_state_dict") or loaded
    if not isinstance(state, dict):
        raise TypeError(f"Unexpected checkpoint payload in {checkpoint_path}")

    config = loaded.get("config") if isinstance(loaded.get("config"), dict) else {}
    obs_len = int(config.get("obs_len", _STT_DEFAULTS.lookback))
    pred_len = int(config.get("pred_len", _STT_DEFAULTS.future_steps))
    hidden_size = int(config.get("hidden_size", _STT_DEFAULTS.hidden_dim))
    embedding_dim = int(config.get("embedding_dim", _STT_DEFAULTS.head_hidden_dim))
    grid_size = int(config.get("grid_size", 4))
    subsample_step = int(config.get("subsample_step", _STT_DEFAULTS.trajectory_stride))
    fps = float(config.get("frame_rate", 25.0))

    held_out_sequence = _extract_held_out_sequence(loaded, checkpoint_path)
    dataset = SocialLSTMDataset(
        baseline_dir=baseline_dir,
        split="train",
        obs_len=obs_len,
        pred_len=pred_len,
        subsample_step=subsample_step,
    )

    val_indices = [
        index
        for index, window in enumerate(dataset.windows)
        if window["sequence"] == held_out_sequence
    ]
    if not val_indices:
        raise RuntimeError(f"No validation windows found for sequence {held_out_sequence}")

    subset = Subset(dataset, val_indices)
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_batch,
        num_workers=min(4, max(1, (os.cpu_count() or 2) // 4)),
        pin_memory=(device.type == "cuda"),
        persistent_workers=False,
    )

    model = SocialLSTM(
        obs_len=obs_len,
        pred_len=pred_len,
        hidden_size=hidden_size,
        embedding_dim=embedding_dim,
        grid_size=grid_size,
    ).to(device)
    model.load_state_dict(state)
    model.eval()

    delta_se_sum = torch.tensor(0.0, device=device)
    pos_se_sum = torch.tensor(0.0, device=device)
    valid_coords_total = torch.tensor(0.0, device=device)

    ade_sum = torch.tensor(0.0, device=device)
    ade_count = torch.tensor(0.0, device=device)
    fde_sum = torch.tensor(0.0, device=device)
    fde_count = torch.tensor(0.0, device=device)

    cv_ade_sum = torch.tensor(0.0, device=device)
    cv_ade_count = torch.tensor(0.0, device=device)
    cv_fde_sum = torch.tensor(0.0, device=device)
    cv_fde_count = torch.tensor(0.0, device=device)

    horizon_totals: dict[str, dict[str, torch.Tensor]] = {}
    for label, seconds in (("0p5s", 0.5), ("1p0s", 1.0), ("2p0s", 2.0), ("full", -1.0)):
        horizon_totals[label] = {
            "ade_sum": torch.tensor(0.0, device=device),
            "ade_count": torch.tensor(0.0, device=device),
            "fde_sum": torch.tensor(0.0, device=device),
            "fde_count": torch.tensor(0.0, device=device),
        }

    sample_count = 0
    with torch.no_grad():
        for batch in loader:
            obs_pos = batch["obs_positions"].to(device, non_blocking=True)
            obs_mask = batch["obs_masks"].to(device, non_blocking=True)
            gt_disp = batch["pred_displacements"].to(device, non_blocking=True)
            pred_mask = batch["pred_masks"].to(device, non_blocking=True)

            pred_disp = model(obs_pos, obs_mask)

            last_obs = obs_pos[:, -1, :, :].unsqueeze(1)
            pred_pos = last_obs + torch.cumsum(pred_disp, dim=1)
            gt_pos = last_obs + torch.cumsum(gt_disp, dim=1)
            cv_pos = _constant_velocity_rollout(obs_pos, obs_mask, pred_len)

            batch_delta_se, batch_valid_coords = _masked_sums(pred_disp, gt_disp, pred_mask)
            batch_pos_se, _ = _masked_sums(pred_pos, gt_pos, pred_mask)
            delta_se_sum += batch_delta_se
            pos_se_sum += batch_pos_se
            valid_coords_total += batch_valid_coords

            a_sum, a_count, f_sum, f_count = _masked_ade_fde(pred_pos, gt_pos, pred_mask)
            ade_sum += a_sum
            ade_count += a_count
            fde_sum += f_sum
            fde_count += f_count

            c_a_sum, c_a_count, c_f_sum, c_f_count = _masked_ade_fde(cv_pos, gt_pos, pred_mask)
            cv_ade_sum += c_a_sum
            cv_ade_count += c_a_count
            cv_fde_sum += c_f_sum
            cv_fde_count += c_f_count

            for label, seconds in (("0p5s", 0.5), ("1p0s", 1.0), ("2p0s", 2.0), ("full", -1.0)):
                if seconds < 0:
                    step_count = pred_len
                else:
                    step_count = _step_for_seconds(
                        seconds=seconds,
                        fps=fps,
                        subsample_step=subsample_step,
                        pred_len=pred_len,
                    )
                h_a_sum, h_a_count, h_f_sum, h_f_count = _masked_ade_fde(
                    pred_pos,
                    gt_pos,
                    pred_mask,
                    step_count=step_count,
                )
                horizon_totals[label]["ade_sum"] += h_a_sum
                horizon_totals[label]["ade_count"] += h_a_count
                horizon_totals[label]["fde_sum"] += h_f_sum
                horizon_totals[label]["fde_count"] += h_f_count

            sample_count += int(obs_pos.size(0))
            if max_samples > 0 and sample_count >= max_samples:
                break

    if valid_coords_total.item() <= 0:
        raise RuntimeError(f"No valid coordinates found while evaluating {checkpoint_path}")

    delta_mse = float((delta_se_sum / valid_coords_total).item())
    pos_mse = float((pos_se_sum / valid_coords_total).item())
    position_rmse = float(torch.sqrt(pos_se_sum / valid_coords_total).item())

    results: dict[str, float | int | str] = {
        "checkpoint": checkpoint_path.name,
        "held_out_sequence": held_out_sequence,
        "val_loss": delta_mse,
        "delta_loss": delta_mse,
        "position_loss": pos_mse,
        "delta_mse": delta_mse,
        "position_rmse": position_rmse,
        "ade": float((ade_sum / ade_count.clamp(min=1.0)).item()),
        "fde": float((fde_sum / fde_count.clamp(min=1.0)).item()),
        "cv_ade": float((cv_ade_sum / cv_ade_count.clamp(min=1.0)).item()),
        "cv_fde": float((cv_fde_sum / cv_fde_count.clamp(min=1.0)).item()),
        "val_samples": sample_count,
    }

    for label in ("0p5s", "1p0s", "2p0s", "full"):
        totals = horizon_totals[label]
        results[f"ade_{label}"] = float((totals["ade_sum"] / totals["ade_count"].clamp(min=1.0)).item())
        results[f"fde_{label}"] = float((totals["fde_sum"] / totals["fde_count"].clamp(min=1.0)).item())

    return results


def evaluate_folder(
    folder: Path,
    *,
    baseline_dir: Path,
    device: torch.device,
    max_samples: int = 0,
) -> None:
    """
    Main writer: Keez Cuijpers
    Reviewer: Ciprian Driscu
    Contributors: 

    Evaluate every checkpoint in a folder and write aggregate results.
    """
    all_results: list[dict[str, float | int | str]] = []
    batch_size = int(os.environ.get("TEST_BATCH_SIZE", "32"))

    for ckpt in sorted(folder.glob("*.pt")):
        print(f"\nLoading checkpoint {ckpt}")
        try:
            metrics = evaluate_checkpoint(
                ckpt,
                baseline_dir=baseline_dir,
                device=device,
                batch_size=batch_size,
                max_samples=max_samples,
            )
        except Exception as exc:
            print(f"Failed evaluating {ckpt.name}: {exc}")
            continue

        print(f"Results for {ckpt.name}: {metrics}")
        all_results.append(metrics)

    if all_results:
        csv_path = folder / "evaluation_results.csv"
        fieldnames = list(all_results[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\n[OK] Evaluation results saved to {csv_path}")
    else:
        print("No checkpoints produced results.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dir", type=Path, required=True)
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument(
        "--baseline-dir",
        type=Path,
        default=Path(__file__).resolve().parents[3] / "Data" / "prepared_baseline",
    )
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    evaluate_folder(
        args.dir,
        baseline_dir=args.baseline_dir,
        device=device,
        max_samples=args.max_samples,
    )
