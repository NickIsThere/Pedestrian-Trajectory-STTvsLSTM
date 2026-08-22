"""
Training script for Social LSTM — aligned to STT config for RQ1.

Alignment with SpaTial Transformer (STT):
  - obs_len=15, pred_len=10, subsample_step=5
    → at MOT20 25fps each step = 0.2s
    → observation = 3.0s, prediction horizon = 2.0s
    → checkpoints at 0.2s intervals, covering 0.5 / 1.0 / 2.0s targets
  - hidden_size=128  (matches STT hidden_dim=128)
  - batch_size=16, num_epochs=100  (matches config.json)
  - Leave-one-out cross-validation over MOT20-01 / 02 / 03
  - Early stopping: patience=10 on val loss
  - LR schedule: ReduceLROnPlateau factor=0.5, patience=5
  - Gradient clipping: max norm 1.0
  - Teacher forcing during training (model.py supports it)

Outputs (under outputs/loo_social_lstm/<run_id>/):
  checkpoints/social_lstm__val_MOT20-0{1,2,3}.pt  — one per fold
  fold_metrics.csv   — per-epoch history for every fold
  summary.csv        — best val loss per fold
  config.json        — full config, compatible with discovery pipeline
checkpoints/social_lstm/best_model.pt  — copy of best fold
"""

from __future__ import annotations

import json
import os
import shutil
import time
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

try:
    from .dataset import SocialLSTMDataset, collate_batch
    from .model import SocialLSTM
except Exception:
    import sys
    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from App.benchmarks.social_lstm.dataset import SocialLSTMDataset, collate_batch
    from App.benchmarks.social_lstm.model import SocialLSTM

from App.model.config import ModelConfig

_STT_DEFAULTS = ModelConfig()


# ── Config ────────────────────────────────────────────────────────────────────
_NUM_EPOCHS    = int(os.environ.get("NUM_EPOCHS",    100))
_BATCH_SIZE    = int(os.environ.get("BATCH_SIZE",    16))
_SUBSAMPLE     = int(os.environ.get("SUBSAMPLE_STEP", str(_STT_DEFAULTS.trajectory_stride)))
_OBS_LEN       = int(os.environ.get("OBS_LEN",       str(_STT_DEFAULTS.lookback)))
_PRED_LEN      = int(os.environ.get("PRED_LEN",      str(_STT_DEFAULTS.future_steps)))
_HIDDEN_SIZE   = int(os.environ.get("HIDDEN_SIZE",   str(_STT_DEFAULTS.hidden_dim)))
_EMBED_DIM     = int(os.environ.get("EMBEDDING_DIM", str(_STT_DEFAULTS.head_hidden_dim)))
_LR            = float(os.environ.get("LEARNING_RATE", "1e-3"))
_LOO           = os.environ.get("LEAVE_ONE_OUT", "true").lower() in {"1", "true", "yes"}
_SEED          = int(os.environ.get("SEED", 42))
_RUN_ID        = os.environ.get("RUN_ID")
_OUTPUT_DIR    = Path(os.environ.get("SOCIAL_LSTM_OUTPUT_DIR", "outputs/loo_social_lstm"))
_CKPT_DIR      = Path(os.environ.get(
    "SOCIAL_LSTM_CHECKPOINT_DIR",
    str(Path(__file__).resolve().parents[3] / "checkpoints" / "social_lstm"),
))

CONFIG: dict = {
    # ── temporal alignment with STT ──────────────────────────────────────────
    "obs_len":        _OBS_LEN,
    "pred_len":       _PRED_LEN,
    "subsample_step": _SUBSAMPLE,
    "frame_rate":     25.0,

    # ── model (hidden_dim matched to STT) ────────────────────────────────────
    "hidden_size":    _HIDDEN_SIZE,
    "embedding_dim":  _EMBED_DIM,
    "grid_size":      int(os.environ.get("SOCIAL_LSTM_GRID_SIZE", "4")),

    # ── training ─────────────────────────────────────────────────────────────
    "batch_size":     _BATCH_SIZE,
    "learning_rate":  _LR,
    "num_epochs":     _NUM_EPOCHS,
    "patience":       10,
    "grad_clip":      1.0,
    "lr_patience":    5,
    "lr_factor":      0.5,

    # ── LOO ──────────────────────────────────────────────────────────────────
    "loo_sequences":  ["MOT20-01", "MOT20-02", "MOT20-03"],
    "leave_one_out":  _LOO,
    "seed":           _SEED,

    # ── paths ─────────────────────────────────────────────────────────────────
    "device":         "cuda" if torch.cuda.is_available() else "cpu",
    "run_id":         _RUN_ID or f"lstm_loo_e{_NUM_EPOCHS}_bs{_BATCH_SIZE}_ss{_SUBSAMPLE}",
    "output_dir":     _OUTPUT_DIR,
    "checkpoint_dir": _CKPT_DIR,
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _amp_ctx(use_amp: bool):
    """
    Main writer: Keez Cuijpers
    Reviewer: Ciprian Driscu
    Contributors: 

    Return an autocast context when AMP is enabled.
    """
    return torch.amp.autocast(device_type="cuda") if use_amp else nullcontext()


def _make_loaders(
    dataset: SocialLSTMDataset,
    train_seqs: list[str],
    val_seqs: list[str],
    batch_size: int,
    use_amp: bool,
) -> tuple[DataLoader, DataLoader]:
    """
    Main writer: Keez Cuijpers
    Reviewer: Ciprian Driscu
    Contributors: 

    Build train and validation data loaders for the selected sequences.
    """
    train_set = set(train_seqs)
    val_set   = set(val_seqs)
    train_idx = [i for i, w in enumerate(dataset.windows) if w["sequence"] in train_set]
    val_idx   = [i for i, w in enumerate(dataset.windows) if w["sequence"] in val_set]
    nw_t = min(8, max(1, (os.cpu_count() or 2) // 2))
    nw_v = min(4, max(1, (os.cpu_count() or 2) // 4))
    return (
        DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True,
                   collate_fn=collate_batch, num_workers=nw_t,
                   pin_memory=use_amp, persistent_workers=(nw_t > 0)),
        DataLoader(Subset(dataset, val_idx),   batch_size=batch_size, shuffle=False,
                   collate_fn=collate_batch, num_workers=nw_v,
                   pin_memory=use_amp, persistent_workers=(nw_v > 0)),
    )


def _train_epoch(model, loader, optimizer, criterion, device, use_amp, scaler, grad_clip):
    """
    Main writer: Keez Cuijpers
    Reviewer: Ciprian Driscu
    Contributors: 

    Train the model for one epoch over the loader.
    """
    model.train()
    total = 0.0
    for batch in tqdm(loader, desc="  train", leave=False):
        obs_pos   = batch["obs_positions"].to(device, non_blocking=True)
        obs_mask  = batch["obs_masks"].to(device, non_blocking=True)
        pred_disp = batch["pred_displacements"].to(device, non_blocking=True)
        pred_mask = batch["pred_masks"].to(device, non_blocking=True)

        # Reconstruct absolute positions for teacher forcing
        last_obs  = obs_pos[:, -1:, :, :]                      # (B, 1, P, 2)
        pred_pos  = last_obs + torch.cumsum(pred_disp, dim=1)  # (B, T, P, 2)

        with _amp_ctx(use_amp):
            # Pass pred_positions for teacher forcing — model.py supports it
            preds = model(obs_pos, obs_mask,
                          pred_positions=pred_pos,
                          pred_masks=pred_mask)
            m    = pred_mask.unsqueeze(-1)
            loss = criterion(preds * m, pred_disp * m)

        optimizer.zero_grad(set_to_none=True)
        if scaler:
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        total += loss.item()
    return total / len(loader)


def _validate(model, loader, criterion, device, use_amp):
    """
    Main writer: Keez Cuijpers
    Reviewer: Ciprian Driscu
    Contributors: 

    Evaluate the model on the validation loader.
    """
    model.eval()
    total = 0.0
    with torch.no_grad():
        for batch in tqdm(loader, desc="  val  ", leave=False):
            obs_pos   = batch["obs_positions"].to(device, non_blocking=True)
            obs_mask  = batch["obs_masks"].to(device, non_blocking=True)
            pred_disp = batch["pred_displacements"].to(device, non_blocking=True)
            pred_mask = batch["pred_masks"].to(device, non_blocking=True)
            with _amp_ctx(use_amp):
                # No teacher forcing at validation — free-running rollout
                preds = model(obs_pos, obs_mask)
                m     = pred_mask.unsqueeze(-1)
                loss  = criterion(preds * m, pred_disp * m)
            total += loss.item()
    return total / len(loader)


def _train_fold(
    fold_idx: int,
    held_out: str,
    train_seqs: list[str],
    dataset: SocialLSTMDataset,
    ckpt_dir: Path,
    fold_rows: list[dict],
) -> tuple[float, Path]:
    """
    Main writer: Keez Cuijpers
    Reviewer: Ciprian Driscu
    Contributors: 

    Train one leave-one-out fold and persist its best checkpoint.
    """
    device  = CONFIG["device"]
    use_amp = (device == "cuda")

    print(f"\n{'='*72}")
    print(f"Fold {fold_idx} | held-out: {held_out} | train: {train_seqs}")
    print(f"{'='*72}")

    train_loader, val_loader = _make_loaders(
        dataset, train_seqs, [held_out], CONFIG["batch_size"], use_amp,
    )
    print(f"  Windows — train: {len(train_loader.dataset)}  val: {len(val_loader.dataset)}")

    model = SocialLSTM(
        obs_len=CONFIG["obs_len"],
        pred_len=CONFIG["pred_len"],
        hidden_size=CONFIG["hidden_size"],
        embedding_dim=CONFIG["embedding_dim"],
        grid_size=CONFIG["grid_size"],
    ).to(device)
    print(f"  Parameters: {model.get_parameter_count():,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=CONFIG["lr_factor"], patience=CONFIG["lr_patience"],
    )
    criterion = nn.MSELoss()
    scaler    = torch.amp.GradScaler("cuda") if use_amp else None

    best_val   = float("inf")
    no_improve = 0
    ckpt_path  = ckpt_dir / f"social_lstm__val_{held_out}.pt"

    print(f"\n  {'Ep':>4} | {'Train':>10} | {'Val':>10} | {'LR':>8} | {'s':>5} | Best")
    print(f"  {'-'*58}")

    for epoch in range(CONFIG["num_epochs"]):
        t0 = time.time()
        tr = _train_epoch(model, train_loader, optimizer, criterion, device, use_amp, scaler, CONFIG["grad_clip"])
        vl = _validate(model, val_loader, criterion, device, use_amp)
        scheduler.step(vl)
        ep_s = time.time() - t0
        lr   = optimizer.param_groups[0]["lr"]

        is_best = vl < best_val
        if is_best:
            best_val   = vl
            no_improve = 0
            torch.save({
                "epoch":              epoch + 1,
                "fold":               fold_idx,
                "held_out_sequence":  held_out,
                "model_state_dict":   model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss":           vl,
                "config":             {k: str(v) if isinstance(v, Path) else v
                                       for k, v in CONFIG.items()},
            }, ckpt_path)
        else:
            no_improve += 1

        fold_rows.append({
            "fold": fold_idx, "held_out_sequence": held_out,
            "epoch": epoch + 1, "train_loss": tr, "val_loss": vl,
            "lr": lr, "is_best": is_best,
        })

        print(f"  {epoch+1:4d} | {tr:10.6f} | {vl:10.6f} | {lr:8.2e} | {ep_s:5.1f}s | {'✓' if is_best else ''}")

        if no_improve >= CONFIG["patience"]:
            print(f"\n  Early stopping at epoch {epoch+1} "
                  f"(no improvement for {CONFIG['patience']} epochs)")
            break

    print(f"\n  Fold {fold_idx} best val loss: {best_val:.6f}  → {ckpt_path.name}")
    return best_val, ckpt_path


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    """
    Main writer: Keez Cuijpers
    Reviewer: Ciprian Driscu
    Contributors: 

    Run the full leave-one-out Social LSTM training workflow.
    """
    torch.manual_seed(CONFIG["seed"])

    run_dir  = CONFIG["output_dir"] / CONFIG["run_id"]
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    CONFIG["checkpoint_dir"].mkdir(parents=True, exist_ok=True)

    fps    = CONFIG["frame_rate"]
    stride = CONFIG["subsample_step"]
    print(f"[{datetime.now()}] Social LSTM — RQ1-aligned Training")
    print(f"CUDA      : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU       : {torch.cuda.get_device_name(0)}")
    print(f"Device    : {CONFIG['device']}")
    print(f"obs_len   : {CONFIG['obs_len']}  ({CONFIG['obs_len']*stride/fps:.1f}s at {fps:.0f}fps)")
    print(f"pred_len  : {CONFIG['pred_len']}  ({CONFIG['pred_len']*stride/fps:.1f}s at {fps:.0f}fps)")
    print(f"stride    : {stride} frames  ({stride/fps:.2f}s per step)")
    print(f"Horizons  : 0.5s=step{int(round(0.5*fps/stride))}  "
          f"1.0s=step{int(round(1.0*fps/stride))}  "
          f"2.0s=step{int(round(2.0*fps/stride))}")
    print(f"LOO       : {CONFIG['leave_one_out']}")
    print(f"Run ID    : {CONFIG['run_id']}")
    torch.backends.cudnn.benchmark = True

    # Resolve baseline_dir — try several candidate locations
    candidates = [
        Path(__file__).resolve().parents[3] / "Data"  / "prepared_baseline",
        Path(__file__).resolve().parents[3] / "data"  / "prepared_baseline",
        Path("Data") / "prepared_baseline",
        Path("data") / "prepared_baseline",
    ]
    baseline_dir = next((p for p in candidates if p.exists()), candidates[0])
    print(f"\nDataset   : {baseline_dir}")

    dataset = SocialLSTMDataset(
        baseline_dir=baseline_dir,
        split="train",
        obs_len=CONFIG["obs_len"],
        pred_len=CONFIG["pred_len"],
        subsample_step=CONFIG["subsample_step"],
    )

    all_seqs = sorted({w["sequence"] for w in dataset.windows})
    print(f"Sequences : {all_seqs}")

    loo_seqs = CONFIG["loo_sequences"]
    missing  = [s for s in loo_seqs if s not in all_seqs]
    if missing:
        raise ValueError(f"LOO sequences not found in dataset: {missing}\n"
                         f"Available: {all_seqs}")

    fold_rows:    list[dict] = []
    fold_results: list[tuple[str, float, Path]] = []

    if CONFIG["leave_one_out"]:
        for fold_idx, held_out in enumerate(loo_seqs):
            train_seqs = [s for s in loo_seqs if s != held_out]
            best_val, ckpt = _train_fold(fold_idx, held_out, train_seqs,
                                         dataset, ckpt_dir, fold_rows)
            fold_results.append((held_out, best_val, ckpt))
    else:
        held_out   = loo_seqs[-1]
        train_seqs = loo_seqs[:-1]
        best_val, ckpt = _train_fold(0, held_out, train_seqs,
                                     dataset, ckpt_dir, fold_rows)
        fold_results.append((held_out, best_val, ckpt))

    # Copy best fold to the repository-level inference checkpoint directory.
    best_held, best_loss, best_ckpt = min(fold_results, key=lambda t: t[1])
    dest = CONFIG["checkpoint_dir"] / "best_model.pt"
    shutil.copy2(best_ckpt, dest)
    print(f"\nBest fold : {best_held}  val_loss={best_loss:.6f}")
    print(f"Copied to : {dest}")

    # ── Write outputs ─────────────────────────────────────────────────────────
    import csv

    with open(run_dir / "fold_metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fold_rows[0].keys())
        writer.writeheader()
        writer.writerows(fold_rows)

    summary_rows = [
        {"fold": i, "held_out_sequence": h, "best_val_loss": v, "checkpoint": str(p)}
        for i, (h, v, p) in enumerate(fold_results)
    ]
    with open(run_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)

    config_out = {
        "model":                  "social_lstm",
        "model_family":           "recurrent",
        "model_variant":          "social_lstm_rq1",
        "train_domain":           "real",
        "test_domain":            "real",
        "run_id":                 CONFIG["run_id"],
        "seed":                   CONFIG["seed"],
        "synthetic_size_fraction": 0.0,
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in CONFIG.items()},
        "folds": [
            {"fold": i, "held_out_sequence": h, "best_val_loss": v}
            for i, (h, v, _) in enumerate(fold_results)
        ],
        "selected_folds":    [h for h, _, _ in fold_results],
        "metric_selection":  "lowest validation loss",
    }
    (run_dir / "config.json").write_text(
        json.dumps(config_out, indent=2, default=str), encoding="utf-8"
    )

    print(f"\nOutputs   : {run_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
