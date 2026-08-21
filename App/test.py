from __future__ import annotations
import os
import sys
import csv
import copy
from pathlib import Path
import torch

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
from data.reader import get_dataset
from train import build_supervised_splits, evaluate_split_for_model, get_model_spec, model_config


def run_eval() -> None:
    dataset = get_dataset("MOT20")
    _, val_split = build_supervised_splits(dataset)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    checkpoints_dir = project_root / "checkpoints"

    #if need be here can had new model with path to pt file
    models_to_test = [
        {
            "name": "transformer_htp",
            "spec_name": "transformer",
            "path": checkpoints_dir / "htp_model.pt",
        },
        {
            "name": "social_lstm",
            "spec_name": "lstm",
            "path": checkpoints_dir / "social_lstm" / "best_model.pt",
        }
    ]

    all_results = []
    batch_size = int(os.environ.get("TEST_BATCH_SIZE", "32"))

    for target in models_to_test:
        ckpt_path = target["path"]

        if not ckpt_path.is_file():
            print(f"pt not found Skipping {target['name']}.")
            continue

        try:
            loaded = torch.load(ckpt_path, map_location=device, weights_only=False)
            state = loaded.get("model_state_dict") or loaded

            cfg = copy.deepcopy(model_config)
            if "config" in loaded:
                raw_cfg = loaded["config"]

                key_mapping = {
                    "obs_len": "lookback",
                    "pred_len": "future_steps",
                    "hidden_size": "hidden_dim",
                    "embedding_dim": "head_hidden_dim",
                }
                for k, v in raw_cfg.items():
                    target_k = key_mapping.get(k, k)
                    if hasattr(cfg, target_k):
                        setattr(cfg, target_k, v)

            model_spec = get_model_spec(target["spec_name"])
            model = model_spec.build_model(cfg, device)

            if hasattr(model, "lstm") and not any(k.startswith("lstm.") for k in state.keys()):
                model.lstm.load_state_dict(state)
            else:
                model.load_state_dict(state)

            model.eval()
        except Exception as e:
            print(f"was not able to load {target['name']} checkpoint: {e}")
            continue

        metrics = evaluate_split_for_model(
            model=model,
            model_spec=model_spec,
            split=val_split,
            config=cfg,
            batch_size=batch_size,
            device=device,
            input_mode="gt",
            progress_desc=f"Evaluating {target['name']}",
        )

        row = {
            "checkpoint": target["name"],
            "held_out_sequence": ",".join(val_split.sequences.keys()),
        }
        row.update(metrics)
        all_results.append(row)

        print(f"{target['name']} -> ADE: {metrics.get('ade', 0):.4f} | FDE: {metrics.get('fde', 0):.4f}")

    if not all_results:
        print("\nNo models were evaluated. check if the .pt files exist.")
        return

    output_csv = project_root / "evaluation_results_standardized.csv"
    fieldnames = list(all_results[0].keys())

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)


if __name__ == "__main__":
    run_eval()