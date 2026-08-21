# Project_2-2_25-26_DSAI_Group-12

Full scientific report visible at Report.pdf

Multi-object trajectory prediction using a transformer-based Hybrid Trajectory Prediction model and a Social LSTM benchmark, with a Flask GUI for visual inspection and comparison.

## Quick Start

```bash
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If you run the project from the command line, set the repository root on `PYTHONPATH` first:

```bash
$env:PYTHONPATH = "."
```

## For Examiners

Use this path if you mainly want to inspect the finished system, compare the models, or check where the outputs are written.

Start the GUI with:

```bash
$env:PYTHONPATH = "."
python -u App/app.py
```

The Flask app in `App/app.py` serves the viewer and exposes API routes for ground-truth tracks, forecast tracks, per-channel statistics, sequence metadata, and frame images.

In the UI, the model selector switches between the available channels. The interface supports the Kalman baseline, the transformer/HTP model, and the Social LSTM benchmark. The frame player lets you step through sequences, and the side panels show combined forecasts, future predictions, and statistics for the currently selected model.

Model checkpoints are loaded from the following locations by default:

- Main model: `checkpoints/htp_model.pt`
- Social LSTM benchmark for the GUI: `App/checkpoints/social_lstm/best_model.pt`
- Social LSTM benchmark for the test harness: `checkpoints/social_lstm/best_model.pt`

If you only want to inspect outputs, the most useful files are:

- `outputs/` for training runs and metrics
- `checkpoints/` for the active inference weights
- `App/checkpoints/social_lstm/best_model.pt` for the Social LSTM file used by the GUI

## For Training

Use this section when you want to rebuild the dataset tables, train either model, or regenerate checkpoints.

### 1. Prepare the data

Raw MOT data is expected under `Dataprep/MOTsource/`. The helper script in `Dataprep/MOT_loader.py` can download MOT20 automatically if it is missing, then builds the tabular dataset used by the project.

Run it with:

```bash
$env:PYTHONPATH = "."
python Dataprep/MOT_loader.py
```

This will:

- download and cache `MOT20` if needed
- parse `seqinfo.ini`, `gt.txt`, and `det.txt`
- generate `data/tables/mot_frame_table.csv`
- generate `data/tables/mot_annotation_table.csv`
- split MOT20 into train, validation, and test groups

The built-in split is:

- train: `MOT20-01`, `MOT20-02`, `MOT20-03`
- val: `MOT20-05`
- test: `MOT20-04`, `MOT20-06`, `MOT20-07`, `MOT20-08`

If you use the notebook in `Dataprep/DataPrep.ipynb`, it follows the same data-loading and table-building flow.

### 2. Train the main model

`App/train.py` is the main trainer. By default it trains the transformer-based HTP model and writes a checkpoint to `checkpoints/htp_model.pt`.

Example:

```bash
$env:PYTHONPATH = "."
$env:MODEL = "transformer"
python -u -m App.train
```

Useful environment variables:

- `MODEL`: `transformer` for the main model, `lstm` for the benchmark wrapper
- `MODEL_LOOKBACK`: observation length, default `15`
- `MODEL_FUTURE_STEPS`: prediction horizon, default `10`
- `TRAJECTORY_STRIDE`: sampling stride, default `5`
- `TRAIN_STEPS`: training steps
- `VAL_INTERVAL`: validation interval
- `BATCH_SIZE`: batch size
- `LEAVE_ONE_OUT`: enable leave-one-out evaluation, default `true`
- `TRAIN_OUTPUT_DIR`: where run folders are written, default `outputs/loo`
- `HTP_CHECKPOINT`: override the main checkpoint path, default `checkpoints/htp_model.pt`

The main trainer writes a run folder under `outputs/loo/<run_id>/` containing the run configuration, per-fold checkpoints, and metrics. The final weights are also saved to `checkpoints/htp_model.pt` for GUI and evaluation use.

### 3. Train the Social LSTM benchmark

To train the benchmark model directly:

```bash
$env:PYTHONPATH = "."
$env:MODEL = "lstm"
$env:NUM_EPOCHS = "100"
$env:BATCH_SIZE = "16"
$env:LEAVE_ONE_OUT = "true"
python -u -m App.train
```

You can also call the benchmark trainer directly:

```bash
$env:PYTHONPATH = "."
python -u -m App.benchmarks.social_lstm.train
```

Key benchmark variables:

- `OBS_LEN`: observation length, default `15`
- `PRED_LEN`: prediction length, default `10`
- `SUBSAMPLE_STEP`: frame stride, default `5`
- `NUM_EPOCHS`: epochs, default `100` in the benchmark script
- `BATCH_SIZE`: batch size, default `16` in the benchmark script
- `LEAVE_ONE_OUT`: leave-one-out cross-validation, default `true`
- `SEED`: random seed, default `42`
- `SOCIAL_LSTM_OUTPUT_DIR`: output root, default `outputs/loo_social_lstm`
- `SOCIAL_LSTM_CHECKPOINT_DIR`: checkpoint copy location, default `App/checkpoints/social_lstm`

The Social LSTM trainer writes:

- `checkpoints/`: per-fold checkpoints such as `social_lstm__val_MOT20-01.pt`
- `fold_metrics.csv`: per-epoch training history
- `summary.csv`: best validation loss per fold
- `config.json`: full run configuration
- `best_model.pt`: copied best fold checkpoint

The benchmark inference code loads `App/checkpoints/social_lstm/best_model.pt` by default, so keep that file in sync with the checkpoint you want the GUI to use.

### 4. Evaluate checkpoints

For the main model, use `App/evaluate.py` to export evaluation units and metrics for a completed run directory.

For the benchmark, use `App/benchmarks/social_lstm/eval_checkpoints.py` to score Social LSTM fold checkpoints with the native benchmark pipeline.

Example benchmark evaluation:

```bash
$env:PYTHONPATH = "."
python -u -m App.benchmarks.social_lstm.eval_checkpoints --dir outputs/loo_social_lstm/<run_id>/checkpoints --max-samples 128
```

This produces `evaluation_results.csv` alongside the checkpoint folder.

### 5. Regenerate the benchmark inference checkpoint

If you want the GUI to use a different Social LSTM fold, copy it into the inference location:

```bash
Copy-Item outputs/loo_social_lstm/<run_id>/checkpoints/social_lstm__val_MOT20-03.pt `
  App/checkpoints/social_lstm/best_model.pt -Force
```

## Project Structure

```
App/
├── app.py                      # Flask GUI and API routes
├── train.py                    # Main training orchestrator
├── evaluate.py                 # Run evaluation/export for completed training runs
├── model/
│   ├── model.py                # HTPModel / transformer-based trajectory model
│   ├── config.py               # Shared model hyperparameters
│   └── loss.py                 # Loss functions
├── benchmarks/
│   ├── evaluate.py             # Generic benchmark evaluator helpers
│   └── social_lstm/
│       ├── train.py            # Social LSTM benchmark trainer
│       ├── eval_checkpoints.py # Benchmark checkpoint evaluation
│       ├── inference.py        # Social LSTM inference for the GUI
│       └── model.py            # Social LSTM architecture
├── data/
│   ├── reader.py               # Sequence / track loading
│   ├── loader.py               # Dataset loading helpers
│   └── feature.py              # Feature extraction and model versioning
├── evaluation/
│   ├── eval.py                 # Generic trajectory evaluation helpers
│   ├── evaluation_units.py     # CSV export / validation helpers
│   └── stats.py                # GUI statistics calculations
├── utils/
│   ├── modelutils.py           # Checkpoint loading / saving helpers
│   ├── horizon_targets.py      # Horizon mapping helpers
│   └── viz.py                  # Visualization helpers
└── checkpoints/
    └── social_lstm/
        └── best_model.pt       # GUI Social LSTM inference checkpoint

Dataprep/
├── DataPrep.ipynb              # Notebook version of the preprocessing flow
└── MOT_loader.py               # MOT20 / MOTSynth parsing and table generation
```

## Checkpoints And Outputs

- `checkpoints/htp_model.pt` is the default main-model checkpoint loaded by the GUI and evaluation helpers.
- `App/checkpoints/social_lstm/best_model.pt` is the default Social LSTM checkpoint used by the GUI.
- `checkpoints/social_lstm/best_model.pt` is the checkpoint path used by the standalone benchmark test harness.
- `outputs/loo/...` stores main-model training runs and metrics.
- `outputs/loo_social_lstm/...` stores benchmark training runs and metrics.

If you retrain a model, update the matching checkpoint file before opening the GUI so the viewer and evaluation code stay aligned with the latest weights.
