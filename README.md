# Pedestrian Trajectory Forecasting with a Spatio-Temporal Transformer

> Comparative multi-agent trajectory forecasting with a Spatio-Temporal Transformer, Social-LSTM, and a Kalman inspection baseline on real and synthetic pedestrian data.

This undergraduate group research project at Maastricht University investigates whether synthetic pedestrian data can substitute for real-world data in trajectory forecasting. We designed a custom crowd-aware **Spatio-Temporal Transformer (STT)**, benchmarked it against a recurrent **Social-LSTM** on **MOT20** (real), and then studied STT training and transfer with **MOTSynth** (synthetic). Models observe 15 sampled timesteps and predict 10 future timesteps; performance is reported with track-level Average Displacement Error (ADE) and Final Displacement Error (FDE). The study tests model choice, the effect of increasing synthetic training volume, and cross-domain generalisation rather than claiming a new state of the art.

## Key Results

The table reports the final report's pooled track-level results. “STT reduction” is the relative error reduction against Social-LSTM on held-out MOT20 sequences. “Synthetic gap” is the relative error increase of the STT trained with the full (100%) MOTSynth curriculum compared with the real-trained STT on real MOT20 data. Lower error is better.

| Prediction horizon | STT ADE reduction vs Social-LSTM | STT FDE reduction vs Social-LSTM | 100% MOTSynth ADE gap vs real-trained | 100% MOTSynth FDE gap vs real-trained |
|---:|---:|---:|---:|---:|
| 0.6 s | 36.5% | 37.3% | +36.8% | +37.9% |
| 1.0 s | 37.3% | 37.9% | +37.4% | +37.6% |
| 2.0 s | 37.6% | 37.3% | +36.3% | +34.9% |

- The STT's improvement over Social-LSTM was significant for every reported ADE/FDE horizon (`Holm-adjusted p = 6.0 × 10⁻⁴`). The report also cautions that the models use different input representations and objectives, so this is a comparison of modelling approaches rather than a controlled architectural ablation.
- None of the 20 MOTSynth training fractions (5% to 100%) entered the pre-specified ±5% practical-equivalence margin relative to real-data training.
- Cross-domain tests showed an asymmetry: the real-trained STT was comparatively robust on both domains, whereas the synthetic-trained STT performed strongly in-domain but degraded on real data.
- These findings are specific to MOT20, MOTSynth, the available compute budget, and this experimental setup; they do not establish that synthetic data is generally ineffective.

Percentages and inferential claims above are taken from the final report (Abstract and Figures 9, 12, and 15). Absolute errors, per-sequence breakdowns, confidence intervals, latency results, and limitations are retained in the [full scientific report](./Report.pdf).

## Research Questions and Experimental Design

1. **Architecture comparison:** Does the crowd-aware STT achieve lower ADE/FDE than a Social-LSTM of comparable scale on MOT20? The reported models have 477,315 and 461,378 trainable parameters, respectively.
2. **Synthetic versus real training:** Can an STT trained on progressively larger MOTSynth fractions reach practical equivalence with the real-trained STT on real data? MOTSynth was split 80/10/10 and evaluated at 20 cumulative fractions from 5% to 100%.
3. **Cross-domain generalisation:** How do real-trained and synthetic-trained STTs behave on real and synthetic held-out test domains in a 2 × 2 train-domain/test-domain design?

The final MOT20 comparison uses sequence-level leave-one-out evaluation over the four annotated sequences `MOT20-01`, `MOT20-02`, `MOT20-03`, and `MOT20-05`. Predictions are evaluated at 0.6 s, 1.0 s, and 2.0 s. The intended 0.5 s target maps to 0.6 s because a stride of five at 25 FPS produces 0.2 s prediction increments.

## Spatio-Temporal Transformer

The STT processes all pedestrians in a scene and predicts future displacements in seven stages:

1. A **motion feature stem** lifts nine kinematic features—normalised foot position, displacement, velocity, speed, and sine/cosine heading—from 9 to 128 dimensions while masking missing observations.
2. A **spatial interaction block** builds a masked `k = 3` nearest-neighbour graph at each observed timestep. An edge MLP scores relative position, velocity, distance, speed, and heading features, then aggregates socially relevant messages.
3. **Temporal patching** uses a 1D convolution (`kernel = stride = 3`) to compress 15 observations into five local-motion tokens.
4. **Crowd-context fusion** masked-mean-pools across pedestrians and injects a lightweight scene-level context into every valid agent.
5. Ten learned **future query tokens** are appended to the five history tokens, with learned history/future positional embeddings.
6. A three-layer, four-head **Transformer encoder** (`d_model = 128`, feed-forward dimension 256) lets future queries attend to the encoded trajectory history.
7. A **regression head** predicts ten 2D displacements. Their cumulative sum from the last observed position reconstructs absolute future coordinates.

Training uses a masked dual Smooth-L1 objective over displacement and reconstructed position. The implementation is in [`App/model/`](./App/model/), with the unified forward pass in [`App/model/model.py`](./App/model/model.py).

## Datasets and Protocol

- **MOT20:** real fixed-camera footage with dense pedestrian crowds and MOTChallenge bounding-box/track annotations. Bottom-centre bounding-box coordinates represent pedestrian foot positions.
- **MOTSynth:** GTA V-based synthetic pedestrian sequences in a MOT-compatible annotation format. It supplies scalable, privacy-preserving synthetic training data for the volume and transfer experiments.
- **Windows:** 15 observed samples and 10 predicted samples at a five-frame stride. At 25 FPS this corresponds to 3.0 s of observation and 2.0 s of prediction.
- **Preprocessing:** coordinates are normalised against the larger image dimension with centring for non-square frames; missing observations are zero-padded and masked. Nine motion features feed the STT, while Social-LSTM uses normalised 2D foot positions.
- **Evaluation:** unseen sequences/domains are evaluated at the track level to reduce dependence from overlapping trajectory windows. The final report should be consulted for the complete split construction and experimental limitations.

## Statistical Evaluation

[`trajectory_stats/`](./trajectory_stats/) contains the analysis framework used to move beyond single aggregate scores:

- schema validation and horizon-aware ADE/FDE aggregation at window, track, or scene level;
- pairing of model errors on matching evaluation units;
- paired sign-flip permutation tests and paired bootstrap confidence intervals for STT-vs-Social-LSTM comparisons;
- Holm family-wise error correction across metrics and horizons;
- paired Two One-Sided Tests (TOST) against a ±5% relative equivalence margin for real-vs-synthetic training;
- a 2 × 2 train/test-domain permutation analysis with 10,000 permutations;
- latency, throughput, and real-time budget summaries where runtime measurements are available.

The exported unit schema records model/run metadata, held-out sequence, track, prediction window, horizon, metric, and runtime information. The final numeric evaluation-unit CSVs are not included in this repository, so the report is the authoritative record of the completed experiments; regenerating the statistical outputs requires newly trained checkpoints and exported evaluation units.

## My Contribution

This was a six-person group research project at Maastricht University, not an individual repository. **Nick Grebe** (the repository owner) conceptualised the STT and its overall architecture; integrated the model's unified forward pass; implemented core model work including the spatial interaction block and compatibility/masking changes; helped design and set up the experiments and training/evaluation workflow; designed and implemented the `trajectory_stats` analysis framework; and wrote substantial portions of the report, including the detailed STT architecture, statistical analysis, limitations, future work, and conclusion.

The implementation was collaborative. Source headers credit **Claire Bams** for the motion stem and future-token implementations, **Noah Nuelandt** for temporal patching and crowd-context fusion, and **Néo Deward** for the Transformer encoder and regression head, with later integration edits by Nick. The Social-LSTM implementation and training/evaluation code are credited primarily to **Keez Cuijpers**, reviewed by **Ciprian Driscu**. The application and visualisation interface was primarily handled by other team members. More granular writer, reviewer, and contributor statements are preserved in the [full report](./Report.pdf) and source-code headers.

## Repository Structure

```text
App/model/                    STT architecture and loss
App/benchmarks/               Social-LSTM and Kalman baselines
App/train.py                  Shared training orchestration
App/evaluate.py               Checkpoint evaluation-unit export
App/evaluation/               Forecast metrics and evaluation schemas
trajectory_stats/             RQ1–RQ3 statistical analysis tooling
Dataprep/                     MOT parsing, feature exploration, and baseline export
Reporting/                    Utilities used to prepare report figures
App/templates/, App/static/   Optional Flask visualisation interface
outputs/                      Small retained run-configuration records
Report.pdf                    Final scientific report
```

## Reproducing the Experiments

### 1. Environment

Python 3.11 or newer is required. The repository's preprocessing notebook records Python 3.11.9; the previous `>=3.14` metadata was not required by the code.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`. Commands below assume the repository root and a POSIX shell. On Windows, set both the repository root and `App` on `PYTHONPATH` using the platform path separator.

### 2. Data preparation

The application data loader attempts to download MOT20 or MOTSynth on first use and caches it under `App/MOTsource/`. These datasets are large (approximately 5 GB and 80 GB as documented in the report); verify the original dataset terms and available storage before downloading.

For the standalone tabular preprocessing flow:

```bash
python Dataprep/MOT_loader.py
```

This downloads MOT20 to `Dataprep/MOTsource/` and writes frame/annotation tables to `data/tables/`. The Social-LSTM trainer additionally expects `data/prepared_baseline/{train,val,test}/baseline_trajectory.csv`; those files are generated by running [`Dataprep/DataPrep.ipynb`](./Dataprep/DataPrep.ipynb) from the repository root after the MOT20 download.

### 3. STT training

```bash
PYTHONPATH=.:App \
MODEL=transformer \
TRAIN_DOMAIN=real \
TEST_DOMAIN=real \
python App/train.py
```

Useful controls include `MODEL_LOOKBACK`, `MODEL_FUTURE_STEPS`, `TRAJECTORY_STRIDE`, `TRAIN_STEPS`, `VAL_INTERVAL`, `BATCH_SIZE`, `LEAVE_ONE_OUT`, `FOLD_SEQUENCE`, `TRAIN_OUTPUT_DIR`, and `EXPORT_EVALUATION_UNITS`. By default, fold checkpoints and run metadata are written below `outputs/loo/<run_id>/`; no pretrained STT checkpoint is included in the repository.

### 4. Social-LSTM training

After generating `data/prepared_baseline/`:

```bash
PYTHONPATH=.:App python -m App.benchmarks.social_lstm.train
```

The benchmark writes fold checkpoints and metrics below `outputs/loo_social_lstm/<run_id>/` and copies its selected inference checkpoint to `checkpoints/social_lstm/best_model.pt`. Relevant controls include `OBS_LEN`, `PRED_LEN`, `SUBSAMPLE_STEP`, `NUM_EPOCHS`, `BATCH_SIZE`, `LEAVE_ONE_OUT`, `SEED`, and `SOCIAL_LSTM_OUTPUT_DIR`.

**Exact-paper caveat:** the final RQ1 report pools four MOT20 held-out sequences (`01`, `02`, `03`, `05`), while the current standalone Social-LSTM trainer defaults to `01`, `02`, and `03` and expects the notebook-generated baseline tables. Reproducing the exact final comparison therefore requires matching the final four-fold prepared data/evaluation setup; it is not a one-command run from a clean clone.

### 5. Evaluation and statistics

Evaluate a completed STT leave-one-out run and export the units consumed by the statistical framework:

```bash
PYTHONPATH=.:App python App/evaluate.py outputs/loo/<run_id> \
  --test-domain real --train-domain real --held-out
```

For Social-LSTM checkpoints, point to its run directory and add `--lstm`. Cross-domain RQ2/RQ3 evaluation requires evaluating the real- and synthetic-trained checkpoints on the same intended domains with correct `--train-domain`, `--test-domain`, model-name, and synthetic-fraction metadata. The analysis entry points are [`trajectory_stats/pipeline.py`](./trajectory_stats/pipeline.py), [`trajectory_stats/rq1.py`](./trajectory_stats/rq1.py), [`trajectory_stats/rq2.py`](./trajectory_stats/rq2.py), and [`trajectory_stats/rq3.py`](./trajectory_stats/rq3.py).

### 6. Optional visualisation interface

The Flask interface can inspect ground truth, a Kalman forecast, and any discoverable trained checkpoints:

```bash
PYTHONPATH=.:App python App/app.py
```

A fresh clone has no `.pt` files, so only the non-learned Kalman channel is available initially. Place a deliberately selected STT or Social-LSTM checkpoint under `checkpoints/` before expecting learned-model inference; do not assume that the reported model weights are bundled.

## Full Report

[Read the full scientific report](./Report.pdf) for the complete methodology, figures, statistical results, limitations, references, and detailed contribution statements.

### Project authors

Ciprian Driscu, Claire Bams, Keez Cuijpers, Néo Deward, Nick Grebe, and Noah Nuelandt — Maastricht University, Data Science & Artificial Intelligence, 2026.
