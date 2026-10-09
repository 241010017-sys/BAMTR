# BAMTR

**Behavior-Aware Mobile Phone Trajectory Recovery via Adaptive Historical Prior Reasoning**

BAMTR recovers missing trajectory coordinates and predicts stay/move semantics from sparse observations and personal mobility history. This PyTorch implementation provides data preparation, model training, evaluation, and trajectory recovery at supplied query timestamps.

## Model

- **Behavior-aware encoding:** a shared BiGRU encodes current and historical trajectories using coordinates, time, and stay/move features.
- **Current-conditioned Historical Attention:** selects relevant information from earlier trajectories of the same user.
- **Context Adaptor:** integrates current observations and historical priors through gated residual adaptation.
- **Dual-context decoding:** combines global context and local observational attention to predict coordinates and mobility semantics jointly.

Calendar context is derived from recording dates using the mainland China 2023 working-day calendar.

## Installation

Use Python 3.10-3.12. Run all commands from the repository root:

```bash
python -m pip install -e .
python -m bamtr --help
```

The reference environment is defined in [environment.yml](environment.yml). The main training configurations use CUDA when available and otherwise use the CPU; the quick-start configuration uses the CPU.

## Quick start

Run the pipeline with the bundled synthetic examples:

```bash
python -m bamtr prepare --config configs/smoke.yaml
python -m bamtr train --config configs/smoke.yaml
python -m bamtr evaluate --checkpoint runs/smoke/best.pt --split test
python -m bamtr plot --run-dir runs/smoke --output runs/smoke/loss.png
```

The example uses a small model and a short training schedule for a pipeline check.

## Data

Place daily CSV files in `data/raw/`. The default configurations read `2023-04-01.csv` through `2023-04-14.csv`.

| Column | Description |
| --- | --- |
| `ID` | User identifier |
| `lng` | Longitude in degrees |
| `lat` | Latitude in degrees |
| `time` | Local recording timestamp |
| `tid` | 30-second time-slot index since midnight |

The loader groups records by user and date and selects strictly earlier same-user trajectories as history. It generates stay/move features automatically. Users are split into training, validation, and test sets at a 6:2:2 ratio; coordinate normalization is fitted on training users.

Observation masks retain the configured fraction of points and preserve endpoints. Only missing points contribute to training losses and evaluation metrics. Calendar features come from dates; stay/move input features come from available observations.

Configure paths, dates, and column names under `data` in the YAML files. See the [data protocol](docs/data_protocol.md) for timestamp handling, sampling, and semantic labels. The public examples are synthetic; local data and training outputs are excluded by `.gitignore`.

## Training and evaluation

| Configuration | Retention rate | Stay duration | Stay radius |
| --- | ---: | ---: | ---: |
| `configs/bamtr_20.yaml` | 20% | 60 min | 150 m |
| `configs/bamtr_40.yaml` | 40% | 90 min | 100 m |
| `configs/bamtr_60.yaml` | 60% | 120 min | 100 m |

Train and evaluate the 40% configuration:

```bash
python -m bamtr prepare --config configs/bamtr_40.yaml
python -m bamtr train --config configs/bamtr_40.yaml
python -m bamtr evaluate --checkpoint runs/bamtr_40/best.pt --split test
python -m bamtr plot --run-dir runs/bamtr_40 --output runs/bamtr_40/loss.png
```

Default training uses Adam, a learning rate of `0.001`, batch size `256`, and `30` epochs. The coordinate and semantic losses are combined with a semantic weight of `10`. Teacher forcing decreases from `0.8` to `0.3`.

Each run saves configurations, logs, and checkpoints. `best.pt` is selected by validation MAE. Evaluation reports MAE, RMSE, Location Recall, and stay/move classification metrics.

Use an unused `--run-dir` for each new experiment. Override individual settings with `--set`, for example:

```bash
python -m bamtr train --config configs/bamtr_40.yaml --set training.batch_size=32 --run-dir runs/bamtr_40_batch32
```

Resume an interrupted run from its saved configuration and latest checkpoint:

```bash
python -m bamtr train --config runs/bamtr_40/config.yaml --resume runs/bamtr_40/latest.pt
```

## Trajectory recovery

After the quick-start run, recover the example query points:

```bash
python -m bamtr predict --checkpoint runs/smoke/best.pt --observations examples/inference/observations.csv --queries examples/inference/queries.csv --history-dir examples/synthetic --output runs/smoke/recovered.csv
```

Query files contain `ID,time`; include `tid` for minute-resolution timestamps. Query timestamps must be unique, lie between the first and last observations, and differ from observed timestamps. Predictions are aligned with these timestamps and merged with unchanged observed points. The output includes an `is_observed` flag and predicted stay probabilities for recovered points.

## Tests

```bash
python -m pip install -e ".[dev]"
python -m pytest
```

## Documentation

- [Model architecture](docs/architecture.md)
- [Data preparation and sampling](docs/data_protocol.md)
- [Implementation settings and reproducibility](docs/reproducibility.md)
- [Validation report](docs/validation_report.md)

## License

The code is released under the [MIT License](LICENSE). Dataset access and redistribution are governed separately.
