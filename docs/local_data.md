# Use the supplied local data

`BAMTR_Local_With_Data.zip` includes the 14 supplied original daily CSV files unchanged, named `2023-04-01.csv` through `2023-04-14.csv`, in `BAMTR/data/raw/`. `BAMTR_GitHub_Complete.zip` contains code, documentation, and synthetic examples only. The `data/` directory is already ignored by Git. The repository's MIT license covers its code and does not grant rights to the supplied trajectory data.

Extract the local archive and open a terminal in its `BAMTR` directory, where `pyproject.toml` and `configs/` are located. The paper-oriented configurations and `configs/real_data_smoke.yaml` already set `data.raw_dir: data/raw`. No `--raw-dir` argument or path edit is required.

## Install, prepare, inspect, and start a new run

Use Python 3.10-3.12 in an activated environment:

```bash
python -m pip install -e .
python -m bamtr prepare --config configs/bamtr_40.yaml
python -m bamtr inspect --config configs/bamtr_40.yaml
python -m bamtr train --config configs/bamtr_40.yaml --run-dir runs/bamtr_40_local
```

Preparation reads the raw CSVs and writes cleaned data and metadata to `data/processed/`; it leaves the originals unchanged. Training requires that preparation has completed. These commands describe how to run the local collection; packaging the files does not imply that a new training experiment has completed or reproduced the paper's results.

The explicit new run directory keeps existing `runs/bamtr_40` outputs intact. If `runs/bamtr_40_local` already contains a run, choose another unused directory. Training refuses to silently overwrite a nonempty run directory. When changing preprocessing settings, choose a separate `--prepared-dir` consistently for preparation, inspection, and training so existing runs retain their original prepared data.

After the new run produces a checkpoint, evaluate or plot it with:

```bash
python -m bamtr evaluate --checkpoint runs/bamtr_40_local/best.pt --split test
python -m bamtr plot --run-dir runs/bamtr_40_local --output runs/bamtr_40_local/loss.png
```

## Continue an existing run

Resume is available only when the run contains `latest.pt`. If an existing run contains only setup artifacts without this checkpoint, start a new run with an unused `--run-dir`, such as `runs/bamtr_40_local`, as shown above.

For an interrupted run that has saved `latest.pt`, use its resolved configuration and checkpoint in the same run directory. For example, once `runs/bamtr_40` contains that checkpoint:

```bash
python -m bamtr train --config runs/bamtr_40/config.yaml --resume runs/bamtr_40/latest.pt
```

For another run, replace both paths with that run's paths. Keep its prepared data and original experimental settings. Resume continues the original epoch schedule; it does not extend a completed run. Do not delete existing checkpoints or replace the saved configuration to start over; use a new run directory for a new experiment.

## How current and historical data are selected

All 14 daily files belong in the same `data/raw/` directory. The loader groups rows by user and date, then constructs each eligible target user-day automatically. The current input contains that day's retained observations. Historical inputs contain eligible trajectories from the same user on strictly earlier dates; same-day and future trajectories are excluded. No manual current/history directory split is needed.

The default requires at least one earlier eligible day. April 1 therefore supplies possible history but is not a target date within this collection. Later dates use the available earlier eligible days, subject to the configured history limits. Users remain separate across training, validation, and test partitions. See [the data protocol](data_protocol.md) for cleaning, observation masks, and history rules.
