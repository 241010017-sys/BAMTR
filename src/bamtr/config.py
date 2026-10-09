"""YAML configuration loading, validation and command-line overrides."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

DEFAULTS = {
    "seed": 42,
    "data": {
        "raw_dir": "data/raw", "file_pattern": "2023-04-*.csv",
        "prepared_dir": "data/processed", "date_start": "2023-04-01",
        "date_end": "2023-04-14",
        "columns": {"user": "ID", "longitude": "lng", "latitude": "lat",
                    "timestamp": "time", "time_id": "tid"},
        "min_points": 4, "max_points": None,
        "max_history_days": None, "min_history_days": 1,
        "split_ratios": [0.6, 0.2, 0.2], "split_seed": 42,
        "retention_rate": 0.4, "mask_seed": 42, "keep_endpoints": True,
        "resample_train_mask": False, "stay_time_minutes": 90.0,
        "stay_radius_m": 100.0, "time_id_seconds": 30.0,
        "timestamp_source": "auto", "holiday_calendar_path": None,
    },
    "model": {
        "encoder_hidden_size": 128, "decoder_hidden_size": 256,
        "num_layers": 1, "dropout": 0.5,
        "semantic_embedding_dim": 30, "environment_embedding_dim": 30,
        "coord_head_dim": 256, "query_time_conditioning": False,
        "use_history": True, "use_context_adaptor": True,
        "use_mobility": True, "use_environment": True,
        "use_local_attention": True,
    },
    "training": {
        "epochs": 30, "batch_size": 256, "learning_rate": 0.001,
        "weight_decay": 0.0, "lambda_semantic": 10.0,
        "teacher_forcing_start": 0.8, "teacher_forcing_end": 0.3,
        "gradient_clip": 5.0, "early_stopping_patience": 0,
        "device": "auto", "num_workers": 0, "cpu_threads": 4,
        "deterministic": True, "run_dir": "runs/bamtr_40",
        "max_train_batches": None, "max_eval_batches": None,
    },
    "evaluation": {
        "distance_metric": "local_euclidean",
        "recall_thresholds_m": [50, 100, 300, 500, 1000, 2000],
        "semantic_threshold": 0.5,
    },
}


def deep_update(base: dict, override: dict, prefix: str = "") -> dict:
    """Merge known settings and reject misspelled keys early."""
    for key, value in override.items():
        if key not in base:
            raise ValueError(f"Unknown setting: {prefix}{key}")
        if isinstance(base[key], dict) and isinstance(value, dict):
            deep_update(base[key], value, f"{prefix}{key}.")
        else:
            base[key] = value
    return base


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict:
    with Path(path).open(encoding="utf-8") as stream:
        supplied = yaml.safe_load(stream) or {}
    if not isinstance(supplied, dict):
        raise ValueError("The YAML configuration must be a mapping.")
    config = deep_update(deepcopy(DEFAULTS), supplied)
    for item in overrides or []:
        key, separator, raw = item.partition("=")
        if not separator:
            raise ValueError("Overrides use key=value, for example training.epochs=2.")
        cursor = config
        keys = key.split(".")
        for component in keys[:-1]:
            if component not in cursor or not isinstance(cursor[component], dict):
                raise ValueError(f"Unknown setting: {key}")
            cursor = cursor[component]
        if keys[-1] not in cursor:
            raise ValueError(f"Unknown setting: {key}")
        cursor[keys[-1]] = yaml.safe_load(raw)
    validate_config(config)
    return config


def validate_config(config: dict) -> None:
    d, m, t, e = (config[key] for key in ("data", "model", "training", "evaluation"))
    if not 0 < d["retention_rate"] < 1:
        raise ValueError("data.retention_rate must be strictly between 0 and 1.")
    if len(d["split_ratios"]) != 3 or abs(sum(d["split_ratios"]) - 1) > 1e-8 or min(d["split_ratios"]) <= 0:
        raise ValueError("Three positive train/validation/test split ratios must sum to one.")
    for name in ("encoder_hidden_size", "decoder_hidden_size", "num_layers",
                 "semantic_embedding_dim", "environment_embedding_dim", "coord_head_dim"):
        if not isinstance(m[name], int) or m[name] < 1:
            raise ValueError(f"model.{name} must be a positive integer.")
    if not 0 <= m["dropout"] < 1:
        raise ValueError("model.dropout must lie in [0, 1).")
    for name in ("epochs", "batch_size", "cpu_threads"):
        if not isinstance(t[name], int) or t[name] < 1:
            raise ValueError(f"training.{name} must be a positive integer.")
    if t["learning_rate"] <= 0 or t["lambda_semantic"] < 0:
        raise ValueError("The learning rate must be positive and the semantic weight nonnegative.")
    for name in ("teacher_forcing_start", "teacher_forcing_end"):
        if not 0 <= t[name] <= 1:
            raise ValueError(f"training.{name} must lie in [0, 1].")
    if e["distance_metric"] not in ("local_euclidean", "haversine"):
        raise ValueError("distance_metric must be local_euclidean or haversine.")
    if not e["recall_thresholds_m"] or min(e["recall_thresholds_m"]) <= 0:
        raise ValueError("Recall thresholds must be positive distances in meters.")


def save_config(config: dict, path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream, sort_keys=False)
