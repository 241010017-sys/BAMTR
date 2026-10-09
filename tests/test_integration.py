"""Small synthetic end-to-end runs; no private data or pretrained files needed."""
from __future__ import annotations

import csv
from copy import deepcopy
from datetime import datetime, timedelta
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from bamtr import engine
from bamtr.config import DEFAULTS
from bamtr.data import TrajectoryDataset, collate_trajectories, prepare_data
from bamtr.model import BAMTR
from bamtr.utils import load_checkpoint


@pytest.fixture
def synthetic_project(tmp_path):
    """Ten invented users, three days, and different daily trajectory lengths."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    truth, lengths = {}, {}
    for day_index in range(3):
        date = datetime(2023, 4, 1) + timedelta(days=day_index)
        with (raw_dir / f"{date:%Y-%m-%d}.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["ID", "lng", "lat", "time", "tid"])
            writer.writeheader()
            for user_index in range(10):
                user = f"synthetic_{user_index:02d}"
                count = 6 + user_index % 3
                lengths[user] = count
                for position in range(count):
                    timestamp = date + timedelta(hours=7, minutes=40 * position)
                    # A stay followed by movement gives both semantic classes.
                    displacement = max(0, position - 3)
                    latitude = 25.45 + 0.001 * user_index + 0.0005 * displacement
                    longitude = 118.20 + 0.0013 * user_index + 0.0007 * displacement + 0.0001 * day_index
                    text = timestamp.isoformat(sep=" ")
                    writer.writerow({"ID": user, "lng": longitude, "lat": latitude,
                                     "time": text, "tid": (7 * 3600 + position * 2400) // 30})
                    truth[(user, text)] = (latitude, longitude)
    config = deepcopy(DEFAULTS)
    config["data"].update({"raw_dir": str(raw_dir),
                           "prepared_dir": str(tmp_path / "processed"),
                           "date_start": "2023-04-01", "date_end": "2023-04-03",
                           "retention_rate": 0.4, "min_history_days": 1,
                           "max_history_days": 2, "min_points": 4,
                           "resample_train_mask": True})
    config["model"].update({"encoder_hidden_size": 8, "decoder_hidden_size": 16,
                            "dropout": 0.1, "semantic_embedding_dim": 4,
                            "environment_embedding_dim": 4, "coord_head_dim": 12})
    config["training"].update({"epochs": 2, "batch_size": 2, "device": "cpu",
                               "num_workers": 0, "cpu_threads": 1,
                               "run_dir": str(tmp_path / "run"),
                               "max_train_batches": None, "max_eval_batches": None,
                               "early_stopping_patience": 0})
    manifest = prepare_data(config)
    return config, manifest, truth, lengths, tmp_path


def read_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def test_prepare_train_evaluate_missing_points_and_checkpoint(synthetic_project):
    config, manifest, source_truth, lengths, temporary = synthetic_project
    users = {split: set(values) for split, values in manifest["users"].items()}
    assert len(set.union(*users.values())) == 10
    assert not users["train"] & users["val"]
    assert not users["train"] & users["test"]
    assert not users["val"] & users["test"]
    trajectories = manifest["trajectories"]
    for split in ("train", "val", "test"):
        # The first day provides history, and only days two and three are targets.
        assert len(manifest["samples"][split]) == 2 * len(users[split])
        for sample in manifest["samples"][split]:
            target = trajectories[sample["trajectory_index"]]
            assert target["user_id"] in users[split]
            for history_index in sample["history_indices"]:
                history = trajectories[history_index]
                assert history["user_id"] == target["user_id"]
                assert history["date"] < target["date"]

    best_path = engine.train(config)
    assert best_path.is_file()
    history_rows = read_rows(Path(config["training"]["run_dir"]) / "history.csv")
    assert [int(row["epoch"]) for row in history_rows] == [1, 2]
    output_dir = temporary / "held_out"
    result = engine.evaluate(best_path, split="test", output_dir=output_dir, device="cpu")
    # Compute expected missing counts from the original CSV lengths and the
    # declared sampling rule, independently of model outputs/collated masks.
    expected_missing = 2 * sum(lengths[user] - max(2, math.floor(0.4 * lengths[user] + 0.5))
                               for user in users["test"])
    assert result["n_missing_points"] == expected_missing
    rows = read_rows(output_dir / "predictions.csv")
    assert len(rows) == expected_missing
    assert {row["user_id"] for row in rows} == users["test"]
    assert len({(row["user_id"], row["timestamp"]) for row in rows}) == expected_missing
    for row in rows:
        np.testing.assert_allclose([float(row["true_lat"]), float(row["true_lng"])],
                                   source_truth[(row["user_id"], row["timestamp"])], atol=1e-8, rtol=0)
    errors = np.asarray([float(row["error_m"]) for row in rows])
    assert result["mae_m"] == pytest.approx(errors.mean())
    assert result["rmse_m"] == pytest.approx(np.sqrt(np.square(errors).mean()))
    assert result["recall_at_m"]["100"] == pytest.approx(np.mean(errors <= 100))

    checkpoint = load_checkpoint(best_path)
    dataset = TrajectoryDataset(config["data"]["prepared_dir"], "test", config["data"])
    assert checkpoint["normalizer"] == dataset.normalizer.to_dict()
    first_batch = collate_trajectories([dataset[0], dataset[1]])
    first = BAMTR(checkpoint["config"]["model"]).eval()
    first.load_state_dict(checkpoint["model_state"])
    second = BAMTR(checkpoint["config"]["model"]).eval()
    second.load_state_dict(load_checkpoint(best_path)["model_state"])
    with torch.no_grad():
        prediction = first(first_batch)
        reloaded = second(first_batch)
    torch.testing.assert_close(prediction["coords"], reloaded["coords"], rtol=0, atol=0)
    torch.testing.assert_close(prediction["stay_logits"], reloaded["stay_logits"], rtol=0, atol=0)
    mask = first_batch["target_mask"].numpy()
    degrees = dataset.normalizer.inverse_transform(prediction["coords"].numpy()[mask])
    csv_degrees = np.asarray([[float(row["pred_lat"]), float(row["pred_lng"])] for row in rows[:len(degrees)]])
    np.testing.assert_allclose(degrees, csv_degrees, rtol=0, atol=1e-9)
    for index in range(len(dataset)):
        sample = dataset[index]
        metadata = sample["metadata"]
        observed = {metadata["full_timestamps"][j] for j in metadata["observed_indices"]}
        missing = set(metadata["target_timestamps"])
        assert not observed & missing
        assert observed | missing == set(metadata["full_timestamps"])


def test_epoch_boundary_resume_matches_uninterrupted_training(synthetic_project, monkeypatch):
    config, _, _, _, temporary = synthetic_project
    uninterrupted_config = deepcopy(config)
    uninterrupted_config["training"]["run_dir"] = str(temporary / "uninterrupted")
    engine.train(uninterrupted_config)
    uninterrupted = load_checkpoint(temporary / "uninterrupted" / "latest.pt")

    interrupted_config = deepcopy(config)
    interrupted_config["training"]["run_dir"] = str(temporary / "interrupted")
    original_run_epoch = engine.run_epoch
    calls = 0

    def interrupt_second_epoch(*args, **kwargs):
        nonlocal calls
        calls += 1
        # Calls one/two are training/validation for epoch one, after which the
        # engine persists model, optimizer and RNG state before call three.
        if calls == 3:
            raise RuntimeError("Synthetic interruption after the first saved epoch")
        return original_run_epoch(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(engine, "run_epoch", interrupt_second_epoch)
        with pytest.raises(RuntimeError, match="Synthetic interruption"):
            engine.train(interrupted_config)
    latest = temporary / "interrupted" / "latest.pt"
    assert load_checkpoint(latest)["epoch"] == 0
    resumed_best = engine.train(interrupted_config, resume=str(latest))
    assert resumed_best.is_file()
    resumed = load_checkpoint(latest)
    assert resumed["epoch"] == uninterrupted["epoch"] == 1
    assert resumed["best_val_mae_m"] == uninterrupted["best_val_mae_m"]
    assert resumed["model_state"].keys() == uninterrupted["model_state"].keys()
    for name, tensor in uninterrupted["model_state"].items():
        torch.testing.assert_close(resumed["model_state"][name], tensor, rtol=0, atol=0,
                                   msg=f"Resume changed parameter {name}")
    full_rows = read_rows(temporary / "uninterrupted" / "history.csv")
    resumed_rows = read_rows(temporary / "interrupted" / "history.csv")
    assert [int(row["epoch"]) for row in resumed_rows] == [1, 2]
    for full, restored in zip(full_rows, resumed_rows):
        for key in full:
            if key != "epoch_seconds":
                assert full[key] == restored[key], f"Resume changed logged field {key}"
