"""Meaningful checks of split isolation, timestamp cleaning and sparse inputs."""
from copy import deepcopy
import csv
from datetime import datetime, timedelta
import hashlib
import json

import numpy as np
import pytest
import torch

from bamtr.config import DEFAULTS
from bamtr.data import (CoordinateNormalizer, TrajectoryDataset, _record_time, _sample,
                        build_inference_sample, collate_trajectories, prepare_data,
                        retained_indices, stay_indicators)


@pytest.fixture
def prepared(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    for day in range(1, 5):
        rows = []
        for user in range(10):
            for point in range(12):
                seconds = 8 * 3600 + point * 900
                moment = datetime(2023, 4, day) + timedelta(seconds=seconds)
                rows.append({"ID": f"synthetic-user-{user:02d}",
                             "lng": 118 + user * .1 + point * .0001,
                             "lat": 25 + user * .1 + point * .0001,
                             "time": moment.isoformat(sep=" "), "tid": seconds // 30})
        with (raw / f"2023-04-{day:02d}.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["ID", "lng", "lat", "time", "tid"])
            writer.writeheader()
            writer.writerows(rows)
    config = deepcopy(DEFAULTS)
    config["data"].update({"raw_dir": str(raw), "prepared_dir": str(tmp_path / "prepared"),
                           "timestamp_source": "auto", "date_end": "2023-04-04"})
    manifest = prepare_data(config)
    return config, manifest


def test_disjoint_users_train_only_scaling_and_earlier_history(prepared):
    config, manifest = prepared
    groups = [set(manifest["users"][name]) for name in ("train", "val", "test")]
    assert [len(group) for group in groups] == [6, 2, 2]
    assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
    dataset = TrajectoryDataset(config["data"]["prepared_dir"], "train", config["data"])
    train_coords = np.concatenate([dataset._trajectory(i)[0] for i, item in
                                  enumerate(manifest["trajectories"]) if item["split"] == "train"])
    np.testing.assert_array_equal(dataset.normalizer.minimum, train_coords.min(0))
    np.testing.assert_array_equal(dataset.normalizer.maximum, train_coords.max(0))
    for descriptors in manifest["samples"].values():
        for sample in descriptors:
            target = manifest["trajectories"][sample["trajectory_index"]]
            for index in sample["history_indices"]:
                history = manifest["trajectories"][index]
                assert history["date"] < target["date"]
                assert history["user_id"] == target["user_id"]
                assert history["split"] == target["split"]
    for item in manifest["files"]:
        expected = hashlib.sha256((__import__("pathlib").Path(config["data"]["raw_dir"]) / item["file"]).read_bytes()).hexdigest()
        assert item["sha256"] == expected


def test_masks_are_independent_of_training_seed_and_validation_epoch(prepared):
    config, _ = prepared
    dataset = TrajectoryDataset(config["data"]["prepared_dir"], "val", config["data"])
    torch.manual_seed(1)
    first = dataset[0]["metadata"]["observed_indices"]
    dataset.set_epoch(8)
    torch.manual_seed(9876)
    assert dataset[0]["metadata"]["observed_indices"] == first
    assert first[0] == 0 and first[-1] == 11
    assert len(first) == 5
    config["data"]["resample_train_mask"] = True
    train = TrajectoryDataset(config["data"]["prepared_dir"], "train", config["data"])
    initial = train[0]["metadata"]["observed_indices"]
    train.set_epoch(1)
    assert train[0]["metadata"]["observed_indices"] != initial


def test_decoder_targets_are_missing_only_and_padding_is_masked(prepared):
    config, _ = prepared
    dataset = TrajectoryDataset(config["data"]["prepared_dir"], "train", config["data"])
    first, later = dataset[0], dataset[2]
    for sample in (first, later):
        observed = set(sample["metadata"]["observed_indices"])
        targets = set(sample["metadata"]["target_indices"])
        assert not observed & targets
        assert observed | targets == set(range(12))
        assert len(sample["target_coords"]) == 12 - len(observed)
        assert len(sample["target_time"]) == len(sample["metadata"]["target_timestamps"])
        full = np.asarray(sample["metadata"]["full_coords"])
        np.testing.assert_allclose(sample["target_coords"], dataset.normalizer.transform(full[sorted(targets)]))
    batch = collate_trajectories([first, later])
    assert batch["target_mask"].sum().item() == 14
    assert batch["missing_mask"].equal(batch["target_mask"])
    assert batch["hist_lengths"][0].tolist() == [12, 0, 0]
    assert batch["hist_mask"][0].tolist() == [True, False, False]


def test_sparse_stay_input_cannot_read_hidden_coordinates():
    coords = np.asarray([[25.0, 118.0], [25.1, 118.1], [25.0, 118.0]], dtype=float)
    seconds = np.asarray([0, 3600, 7200], dtype=float)
    observed = np.asarray([0, 2])
    normalizer = CoordinateNormalizer().fit(np.asarray([[24, 117], [26, 119]], dtype=float))
    settings = deepcopy(DEFAULTS["data"])
    sample = _sample(coords, seconds, observed, [], "2023-04-03", "synthetic", "example", normalizer, settings)
    changed = coords.copy()
    changed[1] = [24, 117]
    other = _sample(changed, seconds, observed, [], "2023-04-03", "synthetic", "example", normalizer, settings)
    np.testing.assert_array_equal(sample["obs_coords"], other["obs_coords"])
    np.testing.assert_array_equal(sample["obs_stay"], other["obs_stay"])
    assert sample["obs_stay"].tolist() == [[1.0], [1.0]]
    assert not np.array_equal(sample["target_coords"], other["target_coords"])
    assert sample["target_stay"].tolist() == [0.0]


def test_auto_timestamp_uses_tid_only_for_minute_precision():
    settings = deepcopy(DEFAULTS["data"])
    settings["timestamp_source"] = "auto"
    row = {"time": "2023/4/1 8:00", "tid": "961"}
    assert _record_time(row, settings) == ("2023-04-01", 28830.0, True)
    row["time"] = "2023-04-02 08:00:47"
    assert _record_time(row, settings) == ("2023-04-02", 28847.0, False)
    row.update(time="2023/4/1 8:00", tid="965")
    with pytest.raises(ValueError, match="disagrees"):
        _record_time(row, settings)


def test_duplicate_keep_last_and_deterministic_max_points(prepared):
    config, _ = prepared
    from pathlib import Path
    source = Path(config["data"]["raw_dir"]) / "2023-04-01.csv"
    with source.open("a", newline="", encoding="utf-8") as stream:
        csv.writer(stream).writerow(["synthetic-user-00", 119.25, 26.25, "2023-04-01 08:00:00", 960])
    config["data"].update(max_points=5, min_history_days=0)
    manifest = prepare_data(config)
    assert manifest["counts"]["duplicate_timestamp_rows_removed"] == 1
    assert manifest["counts"]["conflicting_duplicate_rows_removed"] == 1
    assert manifest["counts"]["max_points_rows_removed"] == 40 * 7
    index = next(i for i, value in enumerate(manifest["trajectories"])
                 if value["user_id"] == "synthetic-user-00" and value["date"] == "2023-04-01")
    split = manifest["trajectories"][index]["split"]
    dataset = TrajectoryDataset(config["data"]["prepared_dir"], split, config["data"])
    coords, times, _ = dataset._trajectory(index)
    np.testing.assert_array_equal(coords[0], [26.25, 119.25])
    assert len(coords) == 5 and times[0] == 28800 and times[-1] == 38700
    first_day = next(dataset[i] for i, descriptor in enumerate(dataset.samples)
                     if descriptor["trajectory_index"] == index)
    batch = collate_trajectories([first_day])
    assert not batch["hist_mask"].any()
    assert batch["hist_coords"].shape == (1, 1, 1, 2)


def test_inference_has_no_hidden_labels_and_rejects_invalid_history():
    settings = deepcopy(DEFAULTS["data"])
    settings["timestamp_source"] = "auto"
    normalizer = CoordinateNormalizer().fit(np.asarray([[24, 117], [26, 119]], dtype=float))
    observed = [{"ID": "synthetic", "lat": 25, "lng": 118, "time": "2023-04-03 08:00:00", "tid": 960},
                {"ID": "synthetic", "lat": 25.01, "lng": 118.01, "time": "2023-04-03 10:00:00", "tid": 1200}]
    query = ["2023-04-03 09:00:00"]
    sample = build_inference_sample(observed, [], query, normalizer, settings)
    assert len(sample["target_time"]) == 1
    assert sample["metadata"]["target_indices"] == [1]
    assert sample["metadata"]["observed_indices"] == [0, 2]
    assert sample["metadata"]["full_coords"][1] is None
    assert not sample["metadata"]["has_ground_truth"]
    np.testing.assert_array_equal(sample["target_coords"], [[0, 0]])
    with pytest.raises(ValueError, match="strictly earlier"):
        build_inference_sample(observed, observed[:1], query, normalizer, settings)
    with pytest.raises(ValueError, match="overlap"):
        build_inference_sample(observed, [], [observed[0]["time"]], normalizer, settings)
    with pytest.raises(ValueError, match="extrapolation"):
        build_inference_sample(observed, [], ["2023-04-03 11:00:00"], normalizer, settings)


def test_normalizer_roundtrip_and_no_test_clipping():
    normalizer = CoordinateNormalizer().fit(np.asarray([[25, 118], [25, 119]], dtype=float))
    outside = np.asarray([[25, 120]], dtype=float)
    transformed = normalizer.transform(outside)
    assert transformed[0, 1] == 2
    np.testing.assert_allclose(normalizer.inverse_transform(transformed), outside)
    reloaded = CoordinateNormalizer.from_dict(json.loads(json.dumps(normalizer.to_dict())))
    torch.testing.assert_close(reloaded.inverse_transform(torch.as_tensor(transformed)), torch.as_tensor(outside).float())


@pytest.mark.parametrize("name,value", [("max_points", 5), ("min_points", 5),
                                      ("split_seed", 99), ("split_ratios", [.5, .3, .2]),
                                      ("min_history_days", 0), ("max_history_days", 2),
                                      ("timestamp_source", "time"), ("time_id_seconds", 60),
                                      ("date_end", "2023-04-03"),
                                      ("file_pattern", "2023-04-0[12].csv"),
                                      ("columns", {"user": "other_id"})])
def test_dataset_rejects_changed_preparation_settings(prepared, name, value):
    config, _ = prepared
    changed = deepcopy(config["data"])
    changed[name] = value
    with pytest.raises(ValueError, match="Re-run prepare"):
        TrajectoryDataset(config["data"]["prepared_dir"], "train", changed)


def test_dataset_accepts_runtime_mask_stay_and_relocated_source_settings(prepared):
    config, _ = prepared
    changed = deepcopy(config["data"])
    changed.update(retention_rate=.6, stay_time_minutes=60, stay_radius_m=150,
                   mask_seed=123, raw_dir="a/relocated/source", prepared_dir="a/relocated/cache")
    dataset = TrajectoryDataset(config["data"]["prepared_dir"], "train", changed)
    assert len(dataset[0]["obs_coords"]) == 7
    assert len(dataset[0]["target_coords"]) == 5
    assert dataset.settings["stay_time_minutes"] == 60
    assert dataset.settings["stay_radius_m"] == 150


def test_inference_rejects_duplicate_query_timestamps():
    settings = deepcopy(DEFAULTS["data"])
    normalizer = CoordinateNormalizer().fit(np.asarray([[24, 117], [26, 119]], dtype=float))
    observed = [{"ID": "synthetic", "lat": 25, "lng": 118, "time": "2023-04-03 08:00:00", "tid": 960},
                {"ID": "synthetic", "lat": 25.01, "lng": 118.01, "time": "2023-04-03 10:00:00", "tid": 1200}]
    with pytest.raises(ValueError, match="duplicate queries"):
        build_inference_sample(observed, [], ["2023-04-03 09:00:00", datetime(2023, 4, 3, 9)],
                               normalizer, settings)
