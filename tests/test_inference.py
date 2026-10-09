"""Public deployment API: immutable observations and no query-label leakage."""
from __future__ import annotations

import csv
from copy import deepcopy
from pathlib import Path

import pytest
import torch

from bamtr.config import DEFAULTS
from bamtr.data import CoordinateNormalizer
from bamtr.inference import predict
from bamtr.model import BAMTR
from bamtr.utils import save_checkpoint


def write_csv(path, fields, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


@pytest.fixture
def deployment_inputs(tmp_path):
    config = deepcopy(DEFAULTS)
    config["model"].update({"encoder_hidden_size": 8, "decoder_hidden_size": 16,
                            "semantic_embedding_dim": 4, "environment_embedding_dim": 4,
                            "coord_head_dim": 12, "dropout": 0.1})
    config["training"].update({"cpu_threads": 1, "device": "cpu"})
    config["data"]["timestamp_source"] = "time"
    torch.manual_seed(407)
    model = BAMTR(config["model"])
    normalizer = CoordinateNormalizer([25.44, 118.19], [25.48, 118.23])
    checkpoint = tmp_path / "random_test_model.pt"
    # This checkpoint is deliberately untrained. These tests check the API's
    # data boundaries, not trajectory recovery quality.
    save_checkpoint({"format_version": 1, "config": config,
                     "model_state": model.state_dict(), "normalizer": normalizer.to_dict()}, checkpoint)

    observations, queries, valid_history = [], [], []
    for user_index, user in enumerate(("synthetic_A", "synthetic_B")):
        for position, hour in enumerate((8, 10, 12, 14)):
            observations.append({"ID": user, "time": f"2023-04-03 {hour:02d}:00:00",
                                 "lat": 25.45000123456789 + 0.003 * user_index + 0.0004 * position,
                                 "lng": 118.20000987654321 + 0.002 * user_index + 0.0003 * position})
            for day in (1, 2):
                valid_history.append({"ID": user, "time": f"2023-04-{day:02d} {hour:02d}:00:00",
                                      "lat": 25.4502 + 0.003 * user_index + 0.0004 * position,
                                      "lng": 118.2001 + 0.002 * user_index + 0.0003 * position})
        for hour in (9, 11, 13):
            queries.append({"ID": user, "time": f"2023-04-03 {hour:02d}:00:00"})
    observations_path = tmp_path / "observations.csv"
    queries_path = tmp_path / "queries.csv"
    poisoned_queries_path = tmp_path / "queries_with_irrelevant_labels.csv"
    write_csv(observations_path, ["ID", "time", "lat", "lng"], observations)
    write_csv(queries_path, ["ID", "time"], queries)
    poisoned = [{**query, "lat": "not-a-coordinate", "lng": "nan", "stay": -999}
                for query in queries]
    write_csv(poisoned_queries_path, ["ID", "time", "lat", "lng", "stay"], poisoned)

    clean_history_dir = tmp_path / "clean_history"
    contaminated_history_dir = tmp_path / "contaminated_history"
    clean_history_dir.mkdir()
    contaminated_history_dir.mkdir()
    contaminants = []
    for position, hour in enumerate((8, 10, 12, 14)):
        # Current/future target-user records must be excluded before the strict
        # same-user, earlier-day inference builder receives historical inputs.
        for day in (3, 4):
            for user in ("synthetic_A", "synthetic_B"):
                contaminants.append({"ID": user, "time": f"2023-04-{day:02d} {hour:02d}:00:00",
                                     "lat": 30.0 + position, "lng": 121.0 + position})
        contaminants.append({"ID": "unrelated_user", "time": f"2023-04-02 {hour:02d}:00:00",
                             "lat": 40.0 + position, "lng": 110.0 + position})
    write_csv(clean_history_dir / "2023-04-01.csv", ["ID", "time", "lat", "lng"], valid_history)
    write_csv(contaminated_history_dir / "2023-04-01.csv", ["ID", "time", "lat", "lng"],
              valid_history + contaminants)
    return {"checkpoint": checkpoint, "observations_path": observations_path,
            "queries_path": queries_path, "poisoned_queries_path": poisoned_queries_path,
            "clean_history_dir": clean_history_dir, "contaminated_history_dir": contaminated_history_dir,
            "observations": observations, "queries": queries, "temporary": tmp_path}


def test_predict_preserves_observations_and_ignores_query_labels(deployment_inputs):
    files = deployment_inputs
    plain_output = files["temporary"] / "plain_predictions.csv"
    poisoned_output = files["temporary"] / "poisoned_predictions.csv"
    summary = predict(files["checkpoint"], files["observations_path"], files["queries_path"],
                      files["clean_history_dir"], plain_output, device="cpu")
    poisoned_summary = predict(files["checkpoint"], files["observations_path"], files["poisoned_queries_path"],
                               files["clean_history_dir"], poisoned_output, device="cpu")
    rows = read_csv(plain_output)
    assert rows == read_csv(poisoned_output)
    assert summary["target_locations_used"] is False
    assert poisoned_summary["target_locations_used"] is False
    assert summary["query_time_conditioning"] is False
    assert summary["groups"] == 2
    assert summary["recovered_points"] == len(files["queries"])
    assert summary["unchanged_observations"] == len(files["observations"])
    assert summary["groups_below_training_history_minimum"] == 0
    assert len(rows) == len(files["observations"]) + len(files["queries"])
    observed_lookup = {(row["ID"], row["time"]): row for row in files["observations"]}
    query_keys = {(row["ID"], row["time"]) for row in files["queries"]}
    predicted_keys = set()
    for row in rows:
        key = (row["ID"], row["time"])
        if row["is_observed"] == "1":
            original = observed_lookup[key]
            # Exact float equality catches a lossy normalization/inverse-
            # normalization round trip of the retained coordinate values.
            assert float(row["lat"]) == float(original["lat"])
            assert float(row["lng"]) == float(original["lng"])
            assert row["stay_probability"] == ""
        else:
            predicted_keys.add(key)
            assert 0 <= float(row["stay_probability"]) <= 1
    assert predicted_keys == query_keys
    assert [(row["ID"], row["time"]) for row in rows] == sorted((row["ID"], row["time"]) for row in rows)
    assert plain_output.with_suffix(".summary.json").is_file()


def test_predict_filters_other_users_and_nonpast_histories(deployment_inputs):
    files = deployment_inputs
    filtered_output = files["temporary"] / "filtered_predictions.csv"
    mixed_output = files["temporary"] / "mixed_predictions.csv"
    filtered = predict(files["checkpoint"], files["observations_path"], files["queries_path"],
                       files["clean_history_dir"], filtered_output, device="cpu")
    mixed = predict(files["checkpoint"], files["observations_path"], files["queries_path"],
                    files["contaminated_history_dir"], mixed_output, device="cpu")
    assert read_csv(mixed_output) == read_csv(filtered_output)
    for key in ("groups", "recovered_points", "unchanged_observations",
                "groups_below_training_history_minimum", "target_locations_used"):
        assert mixed[key] == filtered[key]
