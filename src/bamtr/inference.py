"""Recover explicit missing timestamps without reading their target locations."""
from __future__ import annotations

from collections import defaultdict
import csv
from datetime import datetime, timedelta
from pathlib import Path

import torch

from .data import CoordinateNormalizer, _record_time, build_inference_sample, collate_trajectories
from .model import BAMTR
from .utils import load_checkpoint, seed_everything, select_device, to_device, write_json


def _rows(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        yield from csv.DictReader(stream)


def predict(checkpoint_path, observations_path, queries_path, history_dir, output_path,
            device="auto", calendar_path=None):
    """Predict coordinates/stay probability and merge unchanged observations.

    Queries need only ID,time and, for minute-resolution times in auto mode,
    tid. Query coordinate columns, if present, are not read or passed through.
    Histories are explicitly filtered by user and strictly earlier target day.
    """
    checkpoint = load_checkpoint(checkpoint_path)
    config = checkpoint["config"]
    settings = dict(config["data"])
    if calendar_path:
        settings["holiday_calendar_path"] = str(calendar_path)
    columns = settings["columns"]
    seed_everything(config["seed"], config["training"]["deterministic"], config["training"]["cpu_threads"])
    target_device = select_device(device)
    normalizer = CoordinateNormalizer.from_dict(checkpoint["normalizer"])
    model = BAMTR(config["model"], environment_dim=5).to(target_device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    observed = defaultdict(list)
    for row in _rows(observations_path):
        day, _, _ = _record_time(row, settings)
        observed[(str(row[columns["user"]]).strip(), day)].append(row)
    queries = defaultdict(list)
    for row in _rows(queries_path):
        day, seconds, _ = _record_time(row, settings)
        user = str(row[columns["user"]]).strip()
        instant = datetime.fromisoformat(day) + timedelta(seconds=seconds)
        queries[(user, day)].append(instant.isoformat(sep=" "))
    if not queries:
        raise ValueError("The query CSV is empty.")
    if any(key not in observed for key in queries):
        raise ValueError("Every query user/day needs a matching current observation group.")
    histories = defaultdict(list)
    users = {user for user, _ in queries}
    if history_dir is not None:
        history_paths = sorted(Path(history_dir).glob(settings["file_pattern"]))
        if not history_paths:
            raise FileNotFoundError("No historical CSV files matched the configured file pattern.")
        for path in history_paths:
            for row in _rows(path):
                user = str(row[columns["user"]]).strip()
                if user not in users:
                    continue
                day, _, _ = _record_time(row, settings)
                histories[(user, day)].append(row)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total_missing = total_observed = cold_start_groups = 0
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["ID", "time", "lat", "lng", "is_observed", "stay_probability"])
        writer.writeheader()
        with torch.inference_mode():
            for (user, day), timestamps in sorted(queries.items()):
                earlier = [row for (past_user, past_day), rows in sorted(histories.items())
                           if past_user == user and past_day < day for row in rows]
                sample = build_inference_sample(observed[(user, day)], earlier, timestamps, normalizer, settings)
                batch = to_device(collate_trajectories([sample]), target_device)
                # Drop labels completely, so deployment never depends on dummy supervision.
                batch.pop("target_coords", None)
                batch.pop("target_stay", None)
                result = model(batch, teacher_forcing_ratio=0.0)
                coordinates = normalizer.inverse_transform(result["coords"][0].cpu().numpy())
                probability = result["stay_logits"][0].sigmoid().cpu().numpy()
                metadata = sample["metadata"]
                predicted_by_index = {index: (coordinates[j], probability[j])
                                      for j, index in enumerate(metadata["target_indices"])}
                for index, timestamp in enumerate(metadata["full_timestamps"]):
                    if index in predicted_by_index:
                        coordinate, stay = predicted_by_index[index]
                        observed_flag = 0
                        total_missing += 1
                    else:
                        coordinate, stay = metadata["full_coords"][index], ""
                        observed_flag = 1
                        total_observed += 1
                    writer.writerow({"ID": user, "time": timestamp, "lat": coordinate[0], "lng": coordinate[1],
                                     "is_observed": observed_flag, "stay_probability": stay})
                cold_start_groups += int(not metadata["history_requirement_met"])
    report = {"groups": len(queries), "recovered_points": total_missing,
              "unchanged_observations": total_observed, "groups_below_training_history_minimum": cold_start_groups,
              "query_time_conditioning": config["model"]["query_time_conditioning"],
              "target_locations_used": False, "output": output_path.name}
    write_json(report, output_path.with_suffix(".summary.json"))
    print(f"Recovered {total_missing} query points; preserved {total_observed} observed points. Output: {output_path}", flush=True)
    return report
