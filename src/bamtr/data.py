"""User-disjoint daily trajectory preparation and leakage-safe input construction.

Coordinates always use [latitude, longitude]. Targets are the cleaned observed
records before synthetic masking, not continuous GPS ground truth. Their query
timestamps are assumed known. Personal histories contain earlier days only.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import date as Date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .calendar import environment_features


class CoordinateNormalizer:
    """Independent min-max scaling, fitted exclusively to training users.

    Values outside the training bounding box are not clipped. Constant axes use
    scale one so inverse transformation remains defined.
    """

    def __init__(self, minimum=None, maximum=None):
        self.minimum = None if minimum is None else np.asarray(minimum, dtype=np.float64)
        self.maximum = None if maximum is None else np.asarray(maximum, dtype=np.float64)

    def fit(self, coords: np.ndarray) -> "CoordinateNormalizer":
        coords = np.asarray(coords, dtype=np.float64)
        if coords.ndim != 2 or coords.shape[1] != 2 or not len(coords) or not np.isfinite(coords).all():
            raise ValueError("Normalizer fitting requires finite coordinates with shape [N, 2].")
        self.minimum, self.maximum = coords.min(axis=0), coords.max(axis=0)
        return self

    @property
    def scale(self):
        if self.minimum is None or self.maximum is None:
            raise ValueError("The coordinate normalizer has not been fitted.")
        return np.where(self.maximum > self.minimum, self.maximum - self.minimum, 1.0)

    def transform(self, coords):
        return ((np.asarray(coords, dtype=np.float64) - self.minimum) / self.scale).astype(np.float32)

    def inverse_transform(self, coords):
        if torch.is_tensor(coords):
            scale = torch.as_tensor(self.scale, device=coords.device, dtype=coords.dtype)
            minimum = torch.as_tensor(self.minimum, device=coords.device, dtype=coords.dtype)
            return coords * scale + minimum
        return np.asarray(coords) * self.scale + self.minimum

    def to_dict(self) -> dict:
        _ = self.scale
        return {"coordinate_order": ["latitude", "longitude"],
                "minimum": self.minimum.tolist(), "maximum": self.maximum.tolist()}

    @classmethod
    def from_dict(cls, value: dict) -> "CoordinateNormalizer":
        result = cls(value["minimum"], value["maximum"])
        if result.minimum.shape != (2,) or result.maximum.shape != (2,) or not np.isfinite(result.scale).all():
            raise ValueError("Invalid coordinate normalizer metadata.")
        return result


def _settings(config: dict) -> dict:
    from .config import DEFAULTS
    result = dict(DEFAULTS["data"])
    supplied = config.get("data", config)
    result.update(supplied)
    result["columns"] = {**DEFAULTS["data"]["columns"], **supplied.get("columns", {})}
    return result


def parse_timestamp(value: str | datetime) -> datetime:
    """Parse ISO timestamps and the slash-separated minute format in the data."""
    if isinstance(value, datetime):
        result = value
    else:
        text = str(value).strip()
        if "/" in text:
            result = datetime.strptime(text, "%Y/%m/%d %H:%M:%S" if text.count(":") == 2
                                       else "%Y/%m/%d %H:%M")
        else:
            result = datetime.fromisoformat(text)
    if result.tzinfo is not None:
        raise ValueError("Use consistent local, timezone-naive timestamps for daily trajectories.")
    return result


def _seconds(dt: datetime) -> float:
    return dt.hour * 3600 + dt.minute * 60 + dt.second + dt.microsecond / 1e6


def _record_time(row: dict, settings: dict) -> tuple[str, float, bool]:
    columns = settings["columns"]
    raw = row[columns["timestamp"]]
    dt = parse_timestamp(raw)
    source = settings.get("timestamp_source", "auto")
    if source not in {"auto", "time", "tid"}:
        raise ValueError("timestamp_source must be auto, time, or tid.")
    minute_precision = str(raw).split()[-1].count(":") == 1
    use_tid = source == "tid" or (source == "auto" and minute_precision)
    seconds = _seconds(dt)
    if use_tid:
        candidate = float(row[columns["time_id"]]) * float(settings["time_id_seconds"])
        if not math.isfinite(candidate) or not 0 <= candidate < 86400:
            raise ValueError("Time IDs must denote a valid position within the local day.")
        if source == "auto" and int(candidate // 60) != int(seconds // 60):
            raise ValueError("A minute-resolution timestamp disagrees with its time ID.")
        seconds = candidate
    return dt.date().isoformat(), seconds, use_tid


def _stable_seed(*values) -> int:
    encoded = json.dumps(values, ensure_ascii=True, separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "little")


def retained_indices(length: int, rate: float, seed: int, sample_id: str,
                     keep_endpoints: bool = True, epoch: int = 0) -> np.ndarray:
    """Uniform sampling without replacement, optionally preserving endpoints.

    The retained count is rounded to the nearest integer (half upward), then
    constrained to leave at least one hidden target. For endpoint preservation,
    sequences must contain at least three points. The mask does not depend on
    the training/model random seed or on minibatch order.
    """
    if not 0 < rate < 1 or length < (3 if keep_endpoints else 2):
        raise ValueError("Masking needs a valid retention rate and enough trajectory points.")
    count = max(2 if keep_endpoints else 1, int(math.floor(length * rate + 0.5)))
    count = min(count, length - 1)
    rng = np.random.default_rng(_stable_seed(seed, sample_id, epoch))
    if keep_endpoints:
        middle = rng.choice(np.arange(1, length - 1), size=count - 2, replace=False)
        return np.sort(np.r_[0, middle, length - 1]).astype(np.int64)
    return np.sort(rng.choice(length, size=count, replace=False)).astype(np.int64)


def _distances_m(coords: np.ndarray, anchor: np.ndarray) -> np.ndarray:
    """Haversine distance from an observed anchor, in meters."""
    a, b = np.radians(coords), np.radians(anchor)
    dlat, dlon = a[:, 0] - b[0], a[:, 1] - b[1]
    h = np.sin(dlat / 2) ** 2 + np.cos(a[:, 0]) * np.cos(b[0]) * np.sin(dlon / 2) ** 2
    return 6371008.8 * 2 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def stay_indicators(coords: np.ndarray, seconds: np.ndarray, time_minutes: float = 90.0,
                    radius_m: float = 100.0) -> np.ndarray:
    """Detect contiguous anchor-radius stays using only the supplied points.

    Starting at an anchor, extend until the first point outside its radius.
    Mark the segment if its first-to-last elapsed time meets the threshold;
    otherwise advance the anchor by one. This explicit rule is an implementation
    choice: sparse input indicators and full-reference labels use the same rule
    but are computed independently from their respective available records.
    """
    coords, seconds = np.asarray(coords), np.asarray(seconds)
    if time_minutes <= 0 or radius_m <= 0 or len(coords) != len(seconds):
        raise ValueError("Stay detection requires aligned points and positive thresholds.")
    states = np.zeros(len(coords), dtype=np.float32)
    i = 0
    while i < len(coords):
        distances = _distances_m(coords[i:], coords[i])
        outside = np.flatnonzero(distances > radius_m)
        j = i + int(outside[0]) if len(outside) else len(coords)
        if j > i + 1 and seconds[j - 1] - seconds[i] >= time_minutes * 60:
            states[i:j] = 1
            i = j
        else:
            i += 1
    return states


def _timestamps(day: str, seconds: np.ndarray) -> list[str]:
    start = datetime.combine(Date.fromisoformat(day), datetime.min.time())
    return [(start + timedelta(seconds=float(s))).isoformat(sep=" ") for s in seconds]


def prepare_data(config: dict, output_dir: Path | None = None) -> dict:
    """Read raw CSVs without modification and save NPZ plus JSON metadata.

    Duplicate user/time records are resolved by keeping the last source record
    in sorted-file/input-row order. Optional max_points selects evenly spaced
    positions deterministically, preserving endpoints. Both operations occur
    before reference labels and synthetic masks are constructed.
    """
    settings = _settings(config)
    raw_dir = Path(settings["raw_dir"])
    paths = sorted(raw_dir.glob(settings["file_pattern"]))
    if not paths:
        raise FileNotFoundError(f"No CSV files matched the configured pattern in {raw_dir}.")
    if settings["min_points"] < 3:
        raise ValueError("min_points must be at least three for endpoint-preserving evaluation.")
    maximum = settings["max_points"]
    if maximum is not None and (not isinstance(maximum, int) or maximum < settings["min_points"]):
        raise ValueError("max_points must be null or an integer at least min_points.")
    min_history = settings["min_history_days"]
    max_history = settings["max_history_days"]
    if not isinstance(min_history, int) or min_history < 0:
        raise ValueError("min_history_days must be a nonnegative integer.")
    if max_history is not None and (not isinstance(max_history, int) or max_history < 0 or max_history < min_history):
        raise ValueError("max_history_days must be null or at least min_history_days.")
    grouped = defaultdict(dict)
    counts = Counter({key: 0 for key in ("raw_rows", "timestamps_reconstructed_from_tid",
                    "duplicate_timestamp_rows_removed", "conflicting_duplicate_rows_removed",
                    "short_trajectories_removed", "short_trajectory_rows_removed",
                    "max_points_rows_removed", "rows_outside_date_range")})
    per_file = []
    columns = settings["columns"]
    for path in paths:
        file_rows, file_users = 0, set()
        source_digest = hashlib.sha256()
        with path.open("rb") as source_stream:
            for chunk in iter(lambda: source_stream.read(1024 * 1024), b""):
                source_digest.update(chunk)
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            required = {columns[k] for k in ("user", "latitude", "longitude", "timestamp")}
            if settings.get("timestamp_source", "auto") != "time":
                required.add(columns["time_id"])
            if not required.issubset(reader.fieldnames or []):
                raise ValueError(f"Missing required columns in {path.name}.")
            for row_number, row in enumerate(reader, start=2):
                counts["raw_rows"] += 1
                file_rows += 1
                try:
                    user = str(row[columns["user"]]).strip()
                    if not user:
                        raise ValueError("Empty user identifier")
                    day, seconds, used_tid = _record_time(row, settings)
                    lat, lng = float(row[columns["latitude"]]), float(row[columns["longitude"]])
                    if not math.isfinite(lat) or not math.isfinite(lng) or not -90 <= lat <= 90 or not -180 <= lng <= 180:
                        raise ValueError("Invalid coordinates")
                except (ValueError, KeyError, TypeError) as exc:
                    raise ValueError(f"Invalid record in {path.name}, row {row_number}: {type(exc).__name__}.") from None
                if ((settings.get("date_start") and day < settings["date_start"]) or
                        (settings.get("date_end") and day > settings["date_end"])):
                    counts["rows_outside_date_range"] += 1
                    continue
                file_users.add(user)
                counts["timestamps_reconstructed_from_tid"] += int(used_tid)
                records = grouped[(user, day)]
                if seconds in records:
                    counts["duplicate_timestamp_rows_removed"] += 1
                    counts["conflicting_duplicate_rows_removed"] += int(records[seconds] != (lat, lng))
                records[seconds] = (lat, lng)
        per_file.append({"file": path.name, "sha256": source_digest.hexdigest(),
                         "rows": file_rows, "users_in_date_range": len(file_users)})
    trajectories = []
    for (user, day), records in sorted(grouped.items()):
        ordered = sorted(records)
        if len(ordered) < settings["min_points"]:
            counts["short_trajectories_removed"] += 1
            counts["short_trajectory_rows_removed"] += len(ordered)
            continue
        seconds = np.asarray(ordered, dtype=np.float64)
        coords = np.asarray([records[s] for s in ordered], dtype=np.float64)
        if maximum is not None and len(seconds) > maximum:
            keep = np.linspace(0, len(seconds) - 1, maximum, dtype=np.int64)
            counts["max_points_rows_removed"] += len(seconds) - maximum
            coords, seconds = coords[keep], seconds[keep]
        trajectories.append({"user_id": user, "date": day, "coords": coords, "seconds": seconds})
    # Split all valid users before trajectory-length eligibility. This keeps
    # user assignments stable when max_points or min_points is changed.
    users = sorted({user for user, day in grouped})
    counts["users_without_eligible_trajectories"] = len(set(users) - {item["user_id"] for item in trajectories})
    if len(users) < 3:
        raise ValueError("At least three users with eligible daily trajectories are required.")
    ratios = np.asarray(settings["split_ratios"], dtype=float)
    if ratios.shape != (3,) or np.any(ratios <= 0) or not np.isclose(ratios.sum(), 1):
        raise ValueError("Three positive split ratios must sum to one.")
    shuffled = np.random.default_rng(settings["split_seed"]).permutation(users).tolist()
    train_count = max(1, min(len(users) - 2, int(len(users) * ratios[0])))
    val_count = max(1, min(len(users) - train_count - 1, int(len(users) * ratios[1])))
    split_users = {"train": shuffled[:train_count], "val": shuffled[train_count:train_count + val_count],
                   "test": shuffled[train_count + val_count:]}
    user_split = {user: split for split, entries in split_users.items() for user in entries}
    training_coordinates = [item["coords"] for item in trajectories if user_split[item["user_id"]] == "train"]
    if not training_coordinates:
        raise ValueError("The training-user split has no eligible trajectories.")
    normalizer = CoordinateNormalizer().fit(np.concatenate(training_coordinates))
    offsets = [0]
    trajectory_metadata, samples = [], {"train": [], "val": [], "test": []}
    previous = defaultdict(list)
    for index, item in enumerate(trajectories):
        user, day = item["user_id"], item["date"]
        history = previous[user]
        if max_history is not None:
            history = history[-max_history:] if max_history else []
        split = user_split[user]
        sample_id = hashlib.sha256(f"{user}\0{day}".encode()).hexdigest()[:20]
        trajectory_metadata.append({"user_id": user, "date": day, "split": split,
                                    "length": len(item["seconds"]), "sample_id": sample_id})
        offsets.append(offsets[-1] + len(item["seconds"]))
        if len(history) >= min_history:
            samples[split].append({"trajectory_index": index, "history_indices": list(history)})
        previous[user].append(index)
    if any(not entries for entries in samples.values()):
        raise ValueError("Every split needs eligible targets; reduce min_history_days or provide more days.")
    destination = Path(output_dir or settings["prepared_dir"])
    destination.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination / "trajectories.npz",
                        coords=np.concatenate([x["coords"] for x in trajectories]),
                        seconds=np.concatenate([x["seconds"] for x in trajectories]),
                        offsets=np.asarray(offsets, dtype=np.int64))
    manifest = {"format_version": 1, "coordinate_order": ["latitude", "longitude"],
                "normalizer": normalizer.to_dict(), "data_config": settings,
                "normalizer_fitted_on": "training users only; no clipping of validation/test coordinates",
                "timestamp_policy": "auto uses precise time when available, validated tid for minute-only time",
                "duplicate_policy": "last source record at each user/date/timestamp; never average coordinates",
                "max_points_policy": "deterministic evenly spaced indices including endpoints, before masking",
                "target_policy": "known cleaned original observation timestamps; synthetic location masking",
                "history_policy": "all available cleaned records from strictly earlier days of the same user",
                "counts": dict(counts), "files": per_file,
                "user_counts": {k: len(v) for k, v in split_users.items()},
                "eligible_user_counts": {k: len({item["user_id"] for item in trajectories
                                                    if user_split[item["user_id"]] == k}) for k in split_users},
                "sample_counts": {k: len(v) for k, v in samples.items()},
                "trajectory_count": len(trajectories), "prepared_point_count": offsets[-1],
                "users": split_users, "trajectories": trajectory_metadata, "samples": samples}
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


class TrajectoryDataset(Dataset):
    """One target user-day per item, with fixed or epoch-resampled masks."""

    def __init__(self, prepared_dir: str | Path, split: str, data_config: dict | None = None):
        self.prepared_dir = Path(prepared_dir)
        self.manifest = json.loads((self.prepared_dir / "manifest.json").read_text(encoding="utf-8"))
        if self.manifest.get("format_version") != 1:
            raise ValueError("Unsupported prepared-data format.")
        self.split = "val" if split == "validation" else split
        if self.split not in {"train", "val", "test"}:
            raise ValueError("split must be train, val, or test.")
        merged = dict(self.manifest["data_config"])
        merged.update((data_config or {}).get("data", data_config or {}))
        self.settings = _settings(merged)
        # Prepared trajectories, assignments and history references cannot be
        # changed merely by overriding a training-time setting. File locations
        # may move; source hashes still identify the originally prepared data.
        prepared_settings = _settings(self.manifest["data_config"])
        prepare_keys = ("file_pattern", "columns", "date_start", "date_end",
                        "min_points", "max_points", "min_history_days", "max_history_days",
                        "split_ratios", "split_seed", "time_id_seconds", "timestamp_source")
        changed = [key for key in prepare_keys if self.settings[key] != prepared_settings[key]]
        if changed:
            raise ValueError("Prepared-data settings differ from the manifest: " + ", ".join(changed)
                             + ". Re-run prepare with these settings in a separate prepared_dir.")
        self.normalizer = CoordinateNormalizer.from_dict(self.manifest["normalizer"])
        with np.load(self.prepared_dir / "trajectories.npz", allow_pickle=False) as arrays:
            self.coords = arrays["coords"]
            self.seconds = arrays["seconds"]
            self.offsets = arrays["offsets"]
        self.samples = self.manifest["samples"][self.split]
        self.epoch = 0

    def __len__(self):
        return len(self.samples)

    def set_epoch(self, epoch: int):
        if epoch < 0:
            raise ValueError("epoch must be nonnegative.")
        self.epoch = int(epoch)

    def _trajectory(self, index: int) -> tuple[np.ndarray, np.ndarray, dict]:
        start, stop = self.offsets[index:index + 2]
        return self.coords[start:stop], self.seconds[start:stop], self.manifest["trajectories"][index]

    def __getitem__(self, item: int) -> dict:
        descriptor = self.samples[item]
        coords, seconds, meta = self._trajectory(descriptor["trajectory_index"])
        epoch = self.epoch if self.split == "train" and self.settings["resample_train_mask"] else 0
        observed = retained_indices(len(coords), self.settings["retention_rate"], self.settings["mask_seed"],
                                    meta["sample_id"], self.settings["keep_endpoints"], epoch)
        histories = []
        for history_index in descriptor["history_indices"]:
            hist_coords, hist_seconds, hist_meta = self._trajectory(history_index)
            if hist_meta["user_id"] != meta["user_id"] or hist_meta["date"] >= meta["date"]:
                raise ValueError("Prepared history violates same-user, strictly-earlier-day requirements.")
            histories.append((hist_coords, hist_seconds))
        return _sample(coords, seconds, observed, histories, meta["date"], meta["user_id"],
                       meta["sample_id"], self.normalizer, self.settings)


def _sample(coords, seconds, observed, histories, day, user_id, sample_id, normalizer, settings,
            include_targets=True):
    threshold, radius = settings["stay_time_minutes"], settings["stay_radius_m"]
    target_stay = (stay_indicators(coords, seconds, threshold, radius) if include_targets
                   else np.zeros(len(seconds), dtype=np.float32))
    obs_coords, obs_seconds = coords[observed], seconds[observed]
    obs_stay = stay_indicators(obs_coords, obs_seconds, threshold, radius)
    normalized_history = [{"coords": normalizer.transform(c), "time": (s / 86400).astype(np.float32)[:, None],
                           "stay": stay_indicators(c, s, threshold, radius)[:, None]}
                          for c, s in histories]
    timestamps = _timestamps(day, seconds)
    missing = np.ones(len(coords), dtype=bool)
    missing[observed] = False
    target_indices = np.flatnonzero(missing)
    return {"obs_coords": normalizer.transform(obs_coords),
            "obs_time": (obs_seconds / 86400).astype(np.float32)[:, None], "obs_stay": obs_stay[:, None],
            "histories": normalized_history,
            "target_time": (seconds[missing] / 86400).astype(np.float32)[:, None],
            "target_coords": normalizer.transform(coords[missing]) if include_targets else np.zeros((len(target_indices), 2), dtype=np.float32),
            "target_stay": target_stay[missing],
            "missing_mask": np.ones(len(target_indices), dtype=bool),
            "environment": environment_features(day, float(obs_seconds[0]), settings.get("holiday_calendar_path")),
            "metadata": {"sample_id": sample_id, "user_id": user_id, "date": day,
                         "observed_indices": observed.tolist(), "target_indices": target_indices.tolist(),
                         "target_timestamps": [timestamps[i] for i in target_indices], "full_timestamps": timestamps,
                         "full_coords": [point.tolist() if include_targets or not missing[i] else None
                                         for i, point in enumerate(np.asarray(coords))],
                         "has_ground_truth": bool(include_targets)}}


def build_inference_sample(observed_records: list[dict], history_records: list[dict],
                           query_times: list[str | datetime], normalizer: CoordinateNormalizer,
                           data_config: dict) -> dict:
    """Construct one unlabelled user-day using the same input routines as training.

    Records use the configured CSV columns. Query timestamps must be supplied;
    this function does not infer where observations are missing. The output
    metadata timeline is the sorted union of observations and requested times;
    decoder targets contain only requested missing times. Queries overlapping
    observations or lying outside their temporal span are rejected: this API
    supports interpolation rather than extrapolation.
    Historical records must belong to the same user and strictly earlier dates.
    No dummy target coordinates are used to construct any model input.
    """
    settings = _settings(data_config)
    columns = settings["columns"]
    if not observed_records:
        raise ValueError("Inference requires at least one current observation.")
    if not query_times:
        raise ValueError("Inference requires explicit query timestamps.")
    current, current_users, current_days = {}, set(), set()
    for row in observed_records:
        day, seconds, _ = _record_time(row, settings)
        user = str(row[columns["user"]]).strip()
        coord = (float(row[columns["latitude"]]), float(row[columns["longitude"]]))
        if not user or not np.isfinite(coord).all() or not -90 <= coord[0] <= 90 or not -180 <= coord[1] <= 180:
            raise ValueError("Inference observations require valid identifiers and coordinates.")
        current_users.add(user)
        current_days.add(day)
        current[seconds] = coord
    if len(current_users) != 1 or len(current_days) != 1:
        raise ValueError("An inference sample must contain exactly one user and one target day.")
    user, day = next(iter(current_users)), next(iter(current_days))
    query_seconds = []
    for timestamp in query_times:
        dt = parse_timestamp(timestamp)
        if dt.date().isoformat() != day:
            raise ValueError("All query timestamps must fall on the current target day.")
        query_seconds.append(_seconds(dt))
    if len(set(query_seconds)) != len(query_seconds):
        raise ValueError("Query timestamps must be unique; duplicate queries are not allowed.")
    if set(query_seconds) & set(current):
        raise ValueError("Query timestamps must not overlap existing observations.")
    if min(query_seconds) < min(current) or max(query_seconds) > max(current):
        raise ValueError("Queries must lie between the first and last observations; extrapolation is unsupported.")
    grouped_history = defaultdict(dict)
    for row in history_records:
        hist_day, seconds, _ = _record_time(row, settings)
        if str(row[columns["user"]]).strip() != user or hist_day >= day:
            raise ValueError("History must belong to the current user and strictly earlier dates.")
        coord = (float(row[columns["latitude"]]), float(row[columns["longitude"]]))
        if not np.isfinite(coord).all() or not -90 <= coord[0] <= 90 or not -180 <= coord[1] <= 180:
            raise ValueError("Historical records contain invalid coordinates.")
        grouped_history[hist_day][seconds] = coord
    eligible_days = [d for d in sorted(grouped_history)
                     if len(grouped_history[d]) >= settings["min_points"]]
    maximum_history = settings["max_history_days"]
    if maximum_history is not None:
        eligible_days = eligible_days[-maximum_history:] if maximum_history else []
    histories = []
    for history_day in eligible_days:
        records = grouped_history[history_day]
        hist_seconds = np.asarray(sorted(records), dtype=np.float64)
        hist_coords = np.asarray([records[s] for s in hist_seconds], dtype=np.float64)
        maximum_points = settings["max_points"]
        if maximum_points is not None and len(hist_seconds) > maximum_points:
            indices = np.linspace(0, len(hist_seconds) - 1, maximum_points, dtype=np.int64)
            hist_coords, hist_seconds = hist_coords[indices], hist_seconds[indices]
        histories.append((hist_coords, hist_seconds))
    seconds = np.asarray(sorted(set(current) | set(query_seconds)), dtype=np.float64)
    if not set(query_seconds) - set(current):
        raise ValueError("At least one query timestamp must be unobserved.")
    coords = np.zeros((len(seconds), 2), dtype=np.float64)
    observed = np.asarray([i for i, value in enumerate(seconds) if value in current], dtype=np.int64)
    for index in observed:
        coords[index] = current[seconds[index]]
    sample_id = hashlib.sha256(f"{user}\0{day}".encode()).hexdigest()[:20]
    sample = _sample(coords, seconds, observed, histories, day, user, sample_id, normalizer, settings,
                     include_targets=False)
    sample["metadata"]["query_indices"] = [int(np.searchsorted(seconds, value)) for value in sorted(set(query_seconds))]
    sample["metadata"]["history_days"] = eligible_days
    sample["metadata"]["history_requirement_met"] = len(histories) >= settings["min_history_days"]
    return sample


def collate_trajectories(samples: list[dict]) -> dict:
    """Right-pad observations, histories and targets with explicit valid masks."""
    if not samples:
        raise ValueError("Cannot collate an empty batch.")
    batch_size = len(samples)
    n = max(len(x["obs_coords"]) for x in samples)
    t = max(len(x["target_coords"]) for x in samples)
    k = max(1, max(len(x["histories"]) for x in samples))
    m = max([1] + [len(h["coords"]) for x in samples for h in x["histories"]])
    batch = {"obs_coords": torch.zeros(batch_size, n, 2), "obs_time": torch.zeros(batch_size, n, 1),
             "obs_stay": torch.zeros(batch_size, n, 1), "obs_lengths": torch.zeros(batch_size, dtype=torch.long),
             "hist_coords": torch.zeros(batch_size, k, m, 2), "hist_time": torch.zeros(batch_size, k, m, 1),
             "hist_stay": torch.zeros(batch_size, k, m, 1), "hist_lengths": torch.zeros(batch_size, k, dtype=torch.long),
             "hist_mask": torch.zeros(batch_size, k, dtype=torch.bool),
             "target_time": torch.zeros(batch_size, t, 1), "target_coords": torch.zeros(batch_size, t, 2),
             "target_stay": torch.zeros(batch_size, t), "target_mask": torch.zeros(batch_size, t, dtype=torch.bool),
             "missing_mask": torch.zeros(batch_size, t, dtype=torch.bool),
             "environment": torch.zeros(batch_size, 5), "metadata": []}
    for i, sample in enumerate(samples):
        obs_count, target_count = len(sample["obs_coords"]), len(sample["target_coords"])
        for name in ("obs_coords", "obs_time", "obs_stay"):
            batch[name][i, :obs_count] = torch.as_tensor(sample[name])
        batch["obs_lengths"][i] = obs_count
        for j, history in enumerate(sample["histories"]):
            length = len(history["coords"])
            for field in ("coords", "time", "stay"):
                batch[f"hist_{field}"][i, j, :length] = torch.as_tensor(history[field])
            batch["hist_lengths"][i, j] = length
            batch["hist_mask"][i, j] = True
        for name in ("target_coords", "target_time", "target_stay", "missing_mask"):
            batch[name][i, :target_count] = torch.as_tensor(sample[name])
        batch["target_mask"][i, :target_count] = True
        batch["environment"][i] = torch.as_tensor(sample["environment"])
        batch["metadata"].append(sample["metadata"])
    return batch
