"""Missing-point-only losses and point-weighted spatial/semantic metrics."""
from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn.functional as F

EARTH_RADIUS_M = 6_371_008.8


def multitask_loss(prediction: dict, batch: dict, lambda_semantic: float) -> dict:
    """Mean squared 2-D norm plus mean BCE, both over valid missing points.

    The coordinate term is sum(dx**2 + dy**2)/N, not divided by 2*N.
    No observed points or padded positions contribute to either objective.
    """
    mask = batch["target_mask"].bool()
    count = mask.sum()
    if count.item() == 0:
        raise ValueError("A training batch must contain at least one missing target.")
    coordinates = ((prediction["coords"][mask] - batch["target_coords"][mask]) ** 2).sum(-1)
    semantic = F.binary_cross_entropy_with_logits(
        prediction["stay_logits"][mask], batch["target_stay"][mask].float(), reduction="none")
    coordinate_loss = coordinates.mean()
    semantic_loss = semantic.mean()
    total = coordinate_loss + float(lambda_semantic) * semantic_loss
    return {"loss": total, "coordinate_loss": coordinate_loss,
            "semantic_loss": semantic_loss, "count": int(count.item())}


def distances_m(prediction: np.ndarray, truth: np.ndarray,
                metric: str = "local_euclidean", reference_latitude: float = 0.0) -> np.ndarray:
    """Coordinates are [latitude, longitude] in degrees; output is meters.

    local_euclidean uses a fixed tangent-plane scale at the TRAINING reference
    latitude. haversine is a spherical great-circle alternative, not an MAE
    on angular coordinates. Neither metric silently treats degrees as meters.
    """
    p, t = np.deg2rad(np.asarray(prediction, dtype=np.float64)), np.deg2rad(np.asarray(truth, dtype=np.float64))
    delta = p - t
    delta[..., 1] = (delta[..., 1] + np.pi) % (2 * np.pi) - np.pi
    if metric == "local_euclidean":
        dy = EARTH_RADIUS_M * delta[..., 0]
        dx = EARTH_RADIUS_M * math.cos(math.radians(reference_latitude)) * delta[..., 1]
        return np.hypot(dx, dy)
    if metric == "haversine":
        a = np.sin(delta[..., 0] / 2) ** 2 + np.cos(p[..., 0]) * np.cos(t[..., 0]) * np.sin(delta[..., 1] / 2) ** 2
        return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    raise ValueError(f"Unknown distance metric: {metric}")


class MetricAccumulator:
    """Micro-average over points, independent of padding or batch partitioning."""

    def __init__(self, thresholds_m, metric="local_euclidean", reference_latitude=0.0,
                 semantic_threshold=0.5):
        self.thresholds = np.asarray(thresholds_m, dtype=float)
        self.metric = metric
        self.reference_latitude = float(reference_latitude)
        self.semantic_threshold = semantic_threshold
        self.count = 0
        self.sum_distance = self.sum_squared_distance = 0.0
        self.hits = np.zeros(len(self.thresholds), dtype=np.int64)
        self.tp = self.fp = self.fn = self.tn = 0

    def update(self, prediction_degrees, truth_degrees, stay_probability, stay_truth):
        errors = distances_m(prediction_degrees, truth_degrees, self.metric, self.reference_latitude)
        if not np.isfinite(errors).all():
            raise ValueError("Non-finite predictions prevent meaningful evaluation.")
        self.count += errors.size
        self.sum_distance += float(errors.sum())
        self.sum_squared_distance += float(np.square(errors).sum())
        self.hits += (errors[:, None] <= self.thresholds[None, :]).sum(0)
        predicted = np.asarray(stay_probability) >= self.semantic_threshold
        actual = np.asarray(stay_truth).astype(bool)
        self.tp += int((predicted & actual).sum())
        self.fp += int((predicted & ~actual).sum())
        self.fn += int((~predicted & actual).sum())
        self.tn += int((~predicted & ~actual).sum())
        return errors

    def compute(self):
        if self.count == 0:
            raise ValueError("No missing points were evaluated.")
        precision = self.tp / max(1, self.tp + self.fp)
        recall = self.tp / max(1, self.tp + self.fn)
        return {
            "n_missing_points": self.count,
            "distance_metric": self.metric,
            "reference_latitude_degrees": self.reference_latitude,
            "mae_m": self.sum_distance / self.count,
            "rmse_m": math.sqrt(self.sum_squared_distance / self.count),
            "mae_km": self.sum_distance / self.count / 1000,
            "rmse_km": math.sqrt(self.sum_squared_distance / self.count) / 1000,
            "recall_at_m": {f"{threshold:g}": int(hit) / self.count
                            for threshold, hit in zip(self.thresholds, self.hits)},
            "stay_precision": precision, "stay_recall": recall,
            "stay_f1": 2 * precision * recall / max(1e-15, precision + recall),
            "stay_accuracy": (self.tp + self.tn) / self.count,
            "stay_confusion": {"tp": self.tp, "fp": self.fp, "fn": self.fn, "tn": self.tn},
        }
