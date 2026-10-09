"""Training, validation, checkpointing and held-out evaluation."""
from __future__ import annotations

import csv
from copy import deepcopy
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import save_config
from .data import TrajectoryDataset, collate_trajectories
from .metrics import MetricAccumulator, multitask_loss
from .model import BAMTR
from .utils import (capture_rng_state, environment_info, file_sha256, load_checkpoint,
                    parameter_count, restore_rng_state, save_checkpoint, seed_everything,
                    select_device, to_device, write_json)


def teacher_forcing_ratio(epoch: int, epochs: int, start: float, end: float) -> float:
    """Zero-based epoch; one-epoch runs use the start value."""
    return float(start if epochs == 1 else start + (end - start) * epoch / (epochs - 1))


def _loader(dataset, config, shuffle=False, epoch=0):
    generator = torch.Generator().manual_seed(int(config["seed"]) + epoch)
    return DataLoader(dataset, batch_size=config["training"]["batch_size"], shuffle=shuffle,
                      num_workers=config["training"]["num_workers"],
                      collate_fn=collate_trajectories, generator=generator,
                      pin_memory=torch.cuda.is_available(), drop_last=False)


def _metric_accumulator(dataset, config):
    reference_latitude = float(dataset.normalizer.inverse_transform(np.array([[0.5, 0.5]], dtype=np.float64))[0, 0])
    settings = config["evaluation"]
    return MetricAccumulator(settings["recall_thresholds_m"], settings["distance_metric"], reference_latitude,
                             settings["semantic_threshold"])


def _manifest_hash(config):
    return file_sha256(Path(config["data"]["prepared_dir"]) / "manifest.json")


def run_epoch(model, loader, config, device, optimizer=None, tf_ratio=0.0,
              max_batches=None, prediction_path=None):
    training = optimizer is not None
    model.train(training)
    total, coordinate, semantic, points = 0.0, 0.0, 0.0, 0
    metrics = _metric_accumulator(loader.dataset, config) if not training else None
    stream, writer = None, None
    if prediction_path:
        prediction_path = Path(prediction_path)
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        stream = prediction_path.open("w", newline="", encoding="utf-8")
        writer = csv.DictWriter(stream, fieldnames=["sample_id", "user_id", "date", "timestamp",
                                                   "true_lat", "true_lng", "pred_lat", "pred_lng",
                                                   "true_stay", "stay_probability", "error_m"])
        writer.writeheader()
    try:
        with torch.set_grad_enabled(training):
            for batch_index, batch in enumerate(loader):
                if max_batches is not None and batch_index >= max_batches:
                    break
                batch = to_device(batch, device)
                if training:
                    optimizer.zero_grad(set_to_none=True)
                prediction = model(batch, teacher_forcing_ratio=tf_ratio if training else 0.0)
                losses = multitask_loss(prediction, batch, config["training"]["lambda_semantic"])
                if not torch.isfinite(losses["loss"]):
                    raise FloatingPointError("Loss became non-finite; inspect the data and learning rate.")
                if training:
                    losses["loss"].backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["gradient_clip"], error_if_nonfinite=True)
                    optimizer.step()
                count = losses["count"]
                total += float(losses["loss"].detach()) * count
                coordinate += float(losses["coordinate_loss"].detach()) * count
                semantic += float(losses["semantic_loss"].detach()) * count
                points += count
                if metrics is not None:
                    mask = batch["target_mask"].cpu().numpy().astype(bool)
                    normalized_prediction = prediction["coords"].detach().cpu().numpy()
                    normalized_truth = batch["target_coords"].cpu().numpy()
                    predicted_degrees = loader.dataset.normalizer.inverse_transform(normalized_prediction[mask])
                    truth_degrees = loader.dataset.normalizer.inverse_transform(normalized_truth[mask])
                    probability = prediction["stay_logits"].sigmoid().cpu().numpy()[mask]
                    truth_stay = batch["target_stay"].cpu().numpy()[mask]
                    errors = metrics.update(predicted_degrees, truth_degrees, probability, truth_stay)
                    if writer is not None:
                        offset = 0
                        for i, metadata in enumerate(batch["metadata"]):
                            length = int(mask[i].sum())
                            for j in range(length):
                                k = offset + j
                                writer.writerow({"sample_id": metadata["sample_id"], "user_id": metadata["user_id"],
                                                 "date": metadata["date"], "timestamp": metadata["target_timestamps"][j],
                                                 "true_lat": truth_degrees[k, 0], "true_lng": truth_degrees[k, 1],
                                                 "pred_lat": predicted_degrees[k, 0], "pred_lng": predicted_degrees[k, 1],
                                                 "true_stay": int(truth_stay[k]), "stay_probability": probability[k],
                                                 "error_m": errors[k]})
                            offset += length
    finally:
        if stream:
            stream.close()
    if not points:
        raise ValueError("No valid missing points were processed; inspect the split and batch limits.")
    result = {"loss": total / points, "coordinate_loss": coordinate / points,
              "semantic_loss": semantic / points, "n_missing_points": points}
    if metrics is not None:
        result.update(metrics.compute())
    return result


def train(config: dict, resume: str | None = None):
    settings = config["training"]
    seed_everything(config["seed"], settings["deterministic"], settings["cpu_threads"])
    device = select_device(settings["device"])
    run_dir = Path(settings["run_dir"])
    if run_dir.exists() and any(run_dir.iterdir()) and resume is None:
        raise FileExistsError(f"Run directory is not empty: {run_dir}. Use a new run directory or --resume.")
    run_dir.mkdir(parents=True, exist_ok=True)
    train_dataset = TrajectoryDataset(config["data"]["prepared_dir"], "train", config["data"])
    val_dataset = TrajectoryDataset(config["data"]["prepared_dir"], "val", config["data"])
    if not len(train_dataset) or not len(val_dataset):
        raise ValueError("Training and validation must each contain eligible user-days.")
    model = BAMTR(config["model"], environment_dim=5).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
    fingerprint = _manifest_hash(config)
    best_mae, start_epoch, bad_epochs = float("inf"), 0, 0
    if resume:
        if Path(resume).resolve().parent != run_dir.resolve() or Path(resume).name != "latest.pt":
            raise ValueError("Resume from latest.pt in the same run directory; best.pt is for evaluation.")
        checkpoint = load_checkpoint(resume, device)
        if checkpoint["manifest_sha256"] != fingerprint:
            raise ValueError("Resume requires the same prepared-data manifest.")
        old = checkpoint["config"]
        for section in ("model", "data", "evaluation"):
            if old[section] != config[section]:
                raise ValueError(f"Resume cannot silently change {section} settings.")
        if old["seed"] != config["seed"]:
            raise ValueError("Resume requires the original training seed.")
        # The teacher-forcing schedule is fixed for exact mid-run continuation.
        for key in settings:
            if key in ("device", "num_workers", "cpu_threads", "run_dir"):
                continue
            if old["training"][key] != settings[key]:
                raise ValueError(f"Resume requires the same training.{key}; start a new run to change it.")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = checkpoint["epoch"] + 1
        best_mae = checkpoint["best_val_mae_m"]
        bad_epochs = checkpoint["bad_epochs"]
        restore_rng_state(checkpoint["rng_state"])
        history_path = run_dir / "history.csv"
        if history_path.exists():
            with history_path.open(newline="", encoding="utf-8") as stream:
                reader = csv.DictReader(stream)
                columns = reader.fieldnames
                rows = [row for row in reader if int(row["epoch"]) <= start_epoch]
            # A crash after logging but before checkpointing must not duplicate an epoch.
            with history_path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=columns)
                writer.writeheader()
                writer.writerows(rows)
        best_path = run_dir / "best.pt"
        if checkpoint["validation"]["mae_m"] == best_mae:
            # Repair the short crash window between latest.pt and best.pt writes.
            save_checkpoint(checkpoint, best_path)
            write_json(checkpoint["validation"], run_dir / "best_validation.json")
        elif not best_path.exists():
            raise FileNotFoundError("The earlier best.pt is missing; restore it before resuming this run.")
    if start_epoch >= settings["epochs"]:
        raise ValueError("This checkpoint has already completed the configured training schedule.")
    save_config(config, run_dir / "config.yaml")
    write_json(environment_info(), run_dir / "environment.json")
    write_json(parameter_count(model), run_dir / "parameters.json")
    write_json(train_dataset.normalizer.to_dict(), run_dir / "normalizer.json")
    write_json({"manifest_sha256": fingerprint, "n_train_trajectories": len(train_dataset),
                "n_val_trajectories": len(val_dataset), "selection_metric": "validation MAE in meters",
                "limited_smoke_run": settings["max_train_batches"] is not None or settings["max_eval_batches"] is not None},
               run_dir / "provenance.json")
    print(f"Device: {device}; train/validation trajectories: {len(train_dataset)}/{len(val_dataset)}; "
          f"parameters: {parameter_count(model)['total_parameters']:,}", flush=True)
    val_loader = _loader(val_dataset, config)
    history_path = run_dir / "history.csv"
    fieldnames = ["epoch", "teacher_forcing_ratio", "train_loss", "train_coordinate_loss", "train_semantic_loss",
                  "val_loss", "val_coordinate_loss", "val_semantic_loss", "val_mae_m", "val_rmse_m", "val_recall_100m",
                  "train_missing_points", "val_missing_points", "epoch_seconds"]
    for epoch in range(start_epoch, settings["epochs"]):
        tick = time.perf_counter()
        train_dataset.set_epoch(epoch)
        ratio = teacher_forcing_ratio(epoch, settings["epochs"], settings["teacher_forcing_start"], settings["teacher_forcing_end"])
        trained = run_epoch(model, _loader(train_dataset, config, shuffle=True, epoch=epoch), config, device,
                            optimizer, ratio, settings["max_train_batches"])
        validated = run_epoch(model, val_loader, config, device, max_batches=settings["max_eval_batches"])
        improved = validated["mae_m"] < best_mae
        if improved:
            best_mae, bad_epochs = validated["mae_m"], 0
        else:
            bad_epochs += 1
        row = {"epoch": epoch + 1, "teacher_forcing_ratio": ratio,
               "train_loss": trained["loss"], "train_coordinate_loss": trained["coordinate_loss"],
               "train_semantic_loss": trained["semantic_loss"], "val_loss": validated["loss"],
               "val_coordinate_loss": validated["coordinate_loss"], "val_semantic_loss": validated["semantic_loss"],
               "val_mae_m": validated["mae_m"], "val_rmse_m": validated["rmse_m"],
               "val_recall_100m": validated["recall_at_m"].get("100", ""),
               "train_missing_points": trained["n_missing_points"], "val_missing_points": validated["n_missing_points"],
               "epoch_seconds": time.perf_counter() - tick}
        write_header = not history_path.exists()
        with history_path.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(row)
        checkpoint = {"format_version": 1, "epoch": epoch, "config": deepcopy(config),
                      "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                      "normalizer": train_dataset.normalizer.to_dict(), "manifest_sha256": fingerprint,
                      "best_val_mae_m": best_mae, "bad_epochs": bad_epochs, "validation": validated,
                      "rng_state": capture_rng_state()}
        save_checkpoint(checkpoint, run_dir / "latest.pt")
        if improved:
            save_checkpoint(checkpoint, run_dir / "best.pt")
            write_json(validated, run_dir / "best_validation.json")
        print(f"Epoch {epoch + 1:03d}/{settings['epochs']}: train loss={trained['loss']:.6f}; "
              f"val loss={validated['loss']:.6f}; val MAE={validated['mae_m']:.3f} m; "
              f"time={row['epoch_seconds']:.1f}s", flush=True)
        if settings["early_stopping_patience"] and bad_epochs >= settings["early_stopping_patience"]:
            print("Early stopping: validation MAE did not improve.", flush=True)
            break
    print(f"Best checkpoint: {run_dir / 'best.pt'} (validation MAE {best_mae:.3f} m)", flush=True)
    return run_dir / "best.pt"


def evaluate(checkpoint_path, split="test", output_dir=None, device="auto", prepared_dir=None, max_batches=None):
    checkpoint = load_checkpoint(checkpoint_path)
    config = deepcopy(checkpoint["config"])
    if prepared_dir:
        config["data"]["prepared_dir"] = str(prepared_dir)
    if _manifest_hash(config) != checkpoint["manifest_sha256"]:
        raise ValueError("Evaluation requires the prepared data used for this checkpoint. Use predict for new observations.")
    seed_everything(config["seed"], config["training"]["deterministic"], config["training"]["cpu_threads"])
    target_device = select_device(device)
    dataset = TrajectoryDataset(config["data"]["prepared_dir"], split, config["data"])
    if dataset.normalizer.to_dict() != checkpoint["normalizer"]:
        raise ValueError("The normalizer differs from the training checkpoint.")
    model = BAMTR(config["model"], environment_dim=5).to(target_device)
    model.load_state_dict(checkpoint["model_state"])
    output_dir = Path(output_dir or Path(checkpoint_path).parent / split)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = run_epoch(model, _loader(dataset, config), config, target_device,
                       max_batches=max_batches, prediction_path=output_dir / "predictions.csv")
    result.update({"split": split, "checkpoint_epoch": checkpoint["epoch"] + 1,
                   "checkpoint_sha256": file_sha256(checkpoint_path), "limited_evaluation": max_batches is not None,
                   "retention_rate": config["data"]["retention_rate"]})
    write_json(result, output_dir / "metrics.json")
    print(f"{split}: MAE={result['mae_m']:.3f} m, RMSE={result['rmse_m']:.3f} m; "
          f"missing points={result['n_missing_points']:,}", flush=True)
    return result
