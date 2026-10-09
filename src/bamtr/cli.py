"""Command-line entry points. Run all commands from the repository root."""
from __future__ import annotations

import argparse
from copy import deepcopy
import csv
import json
from pathlib import Path

import numpy as np

from .config import load_config, save_config
from .data import prepare_data
from .engine import evaluate, train
from .inference import predict
from .utils import read_json, write_json


def _configured(args):
    config = load_config(args.config, args.set)
    if getattr(args, "raw_dir", None):
        config["data"]["raw_dir"] = str(Path(args.raw_dir).resolve())
    if getattr(args, "prepared_dir", None):
        config["data"]["prepared_dir"] = str(args.prepared_dir)
    if getattr(args, "run_dir", None):
        config["training"]["run_dir"] = str(args.run_dir)
    if getattr(args, "device", None):
        config["training"]["device"] = args.device
    return config


def _configuration_arguments(parser):
    parser.add_argument("--config", required=True, help="YAML configuration file")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="Override a setting, e.g. --set training.batch_size=16 (repeatable)")
    parser.add_argument("--raw-dir", help="Override the input CSV directory")
    parser.add_argument("--prepared-dir", help="Override the prepared NPZ/JSON directory")


def parser():
    root = argparse.ArgumentParser(description="BAMTR training and missing-trajectory recovery")
    commands = root.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Clean CSVs, split users and fit training-only normalization")
    _configuration_arguments(prepare)
    training = commands.add_parser("train", help="Train and select a checkpoint using validation MAE")
    _configuration_arguments(training)
    training.add_argument("--run-dir")
    training.add_argument("--device", choices=["auto", "cpu", "cuda"])
    training.add_argument("--resume", help="The latest.pt checkpoint from the same run directory")
    evaluation = commands.add_parser("evaluate", help="Evaluate only missing points in a held-out split")
    evaluation.add_argument("--checkpoint", required=True)
    evaluation.add_argument("--split", choices=["train", "val", "test"], default="test")
    evaluation.add_argument("--output-dir")
    evaluation.add_argument("--prepared-dir")
    evaluation.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    evaluation.add_argument("--max-batches", type=int, help="Development-only limit, recorded in metrics")
    prediction = commands.add_parser("predict", help="Recover known query timestamps without target coordinates")
    prediction.add_argument("--checkpoint", required=True)
    prediction.add_argument("--observations", required=True, help="Sparse current observations CSV")
    prediction.add_argument("--queries", required=True, help="CSV containing ID,time and optional tid")
    prediction.add_argument("--history-dir", help="Daily historical CSV folder; omit for current-only fallback")
    prediction.add_argument("--output", required=True, help="Merged reconstructed CSV output")
    prediction.add_argument("--calendar", help="Optional date,is_holiday CSV calendar extension")
    prediction.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    inspect = commands.add_parser("inspect", help="Print aggregate prepared-data statistics without user IDs")
    _configuration_arguments(inspect)
    plot = commands.add_parser("plot", help="Plot actual training logs or measured Recall metrics")
    plot.add_argument("--run-dir")
    plot.add_argument("--metrics", help="metrics.json: plot Recall instead of loss")
    plot.add_argument("--output", help="Output PNG/PDF/SVG (optional for --show)")
    plot.add_argument("--show", action="store_true", help="Display the loss plot interactively")
    sweep = commands.add_parser("sweep", help="Run independent seeds/rates with fixed split and mask seeds")
    _configuration_arguments(sweep)
    sweep.add_argument("--rates", nargs="+", type=float, default=[0.2, 0.4, 0.6])
    sweep.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    sweep.add_argument("--output-dir", default="runs/sweep")
    sweep.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return root


def _check_prepared(config):
    if not (Path(config["data"]["prepared_dir"]) / "manifest.json").is_file():
        raise FileNotFoundError("Prepared data are missing. Run 'python -m bamtr prepare --config ...' first.")


def run_sweep(args):
    base = _configured(args)
    _check_prepared(base)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    stay_settings = {0.2: (60.0, 150.0), 0.4: (90.0, 100.0), 0.6: (120.0, 100.0)}
    if len(set(args.rates)) != len(args.rates) or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Sweep rates and seeds must not contain duplicates.")
    for rate in args.rates:
        if rate not in stay_settings:
            raise ValueError("The paper-rate sweep supports 0.2, 0.4, 0.6. Use train for a custom retention rate.")
        for seed in args.seeds:
            config = deepcopy(base)
            config["seed"] = seed  # User split and observation mask seeds stay fixed.
            config["data"]["retention_rate"] = rate
            config["data"]["stay_time_minutes"], config["data"]["stay_radius_m"] = stay_settings[rate]
            config["training"]["run_dir"] = str(output / f"rho_{int(rate * 100)}_seed_{seed}")
            best = train(config)
            result = evaluate(best, "test", device=args.device)
            rows.append({"retention_rate": rate, "seed": seed, "mae_m": result["mae_m"],
                         "rmse_m": result["rmse_m"], "recall_100m": result["recall_at_m"].get("100")})
    with (output / "individual_runs.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = []
    for rate in args.rates:
        group = [row for row in rows if row["retention_rate"] == rate]
        row = {"retention_rate": rate, "runs": len(group), "std_ddof": 1 if len(group) > 1 else None}
        for name in ("mae_m", "rmse_m", "recall_100m"):
            values = [item[name] for item in group]
            row[f"{name}_mean"] = float(np.mean(values))
            row[f"{name}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
        summary.append(row)
    write_json(summary, output / "summary.json")
    return summary


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == "prepare":
        manifest = prepare_data(_configured(args))
        print(json.dumps({key: manifest[key] for key in ("counts", "user_counts", "sample_counts") if key in manifest}, indent=2))
    elif args.command == "train":
        config = _configured(args)
        _check_prepared(config)
        train(config, args.resume)
    elif args.command == "evaluate":
        evaluate(args.checkpoint, args.split, args.output_dir, args.device, args.prepared_dir, args.max_batches)
    elif args.command == "predict":
        predict(args.checkpoint, args.observations, args.queries, args.history_dir, args.output, args.device, args.calendar)
    elif args.command == "inspect":
        config = _configured(args)
        _check_prepared(config)
        manifest = read_json(Path(config["data"]["prepared_dir"]) / "manifest.json")
        public_summary = {key: manifest[key] for key in ("format_version", "counts", "user_counts", "eligible_user_counts", "sample_counts", "normalizer") if key in manifest}
        print(json.dumps(public_summary, indent=2))
    elif args.command == "plot":
        from .plotting import plot_recall, plot_run
        if args.metrics:
            if not args.output:
                raise ValueError("A Recall plot requires --output.")
            plot_recall(args.metrics, args.output)
        elif args.run_dir:
            if not args.output and not args.show:
                raise ValueError("Use --output or --show for a loss plot.")
            plot_run(args.run_dir, args.output, args.show)
        else:
            raise ValueError("Specify --run-dir for loss curves or --metrics for Recall.")
    elif args.command == "sweep":
        run_sweep(args)


if __name__ == "__main__":
    main()
