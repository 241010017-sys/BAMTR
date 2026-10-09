"""Generate clearly synthetic CSV fixtures without reading private trajectories."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np


def generate(output: Path, users=20, days=6, points=48, seed=123):
    if users < 5 or days < 2 or points < 8:
        raise ValueError("Use at least 5 users, 2 days, and 8 points per day.")
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    fields = ["ID", "lng", "lat", "time", "tid"]
    last_user_records = []
    for day in range(days):
        date = datetime(2023, 4, 1) + timedelta(days=day)
        rows = []
        for user in range(users):
            times = np.sort(rng.choice(np.arange(6 * 120, 22 * 120), size=points, replace=False)) * 30
            fraction = np.linspace(0, 1, points)
            motion = np.where(fraction < .3, 0, np.where(fraction < .5, (fraction - .3) * 5,
                               np.where(fraction < .8, 1, (1 - fraction) * 5)))
            lat = 30.0 + (user % 5) * .012 + motion * (.010 + day * .00015) + rng.normal(0, .00003, points)
            lng = 120.0 + (user // 5) * .015 + motion * .017 + rng.normal(0, .00003, points)
            current = [{"ID": f"synthetic_{user:03d}", "lng": f"{lng[i]:.8f}", "lat": f"{lat[i]:.8f}",
                        "time": (date + timedelta(seconds=int(seconds))).strftime("%Y-%m-%d %H:%M:%S"),
                        "tid": int(seconds // 30)} for i, seconds in enumerate(times)]
            rows.extend(current)
            if day == days - 1 and user == 0:
                last_user_records = current
        with (output / f"{date:%Y-%m-%d}.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    inference = output.parent / "inference"
    inference.mkdir(parents=True, exist_ok=True)
    retained = np.unique(np.r_[0, np.arange(2, points - 1, 3), points - 1])
    with (inference / "observations.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows([last_user_records[i] for i in retained])
    with (inference / "queries.csv").open("w", newline="", encoding="utf-8") as stream:
        query_fields = ["ID", "time", "tid"]
        writer = csv.DictWriter(stream, fieldnames=query_fields)
        writer.writeheader()
        writer.writerows([{key: row[key] for key in query_fields}
                          for i, row in enumerate(last_user_records) if i not in retained])
    print(f"Generated {users * days * points:,} synthetic records in {output}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("examples/synthetic"))
    parser.add_argument("--users", type=int, default=20)
    parser.add_argument("--days", type=int, default=6)
    parser.add_argument("--points", type=int, default=48)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    generate(args.output, args.users, args.days, args.points, args.seed)
