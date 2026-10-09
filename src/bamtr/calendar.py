"""Date-derived context using the mainland China 2023 working-day calendar.

Source: State Council General Office, 8 December 2022:
https://app.www.gov.cn/govdata/gov/202212/08/495070/article.html
The calendar marks non-working days, including ordinary weekends, with
official make-up working days taking precedence. It does not infer weather.
"""
from __future__ import annotations

import csv
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path

import numpy as np

CALENDAR_SOURCE = "https://app.www.gov.cn/govdata/gov/202212/08/495070/article.html"


def _date_range(start: str, end: str) -> set[str]:
    first, last = date.fromisoformat(start), date.fromisoformat(end)
    return {(first + timedelta(days=i)).isoformat() for i in range((last - first).days + 1)}


PUBLIC_HOLIDAYS = set().union(
    _date_range("2023-01-01", "2023-01-02"),
    _date_range("2023-01-21", "2023-01-27"),
    {"2023-04-05"},
    _date_range("2023-04-29", "2023-05-03"),
    _date_range("2023-06-22", "2023-06-24"),
    _date_range("2023-09-29", "2023-10-06"),
)
MAKEUP_WORKDAYS = {"2023-01-28", "2023-01-29", "2023-04-23", "2023-05-06",
                   "2023-06-25", "2023-10-07", "2023-10-08"}


@lru_cache(maxsize=8)
def _external_calendar(path: str) -> dict[str, int]:
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"date", "is_holiday"}.issubset(reader.fieldnames or []):
            raise ValueError("A holiday calendar needs date,is_holiday columns.")
        result = {}
        for row in reader:
            day = date.fromisoformat(row["date"]).isoformat()
            flag = int(row["is_holiday"])
            if flag not in (0, 1) or day in result:
                raise ValueError("Calendar dates must be unique; is_holiday must be 0 or 1.")
            result[day] = flag
    return result


def holiday_status(day: str | date, calendar_path: str | None = None) -> int:
    """Return 0=working, 1=non-working, 2=unknown; never guess other years."""
    day = date.fromisoformat(str(day)[:10])
    key = day.isoformat()
    if calendar_path:
        external = _external_calendar(str(Path(calendar_path).resolve()))
        if key in external:
            return external[key]
    if day.year != 2023:
        return 2
    if key in MAKEUP_WORKDAYS:
        return 0
    return int(key in PUBLIC_HOLIDAYS or day.weekday() >= 5)


def environment_features(day: str | date, first_observed_seconds: float,
                         calendar_path: str | None = None) -> np.ndarray:
    """[observed start hour/24, weekday/6, workday, restday, unknown]."""
    day = date.fromisoformat(str(day)[:10])
    one_hot = np.eye(3, dtype=np.float32)[holiday_status(day, calendar_path)]
    return np.r_[np.float32(first_observed_seconds / 86400.0),
                 np.float32(day.weekday() / 6.0), one_hot].astype(np.float32)
