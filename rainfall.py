# rainfall.py
from __future__ import annotations

from datetime import date, datetime
from typing import Optional

import pandas as pd


__all__ = [
    "_end_doy",
    "prepare_cumulative",
    "prepare_cumulative_rain_days",
]


def _end_doy(end_date: Optional[date] = None) -> int:
    """
    Return the day-of-year (1..366) for `end_date`. If None, uses today's date.
    Accepts `date` or `datetime` and returns an integer DOY.
    """
    if end_date is None:
        end = date.today()
    else:
        end = end_date.date() if isinstance(end_date, datetime) else end_date
    return end.timetuple().tm_yday


def _ensure_doy(df: pd.DataFrame, *, func_name: str) -> pd.DataFrame:
    """
    Ensure there is a 'doy' column; if not, derive from 'date'.
    Returns a copy of the input with a 'doy' column.
    """
    if "doy" in df.columns:
        return df.copy()

    if "date" not in df.columns:
        raise ValueError(f"{func_name} requires 'doy' or 'date' column")

    d = df.copy()
    d["date"] = pd.to_datetime(d["date"], utc=True, errors="coerce")
    d = d.dropna(subset=["date"])
    d["doy"] = d["date"].dt.dayofyear
    return d


def _validate_basics(df: pd.DataFrame, *, required: tuple[str, ...], func_name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{func_name} requires columns: {list(required)} (missing {missing})")


def _clamp_start_day(start_day: int) -> int:
    sd = int(start_day)
    if not 1 <= sd <= 366:
        raise ValueError("start_day must be between 1 and 366")
    return sd


def prepare_cumulative(
    df: pd.DataFrame,
    end_date: Optional[date] = None,
    *,
    start_day: int = 1,
) -> pd.DataFrame:
    """
    Build per-year cumulative precipitation from Jan 1..end_date, starting the series at `start_day` (DOY).

    Input
    -----
    df : DataFrame with at least:
         - 'year' (int)
         - 'prcp' (float)
         - either 'doy' (int 1..366) or 'date' (parseable)
    end_date : date/datetime, optional (default: today). Used to compute end DOY.
    start_day : int (1..366), default 1. The first DOY to include in the series.

    Output
    ------
    DataFrame with columns:
      - 'year' : int
      - 'doy'  : int in [start_day .. end_doy]
      - 'prcp' : float (daily precipitation; missing days filled with 0.0)
      - 'cum'  : float (cumulative precipitation within the year starting at start_day)
    """
    _validate_basics(df, required=("year", "prcp"), func_name="prepare_cumulative")
    work = _ensure_doy(df, func_name="prepare_cumulative")

    end_doy = _end_doy(end_date)
    sd = _clamp_start_day(start_day)
    if sd > end_doy:
        return pd.DataFrame(columns=["year", "doy", "prcp", "cum"])

    if work.empty:
        return pd.DataFrame(columns=["year", "doy", "prcp", "cum"])

    out = []
    for yr, g in work.groupby("year", sort=True):
        gg = g[["doy", "prcp"]].copy()
        gg = gg[gg["doy"].between(sd, end_doy)]
        # Fill any missing days in the window with 0 precip
        gg = gg.set_index("doy").reindex(range(sd, end_doy + 1), fill_value=0.0)
        gg["year"] = yr
        gg["cum"] = gg["prcp"].cumsum()
        gg = gg.reset_index().rename(columns={"index": "doy"})
        out.append(gg[["year", "doy", "prcp", "cum"]])

    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["year", "doy", "prcp", "cum"])


def prepare_cumulative_rain_days(
    df: pd.DataFrame,
    end_date: Optional[date] = None,
    *,
    threshold: float = 0.0,
    start_day: int = 1,
) -> pd.DataFrame:
    """
    Transform daily precipitation into a per-year cumulative *count of rainy days*,
    starting the series at `start_day` (DOY) through `end_date`.

    A day is considered a "rain day" iff prcp > `threshold`. Choose `threshold`
    according to your units (e.g., 0.01 for inches, ~0.25 for mm).

    Input
    -----
    df : DataFrame with at least:
         - 'year' (int)
         - 'prcp' (float)
         - either 'doy' (int 1..366) or 'date' (parseable)
    end_date : date/datetime, optional (default: today). Used to compute end DOY.
    threshold : float, default 0.0. Values > threshold count as a rain day.
    start_day : int (1..366), default 1. The first DOY to include in the series.

    Output
    ------
    DataFrame with columns:
      - 'year'           : int
      - 'doy'            : int in [start_day .. end_doy]
      - 'rain_day'       : 0/1 indicator
      - 'cum_rain_days'  : cumulative count of rain days within the year
    """
    _validate_basics(df, required=("year", "prcp"), func_name="prepare_cumulative_rain_days")
    work = _ensure_doy(df, func_name="prepare_cumulative_rain_days")

    end_doy = _end_doy(end_date)
    sd = _clamp_start_day(start_day)
    if sd > end_doy:
        return pd.DataFrame(columns=["year", "doy", "rain_day", "cum_rain_days"])

    if work.empty:
        return pd.DataFrame(columns=["year", "doy", "rain_day", "cum_rain_days"])

    out = []
    for yr, g in work.groupby("year", sort=True):
        gg = g[["doy", "prcp"]].copy()
        gg = gg[gg["doy"].between(sd, end_doy)]
        # Fill any missing days in the window with 0 precip (=> not a rain day unless threshold < 0)
        gg = gg.set_index("doy").reindex(range(sd, end_doy + 1), fill_value=0.0)
        gg["rain_day"] = (gg["prcp"] > threshold).astype(int)
        gg["cum_rain_days"] = gg["rain_day"].cumsum()
        gg["year"] = yr
        gg = gg.reset_index().rename(columns={"index": "doy"})
        out.append(gg[["year", "doy", "rain_day", "cum_rain_days"]])

    return (
        pd.concat(out, ignore_index=True)
        if out
        else pd.DataFrame(columns=["year", "doy", "rain_day", "cum_rain_days"])
    )
