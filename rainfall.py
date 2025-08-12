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


def _prepare_base_precip(
    df: pd.DataFrame,
    end_date: Optional[date],
    start_day: int,
    *,
    func_name: str,
) -> pd.DataFrame:
    """
    Shared prep step for cumulative series.
    - Validates required columns
    - Ensures 'doy' exists (derives from 'date' if needed)
    - Computes [start_day..end_doy] window
    - Fills missing days with 0.0 precipitation
    Returns a DataFrame with columns ['year', 'doy', 'prcp'].
    """
    _validate_basics(df, required=("year", "prcp"), func_name=func_name)
    work = _ensure_doy(df, func_name=func_name)

    end_doy = _end_doy(end_date)
    sd = _clamp_start_day(start_day)

    if sd > end_doy or work.empty:
        return pd.DataFrame(columns=["year", "doy", "prcp"])

    parts: list[pd.DataFrame] = []
    for yr, g in work.groupby("year", sort=True):
        gg = g[["doy", "prcp"]].copy()
        gg = gg[gg["doy"].between(sd, end_doy)]
        # Fill any missing days in the window with 0 precip
        gg = gg.set_index("doy").reindex(range(sd, end_doy + 1), fill_value=0.0)
        gg["year"] = yr
        gg = gg.reset_index().rename(columns={"index": "doy"})
        parts.append(gg[["year", "doy", "prcp"]])

    if not parts:
        return pd.DataFrame(columns=["year", "doy", "prcp"])

    out = pd.concat(parts, ignore_index=True)
    return out.sort_values(["year", "doy"], kind="stable").reset_index(drop=True)


def prepare_cumulative(
    df: pd.DataFrame,
    end_date: Optional[date] = None,
    *,
    start_day: int = 1,
) -> pd.DataFrame:
    """
    Build per-year cumulative precipitation starting the series at `start_day` (DOY)
    and ending at `end_date` (defaults to today).

    Input
    -----
    df : DataFrame with at least:
         - 'year' (int)
         - 'prcp' (float)
         - either 'doy' (int 1..366) or 'date' (parseable)

    Parameters
    ----------
    end_date : date or datetime, optional
        Last date included for each year. If None, uses today's date.
    start_day : int, default 1
        First DOY to include (1..366).

    Output
    ------
    DataFrame with columns:
      - 'year' : int
      - 'doy'  : int in [start_day .. end_doy]
      - 'prcp' : float (daily precipitation; missing days filled with 0.0)
      - 'cum'  : float (cumulative precipitation within the year starting at start_day)
    """
    base = _prepare_base_precip(df, end_date, start_day, func_name="prepare_cumulative")
    if base.empty:
        return pd.DataFrame(columns=["year", "doy", "prcp", "cum"])

    base["cum"] = base.groupby("year", sort=True)["prcp"].cumsum()
    return base[["year", "doy", "prcp", "cum"]]


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

    A day is considered a "rain day" iff prcp >= `threshold`. Choose `threshold`
    according to your units (e.g., 0.01 for inches, ~0.25 for mm).

    Input
    -----
    df : DataFrame with at least:
         - 'year' (int)
         - 'prcp' (float)
         - either 'doy' (int 1..366) or 'date' (parseable)

    Parameters
    ----------
    threshold : float, default 0.0
        Minimum precipitation to count as a rain day (inclusive).
    end_date : date or datetime, optional
        Last date included for each year. If None, uses today's date.
    start_day : int, default 1
        First DOY to include (1..366).

    Output
    ------
    DataFrame with columns:
      - 'year'          : int
      - 'doy'           : int in [start_day .. end_doy]
      - 'rain_day'      : int in {0,1}
      - 'cum_rain_days' : int (cumulative count within the year starting at start_day)
    """
    base = _prepare_base_precip(df, end_date, start_day, func_name="prepare_cumulative_rain_days")
    if base.empty:
        return pd.DataFrame(columns=["year", "doy", "rain_day", "cum_rain_days"])

    # Inclusive threshold (>=), as requested.
    base["rain_day"] = (base["prcp"] >= threshold).astype(int)
    base["cum_rain_days"] = base.groupby("year", sort=True)["rain_day"].cumsum()
    return base[["year", "doy", "rain_day", "cum_rain_days"]]
