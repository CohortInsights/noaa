# rainfall.py
from __future__ import annotations

from typing import Optional, Union
from datetime import date, datetime
import pandas as pd


def _end_doy(end_date: Optional[date] = None) -> int:
    """
    Return day-of-year for `end_date` (defaults to today).
    Accepts date or datetime.
    """
    if end_date is None:
        end_date = date.today()
    if isinstance(end_date, datetime):
        end_date = end_date.date()
    return end_date.timetuple().tm_yday


def prepare_cumulative(df: pd.DataFrame, end_date: Optional[date] = None) -> pd.DataFrame:
    """
    Prepare a cumulative-precipitation table per year from Jan 1 to `end_date`.

    Input
    -----
    df : DataFrame with at least:
         - 'year' (int)
         - 'prcp' (float)
         - either 'doy' (int 1..366) or 'date' (parseable to Timestamp)
    end_date : optional date limiting the comparison window (default: today)

    Output
    ------
    DataFrame with columns ['year', 'doy', 'prcp', 'cum'] where:
      - days are filled 1..end_doy per year
      - missing days have prcp=0.0
      - 'cum' is per-year cumulative precipitation
    """
    if "year" not in df.columns or "prcp" not in df.columns:
        raise ValueError("prepare_cumulative requires 'year' and 'prcp' columns")

    if "doy" not in df.columns:
        if "date" not in df.columns:
            raise ValueError("prepare_cumulative requires 'doy' or 'date' column")
        d = df.copy()
        d["date"] = pd.to_datetime(d["date"], utc=True, errors="coerce")
        d["doy"] = d["date"].dt.dayofyear
        df = d

    end_doy = _end_doy(end_date)
    if df.empty:
        return pd.DataFrame(columns=["year", "doy", "prcp", "cum"])

    out = []
    for yr, g in df.groupby("year", sort=True):
        gg = g[["doy", "prcp"]].copy()
        gg = gg[gg["doy"].between(1, end_doy)]
        # Fill missing DOYs with 0 precip
        gg = gg.set_index("doy").reindex(range(1, end_doy + 1), fill_value=0.0)
        gg["year"] = yr
        gg["cum"] = gg["prcp"].cumsum()
        gg = gg.reset_index().rename(columns={"index": "doy"})
        out.append(gg[["year", "doy", "prcp", "cum"]])

    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["year", "doy", "prcp", "cum"])


def count_rain_days(
        df: pd.DataFrame,
        start_date: Optional[Union[str, date]] = None,
        end_date: Optional[Union[str, date]] = None,
        *,
        threshold: float = 0.0,
) -> dict:
    """
    Count number of days with no rain vs some rain, filling missing days in the window as 0.

    Parameters
    ----------
    df : pd.DataFrame
        Requires at least ['date', 'prcp'] columns. Multiple years OK.
        'date' may be datetime-like (timezone OK) or string.
    start_date : str | date, optional
        Inclusive start (YYYY-MM-DD). Defaults to min(df['date']).
    end_date : str | date, optional
        Inclusive end (YYYY-MM-DD). Defaults to max(df['date']).
    threshold : float, default 0.0
        Values <= threshold are counted as "no rain". Example: threshold=0.01 (inches).

    Returns
    -------
    dict
        {
            "total_days": int,
            "no_rain_days": int,
            "rain_days": int
        }
    """
    if df.empty:
        return {"total_days": 0, "no_rain_days": 0, "rain_days": 0}
    if not {"date", "prcp"}.issubset(df.columns):
        raise ValueError("count_rain_days requires DataFrame with columns: ['date', 'prcp'].")

    work = df.copy()
    # Normalize dates to midnight and strip tz so we can reindex cleanly
    work["date"] = pd.to_datetime(work["date"], errors="coerce").dt.tz_localize(None)
    work = work.dropna(subset=["date"])

    # Collapse to one value per day (sum is safest; NOAA PRCP is one row/day anyway)
    daily = work.groupby(work["date"].dt.normalize(), as_index=True)["prcp"].sum().sort_index()

    # Determine window
    start = pd.to_datetime(start_date).normalize() if start_date else daily.index.min().normalize()
    end = pd.to_datetime(end_date).normalize() if end_date else daily.index.max().normalize()
    if end < start:
        raise ValueError("end_date must be on/after start_date.")

    # Reindex to ensure every day is present (missing days -> 0 precip)
    all_days = pd.date_range(start, end, freq="D")
    daily = daily.reindex(all_days, fill_value=0.0)

    no_rain_days = int((daily <= threshold).sum())
    rain_days = int((daily > threshold).sum())

    return {
        "total_days": int(len(daily)),
        "no_rain_days": no_rain_days,
        "rain_days": rain_days,
    }


def prepare_cumulative_rain_days(
        df: pd.DataFrame,
        end_date: Optional[date] = None,
        *,
        threshold: float = 0.0,
) -> pd.DataFrame:
    """
    Transform daily precip into a per-year cumulative *count of rainy days*.

    Input
    -----
    df : DataFrame with at least:
         - 'year' (int)
         - 'prcp' (float)
         - either 'doy' (int 1..366) or 'date' (parseable)
    end_date : optional date to clip each year (default: today)
    threshold : float (default 0.0)
        A day is considered a "rain day" iff prcp > threshold
        e.g., threshold=0.01 (inches) or 0.25 (mm) depending on UNITS.

    Output
    ------
    DataFrame with columns:
      - 'year'         : int
      - 'doy'          : day of year (1..end_doy)
      - 'rain_day'     : 0/1 indicator (prcp > threshold)
      - 'cum_rain_days': cumulative count of rain days within the year
    """
    # Validate / infer required columns
    if "year" not in df.columns or "prcp" not in df.columns:
        raise ValueError("prepare_cumulative_rain_days requires 'year' and 'prcp' columns")

    work = df.copy()
    if "doy" not in work.columns:
        if "date" not in work.columns:
            raise ValueError("prepare_cumulative_rain_days requires 'doy' or 'date' column")
        work["date"] = pd.to_datetime(work["date"], utc=True, errors="coerce")
        work["doy"] = work["date"].dt.dayofyear

    # Determine end DOY (mirror behavior of your other prep)
    def _end_doy(ed: Optional[date]) -> int:
        if ed is None:
            ed = date.today()
        if isinstance(ed, datetime):
            ed = ed.date()
        return ed.timetuple().tm_yday

    end_doy = _end_doy(end_date)

    if work.empty:
        return pd.DataFrame(columns=["year", "doy", "rain_day", "cum_rain_days"])

    out = []
    for yr, g in work.groupby("year", sort=True):
        gg = g[["doy", "prcp"]].copy()
        gg = gg[gg["doy"].between(1, end_doy)]
        # Ensure every DOY exists; missing days -> prcp=0.0 (counts as no rain unless threshold < 0)
        gg = gg.set_index("doy").reindex(range(1, end_doy + 1), fill_value=0.0)
        gg["rain_day"] = (gg["prcp"] > threshold).astype(int)
        gg["cum_rain_days"] = gg["rain_day"].cumsum()
        gg["year"] = yr
        gg = gg.reset_index().rename(columns={"index": "doy"})
        out.append(gg[["year", "doy", "rain_day", "cum_rain_days"]])

    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(
        columns=["year", "doy", "rain_day", "cum_rain_days"])
