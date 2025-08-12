# rainfall.py
from __future__ import annotations

from datetime import date, datetime
from typing import Optional
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

