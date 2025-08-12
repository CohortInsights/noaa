# plotting.py
from __future__ import annotations

from datetime import date, datetime
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd


def _end_label(end_date: Optional[date]) -> date:
    if end_date is None:
        end_date = date.today()
    if isinstance(end_date, datetime):
        end_date = end_date.date()
    return end_date


def _unit_label(units: str) -> str:
    return "in" if units == "standard" else "mm"


def plot_cumulative(
    df: pd.DataFrame,
    *,
    units: str = "standard",
    end_date: Optional[date] = None,
    highlight_year: Optional[int] = None,
    station_name: Optional[str] = None,
    title: Optional[str] = None,
    show: bool = True,
    save_path: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot cumulative precipitation curves. Expects a DataFrame with columns:
    ['year','doy','cum'] (e.g., returned by rainfall.prepare_cumulative).

    Parameters
    ----------
    df : pandas.DataFrame
        Must include columns 'year', 'doy', 'cum'. 'prcp' is optional/ignored here.
    units : str
        "standard" (inches) or "metric" (mm) for axis label only.
    end_date : date or datetime, optional
        Used only for the default title label (Jan 1 to <date>).
    highlight_year : int, optional
        Year to emphasize (defaults to current year).
    station_name : str, optional
        If provided, included in the title.
    title : str, optional
        Custom plot title. If None, a default title is constructed.
    show : bool
        Whether to call plt.show(). Set False in tests or automation.
    save_path : str, optional
        If provided, saves the figure to this path (PNG, PDF, etc. based on extension).
    ax : matplotlib.axes.Axes, optional
        Existing axes to plot on; if None, a new figure/axes is created.

    Returns
    -------
    (fig, ax)
        The Matplotlib Figure and Axes objects.
    """
    required = {"year", "doy", "cum"}
    if not required.issubset(df.columns):
        raise ValueError("plot_cumulative expects df with columns: 'year','doy','cum'.")

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    if highlight_year is None:
        highlight_year = date.today().year

    unit_lbl = _unit_label(units)
    end_lbl = _end_label(end_date).isoformat()

    # Plot one line per year
    for yr, g in df.groupby("year", sort=True):
        lw = 2.6 if yr == highlight_year else 1.4
        alpha = 1.0 if yr == highlight_year else 0.8
        z = 3 if yr == highlight_year else 1
        ax.plot(g["doy"], g["cum"], label=str(yr), linewidth=lw, alpha=alpha, zorder=z)

    ax.set_xlabel("Day of Year")
    ax.set_ylabel(f"Cumulative Precipitation ({unit_lbl})")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize=9)

    if title is None:
        base = f"Cumulative Rainfall (Jan 1 to {end_lbl})"
        if station_name:
            base = f"{station_name} — {base}"
        title = base
    ax.set_title(title)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    return fig, ax

