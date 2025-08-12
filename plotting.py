# plotting.py
from __future__ import annotations

from datetime import date, datetime
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd


def _end_label(end_date: Optional[date]) -> date:
    """
    Normalize the end date for titles. If None, use today's date.
    Accepts date or datetime; returns a date.
    """
    if end_date is None:
        end_date = date.today()
    if isinstance(end_date, datetime):
        end_date = end_date.date()
    return end_date


def _unit_label(units: str) -> str:
    """
    Return a short unit label for the y-axis.
    """
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
    Plot cumulative precipitation curves per year.

    Expects a DataFrame (e.g., from rainfall.prepare_cumulative) with columns:
      - 'year' (int)
      - 'doy'  (int)
      - 'cum'  (float)  # cumulative precipitation

    Parameters
    ----------
    units : "standard" for inches, "metric" for mm (axis label only)
    end_date : used for the default title
    highlight_year : emphasize one year's line (defaults to current year)
    station_name : optional prefix for the title
    title : custom plot title; if None a default is constructed
    show : call plt.show() if True
    save_path : if provided, save the figure to this path
    ax : optional Axes to plot on

    Returns
    -------
    (fig, ax)
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

    # One line per year
    for yr, g in df.groupby("year", sort=True):
        lw = 2.6 if yr == highlight_year else 1.4
        alpha = 1.0 if yr == highlight_year else 0.85
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


def plot_cumulative_rain_days(
    df: pd.DataFrame,
    *,
    end_date: Optional[date] = None,
    highlight_year: Optional[int] = None,
    station_name: Optional[str] = None,
    title: Optional[str] = None,
    show: bool = True,
    save_path: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot cumulative *day-counts with rain* per year.

    Expects a DataFrame (e.g., from rainfall.prepare_cumulative_rain_days) with columns:
      - 'year'           (int)
      - 'doy'            (int)
      - 'cum_rain_days'  (int)

    Parameters
    ----------
    end_date : used for the default title
    highlight_year : emphasize one year's line (defaults to current year)
    station_name : optional prefix for the title
    title : custom plot title; if None a default is constructed
    show : call plt.show() if True
    save_path : if provided, save the figure to this path
    ax : optional Axes to plot on

    Returns
    -------
    (fig, ax)
    """
    required = {"year", "doy", "cum_rain_days"}
    if not required.issubset(df.columns):
        raise ValueError("plot_cumulative_rain_days expects df with ['year','doy','cum_rain_days'].")

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    if highlight_year is None:
        highlight_year = date.today().year

    end_lbl = _end_label(end_date).isoformat()

    for yr, g in df.groupby("year", sort=True):
        lw = 2.6 if yr == highlight_year else 1.4
        alpha = 1.0 if yr == highlight_year else 0.85
        z = 3 if yr == highlight_year else 1
        ax.plot(g["doy"], g["cum_rain_days"], label=str(yr), linewidth=lw, alpha=alpha, zorder=z)

    ax.set_xlabel("Day of Year")
    ax.set_ylabel("Cumulative Rain Days")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize=9)

    if title is None:
        base = f"Cumulative Rain Days (Jan 1 to {end_lbl})"
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
