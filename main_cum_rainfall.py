#!/usr/bin/env python3
"""
Main entry point for cumulative rainfall plotting.

Examples:
  poetry run python main.py
  poetry run python main.py --save plot.png --no-show
  poetry run python main.py --years 2020:2025
  poetry run python main.py --years '[2020, 2024, 2025]'
  poetry run python main.py --station GHCND:USW00014837
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, date
from typing import Iterable, List, Optional
import sys

import pandas as pd

import config as cfg
from noaa_api import find_nearby_station, fetch_precip_for_years
from rainfall import prepare_cumulative
from plotting import plot_cumulative


def _get_cfg(name: str, default):
    return getattr(cfg, name, default)


def _parse_years(spec: Optional[str], fallback: Iterable[int]) -> List[int]:
    """
    Accepts:
      - Range:  '2018:2025'
      - CSV:    '2018,2019,2021'
      - List:   '[2018, 2020, 2024]'  (must be quoted in shell)
    """
    if not spec:
        return list(fallback)
    s = spec.strip()
    if s.startswith("[") and s.endswith("]"):
        try:
            seq = ast.literal_eval(s)
            years = [int(x) for x in seq]
            if not years:
                raise ValueError
            return years
        except Exception:
            raise ValueError("Invalid list for --years; e.g., '[2022, 2025, 2027]'")
    if ":" in s:
        a, b = s.split(":", 1)
        start, end = int(a), int(b)
        if end < start:
            raise ValueError("years range end must be >= start")
        return list(range(start, end + 1))
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _parse_date(s: Optional[str]) -> Optional[date]:
    if not s:
        return None
    return datetime.strptime(s, "%Y-%m-%d").date()


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="NOAA cumulative rainfall plotter")
    parser.add_argument("--station", help="Explicit station ID (e.g., GHCND:USW00014837)")
    parser.add_argument("--lat", type=float, help="Latitude (overrides config)")
    parser.add_argument("--lon", type=float, help="Longitude (overrides config)")
    parser.add_argument("--max-nearby", type=int, help="Nearby stations to consider (for auto-pick)")
    parser.add_argument("--units", choices=["standard", "metric"], help="Units for NOAA API + axis label")
    parser.add_argument("--years", help="Years (e.g., '2020:2025', '2020,2024,2025', or '[2020, 2024, 2025]')")
    parser.add_argument("--end-date", help="Clip to this date (YYYY-MM-DD); default: today")
    parser.add_argument("--save", help="Save figure to this path (png/pdf/svg, etc.)")
    parser.add_argument("--no-show", action="store_true", help="Do not display the plot window")

    args = parser.parse_args(argv)

    # Config & defaults
    token = _get_cfg("TOKEN", "")
    if not token or token == "YOUR_NOAA_TOKEN":
        print("ERROR: Set TOKEN in config.py (NOAA CDO API token).", file=sys.stderr)
        return 2

    lat = args.lat if args.lat is not None else _get_cfg("LAT", 42.9908)
    lon = args.lon if args.lon is not None else _get_cfg("LON", -89.5332)
    max_nearby = args.max_nearby if args.max_nearby is not None else _get_cfg("MAX_NEARBY", 12)
    units = args.units if args.units else _get_cfg("UNITS", "standard")
    years_cfg = _get_cfg("YEARS", range(2020, datetime.now().year + 1))
    years = _parse_years(args.years, years_cfg)
    end_date = _parse_date(args.end_date)

    latest_year = max(years)

    # Choose station (command-line overrides auto-pick)
    if args.station:
        station_id = args.station
        print(f"Using station (explicit): {station_id}")
    else:
        try:
            station_id = find_nearby_station(
                token, lat, lon, year=latest_year, end_date=end_date, units=units, max_nearby=max_nearby
            )
        except Exception as e:
            print(f"ERROR: Nearby station search failed: {e}", file=sys.stderr)
            return 3

        if not station_id:
            print(
                "ERROR: No nearby station has full PRCP coverage up to the requested end date.\n"
                "       Try increasing --max-nearby or specify --station explicitly.",
                file=sys.stderr,
            )
            return 3

        print(f"Using station (auto full-coverage): {station_id}")

    print(f"Years: {years}  Units: {units}  End date: {end_date or date.today()}")

    # Fetch data
    try:
        df = fetch_precip_for_years(token, station_id, years, units)
    except Exception as e:
        print(f"ERROR: Fetch failed: {e}", file=sys.stderr)
        return 4

    if df.empty:
        print("WARNING: No precipitation data returned for the requested years/station.")
        return 0

    # Prepare cumulative
    try:
        prep = prepare_cumulative(df, end_date=end_date)
    except Exception as e:
        print(f"ERROR: Failed to prepare cumulative data: {e}", file=sys.stderr)
        return 5

    # Plot
    try:
        fig, ax = plot_cumulative(
            prep,
            units=units,
            end_date=end_date,
            highlight_year=date.today().year,
            station_name=station_id,  # show ID in title for clarity
            show=not args.no_show,
            save_path=args.save,
        )
    except Exception as e:
        print(f"ERROR: Plotting failed: {e}", file=sys.stderr)
        return 6

    if args.save:
        print(f"Saved figure to: {args.save}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

