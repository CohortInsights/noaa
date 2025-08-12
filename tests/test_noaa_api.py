import json
import datetime as dt
import pandas as pd
import pytest
import requests

from noaa_api import find_nearby_stations, fetch_precip_for_years, BASE

# ---------- Helpers for mocking ----------
def _stations_payload():
    # Two stations near (0,0), one exactly at origin
    return {
        "results": [
            {
                "id": "GHCND:STATION_A",
                "name": "Station A",
                "latitude": 0.0,
                "longitude": 0.0,
            },
            {
                "id": "GHCND:STATION_B",
                "name": "Station B",
                "latitude": 0.1,
                "longitude": 0.0,
            },
            {
                "id": "GHCND:STATION_C",
                "name": "Farther Station",
                "latitude": 1.0,
                "longitude": 1.0,
            },
        ]
    }

def _precip_payload(year=2025):
    # Two consecutive days with simple values; API dates are ISO8601 with time
    return {
        "results": [
            {"date": f"{year}-01-01T00:00:00", "value": 1.0},
            {"date": f"{year}-01-02T00:00:00", "value": 2.5},
        ]
    }

# ---------- Tests for find_nearby_stations ----------
def test_find_nearby_stations_returns_sorted_ids(requests_mock):
    token = "dummy"
    lat, lon = 0.0, 0.0
    url = f"{BASE}/stations"

    requests_mock.get(url, json=_stations_payload(), status_code=200)

    ids = find_nearby_stations(token, lat, lon, max_nearby=2, return_details=False)
    # Expect the closest two, sorted by distance: A (0 km), then B (~11 km)
    assert ids == ["GHCND:STATION_A", "GHCND:STATION_B"]

def test_find_nearby_stations_return_details_dataframe(requests_mock):
    token = "dummy"
    lat, lon = 0.0, 0.0
    url = f"{BASE}/stations"

    requests_mock.get(url, json=_stations_payload(), status_code=200)

    df = find_nearby_stations(token, lat, lon, max_nearby=3, return_details=True)
    assert isinstance(df, pd.DataFrame)
    assert set(["id", "name", "latitude", "longitude", "_dist_km"]).issubset(df.columns)
    # Check first row is the closest station
    assert df.iloc[0]["id"] == "GHCND:STATION_A"
    # Distances should be non-decreasing
    assert df["_dist_km"].is_monotonic_increasing or df["_dist_km"].is_monotonic

# ---------- Tests for fetch_precip_for_years ----------
def test_fetch_precip_for_years_basic(requests_mock):
    token = "dummy"
    station_id = "GHCND:STATION_A"
    years = [2025]
    units = "standard"

    url = f"{BASE}/data"
    requests_mock.get(url, json=_precip_payload(2025), status_code=200)

    df = fetch_precip_for_years(token, station_id, years, units)

    assert list(df.columns) == ["date", "year", "prcp", "doy"]
    assert df["year"].nunique() == 1 and df["year"].iloc[0] == 2025
    # DOY should be 1 then 2
    assert df["doy"].tolist() == [1, 2]
    # Types
    assert pd.api.types.is_datetime64_any_dtype(df["date"])
    assert pd.api.types.is_numeric_dtype(df["prcp"])

def test_fetch_precip_for_years_empty_results(requests_mock):
    token = "dummy"
    station_id = "GHCND:STATION_A"
    years = [2024]
    units = "metric"

    url = f"{BASE}/data"
    requests_mock.get(url, json={"results": []}, status_code=200)

    df = fetch_precip_for_years(token, station_id, years, units)
    assert list(df.columns) == ["date", "year", "prcp", "doy"]
    assert df.empty

