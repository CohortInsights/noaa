import pytest
from datetime import date
import pandas as pd

from noaa_api import has_full_prcp_coverage, find_nearby_station

TOKEN = "fake-token"


@pytest.fixture
def station_id():
    return "GHCND:FAKESTATION"


def make_results_for_year(year: int, days: int, start_doy: int = 1):
    """Return a NOAA-like results list with given number of sequential days."""
    base_date = date(year, 1, 1)
    results = []
    for i in range(days):
        d = date.fromordinal(base_date.toordinal() + (start_doy - 1) + i)
        results.append({"date": d.isoformat() + "T00:00:00", "value": 1.0})
    return {"results": results}


def test_has_full_prcp_coverage_true(requests_mock, station_id):
    # 2024 full year (366 days, leap year)
    year = 2024
    expected_days = (date(year, 12, 31) - date(year, 1, 1)).days + 1
    mock_data = make_results_for_year(year, expected_days)
    requests_mock.get(
        "https://www.ncdc.noaa.gov/cdo-web/api/v2/data",
        json=mock_data,
    )

    assert has_full_prcp_coverage(TOKEN, station_id, year, units="standard") is True


def test_has_full_prcp_coverage_false(requests_mock, station_id):
    # Missing 10 days
    year = 2024
    expected_days = (date(year, 12, 31) - date(year, 1, 1)).days + 1
    mock_data = make_results_for_year(year, expected_days - 10)
    requests_mock.get(
        "https://www.ncdc.noaa.gov/cdo-web/api/v2/data",
        json=mock_data,
    )

    assert has_full_prcp_coverage(TOKEN, station_id, year, units="standard") is False


def test_find_nearby_station_returns_first_with_full_coverage(requests_mock):
    year = 2025
    # Mock stations list
    stations_json = {
        "results": [
            {"id": "STATION1", "latitude": 43.0, "longitude": -89.5},
            {"id": "STATION2", "latitude": 43.1, "longitude": -89.4},
        ]
    }
    requests_mock.get(
        "https://www.ncdc.noaa.gov/cdo-web/api/v2/stations",
        json=stations_json,
    )

    # First station: missing data
    requests_mock.get(
        "https://www.ncdc.noaa.gov/cdo-web/api/v2/data",
        [
            {"json": make_results_for_year(year, 100)},  # station 1
            {"json": make_results_for_year(year, (date.today() - date(year, 1, 1)).days + 1)},  # station 2
        ],
    )

    result = find_nearby_station(TOKEN, 43.05, -89.5, year)
    assert result == "STATION2"


def test_find_nearby_station_none_if_no_full_coverage(requests_mock):
    year = 2025
    # Mock stations list
    stations_json = {
        "results": [
            {"id": "STATION1", "latitude": 43.0, "longitude": -89.5},
            {"id": "STATION2", "latitude": 43.1, "longitude": -89.4},
        ]
    }
    requests_mock.get(
        "https://www.ncdc.noaa.gov/cdo-web/api/v2/stations",
        json=stations_json,
    )

    # Both stations: missing data
    requests_mock.get(
        "https://www.ncdc.noaa.gov/cdo-web/api/v2/data",
        [
            {"json": make_results_for_year(year, 50)},  # station 1
            {"json": make_results_for_year(year, 100)}, # station 2
        ],
    )

    result = find_nearby_station(TOKEN, 43.05, -89.5, year)
    assert result is None

