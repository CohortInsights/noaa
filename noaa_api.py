import requests
import pandas as pd
from pandas import DataFrame
from math import radians, sin, cos, asin, sqrt
from typing import List, Union, Optional
from datetime import date, datetime

from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BASE = "https://www.ncdc.noaa.gov/cdo-web/api/v2"


def _session_with_retries(
    total: int = 5,
    backoff_factor: float = 0.8,
    status_forcelist: tuple = (429, 500, 502, 503, 504),
) -> requests.Session:
    """
    Create a requests.Session with exponential backoff retries for transient errors.
    Respects Retry-After headers from the server.
    """
    s = requests.Session()
    retry = Retry(
        total=total,
        read=total,
        connect=total,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
        allowed_methods=frozenset({"GET"}),
        respect_retry_after_header=True,
        raise_on_status=False,  # we'll call r.raise_for_status() ourselves
    )
    adapter = HTTPAdapter(max_retries=retry)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s


_SESSION = _session_with_retries()


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Compute the great-circle distance between two latitude/longitude points.

    Uses the haversine formula on a spherical Earth approximation.

    Parameters
    ----------
    lat1, lon1 : float
        Latitude and longitude of the first point in decimal degrees.
    lat2, lon2 : float
        Latitude and longitude of the second point in decimal degrees.

    Returns
    -------
    float
        Distance in kilometers between the two points.

    Notes
    -----
    - Assumes Earth radius of 6371 km.
    - Adequate for station proximity sorting (small to moderate distances).
    """
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 2 * asin(sqrt(a)) * 6371.0  # km


def find_nearby_stations(
    token: str,
    lat: float,
    lon: float,
    max_nearby: int = 4,
    return_details: bool = False,
) -> Union[List[str], DataFrame]:
    """
    Find GHCND station(s) near a latitude/longitude and return either IDs or full metadata.

    - If return_details=False (default): returns a list of station IDs (strings), sorted by distance.
    - If return_details=True: returns a DataFrame of station details (incl. '_dist_km'), sorted by distance.
    """
    headers = {"token": token}
    delta = 0.3  # ~20–35 km box around the point
    params = {
        "datasetid": "GHCND",
        "extent": f"{lat - delta},{lon - delta},{lat + delta},{lon + delta}",
        "limit": 1000,
    }
    r = _SESSION.get(f"{BASE}/stations", headers=headers, params=params, timeout=30)
    r.raise_for_status()
    stations = r.json().get("results", []) or []

    for s in stations:
        s["_dist_km"] = _haversine(lat, lon, s["latitude"], s["longitude"])

    sorted_stations = sorted(stations, key=lambda x: x["_dist_km"])[:max_nearby]
    if return_details:
        return pd.DataFrame(sorted_stations)
    return [s["id"] for s in sorted_stations]


def has_full_prcp_coverage(
    token: str,
    station_id: str,
    year: int,
    *,
    end_date: Optional[date] = None,
    units: str = "standard",
) -> bool:
    """
    Return True if the station has a PRCP record for **every day** from Jan 1 to end_date (inclusive)
    for the given year. If end_date is None:
      - for past years: uses Dec 31 of that year
      - for the current year: uses today's date
    """
    today = date.today()
    if end_date is None:
        end_dt = date(year, 12, 31) if year < today.year else today
    else:
        end_dt = end_date if isinstance(end_date, date) else end_date.date()
        if end_dt.year != year:
            end_dt = date(year, 12, 31) if year < today.year else today

    expected_days = (end_dt - date(year, 1, 1)).days + 1

    headers = {"token": token}
    params = {
        "datasetid": "GHCND",
        "stationid": station_id,
        "datatypeid": "PRCP",
        "startdate": f"{year}-01-01",
        "enddate": end_dt.isoformat(),
        "limit": 1000,  # daily fits in one call
        "units": units,
    }
    r = _SESSION.get(f"{BASE}/data", headers=headers, params=params, timeout=60)
    r.raise_for_status()
    results = r.json().get("results", []) or []
    if not results:
        return False

    seen = set()
    for row in results:
        d = row["date"][:10]  # YYYY-MM-DD
        if d.startswith(f"{year}-"):
            dt = datetime.strptime(d, "%Y-%m-%d").date()
            if dt <= end_dt:
                seen.add(dt.timetuple().tm_yday)

    return len(seen) >= expected_days


def find_nearby_station(
    token: str,
    lat: float,
    lon: float,
    year: int,
    *,
    end_date: Optional[date] = None,
    units: str = "standard",
    max_nearby: int = 12,
) -> Optional[str]:
    """
    Return the **closest** station ID that has **full PRCP coverage** for the given year
    (i.e., a record for every day from Jan 1..end_date). If none qualify among the
    nearby candidates, return None.
    """
    details = find_nearby_stations(token, lat, lon, max_nearby=max_nearby, return_details=True)
    if isinstance(details, pd.DataFrame) and not details.empty:
        for _, row in details.iterrows():
            sid = row["id"]
            try:
                if has_full_prcp_coverage(token, sid, year, end_date=end_date, units=units):
                    return sid
            except Exception:
                # Skip stations that error/rate-limit, continue to next closest
                continue
    return None


def fetch_precip_for_years(
    token: str,
    station_id: str,
    years: list[int],
    units: str,
) -> DataFrame:
    """
    Fetch daily precipitation (PRCP) for a station across multiple years.

    Returns
    -------
    pandas.DataFrame
        Columns: ["date", "year", "prcp", "doy"] where:
            - date : pandas.Timestamp (UTC date)
            - year : int (calendar year)
            - prcp : float (precip amount in requested units)
            - doy  : int (day of year, 1..366)
    """
    headers = {"token": token}
    all_rows = []

    for year in years:
        params = {
            "datasetid": "GHCND",
            "stationid": station_id,
            "datatypeid": "PRCP",
            "startdate": f"{year}-01-01",
            "enddate": f"{year}-12-31",
            "limit": 1000,  # daily fits in one call
            "units": units,
        }
        r = _SESSION.get(f"{BASE}/data", headers=headers, params=params, timeout=60)
        r.raise_for_status()
        for row in r.json().get("results", []) or []:
            all_rows.append(
                {
                    "date": row["date"][:10],
                    "year": year,
                    "prcp": row.get("value", 0.0),
                }
            )

    df = pd.DataFrame(all_rows)
    if df.empty:
        # Return a consistent, typed empty frame
        return pd.DataFrame(columns=["date", "year", "prcp", "doy"])

    df["date"] = pd.to_datetime(df["date"], utc=True)
    df["doy"] = df["date"].dt.dayofyear
    return df
