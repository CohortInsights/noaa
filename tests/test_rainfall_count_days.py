import pandas as pd
from datetime import date
import pytest

from rainfall import count_rain_days

def test_count_rain_days_basic_fills_missing():
    # Jan 1: 0.0", Jan 3: 0.2"; Jan 2 missing -> treated as 0.0
    df = pd.DataFrame({
        "date": ["2025-01-01", "2025-01-03"],
        "prcp": [0.0, 0.2],
    })
    out = count_rain_days(df, start_date="2025-01-01", end_date="2025-01-03")
    assert out == {"total_days": 3, "no_rain_days": 2, "rain_days": 1}

def test_count_rain_days_threshold_changes_classification():
    df = pd.DataFrame({
        "date": ["2025-01-01", "2025-01-02", "2025-01-03"],
        "prcp": [0.0, 0.02, 0.0],
    })
    # With threshold 0.01, 0.02 counts as "rain"
    out = count_rain_days(df, "2025-01-01", "2025-01-03", threshold=0.01)
    assert out == {"total_days": 3, "no_rain_days": 2, "rain_days": 1}
    # With threshold 0.05, 0.02 counts as "no rain"
    out2 = count_rain_days(df, "2025-01-01", "2025-01-03", threshold=0.05)
    assert out2 == {"total_days": 3, "no_rain_days": 3, "rain_days": 0}

def test_count_rain_days_respects_window_and_groups_multi_rows():
    # Two rows same day should collapse; window trims extra days
    df = pd.DataFrame({
        "date": ["2025-01-01", "2025-01-01", "2025-01-04"],
        "prcp": [0.1, 0.2, 0.0],  # Jan 1 total 0.3
    })
    out = count_rain_days(df, start_date="2025-01-01", end_date="2025-01-03")
    # Days in window: Jan 1..3 -> total 3 days; Jan1 rain, Jan2 no rain (filled 0), Jan3 no rain
    assert out == {"total_days": 3, "no_rain_days": 2, "rain_days": 1}

def test_count_rain_days_empty_input():
    df = pd.DataFrame(columns=["date", "prcp"])
    out = count_rain_days(df, "2025-01-01", "2025-01-03")
    assert out == {"total_days": 0, "no_rain_days": 0, "rain_days": 0}

def test_count_rain_days_raises_without_required_columns():
    with pytest.raises(ValueError):
        count_rain_days(pd.DataFrame({"date": ["2025-01-01"]}))
    with pytest.raises(ValueError):
        count_rain_days(pd.DataFrame({"prcp": [0.1]}))

