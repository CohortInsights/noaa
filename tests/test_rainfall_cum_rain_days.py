import pandas as pd
from datetime import date
import pytest

from rainfall import prepare_cumulative_rain_days

def test_prepare_cumulative_rain_days_basic_and_fill():
    # 2024 has rain on DOY 1, missing DOY 2, no rain on DOY 3
    # 2025 has no rain on DOY 1, rain on DOY 2, missing DOY 3
    raw = pd.DataFrame({
        "year": [2024, 2024, 2025, 2025],
        "doy":  [1,    3,    1,    2   ],
        "prcp": [0.2,  0.0,  0.0,  0.4 ],
    })
    out = prepare_cumulative_rain_days(raw, end_date=date(2025, 1, 3), threshold=0.0)

    # Expect 3 rows per year (DOY 1..3)
    assert (out.groupby("year")["doy"].count() == 3).all()

    y24 = out[out["year"] == 2024].sort_values("doy")
    # rain_day: [1 (0.2>0), 0 (filled 0), 0 (0.0>0 is False)]
    assert y24["rain_day"].tolist() == [1, 0, 0]
    # cum: [1,1,1]
    assert y24["cum_rain_days"].tolist() == [1, 1, 1]

    y25 = out[out["year"] == 2025].sort_values("doy")
    # rain_day: [0,1,0]
    assert y25["rain_day"].tolist() == [0, 1, 0]
    # cum: [0,1,1]
    assert y25["cum_rain_days"].tolist() == [0, 1, 1]

def test_prepare_cumulative_rain_days_threshold_effect():
    raw = pd.DataFrame({
        "year": [2025, 2025, 2025],
        "doy":  [1,    2,    3   ],
        "prcp": [0.00, 0.02, 0.00],
    })
    # threshold=0.01 => DOY2 counts as rain
    out1 = prepare_cumulative_rain_days(raw, end_date=date(2025, 1, 3), threshold=0.01)
    assert out1[out1["doy"]==2]["rain_day"].iloc[0] == 1
    # threshold=0.05 => DOY2 does NOT count as rain
    out2 = prepare_cumulative_rain_days(raw, end_date=date(2025, 1, 3), threshold=0.05)
    assert out2[out2["doy"]==2]["rain_day"].iloc[0] == 0

def test_prepare_cumulative_rain_days_accepts_date_and_infers_doy():
    raw = pd.DataFrame({
        "year": [2024, 2024],
        "date": ["2024-01-01", "2024-01-03"],
        "prcp": [0.1, 0.0],
    })
    out = prepare_cumulative_rain_days(raw, end_date=date(2024, 1, 3))
    y24 = out[out["year"] == 2024].sort_values("doy")
    assert y24["doy"].tolist() == [1, 2, 3]
    assert y24["rain_day"].tolist() == [1, 0, 0]
    assert y24["cum_rain_days"].tolist() == [1, 1, 1]

def test_prepare_cumulative_rain_days_empty_input_typed_empty():
    out = prepare_cumulative_rain_days(pd.DataFrame(columns=["year","doy","prcp"]), end_date=date(2025,1,3))
    assert list(out.columns) == ["year", "doy", "rain_day", "cum_rain_days"]
    assert out.empty

def test_prepare_cumulative_rain_days_missing_required_columns_raises():
    with pytest.raises(ValueError):
        prepare_cumulative_rain_days(pd.DataFrame({"doy":[1], "prcp":[0.1]}))
    with pytest.raises(ValueError):
        prepare_cumulative_rain_days(pd.DataFrame({"year":[2024], "doy":[1]}))

