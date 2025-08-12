import pandas as pd
import pytest
from datetime import date
from rainfall import prepare_cumulative, _end_doy


def test_prepare_cumulative_fills_missing_days_and_cumsum():
    raw = pd.DataFrame({
        "year": [2024, 2024, 2025, 2025],
        "doy":  [1,    3,    1,    2   ],
        "prcp": [1.0,  2.0,  0.5,  1.0 ],
    })
    out = prepare_cumulative(raw, end_date=date(2025, 1, 3))

    assert set(out["year"].unique()) == {2024, 2025}
    # expect 3 rows (DOY 1..3) per year
    assert (out.groupby("year")["doy"].count() == 3).all()

    y24 = out[out["year"] == 2024].sort_values("doy")
    assert y24["prcp"].tolist() == [1.0, 0.0, 2.0]
    assert y24["cum"].tolist() == [1.0, 1.0, 3.0]

    y25 = out[out["year"] == 2025].sort_values("doy")
    assert y25["prcp"].tolist() == [0.5, 1.0, 0.0]
    assert y25["cum"].tolist() == [0.5, 1.5, 1.5]


def test_prepare_cumulative_accepts_date_and_inferrs_doy():
    raw = pd.DataFrame({
        "year": [2024, 2024],
        "date": ["2024-01-01", "2024-01-03"],
        "prcp": [1.0, 2.0],
    })
    out = prepare_cumulative(raw, end_date=date(2024, 1, 3))
    y24 = out[out["year"] == 2024].sort_values("doy")
    assert y24["doy"].tolist() == [1, 2, 3]
    assert y24["prcp"].tolist() == [1.0, 0.0, 2.0]
    assert y24["cum"].tolist() == [1.0, 1.0, 3.0]


def test_prepare_cumulative_empty_input_returns_typed_empty():
    out = prepare_cumulative(pd.DataFrame(columns=["year", "doy", "prcp"]), end_date=date(2025, 1, 3))
    assert list(out.columns) == ["year", "doy", "prcp", "cum"]
    assert out.empty


def test_prepare_cumulative_missing_required_columns_raises():
    with pytest.raises(ValueError):
        prepare_cumulative(pd.DataFrame({"doy": [1, 2]}))
    with pytest.raises(ValueError):
        prepare_cumulative(pd.DataFrame({"year": [2024], "doy": [1]}))


def test__end_doy_handles_datetime_and_default_today():
    # Controlled date
    assert _end_doy(date(2024, 1, 3)) == 3

