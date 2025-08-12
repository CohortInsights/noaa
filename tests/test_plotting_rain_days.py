import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless backend for tests

from datetime import date
import pytest
from plotting import plot_cumulative_rain_days


def test_plot_cumulative_rain_days_basic(tmp_path):
    df = pd.DataFrame({
        "year": [2024, 2024, 2025, 2025],
        "doy":  [1,    2,    1,    2   ],
        "cum_rain_days": [1, 1, 0, 1],
    })
    out_path = tmp_path / "rain_days.png"
    fig, ax = plot_cumulative_rain_days(
        df,
        end_date=date(2025, 1, 2),
        station_name="Test Station",
        show=False,
        save_path=str(out_path),
    )
    assert len(ax.get_lines()) == 2
    assert ax.get_xlabel() == "Day of Year"
    assert ax.get_ylabel() == "Cumulative Rain Days"
    assert "Test Station" in ax.get_title()
    assert out_path.exists() and out_path.stat().st_size > 0

    import matplotlib.pyplot as plt
    plt.close(fig)


def test_plot_cumulative_rain_days_missing_cols_raises():
    bad = pd.DataFrame({"year": [2025], "doy": [1]})
    with pytest.raises(ValueError):
        plot_cumulative_rain_days(bad, show=False)

