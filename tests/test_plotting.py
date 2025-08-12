# tests/test_plotting.py
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless backend for tests

from datetime import date
import pytest

from plotting import plot_cumulative


def test_plot_cumulative_returns_fig_ax_and_plots_two_years(tmp_path):
    df = pd.DataFrame({
        "year": [2024, 2024, 2025, 2025],
        "doy":  [1,    2,    1,    2   ],
        "cum":  [0.5,  0.5,  0.2,  0.5 ],
        # 'prcp' not required by the plotting function
    })
    out_path = tmp_path / "plot.png"
    fig, ax = plot_cumulative(
        df,
        units="standard",
        end_date=date(2025, 1, 2),
        station_name="Test Station",
        show=False,
        save_path=str(out_path),
    )
    # Two lines (one per year)
    assert len(ax.get_lines()) == 2
    # Axis labels and title
    assert ax.get_xlabel() == "Day of Year"
    assert "Cumulative Precipitation (in)" in ax.get_ylabel()
    assert "Test Station" in ax.get_title()
    # File saved
    assert out_path.exists() and out_path.stat().st_size > 0

    # Cleanup
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_plot_cumulative_raises_on_missing_columns():
    df_bad = pd.DataFrame({"year": [2024], "doy": [1]})  # missing 'cum'
    with pytest.raises(ValueError):
        plot_cumulative(df_bad, show=False)

