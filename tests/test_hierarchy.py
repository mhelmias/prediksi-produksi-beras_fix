from pathlib import Path

import pandas as pd

from hierarchy import aggregate_provinces, validate_complete_regions


BASE = Path(__file__).resolve().parents[1]


def test_complete_regions():
    regions = pd.read_csv(BASE / "data" / "regions.csv")
    template = pd.read_csv(
        BASE / "data" / "predictor_template_2026.csv"
    )
    validate_complete_regions(template, regions)
    assert len(template) == 119


def test_bottom_up_sum():
    frame = pd.DataFrame(
        {
            "year": [2026, 2026],
            "province": ["Contoh", "Contoh"],
            "city": ["A", "B"],
            "prediction_ton": [10.0, 15.0],
        }
    )
    result = aggregate_provinces(frame)
    assert result.loc[0, "prediction_ton"] == 25.0
    assert result.loc[0, "child_count"] == 2
