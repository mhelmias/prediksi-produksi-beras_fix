from __future__ import annotations

import pandas as pd


class HierarchyValidationError(ValueError):
    pass


def validate_complete_regions(
    frame: pd.DataFrame,
    regions: pd.DataFrame,
) -> None:
    required_columns = {"tahun", "provinsi", "kabupaten_kota"}
    missing_columns = required_columns.difference(frame.columns)
    if missing_columns:
        raise HierarchyValidationError(
            f"Kolom identitas tidak lengkap: {sorted(missing_columns)}"
        )

    if frame.duplicated(
        subset=["tahun", "provinsi", "kabupaten_kota"]
    ).any():
        duplicates = frame.loc[
            frame.duplicated(
                subset=["tahun", "provinsi", "kabupaten_kota"],
                keep=False,
            ),
            ["tahun", "provinsi", "kabupaten_kota"],
        ]
        raise HierarchyValidationError(
            "Terdapat baris wilayah duplikat: "
            + duplicates.head(10).to_dict(orient="records").__repr__()
        )

    years = frame["tahun"].dropna().unique().tolist()
    if len(years) != 1:
        raise HierarchyValidationError(
            "Satu unggahan hanya boleh berisi satu tahun prediksi."
        )

    expected = set(
        zip(regions["province"], regions["city"], strict=True)
    )
    actual = set(
        zip(
            frame["provinsi"],
            frame["kabupaten_kota"],
            strict=True,
        )
    )

    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        raise HierarchyValidationError(
            "Daftar child tidak lengkap atau tidak sesuai. "
            f"Kurang={missing[:10]}, tambahan={extra[:10]}."
        )

    counts = (
        frame.groupby("provinsi")["kabupaten_kota"]
        .nunique()
        .to_dict()
    )
    expected_counts = (
        regions.groupby("province")["city"].nunique().to_dict()
    )
    if counts != expected_counts:
        raise HierarchyValidationError(
            "Jumlah child per provinsi tidak sesuai. "
            f"Ditemukan={counts}; wajib={expected_counts}."
        )


def aggregate_provinces(city_predictions: pd.DataFrame) -> pd.DataFrame:
    if city_predictions.empty:
        return pd.DataFrame(
            columns=[
                "year",
                "province",
                "prediction_ton",
                "actual_ton",
                "child_count",
            ]
        )

    city_predictions = city_predictions.copy()
    if "actual_ton" not in city_predictions.columns:
        city_predictions["actual_ton"] = pd.NA

    grouped = (
        city_predictions.groupby(
            ["year", "province"],
            as_index=False,
            dropna=False,
        )
        .agg(
            prediction_ton=("prediction_ton", "sum"),
            actual_ton=("actual_ton", "sum"),
            child_count=("city", "nunique"),
        )
        .sort_values(["year", "province"])
    )

    # Jika semua aktual pada kelompok kosong, hasil sum pandas menjadi 0.
    # Kembalikan ke NaN agar tidak disalahartikan sebagai aktual nol.
    actual_availability = (
        city_predictions.groupby(["year", "province"])["actual_ton"]
        .apply(lambda series: series.notna().any())
        .reset_index(name="has_actual")
    )
    grouped = grouped.merge(
        actual_availability,
        on=["year", "province"],
        how="left",
    )
    grouped.loc[~grouped["has_actual"], "actual_ton"] = pd.NA
    return grouped.drop(columns=["has_actual"])
