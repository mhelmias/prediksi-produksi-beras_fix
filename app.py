from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from artifact_loader import resolve_artifacts
from hierarchy import (
    HierarchyValidationError,
    aggregate_provinces,
    validate_complete_regions,
)
from model_runtime import (
    ArtifactNotReadyError,
    ModelRuntime,
    PredictorValidationError,
    RAW_REQUIRED_COLUMNS,
)
from supabase_store import SupabaseStore


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

# Same-year regression:
# prediktor tahun 2025 digunakan untuk mengestimasi produksi tahun 2025.
DEFAULT_PREDICTION_YEAR = 2025
MIN_PREDICTION_YEAR = 2025
MAX_PREDICTION_YEAR = 2100

# Angka konversi nasional GKG menjadi beras.
# Setiap 1 ton GKG dikonversi menjadi 0,6402 ton beras.
GKG_TO_RICE_RATE = 0.6402

st.set_page_config(
    page_title="Estimasi Produksi Beras",
    page_icon="🌾",
    layout="wide",
)


def secret(name: str, default: str = "") -> str:
    try:
        return str(st.secrets.get(name, default))
    except Exception:
        return default


SUPABASE_URL = secret("SUPABASE_URL")
SUPABASE_KEY = secret("SUPABASE_PUBLISHABLE_KEY")
HF_REPO_ID = secret("HF_MODEL_REPO_ID")
HF_REVISION = secret("HF_MODEL_REVISION", "main")
HF_TOKEN = secret("HF_TOKEN")

if not SUPABASE_URL or not SUPABASE_KEY:
    st.error(
        "SUPABASE_URL dan SUPABASE_PUBLISHABLE_KEY belum diisi "
        "pada Streamlit Secrets."
    )
    st.stop()


def public_store() -> SupabaseStore:
    return SupabaseStore(SUPABASE_URL, SUPABASE_KEY)


def authenticated_store() -> SupabaseStore:
    return SupabaseStore(
        SUPABASE_URL,
        SUPABASE_KEY,
        st.session_state.get("access_token"),
        st.session_state.get("refresh_token"),
    )


@st.cache_resource(show_spinner="Memuat model dan preprocessor...")
def load_runtime(
    repo_id: str,
    revision: str,
    token: str,
) -> tuple[ModelRuntime, str]:
    paths = resolve_artifacts(
        repo_id=repo_id or None,
        revision=revision,
        token=token or None,
    )
    runtime = ModelRuntime(
        model_path=paths.model_path,
        preprocessor_path=paths.preprocessor_path,
    )
    return runtime, paths.source


@st.cache_data(ttl=60)
def load_regions() -> pd.DataFrame:
    return public_store().fetch_regions()


@st.cache_data(ttl=60)
def load_years() -> list[int]:
    return public_store().list_years()


@st.cache_data(ttl=60)
def load_city_predictions(
    year: int | None = None,
    province: str | None = None,
) -> pd.DataFrame:
    return public_store().fetch_city_predictions(year, province)


@st.cache_data(ttl=60)
def load_province_predictions(
    year: int | None = None,
    province: str | None = None,
) -> pd.DataFrame:
    return public_store().fetch_province_predictions(
        year,
        province,
    )


def clear_prediction_cache() -> None:
    load_years.clear()
    load_city_predictions.clear()
    load_province_predictions.clear()


def format_ton(value) -> str:
    if value is None or pd.isna(value):
        return "-"
    return (
        f"{float(value):,.2f} ton"
        .replace(",", "X")
        .replace(".", ",")
        .replace("X", ".")
    )


def add_rice_conversion_columns(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Konversi hasil estimasi GKG menjadi beras sebesar 64,02%."""
    if frame.empty:
        return frame.copy()

    result = frame.copy()
    result["gkg_to_rice_rate"] = GKG_TO_RICE_RATE

    if "prediction_ton" in result.columns:
        result["prediction_gkg_ton"] = pd.to_numeric(
            result["prediction_ton"],
            errors="coerce",
        )
        result["prediction_rice_ton"] = (
            result["prediction_gkg_ton"]
            * GKG_TO_RICE_RATE
        )

    if "actual_ton" in result.columns:
        result["actual_gkg_ton"] = pd.to_numeric(
            result["actual_ton"],
            errors="coerce",
        )
        result["actual_rice_ton"] = (
            result["actual_gkg_ton"]
            * GKG_TO_RICE_RATE
        )

    if "coherence_difference_ton" in result.columns:
        result["coherence_difference_rice_ton"] = (
            pd.to_numeric(
                result["coherence_difference_ton"],
                errors="coerce",
            )
            * GKG_TO_RICE_RATE
        )

    result["conversion_percent"] = GKG_TO_RICE_RATE * 100.0
    return result

def validate_admin_frame(
    frame: pd.DataFrame,
    runtime: ModelRuntime,
    regions: pd.DataFrame,
) -> pd.DataFrame:
    result = frame.copy()
    aliases = {
        "Kabupaten/kota": "kabupaten_kota",
        "kabupaten/kota": "kabupaten_kota",
        "kabupaten kota": "kabupaten_kota",
    }
    for old, new in aliases.items():
        if old in result.columns and new not in result.columns:
            result = result.rename(columns={old: new})

    model_ready = set(runtime.feature_names).issubset(
        result.columns
    )
    missing_raw = [
        column
        for column in RAW_REQUIRED_COLUMNS
        if column not in result.columns
    ]
    if missing_raw and not model_ready:
        raise ValueError(
            "Kolom prediktor belum lengkap: "
            + ", ".join(missing_raw)
        )

    required = (
        runtime.feature_names
        if model_ready
        else RAW_REQUIRED_COLUMNS
    )
    empty_columns = []
    for column in required:
        values = result[column]
        missing = values.isna()
        if values.dtype == object:
            missing = missing | (
                values.astype(str).str.strip() == ""
            )
        if bool(missing.any()):
            empty_columns.append(column)
    if empty_columns:
        raise ValueError(
            "Nilai kosong ditemukan pada kolom: "
            + ", ".join(empty_columns)
        )

    validate_complete_regions(result, regions)
    return result


def dashboard_page() -> None:
    st.title("🌾 Estimasi Produksi Beras")
    st.caption(
        "Model mengestimasi produksi padi dalam bentuk Gabah Kering "
        "Giling (GKG) pada 119 kabupaten/kota yang tersebar di 6 provinsi "
        "di Pulau Jawa. Dashboard mengonversi GKG menjadi beras dengan "
        "angka konversi nasional 64,02%, kemudian menampilkan hasil "
        "hierarchical forecasting Bottom-Up hingga tingkat provinsi."
    )

    years = load_years()
    regions = load_regions()
    if regions.empty:
        st.warning(
            "Tabel regions masih kosong. Jalankan seed_regions.sql."
        )
        return
    if not years:
        st.warning(
            "Belum ada hasil estimasi. Login sebagai admin untuk "
            "memasukkan hasil awal atau menjalankan estimasi."
        )
        return

    left, right = st.columns([1, 2])
    year = left.selectbox(
        "Tahun",
        options=sorted(years, reverse=True),
    )
    province_options = sorted(
        regions["province"].unique().tolist()
    )
    province = right.selectbox(
        "Wilayah",
        options=["Semua Provinsi", *province_options],
    )

    if province == "Semua Provinsi":
        frame = load_province_predictions(year)
        if frame.empty:
            st.info("Data provinsi belum tersedia.")
            return

        frame = add_rice_conversion_columns(frame)

        c1, c2 = st.columns(2)
        c1.metric(
            "Total estimasi beras",
            format_ton(frame["prediction_rice_ton"].sum()),
        )
        c2.metric(
            "Total estimasi GKG",
            format_ton(frame["prediction_gkg_ton"].sum()),
        )

        chart = px.bar(
            frame.sort_values("prediction_rice_ton"),
            x="prediction_rice_ton",
            y="province",
            orientation="h",
            labels={
                "prediction_rice_ton": (
                    "Estimasi beras (ton)"
                ),
                "province": "Provinsi",
            },
            title=f"Estimasi Produksi Beras Provinsi Tahun {year}",
        )
        st.plotly_chart(chart, use_container_width=True)

        table_columns = [
            "province",
            "prediction_rice_ton",
            "prediction_gkg_ton",
            "actual_rice_ton",
            "actual_gkg_ton",
            "conversion_percent",
            "child_count",
        ]
        table = frame[table_columns].rename(
            columns={
                "province": "Provinsi",
                "prediction_rice_ton": "Estimasi Beras (ton)",
                "prediction_gkg_ton": "Estimasi GKG (ton)",
                "actual_rice_ton": "Aktual Beras (ton)",
                "actual_gkg_ton": "Aktual GKG (ton)",
                "conversion_percent": (
                    "Konversi GKG ke Beras (%)"
                ),
                "child_count": "Jumlah child",
            }
        )
        st.dataframe(
            table,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Estimasi Beras (ton)": st.column_config.NumberColumn(
                    format="%.2f"
                ),
                "Estimasi GKG (ton)": st.column_config.NumberColumn(
                    format="%.2f"
                ),
                "Aktual Beras (ton)": st.column_config.NumberColumn(
                    format="%.2f"
                ),
                "Aktual GKG (ton)": st.column_config.NumberColumn(
                    format="%.2f"
                ),
                "Konversi GKG ke Beras (%)": (
                    st.column_config.NumberColumn(format="%.4f")
                ),
            },
        )

        with st.expander("Metode konversi GKG menjadi beras"):
            st.write(
                "Beras dihitung dari estimasi GKG menggunakan:"
            )
            st.latex(
                r"\hat{Y}_{beras}="
                r"\hat{Y}_{GKG}\times 0{,}6402"
            )
            st.write(
                "Nilai k_p adalah rendemen GKG ke beras tingkat "
                "provinsi hasil SKGB 2018."
            )
        return

    cities = load_city_predictions(year, province)
    province_frame = load_province_predictions(
        year,
        province,
    )
    if cities.empty or province_frame.empty:
        st.info("Data wilayah belum tersedia.")
        return

    cities = add_rice_conversion_columns(cities)
    province_frame = add_rice_conversion_columns(province_frame)

    province_row = province_frame.iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        f"Estimasi beras {province}",
        format_ton(province_row["prediction_rice_ton"]),
    )
    c2.metric(
        "Estimasi GKG",
        format_ton(province_row["prediction_gkg_ton"]),
    )
    c3.metric(
        "Kabupaten/kota teragregasi",
        int(province_row["child_count"]),
    )
    c4.metric(
        "Selisih koherensi beras",
        format_ton(
            province_row["coherence_difference_rice_ton"]
        ),
    )

    st.caption(
        "Faktor total konversi GKG menjadi beras untuk konsumsi "
        f"pangan penduduk di {province}: "
        f"{province_row['conversion_percent']:.4f}%."
    )

    trend = load_province_predictions(None, province)
    if not trend.empty and trend["year"].nunique() > 1:
        trend = add_rice_conversion_columns(trend)
        line = px.line(
            trend.sort_values("year"),
            x="year",
            y="prediction_rice_ton",
            markers=True,
            labels={
                "year": "Tahun",
                "prediction_rice_ton": (
                    "Produksi beras (ton)"
                ),
            },
            title=f"Tren Estimasi Beras Provinsi {province}",
        )
        st.plotly_chart(line, use_container_width=True)

    bar = px.bar(
        cities.sort_values("prediction_rice_ton"),
        x="prediction_rice_ton",
        y="city",
        orientation="h",
        height=max(500, 28 * len(cities)),
        labels={
            "prediction_rice_ton": (
                "Estimasi beras (ton)"
            ),
            "city": "Kabupaten/Kota",
        },
        title=f"Rincian Estimasi Beras Kabupaten/Kota Tahun {year}",
    )
    st.plotly_chart(bar, use_container_width=True)

    city_table = (
        cities[
            [
                "city",
                "prediction_rice_ton",
                "prediction_gkg_ton",
                "actual_rice_ton",
                "actual_gkg_ton",
                "source",
            ]
        ]
        .rename(
            columns={
                "city": "Kabupaten/Kota",
                "prediction_rice_ton": "Estimasi Beras (ton)",
                "prediction_gkg_ton": "Estimasi GKG (ton)",
                "actual_rice_ton": "Aktual Beras (ton)",
                "actual_gkg_ton": "Aktual GKG (ton)",
                "source": "Sumber",
            }
        )
        .sort_values("Estimasi Beras (ton)", ascending=False)
    )
    st.dataframe(
        city_table,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Estimasi Beras (ton)": st.column_config.NumberColumn(
                format="%.2f"
            ),
            "Estimasi GKG (ton)": st.column_config.NumberColumn(
                format="%.2f"
            ),
            "Aktual Beras (ton)": st.column_config.NumberColumn(
                format="%.2f"
            ),
            "Aktual GKG (ton)": st.column_config.NumberColumn(
                format="%.2f"
            ),
        },
    )


def login_form() -> None:
    st.subheader("Login admin")
    email = st.text_input("Email")
    password = st.text_input("Kata sandi", type="password")
    if st.button("Masuk", type="primary"):
        try:
            store = public_store()
            session = store.sign_in(email, password)
            authed = SupabaseStore(
                SUPABASE_URL,
                SUPABASE_KEY,
                session.access_token,
                session.refresh_token,
            )
            profile = authed.assert_admin(session.user_id)
            st.session_state.update(
                {
                    "user_id": session.user_id,
                    "user_email": session.email,
                    "access_token": session.access_token,
                    "refresh_token": session.refresh_token,
                    "profile": profile,
                }
            )
            st.rerun()
        except Exception as exc:
            st.error(f"Login gagal: {exc}")



def admin_page() -> None:
    st.title("🔐 Admin data prediktor")

    if "access_token" not in st.session_state:
        login_form()
        return

    try:
        store = authenticated_store()
        store.assert_admin(st.session_state["user_id"])
    except Exception as exc:
        st.session_state.clear()
        st.error(f"Sesi admin tidak valid: {exc}")
        return

    top_left, top_right = st.columns([4, 1])
    top_left.success(
        "Login sebagai "
        + str(st.session_state.get("user_email", "admin"))
    )
    if top_right.button("Keluar"):
        try:
            store.sign_out()
        finally:
            st.session_state.clear()
            st.rerun()

    try:
        runtime, artifact_source = load_runtime(
            HF_REPO_ID,
            HF_REVISION,
            HF_TOKEN,
        )
        status = runtime.status()
        st.caption(f"Artefak: {artifact_source}")
        if not (
            status.target_inverse_ready
            and status.feature_preprocessing_ready
        ):
            st.error(status.message)
            return
    except Exception as exc:
        st.error(f"Model belum siap: {exc}")
        return

    regions = load_regions()
    if regions.empty:
        st.error("Tabel regions kosong.")
        return

    with st.expander("Status model dan hierarki"):
        st.json(status.to_dict())
        st.write(
            "Jumlah wilayah wajib:",
            regions.groupby("province")["city"]
            .nunique()
            .to_dict(),
        )

    year = st.number_input(
        "Tahun estimasi",
        min_value=MIN_PREDICTION_YEAR,
        max_value=MAX_PREDICTION_YEAR,
        value=DEFAULT_PREDICTION_YEAR,
        step=1,
        help=(
            "Same-year regression: data prediktor tahun yang dipilih "
            "digunakan untuk mengestimasi produksi pada tahun yang sama."
        ),
    )
    st.caption(
        f"Data prediktor {int(year)} digunakan untuk mengestimasi GKG "
        f"tahun {int(year)}. Hasil GKG kemudian dikonversi menjadi "
        "beras untuk konsumsi pangan penduduk."
    )

    template = pd.read_csv(
        DATA_DIR / "predictor_template_2026.csv"
    )
    template["tahun"] = int(year)
    st.download_button(
        "Unduh template 119 wilayah",
        data=template.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"template_prediktor_{int(year)}.csv",
        mime="text/csv",
    )

    uploaded = st.file_uploader(
        "Unggah CSV prediktor",
        type=["csv"],
    )
    if uploaded is None:
        return

    try:
        frame = pd.read_csv(uploaded)
        st.caption(f"Jumlah baris: {len(frame)}")
        st.dataframe(
            frame.head(20),
            use_container_width=True,
            hide_index=True,
        )
        validated = validate_admin_frame(
            frame,
            runtime,
            regions,
        )

        csv_years = sorted(
            pd.to_numeric(
                validated["tahun"],
                errors="raise",
            )
            .astype(int)
            .unique()
            .tolist()
        )
        selected_year = int(year)

        if csv_years != [selected_year]:
            raise ValueError(
                "Tahun pada CSV harus sama dengan tahun estimasi yang "
                f"dipilih. Pilihan admin={selected_year}, "
                f"tahun dalam CSV={csv_years}."
            )

        st.success(
            "Validasi awal berhasil: 119 wilayah lengkap dan seluruh "
            f"baris menggunakan prediktor tahun {selected_year}."
        )
    except Exception as exc:
        st.error(f"CSV belum valid: {exc}")
        return

    if st.button(
        "Simpan prediktor dan jalankan estimasi",
        type="primary",
    ):
        batch_id = None
        try:
            batch_id = store.create_batch(
                year=int(validated["tahun"].iloc[0]),
                user_id=st.session_state["user_id"],
                filename=uploaded.name,
                row_count=len(validated),
            )
            store.store_predictors(batch_id, validated)
            store.update_batch(batch_id, "processing")

            city_predictions = runtime.predict(validated)
            province_predictions = aggregate_provinces(
                city_predictions
            )

            expected = (
                regions.groupby("province")["city"]
                .nunique()
                .to_dict()
            )
            actual = dict(
                zip(
                    province_predictions["province"],
                    province_predictions["child_count"],
                )
            )
            if actual != expected:
                raise HierarchyValidationError(
                    "Jumlah child hasil estimasi tidak sesuai: "
                    f"{actual}; wajib={expected}."
                )

            store.upsert_predictions(
                batch_id=batch_id,
                city_frame=city_predictions,
                province_frame=province_predictions,
                source="model",
            )
            store.update_batch(batch_id, "completed")
            clear_prediction_cache()

            prediction_year = int(
                validated["tahun"].iloc[0]
            )
            st.success(
                f"Estimasi produksi tahun {prediction_year} untuk "
                "119 kabupaten/kota dan agregasi Bottom-Up menjadi "
                "enam provinsi berhasil."
            )
            province_display = add_rice_conversion_columns(
                province_predictions
            )
            st.dataframe(
                province_display[
                    [
                        "province",
                        "prediction_rice_ton",
                        "prediction_gkg_ton",
                        "conversion_percent",
                        "child_count",
                    ]
                ].rename(
                    columns={
                        "province": "Provinsi",
                        "prediction_rice_ton": "Estimasi Beras (ton)",
                        "prediction_gkg_ton": "Estimasi GKG (ton)",
                        "conversion_percent": (
                            "Konversi GKG ke Beras (%)"
                        ),
                        "child_count": "Jumlah child",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )
        except (
            ValueError,
            PredictorValidationError,
            ArtifactNotReadyError,
            HierarchyValidationError,
        ) as exc:
            if batch_id:
                store.update_batch(batch_id, "failed", str(exc))
            st.error(f"Estimasi gagal: {exc}")
        except Exception as exc:
            if batch_id:
                store.update_batch(batch_id, "failed", str(exc))
            st.error(f"Kesalahan sistem: {exc}")


page = st.sidebar.radio(
    "Navigasi",
    ["Dashboard", "Admin"],
)
st.sidebar.caption(
    "Hierarki: kabupaten/kota → provinsi"
)
st.sidebar.caption(
    "Metode final: Bottom-Up"
)
st.sidebar.caption(
    "Output model: GKG → beras (× 64,02%)"
)

try:
    if page == "Dashboard":
        dashboard_page()
    else:
        admin_page()
except Exception as exc:
    st.error(f"Aplikasi mengalami kesalahan: {exc}")
