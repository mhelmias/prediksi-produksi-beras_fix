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

st.set_page_config(
    page_title="Prediksi Produksi Beras",
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
    st.title("🌾 Prediksi Produksi Beras")
    st.caption(
        "XGBoost kabupaten/kota dan hierarchical forecasting "
        "Bottom-Up hingga tingkat provinsi."
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
            "Belum ada hasil prediksi. Login sebagai admin untuk "
            "memasukkan hasil awal atau menjalankan prediksi."
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

        c1, c2, c3 = st.columns(3)
        c1.metric(
            "Total enam provinsi",
            format_ton(frame["prediction_ton"].sum()),
        )
        c2.metric("Jumlah provinsi", int(len(frame)))
        c3.metric(
            "Jumlah kabupaten/kota",
            int(frame["child_count"].sum()),
        )

        chart = px.bar(
            frame.sort_values("prediction_ton"),
            x="prediction_ton",
            y="province",
            orientation="h",
            labels={
                "prediction_ton": "Prediksi produksi (ton)",
                "province": "Provinsi",
            },
            title=f"Prediksi Produksi Provinsi Tahun {year}",
        )
        st.plotly_chart(chart, use_container_width=True)

        table = frame[
            [
                "province",
                "prediction_ton",
                "actual_ton",
                "child_count",
            ]
        ].rename(
            columns={
                "province": "Provinsi",
                "prediction_ton": "Prediksi (ton)",
                "actual_ton": "Aktual (ton)",
                "child_count": "Jumlah child",
            }
        )
        st.dataframe(
            table,
            use_container_width=True,
            hide_index=True,
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

    province_row = province_frame.iloc[0]
    c1, c2, c3 = st.columns(3)
    c1.metric(
        f"Prediksi {province}",
        format_ton(province_row["prediction_ton"]),
    )
    c2.metric(
        "Kabupaten/kota teragregasi",
        int(province_row["child_count"]),
    )
    c3.metric(
        "Selisih koherensi",
        format_ton(
            province_row["coherence_difference_ton"]
        ),
    )

    trend = load_province_predictions(None, province)
    if not trend.empty and trend["year"].nunique() > 1:
        line = px.line(
            trend.sort_values("year"),
            x="year",
            y="prediction_ton",
            markers=True,
            labels={
                "year": "Tahun",
                "prediction_ton": "Produksi (ton)",
            },
            title=f"Tren Prediksi Provinsi {province}",
        )
        st.plotly_chart(line, use_container_width=True)

    bar = px.bar(
        cities.sort_values("prediction_ton"),
        x="prediction_ton",
        y="city",
        orientation="h",
        height=max(500, 28 * len(cities)),
        labels={
            "prediction_ton": "Prediksi produksi (ton)",
            "city": "Kabupaten/Kota",
        },
        title=f"Rincian Kabupaten/Kota Tahun {year}",
    )
    st.plotly_chart(bar, use_container_width=True)

    st.dataframe(
        cities[
            [
                "city",
                "prediction_ton",
                "actual_ton",
                "source",
            ]
        ]
        .rename(
            columns={
                "city": "Kabupaten/Kota",
                "prediction_ton": "Prediksi (ton)",
                "actual_ton": "Aktual (ton)",
                "source": "Sumber",
            }
        )
        .sort_values("Prediksi (ton)", ascending=False),
        use_container_width=True,
        hide_index=True,
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


def seed_historical(
    store: SupabaseStore,
) -> None:
    seed_path = DATA_DIR / "seed_predictions_2022_2025.csv"
    frame = pd.read_csv(seed_path)
    city = frame[
        [
            "year",
            "province",
            "city",
            "prediction_ton",
            "actual_ton",
        ]
    ].copy()
    province = aggregate_provinces(city)
    store.upsert_predictions(
        batch_id=None,
        city_frame=city,
        province_frame=province,
        source="hasil_model_2022_2025",
    )
    clear_prediction_cache()


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

    if st.button("Masukkan hasil historis 2022–2025"):
        try:
            seed_historical(store)
            st.success("Data historis berhasil disimpan.")
        except Exception as exc:
            st.error(f"Seed gagal: {exc}")

    year = st.number_input(
        "Tahun prediksi",
        min_value=2026,
        max_value=2100,
        value=2026,
        step=1,
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
        st.success(
            "Validasi awal berhasil: 119 wilayah lengkap."
        )
    except Exception as exc:
        st.error(f"CSV belum valid: {exc}")
        return

    if st.button(
        "Simpan prediktor dan jalankan prediksi",
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
                    "Jumlah child hasil prediksi tidak sesuai: "
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

            st.success(
                "Prediksi 119 kabupaten/kota dan agregasi "
                "enam provinsi berhasil."
            )
            st.dataframe(
                province_predictions,
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
            st.error(f"Prediksi gagal: {exc}")
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

try:
    if page == "Dashboard":
        dashboard_page()
    else:
        admin_page()
except Exception as exc:
    st.error(f"Aplikasi mengalami kesalahan: {exc}")
