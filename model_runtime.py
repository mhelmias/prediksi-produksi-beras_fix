from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from pathlib import Path


# Nama kolom yang diterima dari template admin.
RAW_REQUIRED_COLUMNS = [
    "tahun",
    "provinsi",
    "kabupaten_kota",
    "luas panen (ha)",
    "suhu min (oc)",
    "suhu rata-rata (oc)",
    "suhu maks (oc)",
    "kelembapan (%)",
    "curah hujan (mm)",
    "jumlah hari hujan (hari)",
    "kecepatan angin (m/s)",
    "tekanan udara (mb)",
    "radiasi_matahari_kwh_m2_hari",
    "annual_soi_bom",
    "annual_dmi",
    "soi_bom_label",
]


class ArtifactNotReadyError(RuntimeError):
    """Artefak model/preprocessor tidak lengkap atau tidak sesuai."""


class PredictorValidationError(ValueError):
    """Data prediktor mentah tidak sesuai artefak preprocessing."""


@dataclass
class ArtifactStatus:
    model_loaded: bool
    preprocessor_found: bool
    target_inverse_ready: bool
    feature_preprocessing_ready: bool
    n_model_features: int
    scenario_name: str | None
    dataset_mode: str | None
    message: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ModelRuntime:
    """
    Runtime deployment untuk alur:

    prediktor mentah
      -> capping fitur memakai batas Train
      -> Yeo-Johnson fitur
      -> MinMaxScaler fitur
      -> OneHotEncoder
      -> susun 141 fitur model
      -> XGBoost
      -> inverse target scaler
      -> inverse target Yeo-Johnson
      -> clip prediksi negatif menjadi nol
    """

    REQUIRED_PREPROCESSOR_KEYS = {
        "onehot_encoder",
        "scaler",
        "categorical_cols",
        "numeric_cols",
        "feature_transformers",
        "yeojohnson_feature_cols",
        "target_scaler",
        "target_transformer",
    }

    def __init__(
        self,
        model_path: str | Path,
        preprocessor_path: str | Path,
    ) -> None:
        self.model_path = Path(model_path)
        self.preprocessor_path = Path(preprocessor_path)
        if not self.model_path.exists():
            raise ArtifactNotReadyError(
                f"Model tidak ditemukan: {self.model_path}"
            )

        self.model = XGBRegressor()
        self.model.load_model(self.model_path)
        booster = self.model.get_booster()
        self.feature_names = list(booster.feature_names or [])
        if not self.feature_names:
            raise ArtifactNotReadyError(
                "Model XGBoost tidak menyimpan nama fitur."
            )

        self.preprocessor: dict[str, Any] | None = None
        self.onehot_encoder: Any | None = None
        self.feature_scaler: Any | None = None
        self.feature_transformers: dict[str, Any] = {}
        self.target_scaler: Any | None = None
        self.target_transformer: Any | None = None
        self.numeric_cols: list[str] = []
        self.categorical_cols: list[str] = []
        self.yeojohnson_feature_cols: list[str] = []
        self.capping_bounds: pd.DataFrame = pd.DataFrame()
        self.scenario_name: str | None = None
        self.dataset_mode: str | None = None
        self._preprocessor_error: str | None = None
        self._load_preprocessor()

    def _load_preprocessor(self) -> None:
        if not self.preprocessor_path.exists():
            self._preprocessor_error = (
                "preprocessor.joblib belum tersedia pada artifacts/."
            )
            return

        try:
            source = joblib.load(self.preprocessor_path)
            if not isinstance(source, dict):
                raise TypeError(
                    "Preprocessor deployment wajib berupa dictionary."
                )

            missing = sorted(
                self.REQUIRED_PREPROCESSOR_KEYS.difference(source.keys())
            )
            if missing:
                raise KeyError(
                    "Key preprocessor tidak lengkap: " + ", ".join(missing)
                )

            self.preprocessor = source
            self.onehot_encoder = source["onehot_encoder"]
            self.feature_scaler = source["scaler"]
            self.feature_transformers = dict(
                source["feature_transformers"]
            )
            self.target_scaler = source["target_scaler"]
            self.target_transformer = source["target_transformer"]
            self.numeric_cols = list(source["numeric_cols"])
            self.categorical_cols = list(source["categorical_cols"])
            self.yeojohnson_feature_cols = list(
                source["yeojohnson_feature_cols"]
            )
            self.capping_bounds = source.get(
                "capping_bounds", pd.DataFrame()
            )
            self.scenario_name = source.get("scenario_name")
            self.dataset_mode = source.get("dataset_mode")

            self._validate_artifact_compatibility()
        except Exception as exc:
            self._preprocessor_error = (
                f"Gagal memuat preprocessor deployment: {exc}"
            )

    def _validate_artifact_compatibility(self) -> None:
        if self.scenario_name != "01_skenario_target_tanpa_capping":
            raise ArtifactNotReadyError(
                "Preprocessor bukan skenario target tanpa capping: "
                f"{self.scenario_name!r}."
            )
        if self.dataset_mode != "kabkota":
            raise ArtifactNotReadyError(
                f"Preprocessor bukan dataset kabupaten/kota: {self.dataset_mode!r}."
            )

        expected_numeric = list(
            getattr(self.feature_scaler, "feature_names_in_", [])
        )
        if expected_numeric and expected_numeric != self.numeric_cols:
            raise ArtifactNotReadyError(
                "Urutan numeric_cols berbeda dari scaler fitur."
            )

        encoded_names = list(
            self.onehot_encoder.get_feature_names_out(
                self.categorical_cols
            )
        )
        preprocessor_names = self.numeric_cols + encoded_names
        model_equivalent_names = [
            "tahun.1" if name == "tahun" else name
            for name in preprocessor_names
        ]
        if model_equivalent_names != self.feature_names:
            mismatch = [
                (i, left, right)
                for i, (left, right) in enumerate(
                    zip(
                        model_equivalent_names,
                        self.feature_names,
                        strict=False,
                    )
                )
                if left != right
            ]
            raise ArtifactNotReadyError(
                "Urutan/nama fitur preprocessor tidak cocok dengan model. "
                f"Contoh mismatch: {mismatch[:5]}."
            )

        if len(self.feature_names) != 141:
            raise ArtifactNotReadyError(
                "Model deployment diharapkan memiliki 141 fitur, tetapi "
                f"ditemukan {len(self.feature_names)}."
            )

    def status(self) -> ArtifactStatus:
        target_ready = (
            self.target_scaler is not None
            and self.target_transformer is not None
        )
        feature_ready = (
            self.onehot_encoder is not None
            and self.feature_scaler is not None
            and bool(self.feature_transformers)
            and not self._preprocessor_error
        )

        if self._preprocessor_error:
            message = self._preprocessor_error
        elif target_ready and feature_ready:
            message = (
                "Model, preprocessing fitur mentah, dan inverse target siap."
            )
        else:
            message = "Artefak deployment belum lengkap."

        return ArtifactStatus(
            model_loaded=True,
            preprocessor_found=self.preprocessor_path.exists(),
            target_inverse_ready=target_ready,
            feature_preprocessing_ready=feature_ready,
            n_model_features=len(self.feature_names),
            scenario_name=self.scenario_name,
            dataset_mode=self.dataset_mode,
            message=message,
        )

    @staticmethod
    def _canonicalize_raw_columns(frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.copy()
        aliases = {
            "Kabupaten/kota": "kabupaten_kota",
            "kabupaten/kota": "kabupaten_kota",
            "kabupaten kota": "kabupaten_kota",
        }
        for old_name, new_name in aliases.items():
            if old_name in result.columns and new_name not in result.columns:
                result = result.rename(columns={old_name: new_name})
        return result

    def _validate_known_categories(self, raw: pd.DataFrame) -> None:
        encoder_frame = raw.rename(
            columns={"kabupaten_kota": "Kabupaten/kota"}
        )
        for col, known_values in zip(
            self.categorical_cols,
            self.onehot_encoder.categories_,
            strict=True,
        ):
            actual_values = set(
                encoder_frame[col].astype(str).str.strip().unique()
            )
            known = set(map(str, known_values))
            unknown = sorted(actual_values.difference(known))
            if unknown:
                raise PredictorValidationError(
                    f"Kategori tidak dikenal pada {col}: {unknown[:10]}."
                )

    def _apply_feature_capping(self, numeric: pd.DataFrame) -> pd.DataFrame:
        result = numeric.copy()
        if self.capping_bounds.empty:
            return result

        required_bound_columns = {
            "kolom",
            "lower_bound_train",
            "upper_bound_train",
        }
        if not required_bound_columns.issubset(
            self.capping_bounds.columns
        ):
            raise ArtifactNotReadyError(
                "Format capping_bounds pada preprocessor tidak valid."
            )

        bounds = self.capping_bounds.set_index("kolom")
        for col in self.yeojohnson_feature_cols:
            if col not in bounds.index:
                continue
            lower = float(bounds.loc[col, "lower_bound_train"])
            upper = float(bounds.loc[col, "upper_bound_train"])
            result[col] = result[col].clip(lower=lower, upper=upper)
        return result

    def _transform_raw_predictors(
        self,
        frame: pd.DataFrame,
    ) -> pd.DataFrame:
        if self._preprocessor_error:
            raise ArtifactNotReadyError(self._preprocessor_error)

        raw = self._canonicalize_raw_columns(frame)
        missing = [
            col for col in RAW_REQUIRED_COLUMNS if col not in raw.columns
        ]
        if missing:
            raise PredictorValidationError(
                "Kolom prediktor mentah belum lengkap: "
                + ", ".join(missing)
            )

        self._validate_known_categories(raw)

        numeric = raw[self.numeric_cols].copy()
        for col in self.numeric_cols:
            numeric[col] = pd.to_numeric(numeric[col], errors="coerce")
        if numeric.isna().any().any():
            bad = numeric.columns[numeric.isna().any()].tolist()
            raise PredictorValidationError(
                "Nilai numerik kosong/tidak valid pada kolom: "
                + ", ".join(bad)
            )
        if not np.isfinite(numeric.to_numpy(dtype=float)).all():
            raise PredictorValidationError(
                "Prediktor numerik mengandung infinity."
            )

        # Pada skenario tanpa capping, yang tidak dicapping adalah TARGET.
        # Capping fitur tetap mengikuti batas Train yang tersimpan.
        numeric = self._apply_feature_capping(numeric)

        transformed_numeric = numeric.copy()
        for col in self.yeojohnson_feature_cols:
            transformer = self.feature_transformers.get(col)
            if transformer is None:
                raise ArtifactNotReadyError(
                    f"PowerTransformer fitur tidak ditemukan untuk {col}."
                )
            transformed_numeric[col] = transformer.transform(
                numeric[[col]].to_numpy(dtype=float)
            ).reshape(-1)

        scaled_values = self.feature_scaler.transform(
            transformed_numeric[self.numeric_cols]
        )
        scaled_numeric = pd.DataFrame(
            scaled_values,
            columns=self.numeric_cols,
            index=raw.index,
        )

        encoder_input = raw.rename(
            columns={"kabupaten_kota": "Kabupaten/kota"}
        )[self.categorical_cols].copy()
        encoded_values = self.onehot_encoder.transform(encoder_input)
        if hasattr(encoded_values, "toarray"):
            encoded_values = encoded_values.toarray()
        encoded_names = list(
            self.onehot_encoder.get_feature_names_out(
                self.categorical_cols
            )
        )
        encoded = pd.DataFrame(
            np.asarray(encoded_values, dtype=float),
            columns=encoded_names,
            index=raw.index,
        )

        model_input = pd.concat([scaled_numeric, encoded], axis=1)
        model_input = model_input.rename(columns={"tahun": "tahun.1"})

        missing_model = [
            col for col in self.feature_names if col not in model_input.columns
        ]
        extra_model = [
            col for col in model_input.columns if col not in self.feature_names
        ]
        if missing_model or extra_model:
            raise ArtifactNotReadyError(
                "Hasil preprocessing tidak cocok dengan model. "
                f"Missing={missing_model[:10]}, extra={extra_model[:10]}."
            )

        return model_input[self.feature_names].astype(float)

    def _prepare_model_ready(
        self,
        frame: pd.DataFrame,
    ) -> pd.DataFrame:
        # Tetap mendukung CSV 141 fitur model-ready untuk audit/developer.
        if set(self.feature_names).issubset(frame.columns):
            ready = frame[self.feature_names].apply(
                pd.to_numeric, errors="coerce"
            )
            if ready.isna().any().any():
                raise PredictorValidationError(
                    "Input model-ready mengandung nilai kosong/non-numerik."
                )
            return ready.astype(float)

        return self._transform_raw_predictors(frame)

    def _inverse_target(self, predicted_y_model: np.ndarray) -> np.ndarray:
        if (
            self.target_scaler is None
            or self.target_transformer is None
        ):
            raise ArtifactNotReadyError(
                "target_scaler atau target_transformer tidak tersedia."
            )

        values = np.asarray(predicted_y_model, dtype=float).reshape(-1, 1)
        after_minmax = self.target_scaler.inverse_transform(values)
        production_ton = self.target_transformer.inverse_transform(
            after_minmax
        ).reshape(-1)
        if not np.isfinite(production_ton).all():
            raise ArtifactNotReadyError(
                "Hasil inverse target mengandung NaN/infinity."
            )

        # Sama dengan program training:
        # CLIP_NEGATIVE_PREDICTIONS_TO_ZERO = True.
        return np.maximum(production_ton, 0.0)

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        raw = self._canonicalize_raw_columns(frame)
        required_identity = {"tahun", "provinsi", "kabupaten_kota"}
        missing_identity = required_identity.difference(raw.columns)
        if missing_identity:
            raise PredictorValidationError(
                "Kolom identitas prediksi belum lengkap: "
                + ", ".join(sorted(missing_identity))
            )

        model_input = self._prepare_model_ready(raw)
        predicted_y_model = self.model.predict(model_input)
        predicted_ton = self._inverse_target(predicted_y_model)

        return pd.DataFrame(
            {
                "year": raw["tahun"].astype(int).to_numpy(),
                "province": raw["provinsi"].astype(str).to_numpy(),
                "city": raw["kabupaten_kota"].astype(str).to_numpy(),
                "prediction_y_model": np.asarray(
                    predicted_y_model, dtype=float
                ),
                "prediction_ton": predicted_ton,
            }
        )
