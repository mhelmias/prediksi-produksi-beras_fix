from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from supabase import Client, create_client


@dataclass(frozen=True)
class LoginSession:
    user_id: str
    email: str | None
    access_token: str
    refresh_token: str


class SupabaseStore:
    def __init__(
        self,
        url: str,
        publishable_key: str,
        access_token: str | None = None,
        refresh_token: str | None = None,
    ) -> None:
        self.client: Client = create_client(url, publishable_key)
        if access_token and refresh_token:
            self.client.auth.set_session(
                access_token,
                refresh_token,
            )

    def sign_in(self, email: str, password: str) -> LoginSession:
        response = self.client.auth.sign_in_with_password(
            {"email": email, "password": password}
        )
        if response.user is None or response.session is None:
            raise RuntimeError("Supabase tidak mengembalikan sesi login.")
        return LoginSession(
            user_id=str(response.user.id),
            email=response.user.email,
            access_token=response.session.access_token,
            refresh_token=response.session.refresh_token,
        )

    def sign_out(self) -> None:
        self.client.auth.sign_out()

    def get_profile(self, user_id: str) -> dict[str, Any]:
        result = (
            self.client.table("profiles")
            .select("id,email,full_name,role")
            .eq("id", user_id)
            .single()
            .execute()
        )
        return result.data or {}

    def assert_admin(self, user_id: str) -> dict[str, Any]:
        profile = self.get_profile(user_id)
        if profile.get("role") != "admin":
            raise PermissionError(
                "Akun berhasil login tetapi belum memiliki role admin."
            )
        return profile

    def fetch_regions(self) -> pd.DataFrame:
        result = (
            self.client.table("regions")
            .select("province,city,expected_child_count")
            .order("province")
            .order("city")
            .execute()
        )
        return pd.DataFrame(result.data or [])

    def list_years(self) -> list[int]:
        result = (
            self.client.table("city_predictions")
            .select("year")
            .order("year")
            .execute()
        )
        return sorted(
            {int(row["year"]) for row in (result.data or [])},
            reverse=True,
        )

    def fetch_city_predictions(
        self,
        year: int | None = None,
        province: str | None = None,
    ) -> pd.DataFrame:
        query = self.client.table("city_predictions").select("*")
        if year is not None:
            query = query.eq("year", int(year))
        if province:
            query = query.eq("province", province)
        result = (
            query.order("year")
            .order("province")
            .order("city")
            .execute()
        )
        return pd.DataFrame(result.data or [])

    def fetch_province_predictions(
        self,
        year: int | None = None,
        province: str | None = None,
    ) -> pd.DataFrame:
        query = self.client.table("province_predictions").select("*")
        if year is not None:
            query = query.eq("year", int(year))
        if province:
            query = query.eq("province", province)
        result = query.order("year").order("province").execute()
        return pd.DataFrame(result.data or [])

    def create_batch(
        self,
        year: int,
        user_id: str,
        filename: str,
        row_count: int,
    ) -> str:
        result = (
            self.client.table("predictor_batches")
            .insert(
                {
                    "year": int(year),
                    "uploaded_by": user_id,
                    "original_filename": filename,
                    "row_count": int(row_count),
                    "status": "validated",
                }
            )
            .execute()
        )
        return str(result.data[0]["id"])

    @staticmethod
    def _clean_record(record: dict[str, Any]) -> dict[str, Any]:
        cleaned: dict[str, Any] = {}
        for key, value in record.items():
            if pd.isna(value):
                cleaned[key] = None
            elif hasattr(value, "item"):
                cleaned[key] = value.item()
            else:
                cleaned[key] = value
        return cleaned

    def store_predictors(
        self,
        batch_id: str,
        frame: pd.DataFrame,
    ) -> None:
        records = []
        for raw in frame.to_dict(orient="records"):
            row = self._clean_record(raw)
            records.append(
                {
                    "batch_id": batch_id,
                    "year": int(row["tahun"]),
                    "province": str(row["provinsi"]),
                    "city": str(row["kabupaten_kota"]),
                    "payload": row,
                }
            )
        for start in range(0, len(records), 100):
            (
                self.client.table("predictors")
                .insert(records[start : start + 100])
                .execute()
            )

    def update_batch(
        self,
        batch_id: str,
        status: str,
        error_message: str | None = None,
    ) -> None:
        payload: dict[str, Any] = {"status": status}
        if error_message is not None:
            payload["error_message"] = error_message
        (
            self.client.table("predictor_batches")
            .update(payload)
            .eq("id", batch_id)
            .execute()
        )

    def upsert_predictions(
        self,
        batch_id: str | None,
        city_frame: pd.DataFrame,
        province_frame: pd.DataFrame,
        source: str = "model",
    ) -> None:
        city_records = []
        for row in city_frame.to_dict(orient="records"):
            city_records.append(
                {
                    "batch_id": batch_id,
                    "year": int(row["year"]),
                    "province": str(row["province"]),
                    "city": str(row["city"]),
                    "prediction_ton": float(row["prediction_ton"]),
                    "actual_ton": (
                        None
                        if row.get("actual_ton") is None
                        or pd.isna(row.get("actual_ton"))
                        else float(row["actual_ton"])
                    ),
                    "model_name": "XGBoost",
                    "hierarchy_method": "Bottom-Up",
                    "source": source,
                }
            )

        province_records = []
        for row in province_frame.to_dict(orient="records"):
            province_records.append(
                {
                    "batch_id": batch_id,
                    "year": int(row["year"]),
                    "province": str(row["province"]),
                    "prediction_ton": float(row["prediction_ton"]),
                    "actual_ton": (
                        None
                        if row.get("actual_ton") is None
                        or pd.isna(row.get("actual_ton"))
                        else float(row["actual_ton"])
                    ),
                    "child_count": int(row["child_count"]),
                    "coherence_difference_ton": 0.0,
                    "hierarchy_method": "Bottom-Up",
                }
            )

        for start in range(0, len(city_records), 100):
            (
                self.client.table("city_predictions")
                .upsert(
                    city_records[start : start + 100],
                    on_conflict="year,province,city",
                )
                .execute()
            )
        (
            self.client.table("province_predictions")
            .upsert(
                province_records,
                on_conflict="year,province",
            )
            .execute()
        )
