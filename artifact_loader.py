from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import hf_hub_download


@dataclass(frozen=True)
class ArtifactPaths:
    model_path: Path
    preprocessor_path: Path
    source: str


def resolve_artifacts(
    repo_id: str | None,
    token: str | None = None,
    revision: str = "main",
) -> ArtifactPaths:
    local_dir = Path(__file__).resolve().parent / "artifacts"
    local_model = local_dir / "xgboost_model_terbaik.ubj"
    local_preprocessor = local_dir / "preprocessor.joblib"

    if repo_id:
        model_path = Path(
            hf_hub_download(
                repo_id=repo_id,
                filename="xgboost_model_terbaik.ubj",
                repo_type="model",
                revision=revision,
                token=token or None,
            )
        )
        preprocessor_path = Path(
            hf_hub_download(
                repo_id=repo_id,
                filename="preprocessor.joblib",
                repo_type="model",
                revision=revision,
                token=token or None,
            )
        )
        return ArtifactPaths(
            model_path=model_path,
            preprocessor_path=preprocessor_path,
            source=f"Hugging Face: {repo_id}@{revision}",
        )

    if local_model.exists() and local_preprocessor.exists():
        return ArtifactPaths(
            model_path=local_model,
            preprocessor_path=local_preprocessor,
            source="artifacts lokal",
        )

    raise FileNotFoundError(
        "Artefak model tidak ditemukan. Isi HF_MODEL_REPO_ID pada "
        "Streamlit Secrets atau salin model dan preprocessor ke "
        "streamlit_app/artifacts/."
    )
