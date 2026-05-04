from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.linear_model import LinearRegression


@dataclass(frozen=True)
class ModelMetadata:
    model_version: str
    n_features: int
    trained_at_utc: str
    random_seed: int
    n_samples: int
    noise: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelMetadata":
        return cls(
            model_version=str(data["model_version"]),
            n_features=int(data["n_features"]),
            trained_at_utc=str(data["trained_at_utc"]),
            random_seed=int(data["random_seed"]),
            n_samples=int(data["n_samples"]),
            noise=float(data["noise"]),
        )


def train_linear_model(x: np.ndarray, y: np.ndarray) -> LinearRegression:
    model = LinearRegression()
    model.fit(x, y)
    return model


def build_metadata(
    model_version: str, n_features: int, random_seed: int, n_samples: int, noise: float
) -> ModelMetadata:
    return ModelMetadata(
        model_version=model_version,
        n_features=n_features,
        trained_at_utc=datetime.now(timezone.utc).isoformat(),
        random_seed=random_seed,
        n_samples=n_samples,
        noise=noise,
    )


def save_artifacts(
    model: LinearRegression,
    metadata: ModelMetadata,
    artifact_path: Path,
    metadata_path: Path,
) -> None:
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, artifact_path)
    metadata_path.write_text(json.dumps(metadata.to_dict(), indent=2), encoding="utf-8")


def load_artifacts(
    artifact_path: Path, metadata_path: Path
) -> tuple[LinearRegression | None, ModelMetadata | None]:
    if not artifact_path.exists() or not metadata_path.exists():
        return None, None

    try:
        model = joblib.load(artifact_path)
        metadata_data = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata = ModelMetadata.from_dict(metadata_data)
    except Exception:
        return None, None

    if not isinstance(model, LinearRegression):
        return None, None

    return model, metadata
