from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppConfig:
    host: str
    port: int
    artifact_path: Path
    metadata_path: Path
    random_seed: int
    n_samples: int
    n_features: int
    noise: float
    model_version: str

    @classmethod
    def from_env(cls) -> "AppConfig":
        artifacts_dir = Path(os.getenv("ARTIFACT_DIR", "artifacts"))
        return cls(
            host=os.getenv("SERVE_HOST", "0.0.0.0"),
            port=int(os.getenv("SERVE_PORT", "8000")),
            artifact_path=artifacts_dir / "linear_model.joblib",
            metadata_path=artifacts_dir / "metadata.json",
            random_seed=int(os.getenv("TRAIN_RANDOM_SEED", "42")),
            n_samples=int(os.getenv("TRAIN_N_SAMPLES", "300")),
            n_features=int(os.getenv("TRAIN_N_FEATURES", "3")),
            noise=float(os.getenv("TRAIN_NOISE", "0.2")),
            model_version=os.getenv("MODEL_VERSION", "v1"),
        )
