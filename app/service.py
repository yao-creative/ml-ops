from __future__ import annotations

import logging
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from ray import serve
from sklearn.linear_model import LinearRegression

from app.config import AppConfig
from app.data import generate_toy_linear_data
from app.model import ModelMetadata, build_metadata, load_artifacts, save_artifacts, train_linear_model
from app.schemas import PredictRequest, PredictResponse

logger = logging.getLogger(__name__)
api = FastAPI()


@serve.deployment(num_replicas=1)
@serve.ingress(api)
class LinearModelDeployment:
    def __init__(self, config: AppConfig):
        self.config = config
        self.model: LinearRegression | None = None
        self.metadata: ModelMetadata | None = None
        self._load_or_train_model()

    def _load_or_train_model(self) -> None:
        loaded_model, loaded_metadata = load_artifacts(
            self.config.artifact_path, self.config.metadata_path
        )
        if loaded_model is not None and loaded_metadata is not None:
            self.model = loaded_model
            self.metadata = loaded_metadata
            logger.info(
                "Loaded model artifacts from disk",
                extra={
                    "artifact_path": str(self.config.artifact_path),
                    "metadata_path": str(self.config.metadata_path),
                    "model_version": loaded_metadata.model_version,
                },
            )
            return

        logger.info(
            "No valid model artifacts found. Training model from toy data.",
            extra={
                "n_samples": self.config.n_samples,
                "n_features": self.config.n_features,
                "noise": self.config.noise,
                "seed": self.config.random_seed,
            },
        )
        x_train, y_train = generate_toy_linear_data(
            n_samples=self.config.n_samples,
            n_features=self.config.n_features,
            noise=self.config.noise,
            seed=self.config.random_seed,
        )
        self.model = train_linear_model(x_train, y_train)
        self.metadata = build_metadata(
            model_version=self.config.model_version,
            n_features=self.config.n_features,
            random_seed=self.config.random_seed,
            n_samples=self.config.n_samples,
            noise=self.config.noise,
        )
        save_artifacts(
            model=self.model,
            metadata=self.metadata,
            artifact_path=self.config.artifact_path,
            metadata_path=self.config.metadata_path,
        )
        logger.info(
            "Model trained and saved",
            extra={
                "artifact_path": str(self.config.artifact_path),
                "metadata_path": str(self.config.metadata_path),
                "model_version": self.metadata.model_version,
            },
        )

    @api.get("/health")
    def health(self) -> dict[str, Any]:
        ready = self.model is not None and self.metadata is not None
        return {
            "status": "ok" if ready else "not_ready",
            "model_loaded": ready,
            "model_version": self.metadata.model_version if self.metadata else None,
            "n_features": self.metadata.n_features if self.metadata else None,
        }

    @api.post("/predict", response_model=PredictResponse)
    def predict(self, request: PredictRequest) -> PredictResponse:
        if self.model is None or self.metadata is None:
            raise HTTPException(status_code=503, detail="Model not ready")

        rows = self._normalize_features(request.features, self.metadata.n_features)
        predictions = self.model.predict(np.array(rows, dtype=float)).tolist()
        return PredictResponse(
            predictions=predictions,
            model_version=self.metadata.model_version,
            n_features=self.metadata.n_features,
        )

    @staticmethod
    def _normalize_features(
        features: list[float] | list[list[float]], expected_n_features: int
    ) -> list[list[float]]:
        if not features:
            raise HTTPException(status_code=422, detail="features cannot be empty")

        first = features[0]
        if isinstance(first, list):
            rows = features  # type: ignore[assignment]
        elif isinstance(first, (int, float)) and not isinstance(first, bool):
            rows = [features]  # type: ignore[list-item]
        else:
            raise HTTPException(
                status_code=422,
                detail="features must be a list of numbers or list of list of numbers",
            )

        normalized_rows: list[list[float]] = []
        for row in rows:
            if not isinstance(row, list):
                raise HTTPException(status_code=422, detail="each row must be a list")
            if len(row) != expected_n_features:
                raise HTTPException(
                    status_code=422,
                    detail=f"each row must contain exactly {expected_n_features} features",
                )

            parsed_row: list[float] = []
            for value in row:
                if isinstance(value, bool):
                    raise HTTPException(status_code=422, detail="boolean values are not allowed")
                try:
                    parsed_row.append(float(value))
                except (TypeError, ValueError) as exc:
                    raise HTTPException(
                        status_code=422, detail="all feature values must be numeric"
                    ) from exc
            normalized_rows.append(parsed_row)

        return normalized_rows
