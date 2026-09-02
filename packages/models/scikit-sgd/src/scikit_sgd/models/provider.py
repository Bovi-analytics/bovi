"""Construction and local loading provider for scikit SGD models."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import joblib
from bovi_core.ml import ModelProviderRegistry, ResolvedCheckpoint, ResolvedModelArtifact
from sklearn.linear_model import SGDRegressor

from .config import ScikitSGDModelConfig
from .model import ScikitSGDModel

SCIKIT_JOBLIB_FORMAT = "scikit-joblib"


class ScikitSGDModelProvider:
    """Create fresh estimators or restore already-resolved joblib resources."""

    def create(self, config: ScikitSGDModelConfig) -> ScikitSGDModel:
        estimator = SGDRegressor(
            fit_intercept=config.fit_intercept,
            random_state=config.random_state,
        )
        return ScikitSGDModel(native_model=estimator, config=config)

    def restore_checkpoint(
        self,
        config: ScikitSGDModelConfig,
        checkpoint: ResolvedCheckpoint[object],
    ) -> ScikitSGDModel:
        return self._load_resolved(
            config, checkpoint.format, checkpoint.local_path, checkpoint.payload
        )

    def load_artifact(
        self,
        config: ScikitSGDModelConfig,
        artifact: ResolvedModelArtifact[object],
    ) -> ScikitSGDModel:
        return self._load_resolved(config, artifact.format, artifact.local_path, artifact.payload)

    @staticmethod
    def _load_resolved(
        config: ScikitSGDModelConfig,
        resource_format: str,
        local_path: Path | None,
        payload: object | None,
    ) -> ScikitSGDModel:
        if resource_format != SCIKIT_JOBLIB_FORMAT:
            raise ValueError(
                f"Unsupported scikit SGD resource format: {resource_format!r}. "
                f"Expected {SCIKIT_JOBLIB_FORMAT!r}."
            )

        estimator = payload
        if estimator is None:
            if local_path is None or not local_path.is_file():
                raise ValueError(f"Resolved scikit SGD file does not exist: {local_path}")
            estimator = joblib.load(local_path)

        if not isinstance(estimator, SGDRegressor):
            raise TypeError("Scikit SGD resource must contain an SGDRegressor")
        return ScikitSGDModel(native_model=cast(SGDRegressor, estimator), config=config)


ModelProviderRegistry.register("scikit_sgd")(ScikitSGDModelProvider)
