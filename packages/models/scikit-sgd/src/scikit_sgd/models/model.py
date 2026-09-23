"""Runtime wrapper around a scikit-learn SGD regressor."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from bovi_core.ml import Model
from sklearn.linear_model import SGDRegressor

from .config import ScikitSGDModelConfig


class ScikitSGDModel(Model[SGDRegressor, ScikitSGDModelConfig]):
    """An instantiated SGD regressor without storage or training ownership."""

    def __call__(
        self,
        features: npt.ArrayLike,
        **kwargs: object,
    ) -> npt.NDArray[np.float64]:
        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(f"ScikitSGDModel does not accept prediction arguments: {unexpected}")
        return self.native_model.predict(np.asarray(features, dtype=np.float64))
