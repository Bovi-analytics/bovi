from typing import Any, Literal
from unittest.mock import Mock

from bovi_core.ml.predictors import PredictionInterface


class ExamplePredictor(PredictionInterface[object, object, object]):
    def initialize(self) -> None:
        self.initialized_with = self.model

    def predict(
        self,
        data: object,
        return_format: Literal["raw", "base", "rich"] = "raw",
        prompt: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> object:
        return self.model


def test_predictor_receives_model_during_construction():
    model = object()
    config = Mock()

    predictor = ExamplePredictor(model=model, config=config)

    assert predictor.model is model
    assert predictor.config is config
    assert predictor.initialized_with is model
    assert predictor.predict(object()) is model
