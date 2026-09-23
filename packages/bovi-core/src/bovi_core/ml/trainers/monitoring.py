"""Small scalar-regression monitoring helpers, independent of native frameworks."""

from dataclasses import dataclass, field
from math import isfinite
from typing import Literal

import numpy as np

from .results import TrainingStopReason


@dataclass
class MetricMonitor:
    """Track the raw optimum separately from cumulative patience significance."""

    min_delta: float = 0.0
    patience: int | None = None
    target: float | None = None
    mode: Literal["min", "max"] = "min"
    best: float = field(init=False)
    significant_best: float = field(init=False)
    stale: int = 0

    def __post_init__(self) -> None:
        if self.mode not in ("min", "max"):
            raise ValueError("Metric mode must be min or max")
        if not isfinite(self.min_delta) or self.min_delta < 0:
            raise ValueError("min_delta must be finite and nonnegative")
        self.best = self.significant_best = float("inf") if self.mode == "min" else -float("inf")

    def observe(self, score: float) -> tuple[bool, TrainingStopReason | None]:
        if not isfinite(score):
            raise ValueError("Monitored score must be finite")
        direction = 1 if self.mode == "min" else -1
        improved = direction * score < direction * self.best
        if improved:
            self.best = score
        if direction * score < direction * self.significant_best - self.min_delta:
            self.significant_best, self.stale = score, 0
        else:
            self.stale += 1
        stop = None
        if self.target is not None and direction * score <= direction * self.target:
            stop = TrainingStopReason.TARGET_METRIC_REACHED
        elif self.patience is not None and self.stale >= self.patience:
            stop = TrainingStopReason.EARLY_STOPPING
        return improved, stop


@dataclass
class RegressionMetrics:
    """Sample-weighted streaming MSE/MAE for scalar regression."""

    count: int = 0
    squared_error: float = 0.0
    absolute_error: float = 0.0

    def update(self, expected, predicted) -> None:
        expected = np.asarray(expected, dtype=np.float64).reshape(-1)
        predicted = np.asarray(predicted, dtype=np.float64).reshape(-1)
        if expected.shape != predicted.shape or not expected.size:
            raise ValueError("Expected and predicted values must have equal nonempty shapes")
        errors = predicted - expected
        if not np.isfinite(errors).all():
            raise ValueError("Regression values must be finite")
        self.count += len(errors)
        self.squared_error += float(np.sum(errors**2))
        self.absolute_error += float(np.sum(np.abs(errors)))

    def result(self) -> tuple[int, dict[str, float]]:
        if not self.count:
            raise ValueError("Cannot evaluate an empty dataloader")
        return self.count, {
            "mse": self.squared_error / self.count,
            "mae": self.absolute_error / self.count,
        }
