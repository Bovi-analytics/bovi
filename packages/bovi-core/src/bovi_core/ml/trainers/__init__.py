"""Framework-neutral training, evaluation, and result contracts."""

from .config import EvaluationConfig, TrainingConfig
from .context import (
    EvaluationContext,
    ExecutionContext,
    FederatedEvaluationContext,
    FederatedTrainingContext,
    TrainingContext,
)
from .evaluation import (
    EvaluationArtifactReference,
    EvaluationResult,
    EvaluationStatus,
    Evaluator,
)
from .issues import Issue, IssueSeverity
from .logging import (
    LogDestinationResult,
    LogDestinationStatus,
    LogIssue,
    ResultLogOutcome,
    TrainingResultLogger,
)
from .results import (
    EpochResult,
    TrainingResult,
    TrainingStatus,
    TrainingStopReason,
)
from .trainer import Trainer

__all__ = [
    "EpochResult",
    "EvaluationArtifactReference",
    "EvaluationConfig",
    "EvaluationContext",
    "EvaluationResult",
    "EvaluationStatus",
    "Evaluator",
    "ExecutionContext",
    "FederatedEvaluationContext",
    "FederatedTrainingContext",
    "Issue",
    "IssueSeverity",
    "LogDestinationResult",
    "LogDestinationStatus",
    "LogIssue",
    "ResultLogOutcome",
    "Trainer",
    "TrainingConfig",
    "TrainingContext",
    "TrainingResult",
    "TrainingResultLogger",
    "TrainingStatus",
    "TrainingStopReason",
]
