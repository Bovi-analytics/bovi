from pydantic import BaseModel, ConfigDict


class EvaluationResult(BaseModel):
    """Placeholder result returned by trainer evaluation.

    TODO: Define the common evaluation metrics and per-split outputs after the
    concrete trainers agree on their evaluation contract.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
