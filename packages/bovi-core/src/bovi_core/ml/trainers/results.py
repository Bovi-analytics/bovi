from pydantic import BaseModel, ConfigDict


class TrainingResult(BaseModel):
    """Placeholder result returned by a completed training run.

    TODO: Define the common metrics, artifact references, and model metadata once
    the concrete trainer implementations establish the shared result contract.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
