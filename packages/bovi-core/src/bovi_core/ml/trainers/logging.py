from typing import Protocol


class TrainingResultLogger(Protocol):
    """Structural type for optional training-result logging.

    TODO: Add the logging methods after the training result schema and logging
    lifecycle are defined by the concrete trainer implementations.
    """
