"""Universal transform base class for tabular/time-series data."""

from __future__ import annotations

from abc import ABC, abstractmethod


class UniversalTransform(ABC):
    """
    Base class for tabular/time-series transforms.

    NumPy-only: no framework-specific methods needed.
    Subclasses only need to implement __call__ and get_params.

    All transforms are stateless - they process data per-sample without
    requiring a fit() step. This keeps the design simple (YAGNI).

    Example:
        @TransformRegistry.register("imputation")
        class ImputationTransform(UniversalTransform):
            def __init__(self, method: str = "forward_fill"):
                self.method = method

            def __call__(self, data: dict[str, object]) -> dict[str, object]:
                # Apply imputation per-sample
                return self._impute(data)

            def get_params(self) -> dict[str, object]:
                return {"method": self.method}
    """

    @abstractmethod
    def __call__(self, data: dict[str, object]) -> dict[str, object]:
        """
        Apply transform to data dict.

        Args:
            data: Dictionary containing arrays/values to transform.

        Returns:
            Transformed data dictionary.
        """
        pass

    @abstractmethod
    def get_params(self) -> dict[str, object]:
        """
        Return parameters for reproducibility.

        Returns:
            Dictionary of transform parameters.
        """
        pass

    def __repr__(self) -> str:
        """String representation of the transform."""
        params = self.get_params()
        params_str = ", ".join(f"{k}={v!r}" for k, v in params.items())
        return f"{self.__class__.__name__}({params_str})"
