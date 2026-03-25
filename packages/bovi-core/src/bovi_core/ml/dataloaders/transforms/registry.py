"""
Transform registry system.

Allows transforms to self-register from any package, enabling
plugin-style architecture without editing core code.

For vision transforms, use Albumentations directly via register_albumentations().
For tabular transforms, use UniversalTransform base class.
"""

from __future__ import annotations

import inspect
import logging
from collections import OrderedDict
from collections.abc import Callable

logger = logging.getLogger(__name__)


class TransformParameterError(Exception):
    """
    Raised when transform parameters are invalid or missing.

    Provides detailed information about expected parameters to help users
    fix their transform configurations.
    """

    def __init__(
        self,
        transform_name: str,
        transform_class: type[object],
        original_error: Exception,
        provided_params: dict[str, object],
    ) -> None:
        """
        Args:
            transform_name: Name of the transform in the registry.
            transform_class: The transform class being instantiated.
            original_error: The original TypeError from instantiation.
            provided_params: Parameters that were provided by the user.
        """
        # Get the signature of the transform's __init__ method
        sig = inspect.signature(transform_class.__init__)  # type: ignore[misc]

        # Build parameter info
        param_info: list[str] = []
        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue

            # Get type annotation if available
            type_str = (
                str(param.annotation) if param.annotation != inspect.Parameter.empty else "Any"
            )
            type_str = type_str.replace("typing.", "")  # Clean up typing module prefix

            # Check if parameter has a default value
            if param.default != inspect.Parameter.empty:
                param_info.append(
                    f"    {param_name}: {type_str} = {repr(param.default)} (optional)"
                )
            else:
                param_info.append(f"    {param_name}: {type_str} (required)")

        # Build the error message
        provided_str = ", ".join(f"{k}={repr(v)}" for k, v in provided_params.items())

        message = f"""
Invalid parameters for transform '{transform_name}'.

Transform class: {transform_class.__module__}.{transform_class.__name__}

Expected parameters:
{chr(10).join(param_info)}

Provided parameters:
    {provided_str if provided_str else "(none)"}

Original error: {str(original_error)}

Example usage:
    TransformRegistry.create("{transform_name}", param1=..., param2=...)
"""

        super().__init__(message)
        self.transform_name = transform_name
        self.transform_class = transform_class
        self.original_error = original_error
        self.provided_params = provided_params


class TransformRegistry:
    """Global registry for transform classes."""

    _transforms: dict[str, type[object] | Callable[..., object]] = {}
    _factories: dict[str, Callable[..., object]] = {}

    @classmethod
    def register(
        cls, name: str, factory: Callable[..., object] | None = None
    ) -> Callable[[type[object]], type[object]]:
        """
        Register a transform class.

        Usage:
            @TransformRegistry.register("imputation")
            class ImputationTransform(UniversalTransform):
                pass

        Or for direct registration:
            TransformRegistry._transforms["Resize"] = A.Resize

        Args:
            name: Unique identifier for the transform.
            factory: Optional factory function for complex initialization.

        Returns:
            Decorator function.
        """

        def decorator(transform_class: type[object]) -> type[object]:
            if name in cls._transforms:
                logger.warning(f"Transform '{name}' already registered. Overwriting.")

            cls._transforms[name] = transform_class
            if factory:
                cls._factories[name] = factory

            logger.debug(f"Registered transform: {name} -> {transform_class}")
            return transform_class

        if factory is not None:
            cls._factories[name] = factory
            return lambda x: x

        return decorator

    @classmethod
    def register_albumentations(cls) -> None:
        """Auto-register common Albumentations transforms."""
        try:
            import albumentations as A
        except ImportError:
            logger.warning("Albumentations not installed. Vision transforms unavailable.")
            return

        transforms_to_register: dict[str, type[object]] = {
            # Geometric
            "Resize": A.Resize,
            "RandomCrop": A.RandomCrop,
            "CenterCrop": A.CenterCrop,
            "RandomResizedCrop": A.RandomResizedCrop,
            "HorizontalFlip": A.HorizontalFlip,
            "VerticalFlip": A.VerticalFlip,
            "RandomRotate90": A.RandomRotate90,
            "ShiftScaleRotate": A.ShiftScaleRotate,
            "Affine": A.Affine,
            # Color
            "Normalize": A.Normalize,
            "RandomBrightnessContrast": A.RandomBrightnessContrast,
            "RandomGamma": A.RandomGamma,
            "HueSaturationValue": A.HueSaturationValue,
            "CLAHE": A.CLAHE,
            "ToGray": A.ToGray,
            # Blur/Noise
            "GaussianBlur": A.GaussianBlur,
            "GaussNoise": A.GaussNoise,
            "Blur": A.Blur,
            "MotionBlur": A.MotionBlur,
            # Dropout
            "CoarseDropout": A.CoarseDropout,
            "GridDropout": A.GridDropout,
            # Type conversion
            "ToFloat": A.ToFloat,
        }

        for name, transform_class in transforms_to_register.items():
            cls._transforms[name] = transform_class

        logger.info(f"Registered {len(transforms_to_register)} Albumentations transforms")

    @classmethod
    def get(cls, name: str) -> type[object] | Callable[..., object]:
        """
        Get a registered transform class by name.

        Args:
            name: Transform identifier.

        Returns:
            Transform class or callable.

        Raises:
            ValueError: If transform not found.
        """
        if name not in cls._transforms:
            available = ", ".join(sorted(cls._transforms.keys()))
            raise ValueError(
                f"Transform '{name}' not found in registry.\n"
                f"Available transforms: {available}\n"
                f"Hint: Call TransformRegistry.register_albumentations() first."
            )
        return cls._transforms[name]

    @classmethod
    def create(cls, name: str, **params: object) -> object:
        """
        Create a transform instance.

        Args:
            name: Transform identifier.
            **params: Parameters for transform constructor.

        Returns:
            Instantiated transform.

        Raises:
            TransformParameterError: If parameters are invalid or missing.
        """
        if name in cls._factories:
            try:
                return cls._factories[name](**params)
            except TypeError as e:
                # Get the factory function for better error messages
                factory = cls._factories[name]
                raise TransformParameterError(name, type(factory), e, params) from e

        transform_class = cls.get(name)

        try:
            if callable(transform_class):
                return transform_class(**params)
            raise TypeError(f"Transform '{name}' is not callable")
        except TypeError as e:
            # Re-raise with helpful parameter information
            if isinstance(transform_class, type):
                raise TransformParameterError(name, transform_class, e, params) from e
            raise TransformParameterError(name, type(transform_class), e, params) from e

    @classmethod
    def build_vision_pipeline(
        cls, configs: list[dict[str, object]], **compose_kwargs: object
    ) -> object:
        """
        Build Albumentations Compose from config list.

        Args:
            configs: List of {"name": str, "params": dict} dicts.
            **compose_kwargs: Additional args for A.Compose.

        Returns:
            albumentations.Compose instance.
        """
        try:
            import albumentations as A
        except ImportError:
            raise ImportError(
                "Albumentations is required for vision pipelines. "
                "Install with: pip install albumentations"
            )

        transforms: list[object] = []
        for cfg in configs:
            name = str(cfg["name"])
            params_value = cfg.get("params", {})
            params: dict[str, object] = params_value if isinstance(params_value, dict) else {}
            transforms.append(cls.create(name, **params))

        return A.Compose(transforms, **compose_kwargs)

    @classmethod
    def from_config(cls, transform_specs: list[dict[str, object]]) -> OrderedDict[str, object]:
        """
        Build an ordered dict of transforms from configuration specifications.

        This is a convenience method for creating multiple transforms from
        a config file's transform list. Returns an OrderedDict so transforms
        can be accessed by name or iterated in order.

        Args:
            transform_specs: List of transform specifications, each containing:
                - name: The registered transform name
                - params: Optional dict of parameters for the transform

        Returns:
            OrderedDict mapping transform names to instantiated transform objects

        Example:
            >>> # In config.yaml:
            >>> # transforms:
            >>> #   - name: imputation
            >>> #     params:
            >>> #       method: forward_fill
            >>> #   - name: milk_normalization
            >>>
            >>> transforms = TransformRegistry.from_config(
            ...     config.experiment.dataloaders.train.transforms
            ... )
            >>> # Access by name:
            >>> transforms['imputation']
            >>> # Iterate in order:
            >>> for name, transform in transforms.items():
            ...     data = transform(data)
            >>> # Convert to list:
            >>> list(transforms.values())

        """
        transforms: OrderedDict[str, object] = OrderedDict()

        for spec in transform_specs:
            name = str(spec["name"])
            params_value = spec.get("params", {})
            params: dict[str, object] = params_value if isinstance(params_value, dict) else {}
            transform = cls.create(name, **params)
            transforms[name] = transform

        return transforms

    @classmethod
    def list_transforms(cls) -> dict[str, str]:
        """
        List all registered transforms.

        Returns:
            Dict mapping name to module path.
        """
        result: dict[str, str] = {}
        for name, t in cls._transforms.items():
            if hasattr(t, "__module__") and hasattr(t, "__name__"):
                result[name] = f"{t.__module__}.{t.__name__}"
            else:
                result[name] = str(t)
        return result

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """
        Check if a transform is registered.

        Args:
            name: Transform identifier.

        Returns:
            True if registered.
        """
        return name in cls._transforms

    @classmethod
    def clear(cls) -> None:
        """Clear all registrations (mainly for testing)."""
        cls._transforms.clear()
        cls._factories.clear()


# Auto-register Albumentations on module import
TransformRegistry.register_albumentations()
