"""Fresh construction and restoration of portable CPU state dictionaries."""

import torch
from bovi_core.ml import ModelProviderRegistry, ResolvedCheckpoint, ResolvedModelArtifact

from .config import PyTorchLinearModelConfig
from .model import PyTorchLinearModel

FORMAT = "pytorch-linear-state"


class PyTorchLinearModelProvider:
    def create(self, config: PyTorchLinearModelConfig) -> PyTorchLinearModel:
        with torch.random.fork_rng(devices=[]):
            native = torch.nn.Linear(len(config.feature_names), 1, device="cpu")
        torch.nn.init.zeros_(native.weight)
        torch.nn.init.zeros_(native.bias)
        return PyTorchLinearModel(native_model=native, config=config)

    def restore_checkpoint(
        self, config: PyTorchLinearModelConfig, checkpoint: ResolvedCheckpoint[object]
    ) -> PyTorchLinearModel:
        return self._load(config, checkpoint)

    def load_artifact(
        self, config: PyTorchLinearModelConfig, artifact: ResolvedModelArtifact[object]
    ) -> PyTorchLinearModel:
        return self._load(config, artifact)

    def _load(self, config, resource):
        if resource.format != FORMAT or resource.local_path is None:
            raise ValueError("Expected a local pytorch-linear-state checkpoint")
        payload = torch.load(resource.local_path, map_location="cpu", weights_only=True)
        if tuple(payload["feature_names"]) != config.feature_names:
            raise ValueError("Checkpoint feature order differs from model config")
        model = self.create(config)
        model.native_model.load_state_dict(payload["state_dict"])
        return model


ModelProviderRegistry.register("pytorch_linear")(PyTorchLinearModelProvider)
