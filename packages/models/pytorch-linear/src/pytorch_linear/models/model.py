"""CPU inference wrapper; training is owned by the concrete trainer."""

import torch
from bovi_core.ml import Model

from .config import PyTorchLinearModelConfig


class PyTorchLinearModel(Model[torch.nn.Linear, PyTorchLinearModelConfig]):
    def __call__(self, features):
        self.native_model.eval()
        with torch.no_grad():
            return (
                self.native_model(torch.as_tensor(features, dtype=torch.float32, device="cpu"))
                .numpy()
                .reshape(-1)
            )
