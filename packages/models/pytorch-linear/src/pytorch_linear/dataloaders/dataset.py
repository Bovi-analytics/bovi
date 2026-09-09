"""Convert one regression record into a framework-neutral sample."""

import numpy as np
from bovi_core.ml import Dataset


class LinearDataset(Dataset):
    def __init__(self, source, feature_names: tuple[str, ...], target_name: str = "y"):
        super().__init__(source)
        self.feature_names = feature_names
        self.target_name = target_name

    def __len__(self) -> int:
        return len(self.source)

    def __getitem__(self, index: int):
        record = self.source.load_item(index)
        return {
            "features": np.asarray([record[name] for name in self.feature_names], dtype=np.float32),
            "labels": np.float32(record[self.target_name]),
        }
