"""Import isolation for autoencoder Function App tests."""

from __future__ import annotations

import sys
from pathlib import Path

_APP_DIR = Path(__file__).resolve().parents[1]
_APP_MODULES = ("main", "model_assets", "schemas", "settings")

app_path = str(_APP_DIR)
if app_path in sys.path:
    sys.path.remove(app_path)
sys.path.insert(0, app_path)

for module_name in _APP_MODULES:
    loaded = sys.modules.get(module_name)
    loaded_file = getattr(loaded, "__file__", None)
    if loaded_file is not None and Path(loaded_file).resolve().parent != _APP_DIR:
        sys.modules.pop(module_name, None)
