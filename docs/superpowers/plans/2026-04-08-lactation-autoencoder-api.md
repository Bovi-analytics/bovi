# Lactation Autoencoder API Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire up the existing lactation_autoencoder ML package to the Azure Function API with single/batch prediction endpoints, configurable herd stats, and per-model versioning in bovi-core.

**Architecture:** Enrichment (herd stats, event mapping) becomes a transform in the YAML-configured pipeline — not a source responsibility. The API uses `DictSource` to feed HTTP request data into the same pipeline as training. The API is ~5 lines of glue code.

**Tech Stack:** Python 3.12, FastAPI, TensorFlow, bovi-core (Config, transforms, registries), Pydantic, Azure Functions

**Spec:** `docs/superpowers/specs/2026-04-08-lactation-autoencoder-api-design.md`

---

## Key Design Decision: Enrichment is a Transform

Transforms can need reference data (loaded at init from a path in YAML `params`). This is already supported by `UniversalTransform` — no new abstraction needed. Convention: specify the path in `params`, load in `__init__`.

```yaml
transforms:
  - name: herd_stats_enrichment          # NEW: adds herd_stats from pkl files
    params:
      herd_stats_dir: "data/.../pkl/"
  - name: event_tokenization             # REFACTORED: loads its own event_to_idx
    params:
      event_to_idx_path: "data/.../pkl/event_to_idx_dict.pkl"
  - name: imputation
    params:
      method: forward_fill
  - name: milk_normalization
    params:
      max_milk: 80.0
```

This means:
- `LactationPKLSource` becomes a pure JSON DataSource (no herd stats logic)
- The transform pipeline handles ALL data enrichment + processing
- The API doesn't need any special provider/service — just `DictSource → TransformedSource → Dataset → predict`

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `packages/bovi-core/src/bovi_core/config.py` | Modify | Add `{model_version}` injection in `_process_templated_models()` |
| `packages/bovi-core/tests/bovi_core/test_config.py` | Modify | Test for per-model version templating |
| `packages/bovi-core/src/bovi_core/ml/dataloaders/sources/dict_source.py` | Create | Generic in-memory `DictSource` for inference from dicts |
| `packages/bovi-core/tests/bovi_core/ml/dataloaders/sources/test_dict_source.py` | Create | Tests for DictSource |
| `packages/models/lactation-autoencoder/src/lactation_autoencoder/dataloaders/transforms/lactation_transforms.py` | Modify | Add `HerdStatsEnrichmentTransform`, refactor `EventTokenizationTransform` to load own data |
| `packages/models/lactation-autoencoder/src/lactation_autoencoder/dataloaders/sources/lactation_pkl_source.py` | Modify | Remove herd stats logic (moved to transform) |
| `packages/models/lactation-autoencoder/tests/` | Modify | Update tests for new architecture |
| `data/experiments/lactation_autoencoder/versions/v15/config/config.yaml` | Modify | Add enrichment transform, model version, event_to_idx_path |
| `apps/backend/models/lactation-autoencoder/settings.py` | Modify | Slim down to CORS only |
| `apps/backend/models/lactation-autoencoder/.env.example` | Modify | Remove model/herd_stats paths |
| `apps/backend/models/lactation-autoencoder/main.py` | Modify | Real prediction endpoints |
| `apps/backend/models/lactation-autoencoder/tests/conftest.py` | Create | Shared fixtures using `get_project_root()` |
| `apps/backend/models/lactation-autoencoder/tests/test_main.py` | Create | API endpoint tests |
| `apps/backend/api/src/bovi_api/routes/proxy.py` | Modify | Add batch proxy route |

---

### Task 1: Per-model versioning in bovi-core

**Files:**
- Modify: `packages/bovi-core/src/bovi_core/config.py:563-573`
- Modify: `packages/bovi-core/tests/bovi_core/test_config.py`

- [ ] **Step 1: Write failing test for model_version in template context**

Add to `TestTemplatingLogic` class in `packages/bovi-core/tests/bovi_core/test_config.py`:

```python
def test_per_model_version_in_template(self, config_setup):
    """Test that a model's 'version' field is available as {model_version} in templates."""
    path_templates = {
        "versioned_weights": {
            "template": "data/versions/v{model_version}/weights/{weights_file}",
            "uses": "weights_file",
        }
    }
    test_models = {
        "autoencoder": {
            "version": 15,
            "template_vars": {
                "weights_file": {"default": "autoencoder"},
            },
        }
    }
    project_vars = config_setup._flatten_config_node(config_setup.project)
    processed = config_setup._process_templated_models(
        test_models, path_templates, project_vars
    )
    path = processed["autoencoder"]["versioned_weights"]["default"]
    assert "v15" in path
    assert path.endswith("weights/autoencoder")


def test_per_model_version_different_per_model(self, config_setup):
    """Test that different models can have different versions."""
    path_templates = {
        "versioned_weights": {
            "template": "data/versions/v{model_version}/weights/{weights_file}",
            "uses": "weights_file",
        }
    }
    test_models = {
        "model_a": {
            "version": 10,
            "template_vars": {"weights_file": {"default": "a"}},
        },
        "model_b": {
            "version": 20,
            "template_vars": {"weights_file": {"default": "b"}},
        },
    }
    project_vars = config_setup._flatten_config_node(config_setup.project)
    processed = config_setup._process_templated_models(
        test_models, path_templates, project_vars
    )
    assert "v10" in processed["model_a"]["versioned_weights"]["default"]
    assert "v20" in processed["model_b"]["versioned_weights"]["default"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd packages/bovi-core && just test -- -k "test_per_model_version" -v`

- [ ] **Step 3: Implement model_version injection**

In `packages/bovi-core/src/bovi_core/config.py`, in `_process_templated_models()`, after line 573 (`context.update(model_config.get("vars", {}))`), add:

```python
            if "version" in model_config:
                context["model_version"] = model_config["version"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd packages/bovi-core && just test`

- [ ] **Step 5: Commit**

```bash
git add packages/bovi-core/src/bovi_core/config.py packages/bovi-core/tests/bovi_core/test_config.py
git commit -m "feat(bovi-core): add per-model version support in path templates"
```

---

### Task 2: Add DictSource to bovi-core

**Files:**
- Create: `packages/bovi-core/src/bovi_core/ml/dataloaders/sources/dict_source.py`
- Create: `packages/bovi-core/tests/bovi_core/ml/dataloaders/sources/test_dict_source.py`
- Modify: `packages/bovi-core/src/bovi_core/ml/dataloaders/sources/__init__.py`

Generic in-memory DataSource. Serves dicts directly — for HTTP requests, test data, etc.

- [ ] **Step 1: Write failing test**

Create `packages/bovi-core/tests/bovi_core/ml/dataloaders/sources/test_dict_source.py`:

```python
"""Tests for DictSource."""

from bovi_core.ml.dataloaders.sources.dict_source import DictSource


class TestDictSource:
    def test_single_item(self):
        source = DictSource([{"milk": [1.0, 2.0], "parity": 1}])
        assert len(source) == 1
        assert source.load_item(0)["parity"] == 1

    def test_multiple_items(self):
        source = DictSource([{"a": 1}, {"a": 2}, {"a": 3}])
        assert len(source) == 3
        assert source.load_item(2) == {"a": 3}

    def test_get_keys(self):
        source = DictSource([{"x": 1}, {"x": 2}])
        assert source.get_keys() == [0, 1]

    def test_iteration(self):
        data = [{"a": 1}, {"a": 2}]
        assert list(DictSource(data)) == data

    def test_empty(self):
        assert len(DictSource([])) == 0
```

- [ ] **Step 2: Implement DictSource**

Create `packages/bovi-core/src/bovi_core/ml/dataloaders/sources/dict_source.py`:

```python
"""In-memory data source for serving dicts directly."""

from __future__ import annotations

from bovi_core.ml.dataloaders.base.data_source import DataSource


class DictSource(DataSource[dict[str, object]]):
    """Serve pre-built dicts as a DataSource.

    For inference from HTTP requests, test data, or any scenario
    where data is already in memory.

    Args:
        items: List of data dicts to serve.
    """

    def __init__(self, items: list[dict[str, object]]) -> None:
        self._items = items

    def __len__(self) -> int:
        return len(self._items)

    def load_item(self, key: int | str) -> dict[str, object]:
        return self._items[int(key)]

    def get_metadata(self, key: int | str) -> dict[str, object]:
        return {"index": int(key)}

    def get_keys(self) -> list[int | str]:
        return list(range(len(self._items)))
```

- [ ] **Step 3: Export from sources __init__.py**

- [ ] **Step 4: Run tests**

Run: `cd packages/bovi-core && just test`

- [ ] **Step 5: Commit**

```bash
git add packages/bovi-core/src/bovi_core/ml/dataloaders/sources/dict_source.py packages/bovi-core/src/bovi_core/ml/dataloaders/sources/__init__.py packages/bovi-core/tests/bovi_core/ml/dataloaders/sources/test_dict_source.py
git commit -m "feat(bovi-core): add DictSource for in-memory inference"
```

---

### Task 3: HerdStatsEnrichmentTransform + refactor EventTokenizationTransform

**Files:**
- Modify: `packages/models/lactation-autoencoder/src/lactation_autoencoder/dataloaders/transforms/lactation_transforms.py`
- Modify: `packages/models/lactation-autoencoder/src/lactation_autoencoder/dataloaders/sources/lactation_pkl_source.py`
- Modify: `packages/models/lactation-autoencoder/tests/`

This is the core architectural change. Enrichment moves from source to transform.

**Design note — source-agnostic loading:** Transforms that need reference data receive a path via YAML params. For now, loading is local (`open()`). The loading should go through a small loader abstraction (e.g. a `load_file(path, config)` function) so that blob storage support can be added later without changing the transforms. The transform doesn't know or care whether the file is local or in blob — it just asks the loader. Blob implementation is out of scope for now (no container to test against), but the interface must be ready for it.

- [ ] **Step 1: Create `HerdStatsEnrichmentTransform`**

Add to `lactation_transforms.py`:

```python
@TransformRegistry.register("herd_stats_enrichment")
class HerdStatsEnrichmentTransform(UniversalTransform):
    """Enrich data with hierarchical herd statistics from pkl files.

    Loads reference data at init. Adds herd_stats to each data dict
    based on herd_id + parity with 4-level fallback.

    Args:
        herd_stats_dir: Path to herd stats directory (resolved by Config).
    """

    def __init__(self, herd_stats_dir: str) -> None:
        self.herd_stats_dir = Path(herd_stats_dir)
        self._load_reference_data()

    def __call__(self, data: dict[str, object]) -> dict[str, object]:
        herd_id = data.get("herd_id")
        parity = data.get("parity")
        if "herd_stats" not in data or data["herd_stats"] is None:
            data["herd_stats"] = self._get_herd_stats_with_fallback(herd_id, parity)
        return data
```

The `_load_reference_data` (renamed from `_load_herd_stats`), `_get_herd_stats_with_fallback`, and `_convert_stats_to_array` methods are **moved** from `LactationPKLSource`. The file loading within `_load_reference_data` should use a loader function that can be swapped for blob later — not hardcoded `open()` calls.

- [ ] **Step 2: Refactor `EventTokenizationTransform` to load its own data**

Change `EventTokenizationTransform.__init__` to accept `event_to_idx_path`:

```python
def __init__(self, event_to_idx_path: str | None = None, unknown_event: str = "unknown") -> None:
    self.unknown_event = unknown_event
    self.event_to_idx: dict[str, int] | None = None
    if event_to_idx_path is not None:
        self.event_to_idx = self._load_event_mapping(event_to_idx_path)
```

Same principle: `_load_event_mapping` uses a loader function, not direct `open()`. In `__call__`, use `self.event_to_idx` if loaded, otherwise read from `data.get("event_to_idx")`.

- [ ] **Step 3: Simplify `LactationPKLSource`**

Remove herd stats loading and enrichment from `LactationPKLSource`. It becomes a pure JSON DataSource:
- Remove `_load_herd_stats()`, `_get_herd_stats_with_fallback()`, `_convert_stats_to_array()`
- Remove `herd_stats_dir` parameter
- `load_item()` just loads JSON and returns the raw dict
- Herd stats enrichment is now the transform's job

- [ ] **Step 4: Update tests**

Update existing tests in `packages/models/lactation-autoencoder/tests/` to reflect:
- `LactationPKLSource` no longer adds herd_stats
- `HerdStatsEnrichmentTransform` tests for all 4 fallback levels
- `EventTokenizationTransform` tests with `event_to_idx_path`
- Integration tests use the new transform pipeline order

- [ ] **Step 5: Run all tests**

Run: `cd packages/models/lactation-autoencoder && just test`

- [ ] **Step 6: Commit**

```bash
git add packages/models/lactation-autoencoder/
git commit -m "refactor(lactation-autoencoder): move enrichment from source to transforms"
```

---

### Task 4: Update config YAML + commit data

**Files:**
- Modify: `data/experiments/lactation_autoencoder/versions/v15/config/config.yaml`

- [ ] **Step 1: Update YAML with new transform pipeline and model version**

```yaml
path_templates:
  local_weights:
    template: "data/experiments/lactation_autoencoder/versions/v{model_version}/weights/{weights_file}"
    uses: weights_file

dataloaders:
  inference:
    source:
      type: lactation_pkl
      json_root_dir: "data/experiments/lactation_autoencoder/input/inference/json/"
      file_pattern: "*.json"
    transforms:
      - name: herd_stats_enrichment
        params:
          herd_stats_dir: "data/experiments/lactation_autoencoder/input/inference/pkl/"
      - name: event_tokenization
        params:
          event_to_idx_path: "data/experiments/lactation_autoencoder/input/inference/pkl/event_to_idx_dict.pkl"
      - name: imputation
        params:
          method: forward_fill
          fields: ["milk"]
      - name: milk_normalization
        params:
          max_milk: 80.0

models:
  autoencoder:
    version: 15
    type: keras_autoencoder
    framework: tensorflow
    # ... rest stays the same
```

Note: `herd_stats_dir` is removed from `source` section — it's now in the enrichment transform. Also update train/validation dataloaders the same way.

- [ ] **Step 2: Verify data/ is not in .gitignore**

- [ ] **Step 3: Commit**

```bash
git add data/
git commit -m "feat: add data directory, update config for enrichment transforms and model versioning"
```

---

### Task 5: Slim down Settings + verify deps

**Files:**
- Modify: `apps/backend/models/lactation-autoencoder/settings.py`
- Modify: `apps/backend/models/lactation-autoencoder/.env.example`

- [ ] **Step 1: Update settings.py to CORS only**

```python
"""Lactation autoencoder Function App settings."""

from functools import lru_cache

from dotenv import find_dotenv
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Deployment-only configuration. ML config is handled by bovi-core Config."""

    cors_origins: list[str] = ["http://localhost:3000"]

    model_config = {"env_file": find_dotenv(), "env_file_encoding": "utf-8", "extra": "ignore"}


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()
```

- [ ] **Step 2: Update .env.example**

```
# CORS_ORIGINS is set by the root justfile from /.env — do not duplicate here.
```

- [ ] **Step 3: Verify pyproject.toml and function_app.py**

Both already correct — deps present, wiring correct.

- [ ] **Step 4: Commit**

```bash
git add apps/backend/models/lactation-autoencoder/settings.py apps/backend/models/lactation-autoencoder/.env.example
git commit -m "refactor(autoencoder-api): slim settings to CORS only"
```

---

### Task 6: Main API implementation

**Files:**
- Modify: `apps/backend/models/lactation-autoencoder/main.py`

The API is thin glue. Same pipeline as the notebook.

- [ ] **Step 1: Write the complete main.py**

**Startup:**
```python
from bovi_core.config import Config
from bovi_core.ml import create_model
from bovi_core.ml.dataloaders.sources import DictSource, TransformedSource
from bovi_core.ml.dataloaders.transforms.registry import TransformRegistry
from lactation_autoencoder.dataloaders.datasets.lactation_dataset import LactationDataset

config = Config(experiment_name="lactation_autoencoder", project_name="bovi")
model = create_model(config, "autoencoder")
transforms = TransformRegistry.from_config(config.experiment.dataloaders.inference.transforms)
```

**Prediction:**
```python
def _predict_single(request):
    data = request.model_dump()

    # Custom herd_stats from user? Put it in the dict — enrichment transform will skip it
    dict_source = DictSource([data])
    transformed = TransformedSource(dict_source, list(transforms.values()))
    dataset = LactationDataset(source=transformed, config=config)
    features = dataset[0]["features"]

    result = model.predict(features, return_format="rich")
    return AutoencoderPredictResponse(predictions=result.predictions.tolist(), latent_vector=None)
```

That's it. `request.model_dump()` is already a dict. The enrichment transform adds herd_stats (skipping if already provided). The rest of the pipeline (imputation, tokenization, normalization, pad/truncate) is handled by existing transforms + dataset. `return_format="rich"` denormalizes.

For batch: pass multiple dicts to DictSource, iterate dataset.

- [ ] **Step 2: Verify app starts**

Run: `cd apps/backend/models/lactation-autoencoder && uv run python -c "from main import app; print('OK')"`

- [ ] **Step 3: Commit**

```bash
git add apps/backend/models/lactation-autoencoder/main.py
git commit -m "feat(autoencoder-api): implement prediction endpoints"
```

---

### Task 7: API endpoint tests

**Files:**
- Create: `apps/backend/models/lactation-autoencoder/tests/conftest.py`
- Create: `apps/backend/models/lactation-autoencoder/tests/test_main.py`

HTTP contract tests only. Model logic is tested in `packages/models/lactation-autoencoder/tests/`.

- [ ] **Step 1: Create conftest.py**

```python
"""Shared test fixtures."""

from pathlib import Path

import pytest
from bovi_core.utils.path_utils import get_project_root


@pytest.fixture
def project_root():
    return Path(get_project_root(project_name="bovi"))
```

- [ ] **Step 2: Write endpoint tests**

Create `apps/backend/models/lactation-autoencoder/tests/test_main.py`:

```python
"""API endpoint tests — HTTP contract only."""

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


class TestHealthEndpoint:
    def test_health_check(self):
        response = client.get("/")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestPredictEndpoint:
    def test_predict_minimal(self):
        milk = [None] * 304
        for i in range(18):
            milk[i] = 15.0 + i * 1.5
        response = client.post("/predict", json={"milk": milk})
        assert response.status_code == 200
        assert len(response.json()["predictions"]) == 304

    def test_predict_with_all_params(self):
        milk = [25.0] * 200
        response = client.post("/predict", json={
            "milk": milk, "events": ["calving"] + ["pad"] * 199,
            "parity": 2, "herd_id": 2942694, "imputation_method": "linear",
        })
        assert response.status_code == 200
        assert len(response.json()["predictions"]) == 304

    def test_predict_custom_herd_stats(self):
        response = client.post("/predict", json={
            "milk": [25.0] * 100, "herd_stats": [0.5] * 10,
        })
        assert response.status_code == 200

    def test_short_milk_padded(self):
        response = client.post("/predict", json={"milk": [25.0] * 50})
        assert response.status_code == 200
        assert len(response.json()["predictions"]) == 304

    def test_empty_milk_rejected(self):
        assert client.post("/predict", json={"milk": []}).status_code == 422

    def test_bad_imputation_rejected(self):
        response = client.post("/predict", json={"milk": [25.0] * 100, "imputation_method": "invalid"})
        assert response.status_code == 422

    def test_wrong_herd_stats_length(self):
        response = client.post("/predict", json={"milk": [25.0] * 100, "herd_stats": [0.5] * 5})
        assert response.status_code == 422


class TestBatchEndpoint:
    def test_batch_predict(self):
        response = client.post("/predict/batch", json={
            "items": [{"milk": [25.0] * 200, "parity": 1}, {"milk": [30.0] * 150, "parity": 2}],
        })
        assert response.status_code == 200
        assert len(response.json()["results"]) == 2
        assert all(len(r["predictions"]) == 304 for r in response.json()["results"])

    def test_batch_empty_rejected(self):
        assert client.post("/predict/batch", json={"items": []}).status_code == 422
```

- [ ] **Step 3: Run tests**

Run: `cd apps/backend/models/lactation-autoencoder && uv run pytest tests/ -v`

- [ ] **Step 4: Commit**

```bash
git add apps/backend/models/lactation-autoencoder/tests/
git commit -m "test(autoencoder-api): add API endpoint tests"
```

---

### Task 8: Add batch proxy route

**Files:**
- Modify: `apps/backend/api/src/bovi_api/routes/proxy.py`

- [ ] **Step 1: Add batch proxy route**

After `proxy_autoencoder_predict` (line 92), add:

```python
@router.post("/autoencoder/predict/batch")
async def proxy_autoencoder_predict_batch(request: Request) -> JSONResponse:
    """Proxy: batch predict lactation curves via autoencoder."""
    return await _proxy_post(settings.lactation_autoencoder_url, "/predict/batch", request)
```

- [ ] **Step 2: Commit**

```bash
git add apps/backend/api/src/bovi_api/routes/proxy.py
git commit -m "feat(api): add batch proxy route for autoencoder"
```

---

### Task 9: End-to-end smoke test

- [ ] **Step 1: Start the API**

Run: `cd apps/backend/models/lactation-autoencoder && uv run python -m uvicorn main:app --reload --port 8001`

- [ ] **Step 2: Test health**

`curl http://localhost:8001/`

- [ ] **Step 3: Test single prediction**

```bash
curl -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{"milk": [15.0, 26.0, 26.0, 31.0, 35.0, 37.0, 38.0, 39.0, 40.0, 43.0, 42.0, 43.0, 43.0], "parity": 2}'
```

- [ ] **Step 4: Test batch prediction**

```bash
curl -X POST http://localhost:8001/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"items": [{"milk": [25.0, 30.0, 35.0], "parity": 1}, {"milk": [20.0, 25.0], "parity": 3}]}'
```

- [ ] **Step 5: Swagger docs**

Open: `http://localhost:8001/docs`

- [ ] **Step 6: Full test suite**

Run: `cd apps/backend/models/lactation-autoencoder && uv run pytest tests/ -v`
