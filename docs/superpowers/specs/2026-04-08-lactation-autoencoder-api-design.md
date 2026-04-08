# Lactation Autoencoder API — Design Spec

## Goal

Complete the lactation autoencoder Azure Function API so it serves real predictions using the existing `lactation_autoencoder` package and bovi-core Config system. Add support for single and batch inference, configurable parameters (events, parity, herd_id, herd_stats, imputation method), and commit the data directory to version control.

## Current State

- **Package** (`packages/models/lactation-autoencoder/`): Complete — model, predictor, transforms, datasource, results all implemented.
- **Azure Function** (`apps/backend/models/lactation-autoencoder/`): Skeleton — health endpoint works, `/predict` returns zeros.
- **Central API proxy**: Already routes `/autoencoder/predict` to the function app.
- **Data** (`data/experiments/lactation_autoencoder/`): Present locally (~34 MB total including other experiments), not yet in git.

## Architecture

### Config vs Settings

- **Settings** (`settings.py`, Pydantic BaseSettings, `.env`): Deployment-only config — `cors_origins`.
- **Config** (bovi-core, YAML): All ML config — experiment name/version, model paths, data source paths, transform pipeline.

Settings becomes minimal (just CORS). Config handles everything ML-related. The `model_blob_url`, `herd_stats_blob_url`, `model_path`, and `herd_stats_path` fields are removed from Settings — they live in the Config YAML.

### Experiment & Model Version Resolution

There are two levels of versioning:

1. **Experiment version** — which `config.yaml` to load. `Config(experiment_name="lactation_autoencoder")` delegates to `get_run_config_path()` which scans `data/experiments/lactation_autoencoder/versions/` and picks the highest version with a config file (currently v15).

2. **Model version** — which model weights to use, defined **per model** in the YAML. This is a new feature added to bovi-core's `_process_templated_models()`.

```python
config = Config(experiment_name="lactation_autoencoder", project_name="bovi")
# Auto-resolves to data/experiments/lactation_autoencoder/versions/v15/config/config.yaml
```

#### Per-model versioning in YAML

Each model in the `models:` section gets a `version` field. This is injected into the template context as `{model_version}`, allowing each model in a pipeline to point to different weights:

```yaml
path_templates:
  local_weights:
    template: "data/experiments/lactation_autoencoder/versions/v{model_version}/weights/{weights_file}"
    uses: weights_file

models:
  autoencoder:
    version: 15              # ← per-model version, available as {model_version} in templates
    framework: tensorflow
    template_vars:
      weights_file:
        default: "autoencoder"
  
  # Example: a second model in the same pipeline at a different version
  encoder_v2:
    version: 18
    framework: tensorflow
    template_vars:
      weights_file:
        default: "encoder"
```

**Implementation**: small change in bovi-core's `Config._process_templated_models()` — after building the template context, inject `model_version` from the model config:

```python
# In _process_templated_models, after context.update(model_config.get("vars", {})):
if "version" in model_config:
    context["model_version"] = model_config["version"]
```

The existing `experiment_version` continues to work as a fallback for templates that use `{experiment_version}`. The new `{model_version}` takes precedence when defined per model.

### App Startup

Module-level initialization (loaded once, reused per request):

1. **Config** — resolved as described above. Reads the versioned `config.yaml`.
2. **HerdStatsService** — a lightweight service class (created in the API app) that loads only the pkl files from the herd stats directory. Loaded once at startup, kept as a singleton. Provides:
   - `event_to_idx` mapping (for injection into request data before transforms)
   - `get_herd_stats(herd_id, parity)` with hierarchical fallback
   - `global_means` as default when nothing is provided
   
   This is separate from `LactationPKLSource` which is designed for training (requires json_root_dir, builds file index). The API only needs the pkl lookup functionality.
3. **Transforms** — `ImputationTransform`, `EventTokenizationTransform`, `MilkNormalizationTransform` initialized from config. The imputation method comes from the config YAML (default: `forward_fill`).
4. **Model + Predictor** — `LactationPredictor(config=config)` + `LactationAutoencoderModel.from_config(config, predictor)` — TF SavedModel loaded once.

### Endpoints

#### `GET /` — Health check (existing)
Returns `{"status": "ok"}`.

#### `POST /predict` — Single prediction

**Request:**
```json
{
  "milk": [15.0, 26.0, null, null, ..., 50.0, ...],
  "events": ["calving", "pad", ..., "breeding", ...],
  "parity": 2,
  "herd_id": 2942694,
  "herd_stats": null,
  "imputation_method": "forward_fill"
}
```

**Fields:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `milk` | `list[float \| null]` | Yes | — | Daily milk yield (kg). Variable length accepted; truncated or padded with `null` to 304 elements internally. `null` for missing days. Range: 0-80 kg/day. |
| `events` | `list[str]` | No | `["calving"] + 303x ["pad"]` | Daily reproductive/health events. Variable length accepted; truncated or padded with `"pad"` to 304 elements internally. Case-insensitive. |
| `parity` | `int` | No | `1` | Lactation number. 1 = first lactation (heifer). Range: 1-12. |
| `herd_id` | `int` | No | `null` | Herd identifier for hierarchical herd stats lookup. See "Herd Stats Resolution" below. |
| `herd_stats` | `list[float]` | No | `null` | Custom herd statistics (10 floats, normalized 0-1). When provided, overrides any herd_id-based lookup. See "Herd Stats Resolution" below. |
| `imputation_method` | `str` | No | `"forward_fill"` | How to fill missing (null) values in the milk sequence. Options: `forward_fill`, `backward_fill`, `linear`, `zero`, `mean`. |

**Available events (case-insensitive):** `pad`, `calving`, `breeding`, `pregnancypositive`, `pregnancynegative`, `heat`, `dryoff`, `disease`, `mastitis`, `cull`, `died`, `abort`, `donotbreed`, `missingreprostatus`, `unknown`

#### `POST /predict/batch` — Batch prediction

**Request:**
```json
{
  "items": [
    {"milk": [...], "events": [...], "parity": 2, "herd_id": 123},
    {"milk": [...], "parity": 1, "herd_stats": [0.1, 0.2, ...]}
  ],
  "imputation_method": "forward_fill"
}
```

`imputation_method` is set once for the whole batch. Individual items follow the same schema as `/predict`.

**Response (single):**
```json
{
  "predictions": [15.2, 26.1, 27.3, ...],
  "latent_vector": [0.12, -0.34, ...]
}
```

**Response (batch):**
```json
{
  "results": [
    {"predictions": [...], "latent_vector": [...]},
    {"predictions": [...], "latent_vector": [...]}
  ]
}
```

| Field | Type | Description |
|-------|------|-------------|
| `predictions` | `list[float]` | Predicted daily milk yields for 304 days (denormalized, kg/day) |
| `latent_vector` | `list[float] \| null` | Latent-space representation (if available from model) |

---

## Herd Stats Resolution

Herd statistics are 10 numbers that describe the average performance of the herd (farm) a cow belongs to. They give the model context about the production level and management quality of the farm, which helps produce more accurate predictions.

### The 10 herd statistics

| Index | Name | Description | Example |
|-------|------|-------------|---------|
| 0 | Achieved21Milk | Herd average milk production in the first 21 days (kg) | 0.53 |
| 1 | Achieved305Milk | Herd average 305-day total production (kg) | 0.50 |
| 2 | Achieved75Milk | Herd average milk production in the first 75 days (kg) | 0.55 |
| 3 | AchievedMilk | Herd average total lifetime milk production (kg) | 0.41 |
| 4 | DaysDry | Herd average dry period length (days between last milking and next calving) | 0.39 |
| 5 | DaysInMilk | Herd average current days in milk | 0.44 |
| 6 | DaysOpen | Herd average open days (days from calving to conception) | 0.38 |
| 7 | DaysPregnant | Herd average days pregnant at current time | 0.62 |
| 8 | HistoricCalvingInterval | Herd average calving interval (days between consecutive calvings) | 0.54 |
| 9 | QualitySequence | Herd average data quality/completeness score | 0.33 |

All values are normalized to 0-1 range. The example values above are the global means across all herds.

### How herd stats are resolved (4-level fallback)

The system uses a hierarchical fallback to ensure the most specific statistics available are used:

```
Level 1: herd_id + parity combination
    │     Most specific. Uses stats for this exact herd and lactation number.
    │     Example: Herd 2942694, parity 2 → stats for second-lactation cows on that farm.
    │
    ├─ Not found? ──►
    │
Level 2: herd_id only (average across all parities)
    │     Uses the herd's average stats regardless of lactation number.
    │     Example: Herd 2942694 → average stats for all cows on that farm.
    │
    ├─ Not found? ──►
    │
Level 3: parity only (average across all herds)
    │     Uses the global average for that parity.
    │     Example: parity 2 → average stats for all second-lactation cows in the dataset.
    │
    ├─ Not found? ──►
    │
Level 4: global average
          Least specific. Uses the overall average across all herds and parities.
          Values: [0.53, 0.50, 0.55, 0.41, 0.39, 0.44, 0.38, 0.62, 0.54, 0.33]
```

### How users provide herd stats

There are three ways, from most to least specific:

1. **Provide `herd_stats` directly** — 10 normalized floats. The user knows their herd's statistics and provides them. Skips all lookup. Best for: users who have computed their own herd statistics.

2. **Provide `herd_id`** — The system looks up the herd in the pre-loaded dataset (156 known herds). Falls back through levels 1-4 as described above. Best for: users whose herd is in the dataset.

3. **Provide neither** — The system uses global averages (Level 4). Best for: quick predictions without herd context, or when herd information is unknown.

### Frontend implications

The frontend should:
- Show a dropdown or input for `herd_id` (optional, with a list of known herds if desired)
- Show an "Advanced" section where users can manually input 10 herd stats values
- Show a tooltip/help text for each herd stat explaining what it means
- Show which fallback level was used in the response (future enhancement — not in this spec)

---

## Data Flow (detailed)

### Step-by-step for a single `/predict` request

```
1. REQUEST ARRIVES
   ┌─────────────────────────────────────────────────────┐
   │ milk: [15.0, 26.0, null, ..., 50.0, ...]  (316 el) │
   │ events: ["calving", "pad", ..., "breeding"] (300 el)│
   │ parity: 2                                           │
   │ herd_id: 2942694                                    │
   │ imputation_method: "forward_fill"                   │
   └─────────────────────────────────────────────────────┘
                          │
                          ▼
2. TRUNCATE / PAD TO 304 ELEMENTS
   - milk: if len > 304 → truncate to first 304
           if len < 304 → pad with null at the end
   - events: if len > 304 → truncate to first 304
             if len < 304 → pad with "pad" at the end
   Result: milk[304], events[304]
                          │
                          ▼
3. RESOLVE HERD STATS (HerdStatsService, loaded at startup)
   - herd_stats provided in request? → use directly, skip lookup
   - herd_id=2942694, parity=2:
     Level 1: herd_stats_per_parity[2942694]["2"] → FOUND ✓
   Result: herd_stats[10] (normalized floats)
                          │
                          ▼
4. BUILD DATA DICT (matching what transforms expect)
   {
     "milk": [15.0, 26.0, null, ..., 50.0, ...],     # 304 elements
     "events": ["calving", "pad", ..., "breeding"],    # 304 elements
     "parity": 2,
     "herd_stats": [0.51, 0.49, ...],                 # 10 elements
     "event_to_idx": {"pad": 0, "calving": 2, ...}    # injected from singleton
   }
                          │
                          ▼
5. APPLY TRANSFORMS (from registered transform pipeline)
   a) ImputationTransform(method="forward_fill")
      - null values in milk are filled by copying the last valid value forward
      - e.g. [15.0, 26.0, null, null] → [15.0, 26.0, 26.0, 26.0]
      
   b) EventTokenizationTransform
      - "calving" → 2, "pad" → 0, "breeding" → 8, etc. (case-insensitive)
      - Result: events_encoded[304] of integers
      
   c) MilkNormalizationTransform(max_milk=80.0)
      - milk / 80.0 → normalized to 0-1 range
      - e.g. 26.0 → 0.325
                          │
                          ▼
6. PREPARE MODEL INPUTS (LactationPredictor._prepare_inputs)
   - input_11: milk      → tf.Tensor shape (1, 304, 1)  float32
   - input_12: parity    → tf.Tensor shape (1, 1)        float32
   - input_13: events    → tf.Tensor shape (1, 304)      float32
   - input_15: herd_stats → tf.Tensor shape (1, 10)      float32
                          │
                          ▼
7. MODEL INFERENCE
   TF SavedModel serving_default signature
   Output: predictions tensor shape (1, 304) normalized 0-1
                          │
                          ▼
8. DENORMALIZE
   predictions × 80.0 → milk yield in kg/day
                          │
                          ▼
9. RESPONSE
   ┌─────────────────────────────────────────────────────┐
   │ predictions: [15.2, 26.1, 27.3, ..., 18.5]  (304)  │
   │ latent_vector: [0.12, -0.34, ...] or null           │
   └─────────────────────────────────────────────────────┘
```

### Imputation methods explained

The `imputation_method` parameter controls how missing (`null`) values in the milk sequence are filled before the model sees them:

| Method | Description | Example: `[15.0, null, null, 30.0, null]` |
|--------|-------------|-------------------------------------------|
| `forward_fill` (default) | Copy last known value forward | `[15.0, 15.0, 15.0, 30.0, 30.0]` |
| `backward_fill` | Copy next known value backward | `[15.0, 30.0, 30.0, 30.0, 30.0]` |
| `linear` | Linear interpolation between known values | `[15.0, 20.0, 25.0, 30.0, 30.0]` |
| `zero` | Fill with 0.0 | `[15.0, 0.0, 0.0, 30.0, 0.0]` |
| `mean` | Fill with mean of known values | `[15.0, 22.5, 22.5, 30.0, 22.5]` |

The default `forward_fill` is defined in the config YAML and can be overridden per request. All methods are provided by bovi-core's `ImputationTransform`.

---

## Data Organisation

Data lives in `data/` in the repo root and is committed to version control:

```
data/experiments/lactation_autoencoder/
├── input/inference/
│   ├── pkl/                              (1.4 MB - herd stats + metadata pickle files)
│   │   ├── event_to_idx_dict.pkl
│   │   ├── idx_to_herd_par_dict.pkl
│   │   ├── idx_to_herd_dict.pkl
│   │   ├── idx_to_kpi_dict.pkl
│   │   ├── herd_stats_per_parity_dict.pkl
│   │   ├── herd_stats_means_per_herd.pkl
│   │   ├── herd_stats_dict.pkl
│   │   ├── herd_stat_means.pkl
│   │   ├── herd_stat_means_per_parity.pkl
│   │   ├── herd_stat_means_global.pkl
│   │   ├── herd_par_avg_latent_dict.pkl
│   │   ├── par_avg_latent_dict.pkl
│   │   ├── kpi_imputation_dict.pkl
│   │   └── kpi_min_max_dict.pkl
│   └── json/                             (8 KB - example data)
│       └── animal_001.json
├── versions/v15/
│   ├── config/
│   │   ├── config.yaml
│   │   └── pyproject.toml
│   └── weights/                          (~10 MB - TF SavedModel)
│       ├── autoencoder/
│       └── model/
```

The Config YAML references these paths relative to the project root. bovi-core's `_resolve_paths` converts them to absolute paths automatically.

## Files to Change

### Modify
- `apps/backend/models/lactation-autoencoder/main.py` — Replace placeholder with real implementation: new request/response models, HerdStatsService, model loading on startup, single + batch endpoints, truncate/pad to 304 elements
- `apps/backend/models/lactation-autoencoder/settings.py` — Slim down to `cors_origins` only
- `apps/backend/models/lactation-autoencoder/.env.example` — Remove model/herd_stats paths
- `apps/backend/models/lactation-autoencoder/pyproject.toml` — Add `lactation-autoencoder` and `bovi-core` as dependencies if not present
- `apps/backend/models/lactation-autoencoder/function_app.py` — Verify Azure Functions entry point is correctly wired
- `apps/backend/api/src/bovi_api/routes/proxy.py` — Add `/autoencoder/predict/batch` proxy route

### Add
- `apps/backend/models/lactation-autoencoder/herd_stats_service.py` — Lightweight service that loads pkl files and provides herd stats lookup (extracts logic from LactationPKLSource without requiring json_root_dir)
- `data/` — Commit to version control (already present locally)

### Modify (bovi-core)
- `packages/bovi-core/src/bovi_core/config.py` — Add per-model `version` → `{model_version}` injection in `_process_templated_models()`

### Modify (config YAML)
- `data/experiments/lactation_autoencoder/versions/v15/config/config.yaml` — Add `version: 15` to the autoencoder model config, update path template to use `{model_version}` instead of `{experiment_version}`

### No changes needed
- `packages/models/lactation-autoencoder/` — Package is complete

## Out of Scope

- Blob storage integration (future — change YAML paths when ready)
- Frontend changes (separate task — this spec covers the API)
- Training pipeline
- Unity Catalog integration
- Tutorial notebooks for the autoencoder
- Returning the fallback level used for herd stats (future enhancement)
