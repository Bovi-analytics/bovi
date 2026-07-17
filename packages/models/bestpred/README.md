# bestpred-py

Pure Python port of the USDA BESTPRED model for estimating lactation yield,
persistency, data collection rating (DCR), and reliability from test-day data.

The package contains two deliberately separate interfaces:

- a typed Python and pandas API for notebooks and applications;
- a compatibility CLI for the legacy BESTPRED input sources and output files.

The numerical port is validated against output from the current Fortran source.
The historical output distributed with the BESTPRED manual differs from that
current source and is retained as a separate legacy reference.

## Installation

BESTPRED is a member of the Bovi uv workspace and requires Python 3.12.

```bash
cd bovi
uv sync --package bestpred-py
uv run python -c "import bestpred; print(bestpred.__file__)"
```

Package checks can be run from the package directory:

```bash
cd packages/models/bestpred
just test
just lint
just build
```

## Practical notebooks

The numbered notebooks under [`notebooks/`](notebooks/) form an executable
onboarding path. Start Jupyter from the Bovi repository root so imports resolve
through the uv workspace:

```bash
just sync
uv run jupyter notebook packages/models/bestpred/notebooks
```

Work through the series in order:

| Notebook | Purpose |
| --- | --- |
| [`00_start_here.ipynb`](notebooks/00_start_here.ipynb) | Environment, architecture, implementation status, and reading order |
| [`01_dataframe_quickstart.ipynb`](notebooks/01_dataframe_quickstart.ipynb) | Load canonical test-day data and use the recommended DataFrame API |
| [`02_legacy_sources_and_cli.ipynb`](notebooks/02_legacy_sources_and_cli.ipynb) | Inspect sources 10/11/14/15/24, their parsers, and the compatibility CLI |
| [`03_fdd_adapter.ipynb`](notebooks/03_fdd_adapter.ipynb) | Use real or structural FDD Cow/Herd objects and the temporary lactation DTOs |
| [`04_lactationcurve_comparison_and_migration.ipynb`](notebooks/04_lactationcurve_comparison_and_migration.ipynb) | Compare with Bovi `lactationcurve` and review the remaining replacement work |
| [`05_legacy_fortran_oracle.ipynb`](notebooks/05_legacy_fortran_oracle.ipynb) | Run an external Fortran oracle or fall back to the checked-in golden fixture |

The notebooks are repository documentation and are not bundled in the wheel.
They remain executable without optional integrations. To exercise real FDD
objects, install the sibling repository with
`uv pip install -e ../farm-data-definitions`. To run the legacy executable,
set `BESTPRED_FORTRAN_BINARY=/path/to/bestpred`; the executable is deliberately
not committed or used by the Python runtime.

## Python API

The lowest stable computation boundary accepts typed Format-4 records and
BESTPRED parameters:

```python
from pathlib import Path

from bestpred import predict_records
from bestpred.io.parameters import read_parameters
from bestpred.io.source10 import read_source10_records

parameters = read_parameters(Path("tests/fixtures/source10_current/bestpred.par"))
records = read_source10_records(Path("tests/fixtures/source10_current/format4.dat"))
legacy_rows = predict_records(records, parameters, source11_compat=False)
```

`predict_records` returns `DcrResultRow` objects with the original 43 numeric
output positions. Use `prediction_from_dcr_row` for a named result:

```python
from bestpred import prediction_from_dcr_row

prediction = prediction_from_dcr_row(legacy_rows[0], test_id="lactation-1")
print(prediction.milk.yield_305)
print(prediction.milk.yield_reliability)
print(prediction.dcr_milk)
```

## DataFrame API

`predict_dataframe` accepts long-form data with one row per test day and
returns one row per `TestId`.

```python
from pathlib import Path

import pandas as pd

from bestpred import predict_dataframe

test_days = pd.DataFrame(
    {
        "TestId": ["cow-42-l2", "cow-42-l2"],
        "AnimalId": ["HCOW42", "HCOW42"],
        "BirthDate": ["20200101", "20200101"],
        "HerdId": ["35HERD7", "35HERD7"],
        "FreshDate": ["20240203", "20240203"],
        "Parity": [2, 2],
        "LactationLength": [305, 305],
        "HerdMilk305": [20_000, 20_000],
        "HerdFat305": [700, 700],
        "HerdProtein305": [600, 600],
        "DaysInMilk": [30, 60],
        "MilkingYield": [70.0, 75.0],
        "FatPercent": [3.9, 3.8],
        "ProteinPercent": [3.2, 3.1],
        "SCS": [2.1, 2.2],
    }
)

result = predict_dataframe(
    test_days,
    Path("tests/fixtures/source11_current/bestpred.par"),
)
print(result[["TestId", "MilkYield305", "MilkYieldReliability", "DCRMilk"]])
```

Callers with different column names can supply a canonical-to-caller mapping:

```python
result = predict_dataframe(
    test_days.rename(columns={"DaysInMilk": "dim"}),
    parameters,
    column_map={"DaysInMilk": "dim"},
)
```

### Required DataFrame columns

| Column | Meaning |
| --- | --- |
| `TestId` | Stable lactation identifier used for grouping and output |
| `AnimalId` | BESTPRED animal identifier |
| `BirthDate` | Date object, `YYYYMMDD`, or ISO date |
| `HerdId` | BESTPRED herd identifier; its first two characters normally contain the numeric state code |
| `FreshDate` | Calving/fresh date |
| `Parity` | Lactation number, at least 1 |
| `LactationLength` | Target/current lactation length in days |
| `DaysInMilk` | Unique DIM within the lactation |
| `MilkingYield` | Test-day milk yield in `UNITSin` |
| `HerdMilk305` | Herd 305-day milk baseline in `UNITSin` |
| `HerdFat305` | Herd 305-day fat baseline in `UNITSin` |
| `HerdProtein305` | Herd 305-day protein baseline in `UNITSin` |

Identity, dates, parity, lactation length, and herd baselines must remain
constant within each `TestId`.

### Optional DataFrame columns

| Column | Default | Meaning |
| --- | ---: | --- |
| `PreviousDaysOpen` | `140` | Previous lactation days open |
| `FatPercent` | `0.0` | Decimal fat percentage |
| `ProteinPercent` | `0.0` | Decimal protein percentage |
| `SCS` | `0.0` | Somatic cell score |
| `HerdSCS305` | breed fallback | Herd 305-day SCS baseline |
| `Supervised` | `2` | BESTPRED supervision code |
| `Status` | `0` | Test-day status code |
| `TimesMilked` | `2` | Milkings per day |
| `TimesWeighed` | `2` | Weighings per test |
| `TimesSampled` | `2` with component columns, otherwise `0` | Component samples per test |
| `LERDays` | `1` | Labor-efficient-recording interval |
| `PercentShipped` | `100` | Percentage of milk shipped |

If fat, protein, and SCS columns are all absent, the rows are treated as having
no component sample. A measured zero should therefore be supplied explicitly,
preferably together with `TimesSampled`.

### Units and scaling

`MilkingYield` and herd yield baselines use the unit configured by `UNITSin`
in `bestpred.par` (`P` for pounds in the checked-in fixtures). Output yield
columns use `UNITSout`. Percentages and SCS are normal decimal values in the
DataFrame API.

The legacy `Format4Record` boundary stores fixed-width integers:

- milk yield is multiplied by 10;
- fat and protein percentages are multiplied by 10;
- SCS is multiplied by 10.

The DataFrame adapter performs these conversions. Callers constructing
`TestDaySegment` directly must apply the legacy scaling themselves.

## Named output

The DataFrame API returns identifiers, `DCRMilk`, `DCRComponents`, `DCRSCS`,
and the following fields for each of `Milk`, `Fat`, `Protein`, and `SCS`:

- `Yield305`, `Yield365`, `YieldLactation`, and `YieldPartial`;
- `Persistency`;
- `YieldReliability` and `PersistencyReliability`;
- `ExpandedYield`, `Herd305`, and `Bumpiness`.

For example: `MilkYield305`, `ProteinPersistencyReliability`, and
`SCSBumpiness`.

## Compatibility CLI

The CLI supports the ported legacy sources:

| Source | Input | Additional files/output |
| ---: | --- | --- |
| 10 | AIPL Format 4 file | `bestpred.par`, DCR output |
| 11 | `DCRexample.txt` plans | `bestpred.par`, DCR output |
| 14 | DRMS/PCDART file | optional PCDART output |
| 15 | Format 4 file | sibling `format4.means` file |
| 24 | list of source-14 files | optional PCDART output |

Example:

```bash
uv run bestpred run \
  --source 11 \
  --input packages/models/bestpred/tests/fixtures/source11_current/DCRexample.txt \
  --par packages/models/bestpred/tests/fixtures/source11_current/bestpred.par \
  --output /tmp/results_v2.dcr
```

Compatibility output intentionally retains documented current-Fortran quirks,
including source-10/15 header rows and source-14 EOF rows.

## Compare with Bovi lactationcurve

Bovi's existing `lactationcurve.best_predict_method` is a narrower milk-only
DataFrame model. It is not replaced by this package. The comparison command
runs the same source records through BESTPRED and projects their milk test days
into the existing Bovi method:

```bash
uv run bestpred compare-bovi \
  --source 11 \
  --input packages/models/bestpred/tests/fixtures/source11_current/DCRexample.txt \
  --par packages/models/bestpred/tests/fixtures/source11_current/bestpred.par \
  --limit 10
```

The table reports row-level 305-day milk results and deltas in kilograms,
followed by matched-row count, mean absolute delta, and maximum absolute delta.
Material differences are expected because the methods have different data and
model contracts; comparison is an evaluation tool, not an equality test.

## Farm Data Definitions adapter

`bestpred.adapters.farm_data_definitions` provides structural adapters for Cow
and Herd objects plus temporary BESTPRED lactation/test-day DTOs. The module
does not require `farm-data-definitions` to install or run. When that package is
available, its compatible Cow and Herd models can be passed directly.

The permanent FDD lactation, test-day, herd-baseline, and prediction models are
still pending. See `docs/reference/port/bovi_fdd_alignment.md`.

## Documentation and provenance

Start at [`docs/README.md`](docs/README.md). Historical port notes, visual
reports, the BESTPRED manual, original distribution files, and known Fortran
quirks are preserved under `docs/reference`.

The original BESTPRED government work is public domain and carries its own
notice in [`LICENSE`](LICENSE). No warranty is provided.
