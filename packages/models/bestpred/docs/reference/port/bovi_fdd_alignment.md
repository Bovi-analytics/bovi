# Bovi/Farm Data Definitions Alignment

## Current State

`farm-data-definitions` exposes Pydantic v2 models for `Cow`, `Herd`, `Event`,
and event metadata. Its `lactation.py` module is intentionally pending.

The Bovi `lactationcurve` package currently accepts dataframe-oriented inputs
and normalizes columns such as `TestId`, `DaysInMilk`, and `MilkingYield`. It
does not yet model BESTPRED concepts such as component sample flags, DCR,
multi-trait records, herd means, or source-specific test plan metadata.

The current Bovi best-predict function is useful, but materially narrower than
the original BESTPRED contract:

- input is long-form pandas data with `TestId`, `DaysInMilk`, `MilkingYield`;
- only milk is predicted by the public best-predict path;
- the default covariance/curve assets are package-level NumPy arrays;
- there is no typed concept for fat/protein/SCS, sampling frequency, LER,
  supervision, age/PDO adjustment, herd means, DCR, reliability or
  persistency.

The current `farm-data-definitions` package can already carry shared
identity-level objects (`Cow`, `Herd`) and breed composition. Its
`lactation.py` file is still intentionally empty, so lactation/test-day facts
cannot yet be represented upstream without adding new FDD models.

## BESTPRED Needs

BESTPRED requires the following concepts before it can be a complete production
replacement:

- animal identity and breed
- herd identity, state/region, and herd 305-day means
- lactation identity: fresh date, parity, birth date, previous days open
- test-day facts: DIM, milk, fat, protein, SCS, supervision, milking frequency,
  weighing frequency, sampling frequency, LER days, percent shipped
- output facts: projected yields, actual yields, mature equivalent yields, DCR,
  reliability, persistency, bumpiness

## Reuse Strategy

Use `farm-data-definitions` directly for shared animal/herd identity and breed
composition. Add adapters in `bestpred.adapters.farm_data_definitions` rather
than copying ontology classes.

Do not invent a local persistent ontology in `bestpred-py`. The missing
Lactation/TestDay ontology should be proposed upstream in `farm-data-definitions`
after the BESTPRED source-11 and Format 4 requirements are fully proven.

## Implemented Adapter Boundary

`python/src/bestpred/adapters/farm_data_definitions.py` now provides a narrow
adapter boundary:

- `breed_code_from_cow(cow)` maps FDD `Cow.breed` to BESTPRED breed codes.
- `BestpredHerdMeansInput` captures 305-day herd means required by BESTPRED.
- `BestpredTestDayInput` captures BESTPRED-specific test-day fields that FDD
  does not yet expose.
- `BestpredLactationInput` captures fresh date, parity, lactation length,
  previous days open, herd means and test days.
- `format4_record_from_fdd(cow, herd, lactation, ...)` builds the existing
  `Format4Record` boundary consumed by the Python BESTPRED kernel.

This is intentionally an adapter DTO, not a new ontology. It lets us test and
wire the BESTPRED package against FDD identity models today, while keeping a
clear list of fields that should eventually move into upstream
`farm-data-definitions`.

Important adapter rule: BESTPRED still needs a numeric USDA/DHIA-style state
code in the first two characters of `herd_id` for the current age/SCS factor
flow. FDD `Herd.state` can be a free-text region such as `"NY"`, so callers
must pass `state_code` or a `bestpred_herd_id` when `Herd.state` is not already
a numeric BESTPRED state code.

## Upstream FDD Proposal Shape

When we are ready to propose FDD changes, the minimum model set should be:

- `Lactation`: cow/herd reference, calving/fresh date, parity, lactation
  length, previous days open, and optional close/dry-off metadata.
- `TestDay`: lactation reference, DIM, recording date, milk, fat, protein,
  SCS/SCC, supervision/status, milking/weighing/sampling frequency, LER days
  and percent shipped.
- `HerdProductionBaseline`: herd/test-group 305-day milk, fat, protein and SCS
  means used to create BESTPRED herd ratios.
- `LactationPrediction`: output container for 305/365/partial yields,
  mature-equivalent yields, DCR, reliability, persistency and bumpiness.

These should remain unit-explicit. BESTPRED fixtures currently use the legacy
Format-4 scaling conventions, e.g. fat/protein percentages as integer fixed
width fields and SCS as `100 * SCS` inside some C/Fortran boundaries.

## Implication for Bovi Lactationcurve

Once the BESTPRED Python kernel is validated, the Bovi `lactationcurve` package
can migrate from dataframe-only best-predict inputs to shared typed lactation
models. That migration should keep the existing dataframe API as a convenience
wrapper and use typed BESTPRED/FDD models as the canonical internal contract.

Recommended migration path:

1. Keep Bovi's dataframe API as a backwards-compatible wrapper.
2. Convert Bovi dataframe rows into typed FDD/BESTPRED lactation/test-day
   records.
3. Call the validated `bestpred-py` kernel for the official multi-trait path.
4. Return Bovi-friendly dataframe outputs, but preserve typed prediction
   objects internally for traceability.

## Local Comparison CLI

Before changing Bovi, use the comparison CLI in this repo to run the same
source input through:

- the validated `bestpred-py` port, using the current Fortran-compatible
  source records;
- Bovi's existing dataframe `best_predict_method`, using only `TestId`,
  `DaysInMilk` and `MilkingYield`.

Example:

```bash
cd python
uv run bestpred compare-bovi \
  --source 11 \
  --input tests/fixtures/source11_current/DCRexample.txt \
  --par tests/fixtures/source11_current/bestpred.par \
  --limit 10
```

The command prints a row-level table and summary statistics in kilograms. It is
expected that the two implementations can differ materially: Bovi currently
implements a narrower milk-only dataframe method, while `bestpred-py` follows
the original BESTPRED source record preparation and Fortran-compatible kernel.
