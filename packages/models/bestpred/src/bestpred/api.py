"""User-facing Python and pandas interfaces for the BESTPRED kernel."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from bestpred.core.kernel import RESULTS_V2_NUMERIC_FIELD_COUNT, predict_records
from bestpred.io.parameters import read_parameters
from bestpred.models import (
    BestpredModel,
    BestpredParameters,
    DcrResultRow,
    Format4Record,
    TestDaySegment,
)

type TestId = str | int
type ParameterInput = BestpredParameters | str | Path

REQUIRED_COLUMNS = frozenset(
    {
        "TestId",
        "AnimalId",
        "BirthDate",
        "HerdId",
        "FreshDate",
        "Parity",
        "LactationLength",
        "DaysInMilk",
        "MilkingYield",
        "HerdMilk305",
        "HerdFat305",
        "HerdProtein305",
    }
)
LACTATION_COLUMNS = (
    "AnimalId",
    "BirthDate",
    "HerdId",
    "FreshDate",
    "Parity",
    "LactationLength",
    "PreviousDaysOpen",
    "HerdMilk305",
    "HerdFat305",
    "HerdProtein305",
    "HerdSCS305",
)
OPTIONAL_DEFAULTS: dict[str, object] = {
    "PreviousDaysOpen": 140,
    "FatPercent": 0.0,
    "ProteinPercent": 0.0,
    "SCS": 0.0,
    "HerdSCS305": None,
    "Supervised": 2,
    "Status": 0,
    "TimesMilked": 2,
    "TimesWeighed": 2,
    "LERDays": 1,
    "PercentShipped": 100,
}


class TraitPrediction(BestpredModel):
    """Named output metrics for one BESTPRED trait."""

    yield_305: float
    yield_365: float
    yield_lactation: float
    yield_partial: float
    persistency: float
    yield_reliability: float
    persistency_reliability: float
    expanded_yield: float
    herd_305: float
    bumpiness: float


class BestpredPrediction(BestpredModel):
    """Structured representation of one legacy ``results_v2.dcr`` row."""

    test_id: TestId
    animal_id: str
    fresh_date: str
    lactation_length: int
    dcr_milk: float
    dcr_components: float
    dcr_scs: float
    milk: TraitPrediction
    fat: TraitPrediction
    protein: TraitPrediction
    scs: TraitPrediction

    def to_flat_dict(self) -> dict[str, object]:
        """Return a stable, one-row DataFrame representation."""

        row: dict[str, object] = {
            "TestId": self.test_id,
            "AnimalId": self.animal_id,
            "FreshDate": self.fresh_date,
            "LactationLength": self.lactation_length,
            "DCRMilk": self.dcr_milk,
            "DCRComponents": self.dcr_components,
            "DCRSCS": self.dcr_scs,
        }
        for prefix, prediction in (
            ("Milk", self.milk),
            ("Fat", self.fat),
            ("Protein", self.protein),
            ("SCS", self.scs),
        ):
            row.update(
                {
                    f"{prefix}Yield305": prediction.yield_305,
                    f"{prefix}Yield365": prediction.yield_365,
                    f"{prefix}YieldLactation": prediction.yield_lactation,
                    f"{prefix}YieldPartial": prediction.yield_partial,
                    f"{prefix}Persistency": prediction.persistency,
                    f"{prefix}YieldReliability": prediction.yield_reliability,
                    f"{prefix}PersistencyReliability": prediction.persistency_reliability,
                    f"{prefix}ExpandedYield": prediction.expanded_yield,
                    f"{prefix}Herd305": prediction.herd_305,
                    f"{prefix}Bumpiness": prediction.bumpiness,
                }
            )
        return row


@dataclass(frozen=True)
class DataFrameRecords:
    """Format-4 records and their caller-provided DataFrame identifiers."""

    test_ids: tuple[TestId, ...]
    records: tuple[Format4Record, ...]


def prediction_from_dcr_row(
    row: DcrResultRow,
    *,
    test_id: TestId | None = None,
) -> BestpredPrediction:
    """Convert the fixed legacy output positions to named metrics."""

    values = row.numeric_values
    if len(values) != RESULTS_V2_NUMERIC_FIELD_COUNT:
        raise ValueError(
            "BESTPRED DCR rows must contain "
            f"{RESULTS_V2_NUMERIC_FIELD_COUNT} numeric values, got {len(values)}"
        )

    def trait(index: int) -> TraitPrediction:
        return TraitPrediction(
            yield_305=values[3 + index],
            yield_365=values[7 + index],
            yield_lactation=values[11 + index],
            yield_partial=values[15 + index],
            persistency=values[19 + index],
            yield_reliability=values[23 + index],
            persistency_reliability=values[27 + index],
            expanded_yield=values[31 + index],
            herd_305=values[35 + index],
            bumpiness=values[39 + index],
        )

    return BestpredPrediction(
        test_id=row.animal_id if test_id is None else test_id,
        animal_id=row.animal_id,
        fresh_date=row.fresh_date,
        lactation_length=row.dim,
        dcr_milk=values[0],
        dcr_components=values[1],
        dcr_scs=values[2],
        milk=trait(0),
        fat=trait(1),
        protein=trait(2),
        scs=trait(3),
    )


def dataframe_to_records(
    dataframe: pd.DataFrame,
    *,
    column_map: Mapping[str, str] | None = None,
) -> DataFrameRecords:
    """Validate long-form test-day data and build Format-4 kernel records.

    ``column_map`` maps canonical BESTPRED column names to caller column names.
    Milk, component percentages, and SCS are accepted as normal decimal values
    and converted to the fixed-point Format-4 representation internally.
    """

    frame = _canonicalize_columns(dataframe, column_map)
    missing = sorted(REQUIRED_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing required BESTPRED DataFrame columns: {', '.join(missing)}")
    if frame.empty:
        raise ValueError("BESTPRED DataFrame input must contain at least one test-day row")
    required_with_nulls = sorted(
        column for column in REQUIRED_COLUMNS if bool(frame[column].isna().to_numpy().any())
    )
    if required_with_nulls:
        raise ValueError(
            "Required BESTPRED DataFrame columns contain missing values: "
            + ", ".join(required_with_nulls)
        )

    component_columns_supplied = any(
        column in frame.columns for column in ("FatPercent", "ProteinPercent", "SCS")
    )
    frame = frame.copy()
    for column, default in OPTIONAL_DEFAULTS.items():
        if column not in frame.columns:
            frame[column] = default
    if "TimesSampled" not in frame.columns:
        frame["TimesSampled"] = 2 if component_columns_supplied else 0

    test_ids: list[TestId] = []
    records: list[Format4Record] = []
    for raw_test_id, group in frame.groupby("TestId", sort=False, dropna=False):
        test_id = _normalize_test_id(raw_test_id)
        constants = {
            column: _constant_group_value(group, test_id, column) for column in LACTATION_COLUMNS
        }
        dims = [_nonnegative_int(value, "DaysInMilk") for value in group["DaysInMilk"]]
        if len(dims) != len(set(dims)):
            raise ValueError(f"TestId {test_id!r} contains duplicate DaysInMilk values")

        segments = tuple(
            TestDaySegment(
                dim=_nonnegative_int(row["DaysInMilk"], "DaysInMilk"),
                supervised=_nonnegative_int(row["Supervised"], "Supervised"),
                status=_nonnegative_int(row["Status"], "Status"),
                times_milked=_nonnegative_int(row["TimesMilked"], "TimesMilked"),
                times_weighed=_nonnegative_int(row["TimesWeighed"], "TimesWeighed"),
                times_sampled=_nonnegative_int(row["TimesSampled"], "TimesSampled"),
                ler_days=_nonnegative_int(row["LERDays"], "LERDays"),
                percent_shipped=_nonnegative_int(row["PercentShipped"], "PercentShipped"),
                milk_yield=_scaled_nonnegative(row["MilkingYield"], "MilkingYield"),
                fat_percent=_scaled_nonnegative(row["FatPercent"], "FatPercent"),
                protein_percent=_scaled_nonnegative(row["ProteinPercent"], "ProteinPercent"),
                scs=_scaled_nonnegative(row["SCS"], "SCS"),
            )
            for _, row in group.sort_values("DaysInMilk").iterrows()
        )
        records.append(
            Format4Record(
                cow_id=str(constants["AnimalId"]),
                birth_date=_format_date(constants["BirthDate"], "BirthDate"),
                herd_id=str(constants["HerdId"]),
                fresh_date=_format_date(constants["FreshDate"], "FreshDate"),
                parity=_positive_int(constants["Parity"], "Parity"),
                length=_nonnegative_int(constants["LactationLength"], "LactationLength"),
                previous_days_open=_nonnegative_int(
                    constants["PreviousDaysOpen"], "PreviousDaysOpen"
                ),
                herd_me_milk=_nonnegative_int(constants["HerdMilk305"], "HerdMilk305"),
                herd_me_fat=_nonnegative_int(constants["HerdFat305"], "HerdFat305"),
                herd_me_protein=_nonnegative_int(constants["HerdProtein305"], "HerdProtein305"),
                herd_me_scs=_optional_nonnegative_float(constants["HerdSCS305"], "HerdSCS305"),
                segments=segments,
            )
        )
        test_ids.append(test_id)

    return DataFrameRecords(test_ids=tuple(test_ids), records=tuple(records))


def predict_dataframe(
    dataframe: pd.DataFrame,
    parameters: ParameterInput,
    *,
    column_map: Mapping[str, str] | None = None,
    source11_compat: bool = False,
) -> pd.DataFrame:
    """Predict BESTPRED metrics for one or more long-form lactations."""

    if isinstance(parameters, BestpredParameters):
        resolved_parameters = parameters
    else:
        resolved_parameters = read_parameters(Path(parameters))
    converted = dataframe_to_records(dataframe, column_map=column_map)
    rows = predict_records(
        list(converted.records),
        resolved_parameters,
        source11_compat=source11_compat,
    )
    predictions = (
        prediction_from_dcr_row(row, test_id=test_id)
        for test_id, row in zip(converted.test_ids, rows, strict=True)
    )
    return pd.DataFrame(prediction.to_flat_dict() for prediction in predictions)


def _canonicalize_columns(
    dataframe: pd.DataFrame,
    column_map: Mapping[str, str] | None,
) -> pd.DataFrame:
    if column_map is None:
        return dataframe
    known_columns = REQUIRED_COLUMNS | set(OPTIONAL_DEFAULTS) | {"TimesSampled"}
    unknown = sorted(set(column_map).difference(known_columns))
    if unknown:
        raise ValueError(f"Unknown canonical BESTPRED columns in column_map: {', '.join(unknown)}")
    caller_columns = list(column_map.values())
    if len(caller_columns) != len(set(caller_columns)):
        raise ValueError("column_map must not map multiple canonical columns to one caller column")
    return dataframe.rename(columns={caller: canonical for canonical, caller in column_map.items()})


def _constant_group_value(group: pd.DataFrame, test_id: TestId, column: str) -> object:
    values = group[column].drop_duplicates()
    if len(values) != 1:
        raise ValueError(f"{column} must be constant within TestId {test_id!r}")
    return values.iloc[0]


def _normalize_test_id(value: object) -> TestId:
    if hasattr(value, "item"):
        value = value.item()  # type: ignore[union-attr]
    if isinstance(value, (str, int)) and not isinstance(value, bool):
        return value
    return str(value)


def _format_date(value: object, column: str) -> str:
    if isinstance(value, pd.Timestamp):
        value = value.to_pydatetime()
    if isinstance(value, (date, datetime)):
        return value.strftime("%Y%m%d")
    text = str(value).strip()
    for format_string in ("%Y%m%d", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, format_string).strftime("%Y%m%d")
        except ValueError:
            continue
    raise ValueError(f"{column} must be a date or YYYYMMDD/ISO date string")


def _finite_float(value: object, column: str) -> float:
    try:
        numeric = float(str(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{column} must be numeric") from exc
    if not math.isfinite(numeric):
        raise ValueError(f"{column} must be finite")
    return numeric


def _nonnegative_int(value: object, column: str) -> int:
    numeric = _finite_float(value, column)
    if numeric < 0 or not numeric.is_integer():
        raise ValueError(f"{column} must be a non-negative integer")
    return int(numeric)


def _positive_int(value: object, column: str) -> int:
    numeric = _nonnegative_int(value, column)
    if numeric < 1:
        raise ValueError(f"{column} must be at least 1")
    return numeric


def _scaled_nonnegative(value: object, column: str) -> int:
    numeric = _finite_float(value, column)
    if numeric < 0:
        raise ValueError(f"{column} must be non-negative")
    return round(numeric * 10.0)


def _optional_nonnegative_float(value: object, column: str) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(str(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{column} must be numeric") from exc
    if math.isnan(numeric):
        return None
    if not math.isfinite(numeric):
        raise ValueError(f"{column} must be finite")
    if numeric < 0:
        raise ValueError(f"{column} must be non-negative")
    return numeric
