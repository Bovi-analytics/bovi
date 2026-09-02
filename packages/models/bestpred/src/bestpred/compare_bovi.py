"""Compare `bestpred-py` output with Bovi's current dataframe best-predict."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from bestpred.core.kernel import LB_PER_KG, predict_records
from bestpred.core.source11 import simulate_source11_records
from bestpred.io.parameters import read_parameters
from bestpred.io.source10 import read_source10_records
from bestpred.io.source11 import read_source11_examples
from bestpred.models import BestpredParameters, BestpredSource, Format4Record

MILK_FIELD_SCALE: Final = 10.0
BESTPRED_305_MILK_INDEX: Final = 3
DEFAULT_BOVI_PATH: Final = Path(__file__).resolve().parents[3] / "lactationcurve"

BOVI_RUNNER_CODE: Final = """
import json
import sys

import pandas as pd
from lactationcurve.characteristics.best_predict import best_predict_method

input_path = sys.argv[1]
output_path = sys.argv[2]
with open(input_path, encoding="utf-8") as handle:
    rows = json.load(handle)

result = best_predict_method(pd.DataFrame(rows))
with open(output_path, "w", encoding="utf-8") as handle:
    json.dump(result.to_dict(orient="records"), handle)
"""


@dataclass(frozen=True)
class BoviInputRow:
    """One row sent to Bovi's current dataframe API."""

    test_id: str
    days_in_milk: int
    milking_yield: float

    def to_json_row(self) -> dict[str, object]:
        return {
            "TestId": self.test_id,
            "DaysInMilk": self.days_in_milk,
            "MilkingYield": self.milking_yield,
        }


@dataclass(frozen=True)
class BoviComparisonRow:
    """One comparable 305-day milk result."""

    index: int
    test_id: str
    animal_id: str
    fresh_date: str
    dim: int
    test_days: int
    bestpred_305_milk_lb: float
    bestpred_305_milk_kg: float
    bovi_305_milk_kg: float

    @property
    def delta_kg(self) -> float:
        return self.bestpred_305_milk_kg - self.bovi_305_milk_kg

    @property
    def delta_percent(self) -> float:
        if self.bovi_305_milk_kg == 0:
            return float("nan")
        return 100.0 * self.delta_kg / self.bovi_305_milk_kg


@dataclass(frozen=True)
class BoviComparisonSummary:
    """Aggregate comparison statistics for terminal output."""

    total_rows: int
    matched_rows: int
    missing_bovi_rows: int
    mean_abs_delta_kg: float
    max_abs_delta_kg: float


BoviRunner = Callable[[Sequence[BoviInputRow]], dict[str, float]]


def read_records_for_source(
    source: BestpredSource,
    *,
    input_path: Path,
    parameter_path: Path,
) -> tuple[list[Format4Record], BestpredParameters]:
    """Read the input source into the same Format4 records used by the Python port."""

    parameters = read_parameters(parameter_path)
    if source == BestpredSource.DCR_EXAMPLE:
        examples = read_source11_examples(input_path)
        return simulate_source11_records(examples, parameters), parameters
    if source == BestpredSource.FORMAT4:
        return read_source10_records(input_path), parameters
    raise NotImplementedError(
        f"Bovi comparison currently supports source 10 and 11, not source {source.value}."
    )


def build_bovi_dataframe_rows(
    records: Sequence[Format4Record],
    *,
    output_unit: str = "kg",
) -> list[BoviInputRow]:
    """Convert BESTPRED test-day segments to Bovi's dataframe schema."""

    rows: list[BoviInputRow] = []
    for record_index, record in enumerate(records, start=1):
        if not record.segments:
            continue
        test_id = comparison_test_id(record_index, record)
        for segment in record.segments:
            milk_lb = segment.milk_yield / MILK_FIELD_SCALE
            milk = milk_lb / LB_PER_KG if output_unit == "kg" else milk_lb
            rows.append(
                BoviInputRow(
                    test_id=test_id,
                    days_in_milk=segment.dim,
                    milking_yield=milk,
                )
            )
    return rows


def comparison_test_id(record_index: int, record: Format4Record) -> str:
    """Stable comparison id that remains unique when cow ids are repeated."""

    return f"{record_index:04d}:{record.cow_id.strip()}:{record.fresh_date}"


def run_bovi_best_predict(
    rows: Sequence[BoviInputRow],
    *,
    bovi_path: Path = DEFAULT_BOVI_PATH,
    bovi_python: Path | None = None,
) -> dict[str, float]:
    """Run Bovi's current dataframe best-predict implementation in a subprocess."""

    python_executable = bovi_python or _default_bovi_python(bovi_path)
    env = os.environ.copy()
    source_path = str(bovi_path / "src")
    env["PYTHONPATH"] = (
        source_path
        if not env.get("PYTHONPATH")
        else f"{source_path}{os.pathsep}{env['PYTHONPATH']}"
    )

    with tempfile.TemporaryDirectory(prefix="bestpred-bovi-compare-") as tmp:
        input_path = Path(tmp) / "bovi_input.json"
        output_path = Path(tmp) / "bovi_output.json"
        input_path.write_text(
            json.dumps([row.to_json_row() for row in rows]),
            encoding="utf-8",
        )
        subprocess.run(
            [
                str(python_executable),
                "-c",
                BOVI_RUNNER_CODE,
                str(input_path),
                str(output_path),
            ],
            check=True,
            env=env,
        )
        results = json.loads(output_path.read_text(encoding="utf-8"))

    return {str(row["TestId"]): float(row["LactationMilkYield"]) for row in results}


def compare_records_with_bovi(
    records: Sequence[Format4Record],
    parameters: BestpredParameters,
    *,
    bovi_runner: BoviRunner,
    source11_compat: bool,
) -> list[BoviComparisonRow]:
    """Compare the same prepared records in `bestpred-py` and Bovi."""

    comparable_records = [record for record in records if record.segments]
    bovi_input_rows = build_bovi_dataframe_rows(comparable_records, output_unit="kg")
    bovi_results = bovi_runner(bovi_input_rows)
    bestpred_rows = predict_records(
        comparable_records,
        parameters,
        source11_compat=source11_compat,
    )

    comparison: list[BoviComparisonRow] = []
    for index, (record, bestpred_row) in enumerate(
        zip(comparable_records, bestpred_rows, strict=True),
        start=1,
    ):
        test_id = comparison_test_id(index, record)
        bestpred_lb = bestpred_row.numeric_values[BESTPRED_305_MILK_INDEX]
        comparison.append(
            BoviComparisonRow(
                index=index,
                test_id=test_id,
                animal_id=record.cow_id.strip(),
                fresh_date=record.fresh_date,
                dim=record.length,
                test_days=len(record.segments),
                bestpred_305_milk_lb=bestpred_lb,
                bestpred_305_milk_kg=bestpred_lb / LB_PER_KG,
                bovi_305_milk_kg=bovi_results.get(test_id, math.nan),
            )
        )
    return comparison


def format_comparison_table(rows: Sequence[BoviComparisonRow], *, limit: int | None = None) -> str:
    """Render a terminal-friendly comparison table."""

    displayed = list(rows[:limit] if limit is not None else rows)
    headers = (
        "#",
        "animal",
        "dim",
        "td",
        "bestpred_kg",
        "bovi_kg",
        "delta_kg",
        "delta_%",
    )
    table_rows = [
        (
            str(row.index),
            row.animal_id,
            str(row.dim),
            str(row.test_days),
            f"{row.bestpred_305_milk_kg:.2f}",
            f"{row.bovi_305_milk_kg:.2f}",
            f"{row.delta_kg:+.2f}",
            f"{row.delta_percent:+.2f}",
        )
        for row in displayed
    ]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in table_rows))
        if table_rows
        else len(headers[index])
        for index in range(len(headers))
    ]
    lines = [
        "  ".join(header.ljust(widths[index]) for index, header in enumerate(headers)),
        "  ".join("-" * width for width in widths),
    ]
    lines.extend(
        "  ".join(value.ljust(widths[index]) for index, value in enumerate(row))
        for row in table_rows
    )
    if limit is not None and len(rows) > limit:
        lines.append(f"... {len(rows) - limit} additional rows omitted")
    return "\n".join(lines)


def summarize_comparison_rows(rows: Sequence[BoviComparisonRow]) -> BoviComparisonSummary:
    """Summarize finite Bovi-vs-BESTPRED deltas."""

    finite_deltas = [
        abs(row.delta_kg)
        for row in rows
        if math.isfinite(row.bovi_305_milk_kg) and math.isfinite(row.delta_kg)
    ]
    missing_bovi_rows = sum(1 for row in rows if not math.isfinite(row.bovi_305_milk_kg))
    if not finite_deltas:
        return BoviComparisonSummary(
            total_rows=len(rows),
            matched_rows=0,
            missing_bovi_rows=missing_bovi_rows,
            mean_abs_delta_kg=math.nan,
            max_abs_delta_kg=math.nan,
        )
    return BoviComparisonSummary(
        total_rows=len(rows),
        matched_rows=len(finite_deltas),
        missing_bovi_rows=missing_bovi_rows,
        mean_abs_delta_kg=sum(finite_deltas) / len(finite_deltas),
        max_abs_delta_kg=max(finite_deltas),
    )


def format_comparison_summary(summary: BoviComparisonSummary) -> str:
    """Render aggregate comparison statistics for the CLI."""

    return "\n".join(
        (
            "",
            "Summary",
            "-------",
            f"rows: {summary.total_rows}",
            f"matched_rows: {summary.matched_rows}",
            f"missing_bovi_rows: {summary.missing_bovi_rows}",
            f"mean_abs_delta_kg: {summary.mean_abs_delta_kg:.2f}",
            f"max_abs_delta_kg: {summary.max_abs_delta_kg:.2f}",
        )
    )


def _default_bovi_python(bovi_path: Path) -> Path:
    candidates = (
        bovi_path / ".venv/bin/python",
        bovi_path.parents[2] / ".venv/bin/python",
        bovi_path.parents[3] / ".venv/bin/python",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return Path(sys.executable)
