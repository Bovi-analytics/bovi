from __future__ import annotations

import json
import math

import pytest

from bestpred.compare_bovi import (
    build_bovi_dataframe_rows,
    compare_records_with_bovi,
    format_comparison_summary,
    format_comparison_table,
    summarize_comparison_rows,
)
from bestpred.core.kernel import LB_PER_KG
from bestpred.core.source11 import simulate_source11_records
from bestpred.io.parameters import read_parameters
from bestpred.io.source11 import read_source11_examples


def test_build_bovi_dataframe_rows_uses_same_source11_test_days(source11_fixture_dir) -> None:
    parameters = read_parameters(source11_fixture_dir / "bestpred.par")
    records = simulate_source11_records(
        read_source11_examples(source11_fixture_dir / "DCRexample.txt"),
        parameters,
    )

    rows = build_bovi_dataframe_rows(records[:1])

    assert len(rows) == 9
    assert rows[0].test_id == "0001:HOUSA.EX.COW.0001:19990401"
    assert rows[0].days_in_milk == 15
    assert rows[0].milking_yield == pytest.approx(77.9 / LB_PER_KG)


def test_compare_records_with_bovi_aligns_305_milk_outputs(source11_fixture_dir) -> None:
    parameters = read_parameters(source11_fixture_dir / "bestpred.par")
    records = simulate_source11_records(
        read_source11_examples(source11_fixture_dir / "DCRexample.txt"),
        parameters,
    )[:1]

    def fake_bovi_runner(rows):
        payload = [row.to_json_row() for row in rows]
        assert payload[0]["TestId"] == "0001:HOUSA.EX.COW.0001:19990401"
        return {"0001:HOUSA.EX.COW.0001:19990401": 10_000.0}

    comparison = compare_records_with_bovi(
        records,
        parameters,
        bovi_runner=fake_bovi_runner,
        source11_compat=True,
    )

    assert len(comparison) == 1
    assert comparison[0].bestpred_305_milk_lb == pytest.approx(21_899.760897, abs=0.01)
    assert comparison[0].bestpred_305_milk_kg == pytest.approx(21_899.760897 / LB_PER_KG)
    assert comparison[0].bovi_305_milk_kg == 10_000.0
    assert comparison[0].delta_kg == pytest.approx(comparison[0].bestpred_305_milk_kg - 10_000.0)


def test_format_comparison_table_is_terminal_friendly(source11_fixture_dir) -> None:
    parameters = read_parameters(source11_fixture_dir / "bestpred.par")
    records = simulate_source11_records(
        read_source11_examples(source11_fixture_dir / "DCRexample.txt"),
        parameters,
    )[:1]
    comparison = compare_records_with_bovi(
        records,
        parameters,
        bovi_runner=lambda rows: {rows[0].test_id: 10_000.0},
        source11_compat=True,
    )

    table = format_comparison_table(comparison)

    assert "bestpred_kg" in table
    assert "bovi_kg" in table
    assert "HOUSA.EX.COW.0001" in table


def test_summarize_comparison_rows_reports_missing_bovi_outputs(source11_fixture_dir) -> None:
    parameters = read_parameters(source11_fixture_dir / "bestpred.par")
    records = simulate_source11_records(
        read_source11_examples(source11_fixture_dir / "DCRexample.txt"),
        parameters,
    )[:2]
    comparison = compare_records_with_bovi(
        records,
        parameters,
        bovi_runner=lambda rows: {rows[0].test_id: 10_000.0},
        source11_compat=True,
    )

    summary = summarize_comparison_rows(comparison)
    text = format_comparison_summary(summary)

    assert summary.total_rows == 2
    assert summary.matched_rows == 1
    assert summary.missing_bovi_rows == 1
    assert summary.mean_abs_delta_kg == pytest.approx(abs(comparison[0].delta_kg))
    assert "missing_bovi_rows: 1" in text


def test_compare_records_with_bovi_allows_missing_bovi_output(source11_fixture_dir) -> None:
    parameters = read_parameters(source11_fixture_dir / "bestpred.par")
    records = simulate_source11_records(
        read_source11_examples(source11_fixture_dir / "DCRexample.txt"),
        parameters,
    )[24:25]

    comparison = compare_records_with_bovi(
        records,
        parameters,
        bovi_runner=lambda rows: {},
        source11_compat=True,
    )

    assert len(comparison) == 1
    assert comparison[0].test_id == "0001:HOUSA.EX.COW.0025:19990401"
    assert math.isnan(comparison[0].bovi_305_milk_kg)
    assert math.isnan(comparison[0].delta_kg)


def test_fake_bovi_runner_contract(tmp_path) -> None:
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "output.json"
    input_path.write_text(
        json.dumps(
            [
                {"TestId": "cow1", "DaysInMilk": 1, "MilkingYield": 10.0},
                {"TestId": "cow1", "DaysInMilk": 2, "MilkingYield": 12.0},
            ]
        ),
        encoding="utf-8",
    )

    rows = json.loads(input_path.read_text(encoding="utf-8"))
    totals: dict[str, float] = {}
    for row in rows:
        totals.setdefault(row["TestId"], 0.0)
        totals[row["TestId"]] += row["MilkingYield"]
    output_path.write_text(
        json.dumps(
            [{"TestId": test_id, "LactationMilkYield": value} for test_id, value in totals.items()]
        ),
        encoding="utf-8",
    )

    assert json.loads(output_path.read_text(encoding="utf-8")) == [
        {"TestId": "cow1", "LactationMilkYield": 22.0}
    ]
