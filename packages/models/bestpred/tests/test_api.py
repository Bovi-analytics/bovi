from __future__ import annotations

import pandas as pd
import pytest

from bestpred.api import dataframe_to_records, predict_dataframe, prediction_from_dcr_row
from bestpred.core.kernel import predict_records
from bestpred.core.source11 import simulate_source11_records
from bestpred.io.parameters import read_parameters
from bestpred.io.source11 import read_source11_examples


def _source11_dataframe(record) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "TestId": 42,
                "AnimalId": record.cow_id,
                "BirthDate": record.birth_date,
                "HerdId": record.herd_id,
                "FreshDate": record.fresh_date,
                "Parity": record.parity,
                "LactationLength": record.length,
                "PreviousDaysOpen": record.previous_days_open,
                "HerdMilk305": record.herd_me_milk,
                "HerdFat305": record.herd_me_fat,
                "HerdProtein305": record.herd_me_protein,
                "HerdSCS305": record.herd_me_scs,
                "DaysInMilk": segment.dim,
                "MilkingYield": segment.milk_yield / 10.0,
                "FatPercent": segment.fat_percent / 10.0,
                "ProteinPercent": segment.protein_percent / 10.0,
                "SCS": segment.scs / 10.0,
                "Supervised": segment.supervised,
                "Status": segment.status,
                "TimesMilked": segment.times_milked,
                "TimesWeighed": segment.times_weighed,
                "TimesSampled": segment.times_sampled,
                "LERDays": segment.ler_days,
                "PercentShipped": segment.percent_shipped,
            }
            for segment in record.segments
        ]
    )


def test_prediction_from_dcr_row_names_all_legacy_fields(source11_fixture_dir) -> None:
    parameters = read_parameters(source11_fixture_dir / "bestpred.par")
    record = simulate_source11_records(
        read_source11_examples(source11_fixture_dir / "DCRexample.txt"),
        parameters,
    )[0]
    row = predict_records([record], parameters, source11_compat=True)[0]

    prediction = prediction_from_dcr_row(row, test_id="example")

    assert prediction.test_id == "example"
    assert prediction.dcr_milk == pytest.approx(row.numeric_values[0])
    assert prediction.milk.yield_305 == pytest.approx(row.numeric_values[3])
    assert prediction.scs.yield_365 == pytest.approx(row.numeric_values[10])
    assert prediction.protein.persistency_reliability == pytest.approx(row.numeric_values[29])
    assert prediction.scs.bumpiness == pytest.approx(row.numeric_values[42])
    assert prediction.to_flat_dict()["MilkYield305"] == pytest.approx(row.numeric_values[3])


def test_dataframe_to_records_preserves_format4_values(source11_fixture_dir) -> None:
    parameters = read_parameters(source11_fixture_dir / "bestpred.par")
    original = simulate_source11_records(
        read_source11_examples(source11_fixture_dir / "DCRexample.txt"),
        parameters,
    )[0]

    converted = dataframe_to_records(_source11_dataframe(original))

    assert converted.test_ids == (42,)
    assert converted.records == (original,)


def test_predict_dataframe_matches_direct_kernel(source11_fixture_dir) -> None:
    parameters = read_parameters(source11_fixture_dir / "bestpred.par")
    record = simulate_source11_records(
        read_source11_examples(source11_fixture_dir / "DCRexample.txt"),
        parameters,
    )[0]
    expected = predict_records([record], parameters, source11_compat=True)[0]

    result = predict_dataframe(
        _source11_dataframe(record),
        parameters,
        source11_compat=True,
    )

    assert result.loc[0, "TestId"] == 42
    assert result.loc[0, "AnimalId"] == record.cow_id
    assert result.loc[0, "MilkYield305"] == pytest.approx(expected.numeric_values[3])
    assert result.loc[0, "DCRMilk"] == pytest.approx(expected.numeric_values[0])
    assert result.loc[0, "SCSBumpiness"] == pytest.approx(expected.numeric_values[42])


def test_dataframe_api_supports_column_mapping(source11_fixture_dir) -> None:
    parameters = read_parameters(source11_fixture_dir / "bestpred.par")
    record = simulate_source11_records(
        read_source11_examples(source11_fixture_dir / "DCRexample.txt"),
        parameters,
    )[0]
    dataframe = _source11_dataframe(record).rename(
        columns={"TestId": "lactation_id", "DaysInMilk": "dim", "MilkingYield": "milk"}
    )

    result = predict_dataframe(
        dataframe,
        parameters,
        column_map={"TestId": "lactation_id", "DaysInMilk": "dim", "MilkingYield": "milk"},
    )

    assert result.loc[0, "TestId"] == 42


def test_dataframe_api_marks_missing_components_as_unsampled(source11_fixture_dir) -> None:
    parameters = read_parameters(source11_fixture_dir / "bestpred.par")
    record = simulate_source11_records(
        read_source11_examples(source11_fixture_dir / "DCRexample.txt"),
        parameters,
    )[0]
    dataframe = _source11_dataframe(record).drop(
        columns=["FatPercent", "ProteinPercent", "SCS", "TimesSampled"]
    )

    converted = dataframe_to_records(dataframe)

    assert all(segment.times_sampled == 0 for segment in converted.records[0].segments)
    assert all(segment.fat_percent == 0 for segment in converted.records[0].segments)


def test_dataframe_api_rejects_duplicate_days(source11_fixture_dir) -> None:
    parameters = read_parameters(source11_fixture_dir / "bestpred.par")
    record = simulate_source11_records(
        read_source11_examples(source11_fixture_dir / "DCRexample.txt"),
        parameters,
    )[0]
    dataframe = _source11_dataframe(record).iloc[[0, 0]]

    with pytest.raises(ValueError, match="duplicate DaysInMilk"):
        dataframe_to_records(dataframe)


def test_dataframe_api_rejects_inconsistent_lactation_fields(source11_fixture_dir) -> None:
    parameters = read_parameters(source11_fixture_dir / "bestpred.par")
    record = simulate_source11_records(
        read_source11_examples(source11_fixture_dir / "DCRexample.txt"),
        parameters,
    )[0]
    dataframe = _source11_dataframe(record)
    dataframe["Parity"] = [1] + [2] * (len(dataframe) - 1)

    with pytest.raises(ValueError, match="must be constant"):
        dataframe_to_records(dataframe)
