"""Unit tests for the herd stats CSV ingestion utility."""

import pytest
from bovi_api.herd_stats_ingestion import normalize_herd_stats, parse_csv

_COLS = (
    "Achieved21Milk,Achieved305Milk,Achieved75Milk,AchievedMilk,"
    "DaysDry,DaysInMilk,DaysOpen,DaysPregnant,HistoricCalvingInterval,QualitySequence"
)
FULL_HEADER = _COLS
FULL_ROW = "25.0,9000.0,28.0,10000.0,60.0,180.0,100.0,150.0,420.0,0.8"
AGGREGATED_CSV = f"{FULL_HEADER}\n{FULL_ROW}\n".encode()
_ROW1 = "20.0,8000.0,25.0,9000.0,55.0,160.0,90.0,140.0,400.0,0.7"
_ROW2 = "30.0,10000.0,31.0,11000.0,65.0,200.0,110.0,160.0,440.0,0.9"
INDIVIDUAL_CSV = f"{FULL_HEADER}\n{_ROW1}\n{_ROW2}\n".encode()


def test_parse_aggregated_csv():
    raw, fmt, row_count, warnings = parse_csv(AGGREGATED_CSV)
    assert fmt == "aggregated"
    assert row_count == 1
    assert raw["Achieved21Milk"] == pytest.approx(25.0)
    assert warnings == []


def test_parse_individual_csv_computes_mean():
    raw, fmt, row_count, warnings = parse_csv(INDIVIDUAL_CSV)
    assert fmt == "individual"
    assert row_count == 2
    assert raw["Achieved21Milk"] == pytest.approx(25.0)  # mean of 20 and 30
    assert raw["Achieved305Milk"] == pytest.approx(9000.0)  # mean of 8000 and 10000


def test_parse_partial_columns_returns_warning():
    partial = b"Achieved21Milk,Achieved305Milk\n25.0,9000.0\n"
    raw, _, _, warnings = parse_csv(partial)
    assert "Achieved21Milk" in raw
    assert len(warnings) > 0


def test_parse_no_recognised_columns_raises():
    with pytest.raises(ValueError, match="No recognised herd stat columns"):
        parse_csv(b"cow_id,breed\n1,Holstein\n")


def test_parse_unparseable_raises():
    with pytest.raises(ValueError, match="Not a valid CSV"):
        parse_csv(b"\x00\x01\x02\x03")


def test_parse_alias_dim_maps_to_days_in_milk():
    _alias_header = (
        "DIM,Achieved305Milk,Achieved21Milk,Achieved75Milk,AchievedMilk,"
        "DaysDry,DaysOpen,DaysPregnant,HistoricCalvingInterval,QualitySequence"
    )
    _alias_row = "180.0,9000.0,25.0,28.0,10000.0,60.0,100.0,150.0,420.0,0.8"
    alias_csv = f"{_alias_header}\n{_alias_row}\n".encode()
    raw, _, _, _ = parse_csv(alias_csv)
    assert "DaysInMilk" in raw
    assert raw["DaysInMilk"] == pytest.approx(180.0)


def test_parse_row_cap_truncates_with_warning():
    header = f"{FULL_HEADER}\n".encode()
    row = f"{FULL_ROW}\n".encode()
    big_csv = header + row * 5
    _, _, row_count, warnings = parse_csv(big_csv, max_rows=3)
    assert row_count == 3
    assert any("3" in w for w in warnings)


def test_normalize_herd_stats_midpoint():
    ranges = {
        k: (0.0, 100.0)
        for k in [
            "Achieved21Milk",
            "Achieved305Milk",
            "Achieved75Milk",
            "AchievedMilk",
            "DaysDry",
            "DaysInMilk",
            "DaysOpen",
            "DaysPregnant",
            "HistoricCalvingInterval",
            "QualitySequence",
        ]
    }
    raw = {k: 50.0 for k in ranges}
    result = normalize_herd_stats(raw, ranges)
    for val in result.values():
        assert val == pytest.approx(0.5)


def test_normalize_herd_stats_clamps():
    ranges = {
        k: (0.0, 1.0)
        for k in [
            "Achieved21Milk",
            "Achieved305Milk",
            "Achieved75Milk",
            "AchievedMilk",
            "DaysDry",
            "DaysInMilk",
            "DaysOpen",
            "DaysPregnant",
            "HistoricCalvingInterval",
            "QualitySequence",
        ]
    }
    raw = {k: 999.0 for k in ranges}
    result = normalize_herd_stats(raw, ranges)
    for val in result.values():
        assert val == 1.0
