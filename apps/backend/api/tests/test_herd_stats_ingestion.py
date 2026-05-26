"""Unit tests for the herd stats CSV ingestion utility."""

from pathlib import Path

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

COW_DATA_DIR = Path(__file__).resolve().parents[4] / "data" / "cow_data"


def test_parse_aggregated_csv():
    result = parse_csv(AGGREGATED_CSV)
    assert result.format_detected == "aggregated"
    assert result.row_count == 1
    assert result.raw_stats["Achieved21Milk"] == pytest.approx(25.0)
    assert result.cow_count is None
    assert result.detected_parity is None


def test_parse_aggregated_multirow_computes_mean():
    result = parse_csv(INDIVIDUAL_CSV)
    assert result.format_detected == "aggregated"
    assert result.row_count == 2
    assert result.raw_stats["Achieved21Milk"] == pytest.approx(25.0)  # mean of 20 and 30
    assert result.raw_stats["Achieved305Milk"] == pytest.approx(9000.0)  # mean of 8000 and 10000


def test_parse_partial_columns_returns_warning():
    partial = b"Achieved21Milk,Achieved305Milk\n25.0,9000.0\n"
    result = parse_csv(partial)
    assert "Achieved21Milk" in result.raw_stats
    assert len(result.warnings) > 0


def test_parse_no_recognised_format_raises():
    with pytest.raises(ValueError, match="Could not detect CSV format"):
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
    result = parse_csv(alias_csv)
    assert "DaysInMilk" in result.raw_stats
    assert result.raw_stats["DaysInMilk"] == pytest.approx(180.0)


def test_benchmark_test_day_columns_detect_as_test_day_with_mapping_review():
    csv_bytes = (
        b"TestId,herd_id,parity,dim,milk_kg\n"
        b"cow1,2942694,2,10,25.5\n"
        b"cow1,2942694,2,20,28.0\n"
        b"cow2,2942694,3,12,30.0\n"
        b"cow2,2942694,3,24,33.0\n"
    )

    result = parse_csv(csv_bytes)

    assert result.format_detected == "icar_test_day"
    assert result.mapping_required is True
    assert result.column_mapping == {
        "cow_id": "TestId",
        "dim": "dim",
        "milk_kg": "milk_kg",
        "parity": "parity",
        "herd_id": "herd_id",
    }
    assert result.cow_count == 2
    assert result.cows is not None
    assert result.cows[0].cow_id == "cow1"


def test_parse_test_day_with_explicit_column_mapping():
    csv_bytes = (
        b"lactation,days,milk,par\ncow1,10,25.5,2\ncow1,20,28.0,2\ncow2,12,30.0,3\ncow2,24,33.0,3\n"
    )

    result = parse_csv(
        csv_bytes,
        column_mapping={
            "cow_id": "lactation",
            "dim": "days",
            "milk_kg": "milk",
            "parity": "par",
        },
    )

    assert result.format_detected == "icar_test_day"
    assert result.mapping_required is False
    assert result.column_mapping == {
        "cow_id": "lactation",
        "dim": "days",
        "milk_kg": "milk",
        "parity": "par",
    }
    assert result.cow_count == 2
    assert result.detected_parity in (2, 3)


def test_parse_row_cap_truncates_with_warning():
    header = f"{FULL_HEADER}\n".encode()
    row = f"{FULL_ROW}\n".encode()
    big_csv = header + row * 5
    result = parse_csv(big_csv, max_rows=3)
    assert result.row_count == 3
    assert any("3" in w for w in result.warnings)


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


# ---------------------------------------------------------------------------
# Test-day formats
# ---------------------------------------------------------------------------

ICAR_CSV = (
    b"TestId,TestDate,EventType,CalvingDate,BirthDate,Parity,DaysInMilk,DailyMilkingYield\n"
    b"1,2024-01-01,MilkRecording,2023-12-01,2021-01-01,2,15,30.0\n"
    b"1,2024-01-28,MilkRecording,2023-12-01,2021-01-01,2,42,40.0\n"
    b"1,2024-02-25,MilkRecording,2023-12-01,2021-01-01,2,70,38.0\n"
    b"2,2024-01-01,MilkRecording,2023-12-05,2020-03-01,3,10,25.0\n"
    b"2,2024-01-28,MilkRecording,2023-12-05,2020-03-01,3,38,35.0\n"
    b"2,2024-02-25,MilkRecording,2023-12-05,2020-03-01,3,66,33.0\n"
)


def test_parse_icar_returns_per_cow_records():
    result = parse_csv(ICAR_CSV)
    assert result.cows is not None
    assert len(result.cows) == 2
    cow_ids = {c.cow_id for c in result.cows}
    assert cow_ids == {"1", "2"}
    cow1 = next(c for c in result.cows if c.cow_id == "1")
    assert cow1.parity == 2
    assert cow1.dim == [15, 42, 70]
    assert cow1.milk_kg == [30.0, 40.0, 38.0]


def test_parse_aggregated_has_empty_cows_list():
    result = parse_csv(AGGREGATED_CSV)
    assert result.cows is None or result.cows == []


def test_parse_icar_detects_format_and_aggregates():
    result = parse_csv(ICAR_CSV)
    assert result.format_detected == "icar_test_day"
    assert result.cow_count == 2
    assert result.detected_parity in (2, 3)  # tie between cows; Counter keeps first seen
    assert result.row_count == 6
    # AchievedMilk = herd-average of cumulative trapezoidal integral across observed range.
    # Cow 1 (DIM 15→42→70, yield 30→40→38):
    #   ½·15·30 + ½·27·(30+40) + ½·28·(40+38) = 225+945+1092 = 2262
    # Cow 2 (DIM 10→38→66, yield 25→35→33):
    #   ½·10·25 + ½·28·(25+35) + ½·28·(35+33) = 125+840+952 = 1917
    assert result.raw_stats["AchievedMilk"] == pytest.approx((2262 + 1917) / 2)
    # Achieved21Milk: within ±7 days of 21, cow 1 has DIM 15, cow 2 has DIM 10 → mean
    assert "Achieved21Milk" in result.raw_stats
    # DaysInMilk: max per cow (70, 66) → mean = 68
    assert result.raw_stats["DaysInMilk"] == pytest.approx(68.0)
    # Achieved305Milk estimated via trapezoid - just sanity-check it is a large number
    assert result.raw_stats["Achieved305Milk"] > 0


def test_parse_icar_skips_non_milk_events():
    csv_with_events = ICAR_CSV + (b"1,2024-03-01,Calving,2024-03-01,2021-01-01,3,0,0\n")
    result = parse_csv(csv_with_events)
    assert any("EventType" in w for w in result.warnings)
    # AchievedMilk unchanged since the Calving row was skipped (cumulative trapezoid per cow)
    assert result.raw_stats["AchievedMilk"] == pytest.approx((2262 + 1917) / 2)


DAIRYCOM_CSV = (
    b'"ID";"TestDate";"DIM";"MILK";"PCTF";"PCTP";"FCM";"305ME";"RELV";"SCC";"LS";"PEN";\n'
    b"     101 ;09/27/24; 20 ; 90  ;  3,1 ;  3,0 ; 91 ;22000 ;  97 ;   22 ;0,8 ;  6 ;\n"
    b"     101 ;10/25/24; 75 ; 85  ;  3,9 ;  3,1 ;101 ;22500 ; 101 ;   54 ;2,1 ;  6 ;\n"
    b"     101 ;11/22/24;200 ; 55* ;    0 ;    0 ;  0 ;23000 ; 104 ;    0 ;  0 ; 15 ;\n"
    b"     202 ;09/27/24; 25 ; 75  ;  3,2 ;  3,1 ; 81 ;20000 ;  95 ;   25 ;1,0 ;  6 ;\n"
    b"     202 ;10/25/24; 70 ; 70  ;  3,8 ;  3,0 ; 90 ;20200 ;  99 ;   40 ;1,5 ;  6 ;\n"
)


def test_parse_dairycom_detects_and_converts_lbs_to_kg():
    result = parse_csv(DAIRYCOM_CSV, allow_dairy_comp=True)
    assert result.format_detected == "dairycom_test_day"
    assert result.cow_count == 2
    assert result.detected_parity is None
    assert any("lbs to kg" in w for w in result.warnings)
    # AchievedMilk: cumulative trapezoid of MILK (lbs→kg), not daily mean
    assert result.raw_stats["AchievedMilk"] > 1000  # plausible cumulative kg
    # 305ME present on every row → Achieved305Milk is mean of latest 305ME per cow, in kg
    # Cow 101 latest: 23000 lbs; Cow 202 latest: 20200 lbs → mean = 21600 lbs * lbs->kg
    assert result.raw_stats["Achieved305Milk"] == pytest.approx((23000 + 20200) / 2 * 0.45359237)


def test_parse_dairycom_strips_star_flags():
    result = parse_csv(DAIRYCOM_CSV, allow_dairy_comp=True)
    # The "55*" cell in the fixture should still be included (flag stripped)
    # If it were excluded, cow 101's cumulative yield would drop; ensure the value is plausible
    assert result.raw_stats["AchievedMilk"] > 1000


def test_parse_dairycom_is_disabled_by_default():
    with pytest.raises(ValueError, match="Dairy Comp uploads are temporarily disabled"):
        parse_csv(DAIRYCOM_CSV)


# ---------------------------------------------------------------------------
# Real fixture files (skip if the dataset is not checked out locally)
# ---------------------------------------------------------------------------

ICAR_FIXTURE = COW_DATA_DIR / "TestDataSet(in).csv"
DAIRYCOM_FIXTURE = COW_DATA_DIR / "CDREC_TEST_DAY_MILK_.CSV"


@pytest.mark.skipif(not ICAR_FIXTURE.exists(), reason="ICAR fixture not present")
def test_real_icar_dataset():
    result = parse_csv(ICAR_FIXTURE.read_bytes())
    assert result.format_detected == "icar_test_day"
    assert result.cow_count == 407
    assert result.detected_parity is not None and 1 <= result.detected_parity <= 7
    # Cumulative lactation yield per cow (kg) - real herds fall 5k–15k
    assert 3000 < result.raw_stats["AchievedMilk"] < 20000
    assert 5000 < result.raw_stats["Achieved305Milk"] < 15000
    # Per-cow records returned alongside aggregates
    assert result.cows is not None
    assert len(result.cows) == result.cow_count
    sample = result.cows[0]
    assert len(sample.dim) == len(sample.milk_kg) >= 4
    assert sample.parity is not None and 1 <= sample.parity <= 7
    assert all(0 <= d <= 600 for d in sample.dim)
    assert all(0 <= y <= 80 for y in sample.milk_kg)


@pytest.mark.skipif(not DAIRYCOM_FIXTURE.exists(), reason="Dairy Comp fixture not present")
def test_real_dairycom_dataset():
    result = parse_csv(DAIRYCOM_FIXTURE.read_bytes(), allow_dairy_comp=True)
    assert result.format_detected == "dairycom_test_day"
    # Some cows in this export only have zero-milk rows and get filtered out;
    # we expect close to the 1011 unique IDs in the file.
    assert result.cow_count is not None
    assert 1000 <= result.cow_count <= 1011
    assert any("lbs to kg" in w for w in result.warnings)
    # Cumulative yield per cow in kg after lbs→kg conversion
    assert 3000 < result.raw_stats["AchievedMilk"] < 20000
    assert 5000 < result.raw_stats["Achieved305Milk"] < 20000
