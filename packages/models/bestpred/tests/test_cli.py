from __future__ import annotations

import stat

import pytest

from bestpred.cli import main


def _assert_pcdart_output_matches(actual: str, expected: str) -> None:
    actual_lines = actual.splitlines()
    expected_lines = expected.splitlines()
    assert len(actual_lines) == len(expected_lines)

    for line_number, (actual_line, expected_line) in enumerate(
        zip(actual_lines, expected_lines, strict=True),
        start=1,
    ):
        actual_parts = actual_line.split()
        expected_parts = expected_line.split()
        assert actual_parts[:3] == expected_parts[:3], line_number
        assert len(actual_parts) == len(expected_parts), line_number
        for actual_token, expected_token in zip(actual_parts[3:], expected_parts[3:], strict=True):
            assert float(actual_token) == pytest.approx(float(expected_token), abs=1.01), (
                line_number,
                actual_line,
                expected_line,
            )


@pytest.mark.golden
def test_cli_source11_oracle_output(source11_fixture_dir, tmp_path):
    output = tmp_path / "results_v2.dcr"

    exit_code = main(
        [
            "run",
            "--source",
            "11",
            "--input",
            str(source11_fixture_dir / "DCRexample.txt"),
            "--par",
            str(source11_fixture_dir / "bestpred.par"),
            "--output",
            str(output),
            "--oracle-output",
            str(source11_fixture_dir / "results_v2.dcr"),
        ]
    )

    assert exit_code == 0
    assert output.read_text() == (source11_fixture_dir / "results_v2.dcr").read_text()


def test_cli_non_source11_returns_not_implemented(source11_fixture_dir, tmp_path, capsys):
    exit_code = main(
        [
            "run",
            "--source",
            "12",
            "--input",
            str(source11_fixture_dir / "DCRexample.txt"),
            "--par",
            str(source11_fixture_dir / "bestpred.par"),
            "--output",
            str(tmp_path / "results_v2.dcr"),
        ]
    )

    assert exit_code == 2
    assert "Source 12 is not implemented yet" in capsys.readouterr().err


def test_cli_source10_without_oracle_writes_python_output(source10_fixture_dir, tmp_path):
    output = tmp_path / "results_v2.dcr"

    exit_code = main(
        [
            "run",
            "--source",
            "10",
            "--input",
            str(source10_fixture_dir / "format4.dat"),
            "--par",
            str(source10_fixture_dir / "bestpred.par"),
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    text = output.read_text(encoding="utf-8", errors="replace")
    assert "HOUSA00093WNM4798" in text
    assert "27817" in text
    assert "1025" in text


def test_cli_source15_without_oracle_writes_python_output(source15_fixture_dir, tmp_path):
    output = tmp_path / "results_v2.dcr"
    input_path = tmp_path / "format4.dat"
    means_path = tmp_path / "format4.means"
    input_path.write_text((source15_fixture_dir / "format4.dat").read_text(), encoding="utf-8")
    means_path.write_text((source15_fixture_dir / "format4.means").read_text(), encoding="utf-8")

    exit_code = main(
        [
            "run",
            "--source",
            "15",
            "--input",
            str(input_path),
            "--par",
            str(source15_fixture_dir / "bestpred.par"),
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    text = output.read_text(encoding="utf-8", errors="replace")
    assert "HOUSA00093WNM4798" in text
    assert "24000" in text
    assert "27680" in text


def test_cli_source14_without_oracle_writes_python_output(source14_fixture_dir, tmp_path):
    output = tmp_path / "results_v2.dcr"
    pcdart_output = tmp_path / "pcdart.bpo"

    exit_code = main(
        [
            "run",
            "--source",
            "14",
            "--input",
            str(source14_fixture_dir / "test241.txt"),
            "--par",
            str(source14_fixture_dir / "bestpred.par"),
            "--output",
            str(output),
            "--pcdart-output",
            str(pcdart_output),
        ]
    )

    assert exit_code == 0
    text = output.read_text(encoding="utf-8", errors="replace")
    assert "1000001" in text
    assert "22154" in text
    assert "26159" in text
    _assert_pcdart_output_matches(
        pcdart_output.read_text(encoding="utf-8"),
        (source14_fixture_dir / "pcdart.bpo").read_text(encoding="utf-8"),
    )


def test_cli_source24_without_oracle_writes_python_output(source24_fixture_dir, tmp_path):
    output = tmp_path / "results_v2.dcr"
    pcdart_output = tmp_path / "pcdart.bpo"
    list_path = tmp_path / "pcdart_files.txt"
    test241_path = tmp_path / "test241.txt"
    test242_path = tmp_path / "test242.txt"
    list_path.write_text((source24_fixture_dir / "pcdart_files.txt").read_text(), encoding="utf-8")
    test241_path.write_text((source24_fixture_dir / "test241.txt").read_text(), encoding="utf-8")
    test242_path.write_text((source24_fixture_dir / "test242.txt").read_text(), encoding="utf-8")

    exit_code = main(
        [
            "run",
            "--source",
            "24",
            "--input",
            str(list_path),
            "--par",
            str(source24_fixture_dir / "bestpred.par"),
            "--output",
            str(output),
            "--pcdart-output",
            str(pcdart_output),
        ]
    )

    assert exit_code == 0
    text = output.read_text(encoding="utf-8", errors="replace")
    assert "1000001" in text
    assert "2000001" in text
    assert "29282" in text
    _assert_pcdart_output_matches(
        pcdart_output.read_text(encoding="utf-8"),
        (source24_fixture_dir / "pcdart.bpo").read_text(encoding="utf-8"),
    )


def test_cli_source11_without_oracle_writes_partial_python_output(source11_fixture_dir, tmp_path):
    output = tmp_path / "results_v2.dcr"
    input_file = tmp_path / "DCRexample-mini.txt"
    input_file.write_text(
        "    1      2      2     2      1      30     15  255   5  255-day RIP\n",
        encoding="utf-8",
    )

    exit_code = main(
        [
            "run",
            "--source",
            "11",
            "--input",
            str(input_file),
            "--par",
            str(source11_fixture_dir / "bestpred.par"),
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    text = output.read_text()
    assert "HOUSA.EX.COW.0001" in text
    assert "21900" in text
    assert "3.08" in text
    assert "0.29" in text
    assert "nan" not in text


def test_cli_compare_bovi_uses_same_source11_input(source11_fixture_dir, tmp_path, capsys):
    fake_bovi_python = tmp_path / "fake_bovi_python"
    fake_bovi_python.write_text(
        """#!/usr/bin/env python3
import json
import sys

input_path = sys.argv[-2]
output_path = sys.argv[-1]
rows = json.load(open(input_path, encoding="utf-8"))
totals = {}
for row in rows:
    totals.setdefault(row["TestId"], 0.0)
    totals[row["TestId"]] += row["MilkingYield"]
json.dump(
    [{"TestId": test_id, "LactationMilkYield": value} for test_id, value in totals.items()],
    open(output_path, "w", encoding="utf-8"),
)
""",
        encoding="utf-8",
    )
    fake_bovi_python.chmod(fake_bovi_python.stat().st_mode | stat.S_IXUSR)
    input_file = tmp_path / "DCRexample-mini.txt"
    input_file.write_text(
        "    1      2      2     2      1      30     15  255   5  255-day RIP\n",
        encoding="utf-8",
    )

    exit_code = main(
        [
            "compare-bovi",
            "--source",
            "11",
            "--input",
            str(input_file),
            "--par",
            str(source11_fixture_dir / "bestpred.par"),
            "--bovi-python",
            str(fake_bovi_python),
        ]
    )

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "bestpred_kg" in output
    assert "bovi_kg" in output
    assert "Summary" in output
    assert "matched_rows: 1" in output
    assert "HOUSA.EX.COW.0001" in output
