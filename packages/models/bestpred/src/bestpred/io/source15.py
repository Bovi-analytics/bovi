"""Parser for source-15 `format4.dat` plus `format4.means`."""

from __future__ import annotations

from pathlib import Path

from bestpred.io.source10 import (
    build_current_fortran_source10_rows,
    parse_source10_record_line,
)
from bestpred.models import Format4MeansRecord, Format4Record


def _slice(text: str, start: int, stop: int) -> str:
    """1-based inclusive slicing like the Fortran fixed-width formats."""

    return text[start - 1 : stop]


def _parse_int(field: str) -> int:
    stripped = field.strip()
    return int(stripped) if stripped else 0


def parse_source15_means_line(line: str) -> Format4MeansRecord:
    """Parse one source-15 means row.

    Fortran format:
    `a17,1x,a8,1x,f5.0,1x,f4.0,1x,f4.0,2x,f3.0`
    """

    return Format4MeansRecord(
        cow_id=_slice(line, 1, 17),
        fresh_date=_slice(line, 19, 26),
        herd_me_milk=_parse_int(_slice(line, 28, 32)),
        herd_me_fat=_parse_int(_slice(line, 34, 37)),
        herd_me_protein=_parse_int(_slice(line, 39, 42)),
        herd_me_scs=_parse_int(_slice(line, 45, 47)) / 100.0,
    )


def _resolve_current_fortran_means(
    record: Format4Record,
    means: Format4MeansRecord,
) -> Format4MeansRecord:
    """Mirror current Fortran source-15 mismatch behavior.

    `bestpred_main.f90` zeroes the means when cow ID or fresh date do not
    match. The Python port mirrors that state instead of raising.
    """

    if record.cow_id == means.cow_id and record.fresh_date == means.fresh_date:
        return means

    return Format4MeansRecord(
        cow_id=record.cow_id,
        fresh_date=record.fresh_date,
        herd_me_milk=0,
        herd_me_fat=0,
        herd_me_protein=0,
        herd_me_scs=0.0,
    )


def read_source15_records(data_path: Path, means_path: Path) -> list[Format4Record]:
    """Read source-15 rows using the current Fortran two-row flow."""

    data_lines = [
        line.rstrip("\n")
        for line in data_path.read_text(encoding="utf-8", errors="replace").splitlines()
        if line.strip()
    ]
    means_lines = [
        line.rstrip("\n")
        for line in means_path.read_text(encoding="utf-8", errors="replace").splitlines()
        if line.strip()
    ]

    if len(means_lines) < len(data_lines):
        raise ValueError(
            "source 15 requires at least one non-empty means row per format4 row: "
            f"{len(data_lines)} data rows, {len(means_lines)} means rows"
        )

    records: list[Format4Record] = []
    for raw_line, means_line in zip(data_lines, means_lines, strict=False):
        parsed = parse_source10_record_line(raw_line)
        means = _resolve_current_fortran_means(parsed, parse_source15_means_line(means_line))
        records.extend(build_current_fortran_source10_rows(parsed, detail_means=means))
    return records
