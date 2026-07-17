"""Parser and compatibility writer for BESTPRED DCR output files."""

from __future__ import annotations

import math
from pathlib import Path

from bestpred.models import DcrResultRow


def _parse_float(token: str) -> float:
    if token.lower() in {"nan", "+nan", "-nan"}:
        return math.nan
    return float(token)


def parse_dcr_result_line(line: str) -> DcrResultRow:
    """Parse one whitespace-tokenized DCR output row."""

    parts = line.split()
    if len(parts) < 4:
        raise ValueError(f"Invalid DCR output row: {line!r}")
    return DcrResultRow(
        animal_id=parts[0],
        fresh_date=parts[1],
        dim=int(parts[2]),
        numeric_values=tuple(_parse_float(token) for token in parts[3:]),
        raw_line=line.rstrip("\n"),
    )


def read_dcr_results(path: Path) -> list[DcrResultRow]:
    """Read `results_v2.dcr`-style rows."""

    return [
        parse_dcr_result_line(line)
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
        if line.strip()
    ]


def write_compatibility_dcr(path: Path, rows: list[DcrResultRow]) -> None:
    """Write rows using their original Fortran-compatible raw lines.

    This writer is intended for golden-test/debug compatibility. Production
    users should consume the structured models instead.
    """

    path.write_text("".join(f"{row.raw_line}\n" for row in rows), encoding="utf-8")
