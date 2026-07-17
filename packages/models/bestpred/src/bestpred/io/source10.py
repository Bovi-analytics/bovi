"""Parser helpers for source-10/source-15 `format4.dat` records."""

from __future__ import annotations

from pathlib import Path

from bestpred.models import Format4MeansRecord, Format4Record, TestDaySegment


def _slice(text: str, start: int, stop: int) -> str:
    """1-based inclusive slicing like the Fortran format offsets."""

    return text[start - 1 : stop]


def _parse_int(field: str) -> int:
    stripped = field.strip()
    return int(stripped) if stripped else 0


def _parse_segment(segment: str) -> TestDaySegment:
    return TestDaySegment(
        dim=_parse_int(_slice(segment, 1, 3)),
        supervised=_parse_int(_slice(segment, 4, 4)),
        status=_parse_int(_slice(segment, 5, 5)),
        times_milked=_parse_int(_slice(segment, 6, 6)),
        times_weighed=_parse_int(_slice(segment, 7, 7)),
        times_sampled=_parse_int(_slice(segment, 8, 8)),
        ler_days=_parse_int(_slice(segment, 9, 10)),
        percent_shipped=_parse_int(_slice(segment, 11, 13)),
        milk_yield=_parse_int(_slice(segment, 14, 17)),
        fat_percent=_parse_int(_slice(segment, 18, 19)),
        protein_percent=_parse_int(_slice(segment, 20, 21)),
        scs=_parse_int(_slice(segment, 22, 23)),
    )


def parse_source10_record_line(line: str) -> Format4Record:
    cow_id = _slice(line, 3, 19)
    birth_date = _slice(line, 71, 78)
    herd_id = _slice(line, 107, 114)
    fresh_date = _slice(line, 128, 135)
    header_length = _parse_int(_slice(line, 136, 138))
    parity = _parse_int(_slice(line, 159, 160))
    previous_days_open = _parse_int(_slice(line, 246, 248))
    segment_count = _parse_int(_slice(line, 249, 250))
    me_milk = _parse_int(_slice(line, 188, 192))
    me_fat = _parse_int(_slice(line, 193, 196))
    me_protein = _parse_int(_slice(line, 197, 200))

    segments = tuple(
        _parse_segment(_slice(line, 251 + index * 23, 273 + index * 23))
        for index in range(segment_count)
    )

    return Format4Record(
        cow_id=cow_id,
        birth_date=birth_date,
        herd_id=herd_id,
        fresh_date=fresh_date,
        parity=parity,
        length=header_length,
        previous_days_open=previous_days_open,
        herd_me_milk=me_milk,
        herd_me_fat=me_fat,
        herd_me_protein=me_protein,
        herd_me_scs=0.0,
        segments=segments,
    )


def build_current_fortran_source10_rows(
    parsed: Format4Record,
    *,
    detail_means: Format4MeansRecord | None = None,
) -> tuple[Format4Record, Format4Record]:
    """Expand one parsed Format-4 row into the current two-row main-loop flow."""

    detail_update: dict[str, object] = {
        "length": max((segment.dim for segment in parsed.segments), default=parsed.length)
    }
    if detail_means is not None:
        detail_update.update(
            {
                "herd_me_milk": detail_means.herd_me_milk,
                "herd_me_fat": detail_means.herd_me_fat,
                "herd_me_protein": detail_means.herd_me_protein,
                "herd_me_scs": detail_means.herd_me_scs,
            }
        )

    return (
        parsed.model_copy(update={"segments": tuple()}),
        parsed.model_copy(update=detail_update),
    )


def read_source10_records(path: Path) -> list[Format4Record]:
    """Read source-10 `format4.dat` rows.

    The current Fortran `bestpred_main.f90` processes each source-10 line twice:
    first as a header-only zero-test record, then again with the assembled
    segments after the EOF-driven flush of the pending record. The Python
    fixture/oracle follows that current behavior.
    """

    records: list[Format4Record] = []
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not raw_line.strip():
            continue
        parsed = parse_source10_record_line(raw_line.rstrip("\n"))
        records.extend(build_current_fortran_source10_rows(parsed))
    return records
