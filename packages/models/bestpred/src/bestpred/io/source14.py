"""Parsers for source-14 and source-24 PCDART-style inputs."""

from __future__ import annotations

from pathlib import Path

from bestpred.models import Format4Record, TestDaySegment


def _slice(text: str, start: int, stop: int) -> str:
    return text[start - 1 : stop]


def _pad(text: str, width: int) -> str:
    return text.ljust(width)


def _parse_int(field: str) -> int:
    stripped = field.strip()
    return int(stripped) if stripped else 0


def _parse_segment(segment: str) -> TestDaySegment:
    padded = _pad(segment, 21)
    dim = min(_parse_int(_slice(padded, 1, 4)), 999)
    return TestDaySegment(
        dim=dim,
        supervised=_parse_int(_slice(padded, 5, 5)),
        status=_parse_int(_slice(padded, 6, 6)),
        times_milked=_parse_int(_slice(padded, 7, 7)),
        times_weighed=_parse_int(_slice(padded, 8, 8)),
        times_sampled=_parse_int(_slice(padded, 9, 9)),
        ler_days=_parse_int(_slice(padded, 10, 11)),
        percent_shipped=0,
        milk_yield=_parse_int(_slice(padded, 12, 15)),
        fat_percent=_parse_int(_slice(padded, 16, 17)),
        protein_percent=_parse_int(_slice(padded, 18, 19)),
        scs=_parse_int(_slice(padded, 20, 21)),
    )


def _parse_header(line: str) -> tuple[str, int, int, int, float]:
    padded = _pad(line.rstrip("\n"), 23)
    herd_id = _slice(padded, 1, 8)
    return (
        herd_id,
        _parse_int(_slice(padded, 9, 13)),
        _parse_int(_slice(padded, 14, 17)),
        _parse_int(_slice(padded, 18, 21)),
        _parse_int(_slice(padded, 22, 23)) / 100.0,
    )


def _detail_cow_id(breed_code: str, cow_id7: str) -> str:
    return f"{breed_code}   {cow_id7}".ljust(17)


def _parse_detail_line(
    line: str,
    *,
    herd_id: str,
    herd_me_milk: int,
    herd_me_fat: int,
    herd_me_protein: int,
    herd_me_scs: float,
) -> Format4Record:
    padded = _pad(line.rstrip("\n"), 44)
    cow_id7 = _slice(padded, 9, 15)
    breed_code = _slice(padded, 16, 17)
    birth_date = _slice(padded, 18, 25)
    previous_days_open = min(_parse_int(_slice(padded, 26, 28)), 99)
    parity = min(_parse_int(_slice(padded, 29, 30)), 9)
    fresh_date = _slice(padded, 31, 38)
    segment_count = _parse_int(_slice(padded, 43, 44))
    segments = tuple(
        _parse_segment(_slice(line, 45 + index * 21, 65 + index * 21))
        for index in range(segment_count)
    )
    return Format4Record(
        cow_id=_detail_cow_id(breed_code, cow_id7),
        birth_date=birth_date,
        herd_id=herd_id,
        fresh_date=fresh_date,
        parity=parity,
        length=max((segment.dim for segment in segments), default=0),
        previous_days_open=previous_days_open,
        herd_me_milk=herd_me_milk,
        herd_me_fat=herd_me_fat,
        herd_me_protein=herd_me_protein,
        herd_me_scs=herd_me_scs,
        segments=segments,
    )


def _build_eof_zero_row(
    *,
    template: Format4Record,
    herd_me_milk: int,
    herd_me_fat: int,
    herd_me_protein: int,
    herd_me_scs: float,
) -> Format4Record:
    return template.model_copy(
        update={
            "length": 0,
            "herd_me_milk": herd_me_milk,
            "herd_me_fat": herd_me_fat,
            "herd_me_protein": herd_me_protein,
            "herd_me_scs": herd_me_scs,
            "compatibility_tag": "source14_eof_zero",
            "segments": tuple(),
        }
    )


def read_source14_records(path: Path) -> list[Format4Record]:
    """Read one PCDART-style source-14 file.

    The current Fortran main loop emits an additional EOF/zero-test row per
    source-14 file after the final real cow record. The Python port preserves
    that compatibility artifact because it affects the current `results_v2.dcr`
    oracle for deterministic test files such as `test241.txt` and `test242.txt`.
    """

    lines = [
        line.rstrip("\n")
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
        if line.strip()
    ]
    if not lines:
        return []

    herd_id, herd_me_milk, herd_me_fat, herd_me_protein, herd_me_scs = _parse_header(lines[0])
    detail_records = [
        _parse_detail_line(
            line,
            herd_id=herd_id,
            herd_me_milk=herd_me_milk,
            herd_me_fat=herd_me_fat,
            herd_me_protein=herd_me_protein,
            herd_me_scs=herd_me_scs,
        )
        for line in lines[1:]
    ]
    if not detail_records:
        return []

    return [
        *detail_records,
        _build_eof_zero_row(
            template=detail_records[-1],
            herd_me_milk=herd_me_milk,
            herd_me_fat=herd_me_fat,
            herd_me_protein=herd_me_protein,
            herd_me_scs=herd_me_scs,
        ),
    ]


def read_source24_records(path: Path) -> list[Format4Record]:
    """Read the source-24 file list and flatten its source-14 records."""

    records: list[Format4Record] = []
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        filename = raw_line.strip()
        if not filename:
            continue
        records.extend(read_source14_records(path.parent / filename))
    return records
