"""Source-11 simulation as implemented in `bestpred_main.f90`."""

from __future__ import annotations

import math

from bestpred.models import (
    BestpredParameters,
    Format4Record,
    Source11Example,
    TestDaySegment,
)

# Source 11 is the built-in Fortran demo data generator, not a production input
# format. These constants mirror `bestpred_main.f90:844-846` and the source-11
# caller state passed through `herd305` for oracle compatibility.
SOURCE11_HERD_ME_MILK = 20_000
SOURCE11_HERD_ME_FAT = 700
SOURCE11_HERD_ME_PROTEIN = 600
SOURCE11_HERD_ME_SCS = 3.08
SOURCE11_HERD_ID = "12345678"
SOURCE11_FRESH_DATE = "19990401"
SOURCE11_PREVIOUS_DAYS_OPEN = 140


def _fortran_int(value: float) -> int:
    """Match Fortran real-to-integer assignment truncation."""

    return math.trunc(value)


def _birth_date_for(fresh_date: str, parity: int) -> str:
    year = int(fresh_date[:4])
    return f"{year - parity - 1:04d}{fresh_date[4:]}"


def _segment_for(dim: int, line: Source11Example, line_index: int) -> TestDaySegment:
    plan = line.plan_lines[line_index]
    milk = (
        15.0
        + (
            SOURCE11_HERD_ME_MILK * 0.003
            - 0.12 * (dim - 150)
            + 15.0 * (0.7 - (0.7 / plan.ler_days) * math.sin(dim / 11.0))
        )
        * 10.0
    )
    fat = (3.6 + 0.4 * math.sin((11 - dim) / 11.0)) * 10.0
    protein = (3.2 + 0.3 * math.sin((11 - dim) / 11.0)) * 10.0
    scs = (3.3 + 2.0 * math.sin((60 - dim) / 60.0)) * 10.0

    if plan.times_sampled == 0:
        fat = 0.0
        protein = 0.0
        scs = 0.0

    return TestDaySegment(
        dim=dim,
        supervised=plan.supervised,
        status=0,
        times_milked=plan.times_milked,
        times_weighed=plan.times_weighed,
        times_sampled=plan.times_sampled,
        ler_days=plan.ler_days,
        percent_shipped=100,
        milk_yield=_fortran_int(milk),
        fat_percent=_fortran_int(fat),
        protein_percent=_fortran_int(protein),
        scs=_fortran_int(scs),
    )


def simulate_source11_record(
    example: Source11Example,
    parameters: BestpredParameters,
    *,
    herd_me_scs: float = SOURCE11_HERD_ME_SCS,
) -> Format4Record:
    """Construct the Format 4-style record generated for one source-11 example."""

    segments: list[TestDaySegment] = []
    length = 0

    for line_index, plan in enumerate(example.plan_lines):
        for dim in range(plan.first_test, plan.last_test + 1, plan.test_interval):
            segment = _segment_for(dim, example, line_index)
            segments.append(segment)
            length = max(length, dim)

    parity = example.plan_lines[-1].parity
    cow_id = f"{parameters.source11_breed.value}USA.EX.COW."
    cow_id = f"{cow_id[:13]}{example.number:04d}"

    return Format4Record(
        cow_id=cow_id,
        birth_date=_birth_date_for(SOURCE11_FRESH_DATE, parity),
        herd_id=SOURCE11_HERD_ID,
        fresh_date=SOURCE11_FRESH_DATE,
        parity=parity,
        length=length,
        previous_days_open=SOURCE11_PREVIOUS_DAYS_OPEN,
        herd_me_milk=SOURCE11_HERD_ME_MILK,
        herd_me_fat=SOURCE11_HERD_ME_FAT,
        herd_me_protein=SOURCE11_HERD_ME_PROTEIN,
        herd_me_scs=herd_me_scs,
        segments=tuple(segments),
    )


def simulate_source11_records(
    examples: list[Source11Example],
    parameters: BestpredParameters,
) -> list[Format4Record]:
    """Construct all source-11 Format 4-style records."""

    records: list[Format4Record] = []
    herd_me_scs = SOURCE11_HERD_ME_SCS
    for example in examples:
        records.append(
            simulate_source11_record(
                example,
                parameters,
                herd_me_scs=herd_me_scs,
            )
        )
        # `bestpred_fmt4.f90:315` mutates the caller's SCS herd value in-place.
        # Source 11 reuses that caller array across example records, so the
        # effective SCS herd mean is divided by 100 after each processed row.
        herd_me_scs /= 100.0
    return records
