from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from uuid import UUID, uuid4

import pytest

from bestpred.adapters.farm_data_definitions import (
    BestpredHerdMeansInput,
    BestpredLactationInput,
    BestpredTestDayInput,
    breed_code_from_cow,
    format4_record_from_fdd,
)
from bestpred.models import BreedCode


@dataclass(frozen=True)
class BreedPartStub:
    breed_code: str
    proportion: float


@dataclass(frozen=True)
class CowStub:
    animal_id: UUID
    herd_id: UUID
    birth_date: date
    breed: tuple[BreedPartStub, ...]
    animal_dhia: str | None = None
    animal_usda: str | None = None
    animal_legal_id: str | None = None
    animal_ear_tag: str | None = None
    animal_farm_name: str | None = None


@dataclass(frozen=True)
class HerdStub:
    herd_id: UUID
    state: str | None
    registration_number: str | None = None
    source_id: UUID | int | str | None = None


def _cow(*, breed: str = "HOL", animal_dhia: str | None = None) -> CowStub:
    return CowStub(
        animal_id=uuid4(),
        herd_id=uuid4(),
        birth_date=date(2020, 1, 1),
        breed=(BreedPartStub(breed_code=breed, proportion=1.0),),
        animal_dhia=animal_dhia,
    )


def _lactation(*, length: int = 305) -> BestpredLactationInput:
    return BestpredLactationInput(
        fresh_date=date(2024, 2, 3),
        parity=2,
        length=length,
        previous_days_open=120,
        herd_means=BestpredHerdMeansInput(milk=20_000, fat=700, protein=600, scs=3.08),
        test_days=(
            BestpredTestDayInput(
                dim=60,
                milk_yield=75,
                fat_percent=38,
                protein_percent=31,
                scs=22,
            ),
            BestpredTestDayInput(
                dim=30,
                milk_yield=70,
                fat_percent=39,
                protein_percent=32,
                scs=21,
            ),
        ),
    )


def test_adapter_module_does_not_require_farm_data_definitions() -> None:
    cow = _cow()

    assert breed_code_from_cow(cow) == BreedCode.HOLSTEIN


def test_format4_record_from_fdd_builds_bestpred_boundary_record() -> None:
    cow = _cow(animal_dhia="cow-42")
    herd = HerdStub(herd_id=cow.herd_id, state="35", registration_number="herd-7")

    record = format4_record_from_fdd(cow, herd, _lactation())

    assert record.cow_id == "HCOW42"
    assert record.birth_date == "20200101"
    assert record.herd_id == "35HERD7"
    assert record.fresh_date == "20240203"
    assert record.parity == 2
    assert record.previous_days_open == 120
    assert record.herd_me_scs == pytest.approx(3.08)
    assert [segment.dim for segment in record.segments] == [30, 60]
    assert record.segments[0].to_fortran_segment() == " 3020222 1100  70393221"


def test_format4_record_from_fdd_requires_bestpred_state_code() -> None:
    cow = _cow(breed="JER")
    herd = HerdStub(herd_id=cow.herd_id, state="NY")

    with pytest.raises(ValueError, match="state_code is required"):
        format4_record_from_fdd(cow, herd, _lactation(length=0))


def test_format4_record_from_fdd_accepts_explicit_state_and_ids() -> None:
    cow = _cow(breed="JER")
    herd = HerdStub(herd_id=cow.herd_id, state="NY")

    record = format4_record_from_fdd(
        cow,
        herd,
        _lactation(length=0),
        state_code=12,
        bestpred_cow_id="JEXPLICIT",
        bestpred_herd_id="12EXPLICIT",
    )

    assert record.cow_id == "JEXPLICIT"
    assert record.herd_id == "12EXPLICIT"
