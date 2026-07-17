"""Integration points for `farm-data-definitions`.

The current farm-data-definitions package has Cow, Herd, Event, and typed event
metadata models, but its Lactation module is intentionally pending. These
helpers keep the dependency explicit while avoiding invented ontology fields.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from datetime import date
from typing import Final, Protocol
from uuid import UUID

from pydantic import Field, field_validator

from bestpred.models import BestpredModel, BreedCode, Format4Record, TestDaySegment

FDD_TO_BESTPRED_BREED: Final[dict[str, BreedCode]] = {
    "HOL": BreedCode.HOLSTEIN,
    "JER": BreedCode.JERSEY,
    "BSW": BreedCode.BROWN_SWISS,
    "GUE": BreedCode.GUERNSEY,
    "RDC": BreedCode.AYRSHIRE,
}


class BreedPartLike(Protocol):
    """Structural subset of a farm-data-definitions breed part."""

    @property
    def breed_code(self) -> object: ...

    @property
    def proportion(self) -> float: ...


class CowLike(Protocol):
    """Structural Cow contract used without a hard FDD dependency."""

    @property
    def animal_id(self) -> UUID | int | str: ...

    @property
    def birth_date(self) -> date: ...

    @property
    def breed(self) -> Sequence[BreedPartLike]: ...

    @property
    def animal_dhia(self) -> str | None: ...

    @property
    def animal_usda(self) -> str | None: ...

    @property
    def animal_legal_id(self) -> str | None: ...

    @property
    def animal_ear_tag(self) -> str | None: ...

    @property
    def animal_farm_name(self) -> str | None: ...


class HerdLike(Protocol):
    """Structural Herd contract used without a hard FDD dependency."""

    @property
    def herd_id(self) -> UUID | int | str: ...

    @property
    def state(self) -> str | None: ...

    @property
    def registration_number(self) -> str | None: ...

    @property
    def source_id(self) -> UUID | int | str | None: ...


class BestpredHerdMeansInput(BestpredModel):
    """305-day herd means needed by the BESTPRED Format-4 boundary."""

    milk: int = Field(ge=0)
    fat: int = Field(ge=0)
    protein: int = Field(ge=0)
    scs: float | None = Field(default=None, ge=0)


class BestpredTestDayInput(BestpredModel):
    """BESTPRED-compatible test-day facts missing from the current FDD ontology."""

    dim: int = Field(ge=0)
    milk_yield: int = Field(ge=0)
    fat_percent: int = Field(default=0, ge=0)
    protein_percent: int = Field(default=0, ge=0)
    scs: int = Field(default=0, ge=0)
    supervised: int = Field(default=2, ge=0, le=9)
    status: int = Field(default=0, ge=0, le=9)
    times_milked: int = Field(default=2, ge=0, le=9)
    times_weighed: int = Field(default=2, ge=0, le=9)
    times_sampled: int = Field(default=2, ge=0, le=9)
    ler_days: int = Field(default=1, ge=0, le=99)
    percent_shipped: int = Field(default=100, ge=0, le=999)

    def to_segment(self) -> TestDaySegment:
        """Convert to the existing Format-4 test-day segment model."""

        return TestDaySegment(
            dim=self.dim,
            supervised=self.supervised,
            status=self.status,
            times_milked=self.times_milked,
            times_weighed=self.times_weighed,
            times_sampled=self.times_sampled,
            ler_days=self.ler_days,
            percent_shipped=self.percent_shipped,
            milk_yield=self.milk_yield,
            fat_percent=self.fat_percent,
            protein_percent=self.protein_percent,
            scs=self.scs,
        )


class BestpredLactationInput(BestpredModel):
    """Temporary adapter DTO until FDD exposes Lactation and TestDay models."""

    fresh_date: date
    parity: int = Field(ge=1)
    length: int = Field(ge=0)
    herd_means: BestpredHerdMeansInput
    test_days: tuple[BestpredTestDayInput, ...]
    previous_days_open: int = Field(default=0, ge=0)

    @field_validator("test_days")
    @classmethod
    def require_sorted_unique_dims(
        cls,
        value: tuple[BestpredTestDayInput, ...],
    ) -> tuple[BestpredTestDayInput, ...]:
        dims = [test_day.dim for test_day in value]
        if len(dims) != len(set(dims)):
            raise ValueError("test_days must not contain duplicate DIM values")
        return tuple(sorted(value, key=lambda test_day: test_day.dim))


def breed_code_from_cow(cow: CowLike) -> BreedCode | None:
    """Map a farm-data-definitions Cow breed composition to a BESTPRED breed."""

    if not cow.breed:
        return None

    primary = max(cow.breed, key=lambda part: part.proportion)
    return FDD_TO_BESTPRED_BREED.get(str(primary.breed_code))


def format4_record_from_fdd(
    cow: CowLike,
    herd: HerdLike,
    lactation: BestpredLactationInput,
    *,
    state_code: int | None = None,
    bestpred_cow_id: str | None = None,
    bestpred_herd_id: str | None = None,
) -> Format4Record:
    """Build a BESTPRED Format-4 record from FDD Cow/Herd plus adapter DTOs."""

    resolved_state_code = _resolve_state_code(herd, state_code, bestpred_herd_id)
    return Format4Record(
        cow_id=bestpred_cow_id or _bestpred_cow_id(cow),
        birth_date=_format_date(cow.birth_date),
        herd_id=bestpred_herd_id or _bestpred_herd_id(herd, resolved_state_code),
        fresh_date=_format_date(lactation.fresh_date),
        parity=lactation.parity,
        length=lactation.length,
        previous_days_open=lactation.previous_days_open,
        herd_me_milk=lactation.herd_means.milk,
        herd_me_fat=lactation.herd_means.fat,
        herd_me_protein=lactation.herd_means.protein,
        herd_me_scs=lactation.herd_means.scs,
        segments=tuple(test_day.to_segment() for test_day in lactation.test_days),
    )


def _format_date(value: date) -> str:
    return value.strftime("%Y%m%d")


def _bestpred_cow_id(cow: CowLike) -> str:
    breed = breed_code_from_cow(cow) or BreedCode.HOLSTEIN
    external_id = (
        cow.animal_dhia
        or cow.animal_usda
        or cow.animal_legal_id
        or cow.animal_ear_tag
        or cow.animal_farm_name
        or str(cow.animal_id)
    )
    return f"{breed.fortran_trait_prefix}{_compact_identifier(external_id)}"


def _bestpred_herd_id(herd: HerdLike, state_code: int) -> str:
    external_id = herd.registration_number or str(herd.source_id or "") or str(herd.herd_id)
    return f"{state_code:02d}{_compact_identifier(external_id)}"


def _compact_identifier(value: UUID | int | str) -> str:
    compact = re.sub(r"[^A-Za-z0-9]", "", str(value).strip().upper())
    return compact or "UNKNOWN"


def _resolve_state_code(
    herd: HerdLike,
    state_code: int | None,
    bestpred_herd_id: str | None,
) -> int:
    if state_code is not None:
        return _validate_state_code(state_code)

    if bestpred_herd_id is not None:
        prefix = bestpred_herd_id[:2]
        if prefix.isdigit():
            return _validate_state_code(int(prefix))
        raise ValueError("bestpred_herd_id must start with a two-digit BESTPRED state code")

    if herd.state is not None and herd.state.strip().isdigit():
        return _validate_state_code(int(herd.state.strip()))

    raise ValueError("state_code is required when Herd.state is not a numeric BESTPRED state code")


def _validate_state_code(value: int) -> int:
    if not 0 <= value <= 96:
        raise ValueError("BESTPRED state_code must be between 0 and 96")
    return value
