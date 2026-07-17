# pyright: reportAny=false, reportExplicitAny=false, reportUnknownMemberType=false

from __future__ import annotations

from datetime import date
from typing import Any
from uuid import uuid4

import pytest

from bestpred.adapters.farm_data_definitions import breed_code_from_cow
from bestpred.models import BreedCode

fdd: Any = pytest.importorskip("farm_data_definitions")


def test_installed_fdd_cow_satisfies_structural_adapter() -> None:
    cow = fdd.Cow(
        animal_id=uuid4(),
        herd_id=uuid4(),
        gender=fdd.AnimalGender.FEMALE,
        birth_date=date(2020, 1, 1),
        breed=[fdd.BreedPart(breed_code=fdd.Breed.HOL, proportion=1.0)],
    )

    assert breed_code_from_cow(cow) == BreedCode.HOLSTEIN
