"""Age, season, and previous-days-open adjustments from `aiplage.c`."""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Final

import numpy as np
import numpy.typing as npt

from bestpred.models import Format4Record

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int_]

DOMEAN: Final = 120
BREEDS: Final = "ABGHJMW"
REGION_BREED_INDEX: Final[tuple[int, ...]] = (0, 0, 0, 1, 2, 3, 2)
REGION_DAYS_OPEN: Final[tuple[int, ...]] = (0, 0, 1, 4, 4, 1, 3, 2, 3, 5, 5, 6)
AGE_LIMITS: Final[tuple[tuple[int, int], ...]] = (
    (16, 36),
    (28, 56),
    (40, 74),
    (52, 86),
    (64, 98),
    (76, 165),
)
DEFAULT_AIPLAGE_HEADER: Final[Path | None] = None
DATA_PACKAGE: Final[str] = "bestpred.data"


@dataclass(frozen=True)
class AgeAdjustmentFactors:
    """Milk, fat, and protein factors returned by `aiplage`."""

    milk: float
    fat: float
    protein: float


@dataclass(frozen=True)
class AiplageData:
    """Parsed coefficient arrays from `aiplage.h`."""

    floats: dict[str, FloatArray]
    region: IntArray


def aiplage_factors(
    *,
    breed: str,
    age_at_freshening_months: int,
    fresh_year: int,
    fresh_month: int,
    parity: int,
    state: int,
    previous_days_open: int,
    agebase: int = 0,
    data: AiplageData | None = None,
) -> AgeAdjustmentFactors:
    """Pure Python port of `aiplage.c`.

    The returned factors standardize 305-day milk, fat, and protein yields.
    """

    coefficients = load_aiplage_data() if data is None else data
    breed_index = _breed_index(breed)
    region = int(coefficients.region[state, REGION_BREED_INDEX[breed_index]]) - 1
    if parity == 0:
        raise ValueError("parity must not be 0")

    lac = min(parity - 1, 5)
    age = min(max(age_at_freshening_months, AGE_LIMITS[lac][0]), AGE_LIMITS[lac][1])
    month_index = fresh_month - 1
    pdo = previous_days_open
    if (parity > 1 and pdo < 20) or pdo > 305:
        pdo = DOMEAN

    milk = 0.0
    fat = 0.0
    protein = 0.0

    for pass_number in (1, 2):
        months = 1
        if pass_number == 2:
            if agebase == 0:
                continue
            age = agebase
            lac = _c_int_div(agebase - 18, 13)
            pdo = 90
            months = 12
            average_milk = 0.0
            average_fat = 0.0
            average_protein = 0.0
        else:
            average_milk = 0.0
            average_fat = 0.0
            average_protein = 0.0

        for month in range(1, months + 1):
            if pass_number == 2:
                month_index = month - 1

            pdo_milk, pdo_fat, pdo_protein = _previous_days_open_effects(
                coefficients=coefficients,
                breed_index=breed_index,
                region=region,
                lac=lac,
                pdo=pdo,
            )
            yrgp, yrgpp = _year_groups(fresh_year)
            age2 = age * age
            fac_milk, fac_fat, fac_protein = _base_factors(
                coefficients=coefficients,
                breed_index=breed_index,
                region=region,
                yrgp=yrgp,
                yrgpp=yrgpp,
                lac=lac,
                month_index=month_index,
                age=age,
                age2=age2,
                pdo_milk=pdo_milk,
                pdo_fat=pdo_fat,
                pdo_protein=pdo_protein,
            )

            if pass_number == 1:
                milk = fac_milk
                fat = fac_fat
                protein = fac_protein
            else:
                average_milk += fac_milk / months
                average_fat += fac_fat / months
                average_protein += fac_protein / months
                if month == 12:
                    milk /= average_milk
                    fat /= average_fat
                    protein /= average_protein

    return AgeAdjustmentFactors(milk=milk, fat=fat, protein=protein)


def format4_age_factors(record: Format4Record, *, agebase: int = 0) -> AgeAdjustmentFactors:
    """Compute M/F/P age factors for a prepared Format 4 record."""

    fresh_year = int(record.fresh_date[:4])
    fresh_month = int(record.fresh_date[4:6])
    state = int(record.herd_id[:2])
    age = int((perpetual_day(record.fresh_date) - perpetual_day(record.birth_date)) / 30.5)
    previous_days_open = record.previous_days_open
    if previous_days_open < 1:
        previous_days_open = 140
    if record.parity == 1:
        previous_days_open = 0
    return aiplage_factors(
        breed=record.cow_id[:1],
        age_at_freshening_months=age,
        fresh_year=fresh_year,
        fresh_month=fresh_month,
        parity=record.parity,
        state=state,
        previous_days_open=previous_days_open,
        agebase=agebase,
    )


def perpetual_day(date8: str) -> int:
    """Port of `bestpred_fmt4.f90::pday`."""

    month_days = (
        (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31),
        (31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31),
    )
    year = int(date8[:4]) - 1900
    month = int(date8[4:6])
    day = int(date8[6:8])
    if year <= 0:
        return -21916

    leap_count = _c_int_div(year, 4)
    number_of_years = year - 60
    year_groups = abs(number_of_years) // 4
    balance_years = abs(number_of_years) - (year_groups * 4)
    leap_index = 1 if year == leap_count * 4 else 0
    month = min(max(month, 1), 12)
    if day == 0:
        day = 1
    day = min(day, month_days[leap_index][month - 1])
    day += sum(month_days[leap_index][: month - 1])

    if number_of_years <= -1:
        day = 365 - day
        pday = -((year_groups * 1461) + day + (balance_years - 1) * 365)
    else:
        if balance_years > 0:
            balance_years = ((balance_years - 1) * 365) + 366
        pday = (year_groups * 1461) + day + balance_years
    return pday - 1


@lru_cache(maxsize=1)
def load_aiplage_data(header_path: Path | None = DEFAULT_AIPLAGE_HEADER) -> AiplageData:
    """Load coefficient arrays from the original C header."""

    text = (
        resources.files(DATA_PACKAGE).joinpath("aiplage.h").read_text(encoding="utf-8")
        if header_path is None
        else header_path.read_text(encoding="utf-8")
    )
    arrays: dict[str, FloatArray] = {}
    names = (
        "amregtm",
        "afregtm",
        "apregtm",
        "bmregtm",
        "bfregtm",
        "bpregtm",
        "gmregtm",
        "gfregtm",
        "gpregtm",
        "hmregtm",
        "hfregtm",
        "hpregtm",
        "jmregtm",
        "jfregtm",
        "jpregtm",
        "mmregtm",
        "mfregtm",
        "mpregtm",
        "mdop",
        "fdop",
        "pdop",
        "hmdop",
        "hfdop",
        "hpdop",
        "amequ0",
        "afequ0",
        "apequ0",
        "amequ",
        "afequ",
        "apequ",
        "bmequ0",
        "bfequ0",
        "bpequ0",
        "bmequ",
        "bfequ",
        "bpequ",
        "jmequ0",
        "jfequ0",
        "jpequ0",
        "gmequ0",
        "gfequ0",
        "gpequ0",
        "gmequ",
        "gfequ",
        "gpequ",
        "jmequ",
        "jfequ",
        "jpequ",
        "mmequ0",
        "mfequ0",
        "mpequ0",
        "mmequ",
        "mfequ",
        "mpequ",
        "hmequ0",
        "hmequ",
        "hfequ0",
        "hfequ",
        "hpequ0",
        "hpequ",
    )
    for name in names:
        arrays[name] = _parse_float_array(text, name)
    return AiplageData(floats=arrays, region=_parse_int_array(text, "region"))


def _previous_days_open_effects(
    *,
    coefficients: AiplageData,
    breed_index: int,
    region: int,
    lac: int,
    pdo: int,
) -> tuple[float, float, float]:
    if lac <= 0:
        return 0.0, 0.0, 0.0

    lacdo = min(lac, 2) - 1
    pdo2 = pdo * pdo
    if breed_index == 3:
        regdo = REGION_DAYS_OPEN[region]
        milk_coefficients = coefficients.floats["hmdop"][regdo, lacdo]
        fat_coefficients = coefficients.floats["hfdop"][regdo, lacdo]
        protein_coefficients = coefficients.floats["hpdop"][regdo, lacdo]
    else:
        brddo = breed_index if breed_index < 3 else breed_index - 1
        milk_coefficients = coefficients.floats["mdop"][brddo, lacdo]
        fat_coefficients = coefficients.floats["fdop"][brddo, lacdo]
        protein_coefficients = coefficients.floats["pdop"][brddo, lacdo]

    return (
        _pdo_effect(milk_coefficients, pdo, pdo2),
        _pdo_effect(fat_coefficients, pdo, pdo2),
        _pdo_effect(protein_coefficients, pdo, pdo2),
    )


def _pdo_effect(coefficients: FloatArray, pdo: int, pdo2: int) -> float:
    value = float(coefficients[0] + coefficients[1] * pdo + coefficients[2] * pdo2)
    if pdo > 150:
        value += float(coefficients[3] * (pdo - 150) * (pdo - 150))
    return value


def _year_groups(fresh_year: int) -> tuple[int, int]:
    if fresh_year >= 1987:
        return 4, 1
    if fresh_year >= 1969:
        return _c_int_div(fresh_year - 1963, 6), 0
    return 0, 0


def _base_factors(
    *,
    coefficients: AiplageData,
    breed_index: int,
    region: int,
    yrgp: int,
    yrgpp: int,
    lac: int,
    month_index: int,
    age: int,
    age2: int,
    pdo_milk: float,
    pdo_fat: float,
    pdo_protein: float,
) -> tuple[float, float, float]:
    data = coefficients.floats
    match breed_index:
        case 3:
            milk = _factor(
                data["hmregtm"][region, yrgp],
                pdo_milk,
                data["hmequ0"][yrgp, region, lac, month_index],
                data["hmequ"][yrgp, region, lac],
                age,
                age2,
            )
            fat = _factor(
                data["hfregtm"][region, yrgp],
                pdo_fat,
                data["hfequ0"][yrgp, region, lac, month_index],
                data["hfequ"][yrgp, region, lac],
                age,
                age2,
            )
            protein = _factor(
                data["hpregtm"][region, yrgpp],
                pdo_protein,
                data["hpequ0"][yrgpp, region, lac, month_index],
                data["hpequ"][yrgpp, region, lac],
                age,
                age2,
            )
        case 4:
            milk = _factor(
                data["jmregtm"][region, yrgp],
                pdo_milk,
                data["jmequ0"][yrgp, region, lac, month_index],
                data["jmequ"][yrgp, region, lac],
                age,
                age2,
            )
            fat = _factor(
                data["jfregtm"][region, yrgp],
                pdo_fat,
                data["jfequ0"][yrgp, region, lac, month_index],
                data["jfequ"][yrgp, region, lac],
                age,
                age2,
            )
            protein = _factor(
                data["jpregtm"][region, yrgpp],
                pdo_protein,
                data["jpequ0"][yrgpp, region, lac, month_index],
                data["jpequ"][yrgpp, region, lac],
                age,
                age2,
            )
        case 2:
            milk = _factor(
                data["gmregtm"][region, yrgp],
                pdo_milk,
                data["gmequ0"][yrgp, region, lac, month_index],
                data["gmequ"][yrgp, region, lac],
                age,
                age2,
            )
            fat = _factor(
                data["gfregtm"][region, yrgp],
                pdo_fat,
                data["gfequ0"][yrgp, region, lac, month_index],
                data["gfequ"][yrgp, region, lac],
                age,
                age2,
            )
            protein = _factor(
                data["gpregtm"][region, yrgpp],
                pdo_protein,
                data["gpequ0"][yrgpp, region, lac, month_index],
                data["gpequ"][yrgpp, region, lac],
                age,
                age2,
            )
        case 1:
            milk = _factor(
                data["bmregtm"][region, yrgp],
                pdo_milk,
                data["bmequ0"][yrgp, region, lac, month_index],
                data["bmequ"][yrgp, region, lac],
                age,
                age2,
            )
            fat = _factor(
                data["bfregtm"][region, yrgp],
                pdo_fat,
                data["bfequ0"][yrgp, region, lac, month_index],
                data["bfequ"][yrgp, region, lac],
                age,
                age2,
            )
            protein = _factor(
                data["bpregtm"][region, yrgpp],
                pdo_protein,
                data["bpequ0"][yrgpp, region, lac, month_index],
                data["bpequ"][yrgpp, region, lac],
                age,
                age2,
            )
        case 0:
            milk = _factor(
                data["amregtm"][region, yrgp],
                pdo_milk,
                data["amequ0"][yrgp, region, lac, month_index],
                data["amequ"][yrgp, region, lac],
                age,
                age2,
            )
            fat = _factor(
                data["afregtm"][region, yrgp],
                pdo_fat,
                data["afequ0"][yrgp, region, lac, month_index],
                data["afequ"][yrgp, region, lac],
                age,
                age2,
            )
            protein = _factor(
                data["apregtm"][region, yrgpp],
                pdo_protein,
                data["apequ0"][yrgpp, region, lac, month_index],
                data["apequ"][yrgpp, region, lac],
                age,
                age2,
            )
        case 5:
            milk = _factor(
                data["mmregtm"][yrgp],
                pdo_milk,
                data["mmequ0"][yrgp, lac, month_index],
                data["mmequ"][yrgp, lac],
                age,
                age2,
            )
            fat = _factor(
                data["mfregtm"][yrgp],
                pdo_fat,
                data["mfequ0"][yrgp, lac, month_index],
                data["mfequ"][yrgp, lac],
                age,
                age2,
            )
            protein = _factor(
                data["mpregtm"][yrgpp],
                pdo_protein,
                data["mpequ0"][yrgpp, lac, month_index],
                data["mpequ"][yrgpp, lac],
                age,
                age2,
            )
        case _:
            raise ValueError(f"Unsupported breed index: {breed_index}")
    return float(milk), float(fat), float(protein)


def _factor(
    regional_mean: float,
    pdo_effect: float,
    intercept: float,
    age_coefficients: FloatArray,
    age: int,
    age2: int,
) -> float:
    return float(
        regional_mean
        / (
            regional_mean
            + pdo_effect
            + intercept
            + age_coefficients[0] * age
            + age_coefficients[1] * age2
        )
    )


def _breed_index(breed: str) -> int:
    breed_character = breed.strip().upper()[:1]
    index = BREEDS.find(breed_character)
    if index == -1:
        return 3
    return index


def _parse_float_array(text: str, name: str) -> FloatArray:
    body, shape = _array_body_and_shape(text, name)
    values = [float(value) for value in _numeric_tokens(_strip_comments(body))]
    expected = int(np.prod(shape))
    if len(values) != expected:
        raise ValueError(f"Array {name} has {len(values)} values, expected {expected}")
    return np.asarray(values, dtype=np.float64).reshape(shape)


def _parse_int_array(text: str, name: str) -> IntArray:
    body, shape = _array_body_and_shape(text, name)
    values = [int(float(value)) for value in _numeric_tokens(_strip_comments(body))]
    expected = int(np.prod(shape))
    if len(values) != expected:
        raise ValueError(f"Array {name} has {len(values)} values, expected {expected}")
    return np.asarray(values, dtype=np.int_).reshape(shape)


def _array_body_and_shape(text: str, name: str) -> tuple[str, tuple[int, ...]]:
    pattern = re.compile(
        rf"(?:float|char)\s+{re.escape(name)}\s*((?:\[\s*\d+\s*\]\s*)+)\s*=\s*\{{(?P<body>.*?)\}};",
        re.DOTALL,
    )
    match = pattern.search(text)
    if match is None:
        raise ValueError(f"Could not find array {name!r} in aiplage header")
    shape = tuple(int(value) for value in re.findall(r"\[\s*(\d+)\s*\]", match.group(1)))
    return match.group("body"), shape


def _strip_comments(text: str) -> str:
    return re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)


def _numeric_tokens(text: str) -> list[str]:
    return re.findall(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", text)


def _c_int_div(left: int, right: int) -> int:
    return int(left / right)
