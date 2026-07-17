"""Fortran-compatible standard curve interpolation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from math import exp
from pathlib import Path
from typing import Final

import numpy as np
import numpy.typing as npt

from bestpred.models import Trait

FloatArray = npt.NDArray[np.float64]
CurveParams3 = tuple[float, float, float]
CurveParams4 = tuple[float, float, float, float]
TraitKey = int
ParityGroup = int
BreedIndex = int
FortranShape = tuple[int, ...]

MAX_SOURCE_BREED: Final = 6
HOLSTEIN_BREED: Final = 4
FIRST_PARITY_GROUP: Final = 1
LATER_PARITY_GROUP: Final = 2
MAX_REGION: Final = 7
MAX_SEASON: Final = 4
DEFAULT_FORTRAN_SOURCE: Final[Path | None] = None
DATA_PACKAGE: Final[str] = "bestpred.data"
TRAIT_TO_FORTRAN: Final[dict[Trait, int]] = {
    Trait.MILK: 1,
    Trait.FAT: 2,
    Trait.PROTEIN: 3,
    Trait.SCS: 4,
}

MONTHLY_DAILY_YIELD: Final[dict[ParityGroup, dict[TraitKey, tuple[float, ...]]]] = {
    1: {
        1: (53.3, 63.0, 67.2, 68.2, 67.2, 65.6, 63.8, 61.9, 59.7, 57.3, 54.8, 52.1),
        2: (1.98, 2.35, 2.54, 2.62, 2.62, 2.58, 2.52, 2.46, 2.39, 2.31, 2.24, 2.16),
        3: (1.64, 1.90, 2.00, 2.03, 2.02, 1.98, 1.94, 1.89, 1.83, 1.77, 1.71, 1.65),
        4: (3.27, 3.27, 3.27, 3.27, 3.27, 3.27, 3.27, 3.27, 3.27, 3.27, 3.27, 3.27),
    },
    2: {
        1: (76.3, 86.0, 87.9, 86.7, 83.9, 80.7, 77.4, 74.2, 71.1, 68.1, 65.1, 62.3),
        2: (2.90, 3.31, 3.43, 3.43, 3.37, 3.28, 3.18, 3.08, 2.98, 2.87, 2.77, 2.67),
        3: (2.35, 2.59, 2.64, 2.60, 2.53, 2.45, 2.36, 2.27, 2.19, 2.10, 2.02, 1.94),
        4: (3.27, 3.27, 3.27, 3.27, 3.27, 3.27, 3.27, 3.27, 3.27, 3.27, 3.27, 3.27),
    },
}

MONTHLY_SD: Final[dict[ParityGroup, dict[TraitKey, tuple[float, ...]]]] = {
    1: {
        1: (10.8, 10.5, 10.7, 10.8, 10.8, 10.6, 10.3, 10.0, 9.7, 9.4, 9.1, 8.8),
        2: (0.62, 0.56, 0.53, 0.51, 0.50, 0.49, 0.49, 0.48, 0.47, 0.46, 0.46, 0.45),
        3: (0.37, 0.34, 0.34, 0.34, 0.34, 0.34, 0.33, 0.33, 0.32, 0.32, 0.31, 0.31),
        4: (1.52, 1.52, 1.52, 1.52, 1.52, 1.52, 1.52, 1.52, 1.52, 1.52, 1.52, 1.52),
    },
    2: {
        1: (16.5, 15.9, 16.1, 16.2, 16.1, 15.8, 15.4, 15.0, 14.5, 14.1, 13.6, 13.2),
        2: (0.91, 0.79, 0.74, 0.70, 0.67, 0.65, 0.63, 0.61, 0.59, 0.57, 0.56, 0.54),
        3: (0.56, 0.48, 0.46, 0.45, 0.44, 0.43, 0.42, 0.41, 0.40, 0.39, 0.38, 0.37),
        4: (1.52, 1.52, 1.52, 1.52, 1.52, 1.52, 1.52, 1.52, 1.52, 1.52, 1.52, 1.52),
    },
}

WOOD_MEANS: Final[dict[BreedIndex, dict[ParityGroup, dict[TraitKey, CurveParams3]]]] = {
    1: {
        1: {
            1: (13.7957, 0.1904, 0.00268),
            2: (0.6496, 0.1217, 0.0016),
            3: (0.513, 0.1244, 0.00152),
        },
        2: {
            1: (18.195, 0.1997, 0.00421),
            2: (0.9498, 0.1033, 0.00293),
            3: (0.7401, 0.1052, 0.00267),
        },
    },
    2: {
        1: {
            1: (15.0619, 0.1619, 0.00182),
            2: (0.6908, 0.1165, 0.00116),
            3: (0.4847, 0.1516, 0.00121),
        },
        2: {
            1: (23.3824, 0.13, 0.00274),
            2: (1.1891, 0.0569, 0.00185),
            3: (0.8221, 0.0961, 0.00186),
        },
    },
    3: {
        1: {
            1: (15.0366, 0.146, 0.00214),
            2: (0.6033, 0.1407, 0.00136),
            3: (0.4972, 0.1203, 0.00129),
        },
        2: {1: (21.2814, 0.12, 0.00297), 2: (0.962, 0.0907, 0.00204), 3: (0.7883, 0.0677, 0.00184)},
    },
    4: {
        1: {
            1: (13.0097, 0.2673, 0.00262),
            2: (0.7842, 0.1199, 0.0013),
            3: (0.4625, 0.2033, 0.00161),
        },
        2: {
            1: (22.0087, 0.2155, 0.00357),
            2: (1.2874, 0.0731, 0.00213),
            3: (0.8539, 0.1317, 0.00232),
        },
    },
    5: {
        1: {
            1: (11.5338, 0.2022, 0.00222),
            2: (0.5027, 0.1954, 0.00161),
            3: (0.4096, 0.1797, 0.00148),
        },
        2: {
            1: (17.399, 0.1766, 0.00316),
            2: (0.7902, 0.1634, 0.00249),
            3: (0.7146, 0.1231, 0.00219),
        },
    },
    6: {
        1: {
            1: (13.6246, 0.1877, 0.00231),
            2: (0.6878, 0.0805, 0.00108),
            3: (0.5389, 0.0914, 0.00133),
        },
        2: {
            1: (22.4251, 0.1434, 0.00348),
            2: (1.2134, 0.0154, 0.00199),
            3: (0.8634, 0.0551, 0.00224),
        },
    },
}

WOOD_SD: Final[dict[BreedIndex, dict[ParityGroup, dict[TraitKey, CurveParams3]]]] = {
    1: {
        1: {
            1: (3.8737, 0.1027, 0.000381),
            2: (0.2259, 0.034, 9.3e-05),
            3: (0.1229, 0.0875, -4e-05),
        },
        2: {
            1: (5.8246, 0.0935, 0.00136),
            2: (0.3686, 0.0121, 0.001),
            3: (0.2229, 0.0216, 0.000421),
        },
    },
    2: {
        1: {
            1: (4.3308, 0.0905, 0.000127),
            2: (0.2985, 0.0025, -6e-05),
            3: (0.1313, 0.098, -0.00011),
        },
        2: {
            1: (7.0287, 0.0562, 0.000583),
            2: (0.495, -0.025, 0.000459),
            3: (0.2418, 0.0281, 6e-05),
        },
    },
    3: {
        1: {
            1: (4.7574, 0.065, 0.000308),
            2: (0.3265, -0.0553, -0.00058),
            3: (0.1574, 0.035, -0.00023),
        },
        2: {
            1: (6.5111, 0.0527, 0.000974),
            2: (0.402, -0.0215, 0.000432),
            3: (0.2496, -0.0133, 9.1e-05),
        },
    },
    4: {
        1: {
            1: (5.3807, 0.054, -0.0003),
            2: (0.3798, -0.0612, -0.00061),
            3: (0.1803, 0.0168, -0.00072),
        },
        2: {
            1: (8.7545, 0.0282, 0.000439),
            2: (0.5536, -0.0526, 0.000304),
            3: (0.327, -0.0363, -0.00022),
        },
    },
    5: {
        1: {
            1: (3.8016, 0.074, -0.00012),
            2: (0.2245, 0.045, -0.00023),
            3: (0.1255, 0.0761, -0.0004),
        },
        2: {
            1: (5.114, 0.0797, 0.000618),
            2: (0.2911, 0.069, 0.000659),
            3: (0.2027, 0.0399, 7.7e-05),
        },
    },
    6: {
        1: {
            1: (3.3264, 0.1834, 0.000397),
            2: (0.3183, -0.0446, -0.00092),
            3: (0.1407, 0.0583, -0.00022),
        },
        2: {
            1: (7.7328, 0.0241, 3.4e-05),
            2: (0.5617, -0.124, -0.00082),
            3: (0.3494, -0.1302, -0.00139),
        },
    },
}

MANDG_MEANS: Final[dict[BreedIndex, dict[ParityGroup, CurveParams4]]] = {
    1: {1: (1.7911, -0.00233, 2.157e-06, 14.2048), 2: (1.7716, -0.00791, -1e-05, 17.8304)},
    2: {1: (1.6126, -0.00475, -5.28e-06, 17.8126), 2: (2.2231, -0.00594, -8.51e-06, 13.5807)},
    3: {1: (2.3295, -0.00316, -4.36e-06, 14.8032), 2: (2.4706, -0.00426, -5.2e-06, 8.8076)},
    4: {1: (1.9798, -0.00344, -3.42e-06, 16.8829), 2: (2.5072, -0.00431, -4.59e-06, 8.9804)},
    5: {1: (2.159, -0.00354, -3.68e-06, 19.653), 2: (2.2129, -0.00572, -8.02e-06, 14.9249)},
    6: {1: (1.8987, -0.00308, -1.56e-06, 14.1516), 2: (1.2378, -0.0115, -3e-05, 20.7716)},
}

MANDG_SD: Final[dict[BreedIndex, dict[ParityGroup, CurveParams4]]] = {
    1: {1: (1.6442, -0.00055, -3.14e-06, 1.476), 2: (2.1741, 0.00243, 4.193e-06, -1.3917)},
    2: {1: (1.9034, 0.00141, 4.422e-06, 0.2612), 2: (2.2805, 0.00285, 6.958e-06, -2.7067)},
    3: {1: (1.647, -0.00064, -1.92e-06, 2.6782), 2: (2.243, 0.00131, 1.45e-06, -3.0148)},
    4: {1: (1.9551, 0.000248, -1.22e-07, -1.5154), 2: (2.4849, 0.00229, 3.454e-06, -6.3911)},
    5: {1: (1.7515, -5e-05, -3.91e-07, 3.1323), 2: (2.3131, 0.00134, 1.457e-06, -2.5612)},
    6: {1: (1.6281, -0.00075, -2.87e-06, 2.2243), 2: (2.4802, 0.00184, 2.61e-06, -1.5926)},
}


@dataclass(frozen=True)
class InterpolatedCurve:
    """One Fortran `interpolate` result for a trait/parity/breed/method."""

    daily_yield: FloatArray
    cumulative_yield: FloatArray
    daily_sd: FloatArray
    mean_persistency_numerator: float
    method_used: str
    breed_used: int
    trait: int
    parity_group: int


def interpolate_curve(
    *,
    trait: Trait | int,
    parity_group: int,
    breed: int,
    method: str,
    maxlen: int = 365,
    region: int = 1,
    season: int = 1,
) -> InterpolatedCurve:
    """Port of Fortran `interpolate`.

    Returned arrays are zero-indexed: DIM `i` is stored at index `i - 1`.
    """

    trait_id = _normalize_trait(trait)
    parity_id = _normalize_parity_group(parity_group)
    breed_id = _normalize_breed(breed)
    method_id = _normalize_method(method, trait_id)
    region_id = _normalize_region(region)
    season_id = _normalize_season(season)

    if maxlen < 1:
        raise ValueError("maxlen must be at least 1")

    match method_id:
        case "L":
            daily, daily_sd, meanp = _linear_curve(trait_id, parity_id, maxlen)
        case "W":
            daily, daily_sd, meanp = _wood_curve(trait_id, parity_id, breed_id, maxlen)
        case "G":
            daily, daily_sd, meanp = _mandg_curve(trait_id, parity_id, breed_id, maxlen)
        case "R":
            daily, daily_sd, meanp = _regional_wood_curve(trait_id, parity_id, region_id, maxlen)
        case "C":
            daily, daily_sd, meanp = _calving_wood_curve(trait_id, parity_id, season_id, maxlen)
        case "T":
            daily, daily_sd, meanp = _seasonal_wood_curve(
                trait_id, parity_id, season_id, region_id, maxlen
            )
        case "S":
            daily, daily_sd, meanp = _regional_mandg_curve(parity_id, region_id, maxlen)
        case "D":
            daily, daily_sd, meanp = _calving_mandg_curve(parity_id, season_id, maxlen)
        case "U":
            daily, daily_sd, meanp = _seasonal_mandg_curve(parity_id, season_id, region_id, maxlen)
        case _:
            raise NotImplementedError(f"Interpolation method {method_id!r} is not ported yet.")

    return InterpolatedCurve(
        daily_yield=daily,
        cumulative_yield=np.cumsum(daily, dtype=np.float64),
        daily_sd=daily_sd,
        mean_persistency_numerator=meanp,
        method_used=method_id,
        breed_used=breed_id,
        trait=trait_id,
        parity_group=parity_id,
    )


def _linear_curve(
    trait: int, parity_group: int, maxlen: int
) -> tuple[FloatArray, FloatArray, float]:
    daily = np.empty(maxlen, dtype=np.float64)
    daily_sd = np.empty(maxlen, dtype=np.float64)
    monthly_yield = MONTHLY_DAILY_YIELD[parity_group][trait]
    monthly_sd = MONTHLY_SD[parity_group][trait]
    meanp = 0.0

    for day in range(1, maxlen + 1):
        month = min(max(1, (day + 15) // 30), 11)
        value = (
            (day - month * 30 + 15) * monthly_yield[month]
            + ((month + 1) * 30 - 15 - day) * monthly_yield[month - 1]
        ) / 30.0
        sd = (
            (day - month * 30 + 15) * monthly_sd[month]
            + ((month + 1) * 30 - 15 - day) * monthly_sd[month - 1]
        ) / 30.0
        if day > 365:
            value = monthly_yield[11]
            sd = monthly_sd[11]
        daily[day - 1] = value
        daily_sd[day - 1] = sd
        if day <= 305:
            meanp += value * day

    return daily, daily_sd, meanp


def _wood_curve(
    trait: int, parity_group: int, breed: int, maxlen: int
) -> tuple[FloatArray, FloatArray, float]:
    if trait == 4:
        raise ValueError("Wood interpolation is only valid for milk, fat, and protein.")

    daily = np.empty(maxlen, dtype=np.float64)
    daily_sd = np.empty(maxlen, dtype=np.float64)
    mean_params = WOOD_MEANS[breed][parity_group][trait]
    sd_params = WOOD_SD[breed][parity_group][trait]
    meanp = 0.0

    for day in range(1, maxlen + 1):
        value = _wood_value(day, mean_params)
        sd = _wood_value(day, sd_params)
        daily[day - 1] = value
        daily_sd[day - 1] = sd
        meanp += value * day

    return daily, daily_sd, meanp


def _mandg_curve(
    trait: int, parity_group: int, breed: int, maxlen: int
) -> tuple[FloatArray, FloatArray, float]:
    if trait != 4:
        raise ValueError("Morant/Gnanasakthy interpolation is only valid for SCS.")

    daily = np.empty(maxlen, dtype=np.float64)
    daily_sd = np.empty(maxlen, dtype=np.float64)
    mean_params = MANDG_MEANS[breed][parity_group]
    sd_params = MANDG_SD[breed][parity_group]
    meanp = 0.0

    for day in range(1, maxlen + 1):
        dim = float(day + 10)
        value = _mandg_mean_value(dim, mean_params)
        sd = _mandg_sd_value(dim, sd_params)
        daily[day - 1] = value
        daily_sd[day - 1] = sd
        meanp += value * day

    return daily, daily_sd, meanp


def _regional_wood_curve(
    trait: int, parity_group: int, region: int, maxlen: int
) -> tuple[FloatArray, FloatArray, float]:
    if trait == 4:
        raise ValueError("Regional Wood interpolation is only valid for milk, fat, and protein.")

    tables = load_regional_curve_tables()
    mean_params = _params3(
        tables["regional_woods_means"][:, trait - 1, parity_group - 1, region - 1]
    )
    sd_params = _params3(tables["regional_woods_sd"][:, trait - 1, parity_group - 1, region - 1])
    return _wood_curve_from_params(mean_params, sd_params, maxlen)


def _calving_wood_curve(
    trait: int, parity_group: int, season: int, maxlen: int
) -> tuple[FloatArray, FloatArray, float]:
    if trait == 4:
        raise ValueError(
            "Calving-season Wood interpolation is only valid for milk, fat, and protein."
        )

    tables = load_regional_curve_tables()
    mean_params = _params3(
        tables["calving_woods_means"][:, trait - 1, parity_group - 1, season - 1]
    )
    sd_params = _params3(tables["calving_woods_sd"][:, trait - 1, parity_group - 1, season - 1])
    return _wood_curve_from_params(mean_params, sd_params, maxlen)


def _seasonal_wood_curve(
    trait: int, parity_group: int, season: int, region: int, maxlen: int
) -> tuple[FloatArray, FloatArray, float]:
    if trait == 4:
        raise ValueError("Seasonal Wood interpolation is only valid for milk, fat, and protein.")

    tables = load_regional_curve_tables()
    mean_params = _params3(
        tables["seasonal_woods_means"][:, trait - 1, parity_group - 1, season - 1, region - 1]
    )
    sd_params = _params3(
        tables["seasonal_woods_sd"][:, trait - 1, parity_group - 1, season - 1, region - 1]
    )
    return _wood_curve_from_params(mean_params, sd_params, maxlen)


def _regional_mandg_curve(
    parity_group: int, region: int, maxlen: int
) -> tuple[FloatArray, FloatArray, float]:
    tables = load_regional_curve_tables()
    mean_params = _params4(tables["regional_mandg_means"][:, 0, parity_group - 1, region - 1])
    sd_params = _params4(tables["regional_mandg_sd"][:, 0, parity_group - 1, region - 1])
    return _mandg_curve_from_params(mean_params, sd_params, maxlen)


def _calving_mandg_curve(
    parity_group: int, season: int, maxlen: int
) -> tuple[FloatArray, FloatArray, float]:
    tables = load_regional_curve_tables()
    mean_params = _params4(tables["calving_mandg_means"][:, 0, parity_group - 1, season - 1])
    sd_params = _params4(tables["calving_mandg_sd"][:, 0, parity_group - 1, season - 1])
    return _mandg_curve_from_params(mean_params, sd_params, maxlen)


def _seasonal_mandg_curve(
    parity_group: int, season: int, region: int, maxlen: int
) -> tuple[FloatArray, FloatArray, float]:
    tables = load_regional_curve_tables()
    mean_params = _params4(
        tables["seasonal_mandg_means"][:, 0, parity_group - 1, season - 1, region - 1]
    )
    sd_params = _params4(
        tables["seasonal_mandg_sd"][:, 0, parity_group - 1, season - 1, region - 1]
    )
    return _mandg_curve_from_params(mean_params, sd_params, maxlen)


def _wood_curve_from_params(
    mean_params: CurveParams3, sd_params: CurveParams3, maxlen: int
) -> tuple[FloatArray, FloatArray, float]:
    daily = np.empty(maxlen, dtype=np.float64)
    daily_sd = np.empty(maxlen, dtype=np.float64)
    meanp = 0.0

    for day in range(1, maxlen + 1):
        value = _wood_value(day, mean_params)
        sd = _wood_value(day, sd_params)
        daily[day - 1] = value
        daily_sd[day - 1] = sd
        meanp += value * day

    return daily, daily_sd, meanp


def _mandg_curve_from_params(
    mean_params: CurveParams4, sd_params: CurveParams4, maxlen: int
) -> tuple[FloatArray, FloatArray, float]:
    daily = np.empty(maxlen, dtype=np.float64)
    daily_sd = np.empty(maxlen, dtype=np.float64)
    meanp = 0.0

    for day in range(1, maxlen + 1):
        dim = float(day + 10)
        value = _mandg_mean_value(dim, mean_params)
        sd = _mandg_sd_value(dim, sd_params)
        daily[day - 1] = value
        daily_sd[day - 1] = sd
        meanp += value * day

    return daily, daily_sd, meanp


def _wood_value(day: int, params: CurveParams3) -> float:
    a, b, c = params
    dim = float(day)
    return a * dim**b * exp(-dim * c)


def _mandg_mean_value(dim: float, params: CurveParams4) -> float:
    a, b, c, d = params
    return a - b * dim + c * dim**2 + d / dim


def _mandg_sd_value(dim: float, params: CurveParams4) -> float:
    a, b, c, d = params
    return a - b * dim + (c * dim**2) / 2.0 + d / dim


def _normalize_trait(trait: Trait | int) -> int:
    if isinstance(trait, Trait):
        return TRAIT_TO_FORTRAN[trait]
    if 1 <= trait <= 4:
        return trait
    raise ValueError(f"Unsupported trait: {trait!r}")


def _normalize_parity_group(parity_group: int) -> int:
    if parity_group in {FIRST_PARITY_GROUP, LATER_PARITY_GROUP}:
        return parity_group
    raise ValueError(f"Unsupported parity group: {parity_group!r}")


def _normalize_breed(breed: int) -> int:
    if 1 <= breed <= MAX_SOURCE_BREED:
        return breed
    return HOLSTEIN_BREED


def _normalize_region(region: int) -> int:
    if 1 <= region <= MAX_REGION:
        return region
    raise ValueError(f"Unsupported region: {region!r}")


def _normalize_season(season: int) -> int:
    if 1 <= season <= MAX_SEASON:
        return season
    raise ValueError(f"Unsupported season: {season!r}")


def _normalize_method(method: str, trait: int) -> str:
    method_id = method.strip().upper()[:1]
    valid_methods = {"L", "W", "R", "T", "C"} if trait < 4 else {"L", "G", "S", "U", "D"}
    if method_id in valid_methods:
        return method_id
    return "L"


@lru_cache(maxsize=1)
def load_regional_curve_tables(
    source_path: Path | None = DEFAULT_FORTRAN_SOURCE,
) -> dict[str, FloatArray]:
    """Parse regional and seasonal curve parameter arrays from `bestpred.f90`."""

    source = (
        resources.files(DATA_PACKAGE).joinpath("bestpred.f90").read_text(encoding="utf-8")
        if source_path is None
        else source_path.read_text(encoding="utf-8")
    )
    specs: dict[str, FortranShape] = {
        "regional_woods_means": (3, 3, 2, 7),
        "regional_woods_sd": (3, 3, 2, 7),
        "regional_mandg_means": (4, 1, 2, 7),
        "regional_mandg_sd": (4, 1, 2, 7),
        "calving_woods_means": (3, 3, 2, 4),
        "calving_woods_sd": (3, 3, 2, 4),
        "calving_mandg_means": (4, 1, 2, 4),
        "calving_mandg_sd": (4, 1, 2, 4),
        "seasonal_woods_means": (3, 3, 2, 4, 7),
        "seasonal_woods_sd": (3, 3, 2, 4, 7),
        "seasonal_mandg_means": (4, 1, 2, 4, 7),
        "seasonal_mandg_sd": (4, 1, 2, 4, 7),
    }
    return {name: _parse_fortran_data_array(source, name, shape) for name, shape in specs.items()}


def _parse_fortran_data_array(source: str, name: str, shape: FortranShape) -> FloatArray:
    pattern = re.compile(
        rf"\bdata\s+{re.escape(name)}\s*/(?P<body>.*?)/", re.IGNORECASE | re.DOTALL
    )
    match = pattern.search(source)
    if match is None:
        raise ValueError(f"Could not find Fortran data block {name!r}")

    body = _strip_fortran_comments(match.group("body")).replace("&", " ")
    values = [float(token) for token in _numeric_tokens(body)]
    expected = int(np.prod(shape))
    if len(values) != expected:
        raise ValueError(
            f"Fortran data block {name!r} has {len(values)} values, expected {expected}"
        )
    return np.asarray(values, dtype=np.float64).reshape(shape, order="F")


def _strip_fortran_comments(text: str) -> str:
    return "\n".join(line.split("!", 1)[0] for line in text.splitlines())


def _numeric_tokens(text: str) -> list[str]:
    return re.findall(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eEdD][-+]?\d+)?", text)


def _params3(values: FloatArray) -> CurveParams3:
    return float(values[0]), float(values[1]), float(values[2])


def _params4(values: FloatArray) -> CurveParams4:
    return float(values[0]), float(values[1]), float(values[2]), float(values[3])
