"""Read `bestpred.par` Fortran namelist files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import f90nml

from bestpred.models import BestpredParameters


def _normalize_namelist(raw: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    dim0: list[int | None] = [None] * 8
    grafplot: list[int | None] = [None] * 4
    aliases = {
        "use3x": "use3X",
        "globalmtrait": "GLOBALmtrait",
        "unitsin": "UNITSin",
        "unitsout": "UNITSout",
        "persfloor": "PERSfloor",
        "persceiling": "PERSceiling",
        "breedunk": "breedUNK",
        "writecurve": "WRITEcurve",
        "curvefile": "CURVEfile",
        "curvesmall": "CURVEsmall",
        "curvesingle": "CURVEsingle",
        "writedata": "WRITEdata",
        "datafile": "DATAfile",
        "infile": "INfile",
        "outfile": "OUTfile",
        "onscreen": "ONscreen",
        "intmethod": "INTmethod",
        "intmethodscs": "INTmethodSCS",
        "debugparms": "DEBUGparms",
        "debugmsgs": "DEBUGmsgs",
        "logon": "LOGon",
        "logfile": "LOGfile",
        "logfreq": "LOGfreq",
    }

    for key, value in raw.items():
        lowered = key.lower()
        if lowered == "dim0":
            sequence = list(value)
            dim0 = [int(item) for item in sequence[:8]]
            continue
        if lowered == "grafplot":
            sequence = list(value)
            grafplot = [int(item) for item in sequence[:4]]
            continue
        normalized[aliases.get(lowered, key)] = value

    if all(value is not None for value in dim0):
        normalized["dim0"] = tuple(int(value) for value in dim0 if value is not None)
    if all(value is not None for value in grafplot):
        normalized["GRAFplot"] = tuple(int(value) for value in grafplot if value is not None)

    return normalized


def read_parameters(path: Path) -> BestpredParameters:
    """Read a BESTPRED Fortran namelist parameter file."""

    namelist = f90nml.read(path)
    bestpred = namelist.get("bestpred")
    if bestpred is None:
        raise ValueError(f"No &bestpred namelist found in {path}")
    return BestpredParameters.model_validate(_normalize_namelist(dict(bestpred)))
