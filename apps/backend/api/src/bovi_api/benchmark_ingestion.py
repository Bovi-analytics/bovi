"""CSV ingestion for benchmark uploads (own-method, test-day, actual-yields)."""

from __future__ import annotations

import csv
import io
from collections.abc import Sequence
from typing import Literal, overload

# Centralized column aliases - accepted in any case across all parsers.
COW_ID_ALIASES = ("cow_id", "testid", "id")
PARITY_ALIASES = ("parity", "lact", "lactation")
HERD_ID_ALIASES = ("herd_id", "herdid", "herd", "farm_id", "farmid")
DIM_ALIASES = ("dim", "daysinmilk", "days_in_milk")
MILK_ALIASES = ("milk_kg", "milk", "dailymilkingyield", "milkrecording")
YIELD_ALIASES = (
    "yield_305day",
    "total_305_yield",
    "totalactualproduction",
    "actualproduction",
    "calculatedmilkyield",
)


def _resolve(headers: Sequence[str], aliases: tuple[str, ...]) -> str | None:
    """Find the first matching header (case-insensitive) from a list of aliases."""
    lowered = {h.strip().lower(): h for h in headers}
    for alias in aliases:
        if alias in lowered:
            return lowered[alias]
    return None


def _decode(content: bytes) -> str:
    if not content.strip():
        raise ValueError("CSV file is empty.")
    try:
        return content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("CSV file is not valid UTF-8.") from exc


def _sniff_reader(text: str) -> csv.DictReader:
    sample = text[:2048]
    sep = ";" if sample.count(";") > sample.count(",") else ","
    return csv.DictReader(io.StringIO(text), delimiter=sep)


@overload
def parse_submission_csv(
    content: bytes, return_failed: Literal[False] = False
) -> dict[str, float]: ...


@overload
def parse_submission_csv(
    content: bytes, return_failed: Literal[True]
) -> tuple[dict[str, float], list[str]]: ...


def parse_submission_csv(
    content: bytes,
    return_failed: bool = False,
) -> dict[str, float] | tuple[dict[str, float], list[str]]:
    """Parse own-method CSV: cow_id + 305-day yield.

    Accepts column-name aliases for both cow_id and yield columns.
    """
    text = _decode(content)
    reader = _sniff_reader(text)
    if reader.fieldnames is None:
        raise ValueError("CSV file has no headers.")

    cow_col = _resolve(reader.fieldnames, COW_ID_ALIASES)
    yield_col = _resolve(reader.fieldnames, YIELD_ALIASES)
    if cow_col is None or yield_col is None:
        raise ValueError(
            "Missing required columns. Expected cow_id and a yield column "
            f"(any of: {', '.join(YIELD_ALIASES)})."
        )

    results: dict[str, float] = {}
    failed: list[str] = []
    for row in reader:
        cow_id = (row.get(cow_col) or "").strip()
        if not cow_id:
            continue
        try:
            value = float(str(row[yield_col]).replace(",", "."))
        except (ValueError, TypeError, KeyError):
            failed.append(cow_id)
            continue
        if value < 0:
            raise ValueError(f"negative yield for cow '{cow_id}': {value}")
        results[cow_id] = value

    if return_failed:
        return results, failed
    return results


def parse_test_day_csv(content: bytes) -> dict[str, dict]:
    """Parse a test-day CSV into {cow_id: {parity, herd_id, dim[], milk_kg[]}}.

    Required columns: cow_id, dim, milk_kg. Optional: parity, herd_id.
    Multiple rows per cow are grouped together.
    """
    text = _decode(content)
    reader = _sniff_reader(text)
    if reader.fieldnames is None:
        raise ValueError("CSV file has no headers.")

    cow_col = _resolve(reader.fieldnames, COW_ID_ALIASES)
    dim_col = _resolve(reader.fieldnames, DIM_ALIASES)
    milk_col = _resolve(reader.fieldnames, MILK_ALIASES)
    parity_col = _resolve(reader.fieldnames, PARITY_ALIASES)
    herd_id_col = _resolve(reader.fieldnames, HERD_ID_ALIASES)
    if cow_col is None or dim_col is None or milk_col is None:
        raise ValueError(
            "Test-day CSV missing required columns. Expected cow_id, dim, milk_kg "
            "(or aliases). Optional: parity, herd_id."
        )

    out: dict[str, dict] = {}
    for row in reader:
        cow_id = (row.get(cow_col) or "").strip()
        if not cow_id:
            continue
        try:
            d = int(float(str(row[dim_col]).replace(",", ".")))
            m = float(str(row[milk_col]).replace(",", "."))
        except (ValueError, TypeError, KeyError):
            continue
        if d < 0 or m <= 0:
            continue
        parity: int | None = None
        if parity_col and row.get(parity_col):
            try:
                parity = int(float(str(row[parity_col])))
            except (ValueError, TypeError):
                parity = None
        herd_id: int | None = None
        if herd_id_col and row.get(herd_id_col):
            try:
                herd_id = int(float(str(row[herd_id_col])))
            except (ValueError, TypeError):
                herd_id = None
        entry = out.setdefault(
            cow_id,
            {"parity": parity, "herd_id": herd_id, "dim": [], "milk_kg": []},
        )
        if parity is not None and entry["parity"] is None:
            entry["parity"] = parity
        if herd_id is not None and entry["herd_id"] is None:
            entry["herd_id"] = herd_id
        entry["dim"].append(d)
        entry["milk_kg"].append(round(m, 2))

    if not out:
        raise ValueError("No valid test-day rows found.")
    # Sort each cow's records by DIM
    for entry in out.values():
        order = sorted(range(len(entry["dim"])), key=lambda i: entry["dim"][i])
        entry["dim"] = [entry["dim"][i] for i in order]
        entry["milk_kg"] = [entry["milk_kg"][i] for i in order]
    return out


def parse_actual_yields_csv(content: bytes) -> dict[str, float]:
    """Parse an actual-yields CSV into {cow_id: total_305_yield}."""
    text = _decode(content)
    reader = _sniff_reader(text)
    if reader.fieldnames is None:
        raise ValueError("CSV file has no headers.")

    cow_col = _resolve(reader.fieldnames, COW_ID_ALIASES)
    yield_col = _resolve(reader.fieldnames, YIELD_ALIASES)
    if cow_col is None or yield_col is None:
        raise ValueError(
            "Actual-yields CSV missing required columns. Expected cow_id and a yield column."
        )

    out: dict[str, float] = {}
    for row in reader:
        cow_id = (row.get(cow_col) or "").strip()
        if not cow_id:
            continue
        try:
            value = float(str(row[yield_col]).replace(",", "."))
        except (ValueError, TypeError, KeyError):
            continue
        if value < 0:
            continue
        out[cow_id] = value

    if not out:
        raise ValueError("No valid actual-yield rows found.")
    return out
