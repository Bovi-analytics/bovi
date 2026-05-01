"""CSV ingestion and normalization utility for herd stats upload.

Three input formats are supported, auto-detected from the header row:

1. ``aggregated`` — one row per herd profile, columns named after canonical
   stats (e.g. ``Achieved305Milk``, ``DaysInMilk``). The original format.
2. ``icar_test_day`` — one row per cow per milk recording, as produced by the
   ICAR platform. Required columns: ``TestId``, ``DaysInMilk``,
   ``DailyMilkingYield`` (and optionally ``Parity``, ``EventType``).
3. ``dairycom_test_day`` — DairyCom (Cornell-style) export with European
   decimals (``,``) and milk in **lbs**. Required columns: ``ID``, ``DIM``,
   ``MILK`` (and optionally ``305ME`` with the 305-d mature equivalent).

For the two test-day formats the parser aggregates raw records into herd-level
stats: per-cow averages at DIM windows (21, 75), overall mean daily yield, mean
days-in-milk per cow, and a 305-day yield estimate (DairyCom uses the ``305ME``
column directly; ICAR uses a trapezoidal test-interval integration).

Normalization is inlined (not imported from lactation_autoencoder) to keep the
central API free of ML framework dependencies.
"""

import csv
import io
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Literal

FormatDetected = Literal["aggregated", "icar_test_day", "dairycom_test_day"]

CANONICAL_NAMES: list[str] = [
    "Achieved21Milk",
    "Achieved305Milk",
    "Achieved75Milk",
    "AchievedMilk",
    "DaysDry",
    "DaysInMilk",
    "DaysOpen",
    "DaysPregnant",
    "HistoricCalvingInterval",
    "QualitySequence",
]

# Case-insensitive column header → canonical stat name (aggregated format only)
_ALIASES: dict[str, str] = {name.lower(): name for name in CANONICAL_NAMES}
_ALIASES.update(
    {
        "dim": "DaysInMilk",
        "305milk": "Achieved305Milk",
        "21milk": "Achieved21Milk",
        "75milk": "Achieved75Milk",
        "totalmilk": "AchievedMilk",
        "calvinginterval": "HistoricCalvingInterval",
        "quality": "QualitySequence",
    }
)

LBS_TO_KG = 0.45359237

# Windows used to sample test-day yield around DIM 21 and DIM 75
_DIM_21_WINDOW = 7
_DIM_75_WINDOW = 10


@dataclass
class _CowLactation:
    """Intermediate shape produced by the test-day parsers."""

    cow_id: str
    parity: int | None
    test_days: list[tuple[int, float]]  # (DIM, milk_kg)
    milk_305d: float | None  # None unless the source provided a 305-d column


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------


def _normalise_header(h: str) -> str:
    return h.strip().lower().replace(" ", "").replace("_", "")


def _detect_format(header: list[str]) -> FormatDetected:
    """Classify a CSV header row into one of the supported formats.

    Args:
        header (list[str]): Raw header fields from the first row of the CSV.

    Returns:
        FormatDetected: ``"icar_test_day"``, ``"dairycom_test_day"`` or
        ``"aggregated"``.

    Raises:
        ValueError: If the header does not look like any supported format.

    """
    norm = {_normalise_header(h) for h in header}

    icar_required = {"testid", "daysinmilk", "dailymilkingyield"}
    icar_signals = {"eventtype", "calvingdate", "dailymilkingyield"}
    if icar_required <= norm and norm & icar_signals:
        return "icar_test_day"

    dairycom_required = {"id", "dim", "milk"}
    dairycom_signals = {"305me", "pctf", "pctp", "fcm", "relv", "scc", "pen"}
    if dairycom_required <= norm and norm & dairycom_signals:
        return "dairycom_test_day"

    if any(_resolve_aggregated_column(h) for h in header):
        return "aggregated"

    raise ValueError(
        "Could not detect CSV format. Expected aggregated herd stats "
        "(columns like Achieved305Milk), ICAR test-day records (TestId, "
        "DaysInMilk, DailyMilkingYield) or a DairyCom export (ID, DIM, MILK, "
        "305ME)."
    )


def _resolve_aggregated_column(header: str) -> str | None:
    """Map a CSV column header to a canonical stat name, or None."""
    return _ALIASES.get(_normalise_header(header))


def _is_binary_content(content: bytes) -> bool:
    """Return True if content looks like non-text binary data."""
    if b"\x00" in content:
        return True
    non_printable = sum(1 for b in content if b < 0x09 or (0x0E <= b <= 0x1F) or b == 0x7F)
    return len(content) > 0 and non_printable / len(content) > 0.3


def _sniff_delimiter(sample: str) -> str:
    """Pick ',' or ';' by comparing their counts in the first non-empty line."""
    for line in sample.splitlines():
        if line.strip():
            return ";" if line.count(";") > line.count(",") else ","
    return ","


def _parse_number(cell: str) -> float | None:
    """Parse a number, tolerating European decimals and DairyCom flag markers."""
    if cell is None:
        return None
    s = cell.strip().rstrip("*")
    if not s:
        return None
    if "," in s and "." not in s:
        s = s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@dataclass
class CowRecord:
    """Per-cow test-day records returned alongside aggregated herd stats."""

    cow_id: str
    parity: int | None
    dim: list[int]
    milk_kg: list[float]


@dataclass
class IngestionResult:
    """Result of parsing a CSV upload."""

    raw_stats: dict[str, float]
    format_detected: FormatDetected
    row_count: int
    warnings: list[str]
    cow_count: int | None = None
    detected_parity: int | None = None
    cows: list[CowRecord] | None = None


def parse_csv(content: bytes, max_rows: int = 200_000) -> IngestionResult:
    """Parse uploaded CSV bytes into raw herd stats.

    The header row is used to auto-detect the format; the appropriate parser
    is then dispatched. Test-day formats are aggregated into herd-level means.

    Args:
        content (bytes): Raw CSV bytes from the uploaded file.
        max_rows (int): Maximum rows to process; excess rows are truncated
            with a warning.

    Returns:
        IngestionResult: Parsed stats plus metadata. ``raw_stats`` contains
        only the canonical keys the parser was able to derive; the rest are
        left for the caller (or the user) to fill in.

    Raises:
        ValueError: If the bytes are not parseable as CSV, if the header does
            not match any supported format, or if there are no data rows.

    """
    if _is_binary_content(content):
        raise ValueError("Not a valid CSV file — binary content detected")

    text = content.decode("utf-8", errors="replace")
    # Strip UTF-8 BOM if present
    if text.startswith("﻿"):
        text = text[1:]
    if not text.strip():
        raise ValueError("Not a valid CSV file — empty file")

    delimiter = _sniff_delimiter(text[:4096])
    try:
        reader = csv.reader(io.StringIO(text), delimiter=delimiter)
        header = next(reader)
        data_rows = list(reader)
    except (StopIteration, csv.Error) as exc:
        raise ValueError("Not a valid CSV file") from exc

    if not header:
        raise ValueError("Not a valid CSV file — header row is empty")
    if not data_rows:
        raise ValueError("Not a valid CSV file — no data rows found")

    fmt = _detect_format(header)

    warnings: list[str] = []
    if len(data_rows) > max_rows:
        data_rows = data_rows[:max_rows]
        warnings.append(
            f"File had more rows than the limit of {max_rows}; only the first {max_rows} were used."
        )

    if fmt == "aggregated":
        raw_stats, row_count, parse_warnings = _parse_aggregated(header, data_rows)
        warnings.extend(parse_warnings)
        return IngestionResult(
            raw_stats=raw_stats,
            format_detected="aggregated",
            row_count=row_count,
            warnings=warnings,
        )

    if fmt == "icar_test_day":
        cows, parse_warnings = _parse_icar_test_day(header, data_rows)
    else:  # dairycom_test_day
        cows, parse_warnings = _parse_dairycom_test_day(header, data_rows)
        warnings.append("Milk values converted from lbs to kg (DairyCom export).")
    warnings.extend(parse_warnings)

    if not cows:
        raise ValueError("No usable cow records could be extracted from the test-day file.")

    stats, aggregate_warnings = _aggregate_test_days(cows)
    warnings.extend(aggregate_warnings)

    parity_values = [c.parity for c in cows if c.parity is not None]
    detected_parity: int | None
    if parity_values:
        detected_parity = Counter(parity_values).most_common(1)[0][0]
    else:
        detected_parity = None

    # Cap per-cow payload so a rogue upload doesn't balloon the response
    max_cows_in_payload = 2000
    trimmed_cows = cows[:max_cows_in_payload]
    if len(cows) > max_cows_in_payload:
        warnings.append(
            f"Only the first {max_cows_in_payload} cow records are returned to the client "
            f"(the full upload had {len(cows)} cows)."
        )

    cow_records = [
        CowRecord(
            cow_id=c.cow_id,
            parity=c.parity,
            dim=[d for d, _ in c.test_days],
            milk_kg=[y for _, y in c.test_days],
        )
        for c in trimmed_cows
    ]

    return IngestionResult(
        raw_stats=stats,
        format_detected=fmt,
        row_count=len(data_rows),
        warnings=warnings,
        cow_count=len(cows),
        detected_parity=detected_parity,
        cows=cow_records,
    )


def normalize_herd_stats(
    raw: dict[str, float],
    stat_ranges: dict[str, tuple[float, float]],
) -> dict[str, float]:
    """Map raw domain values to 0–1 using provided ranges.

    Values outside the range are clamped. Only keys present in both ``raw``
    and ``stat_ranges`` are normalized; missing keys are omitted from the
    output.

    Args:
        raw (dict[str, float]): Dict of canonical stat name → raw domain value.
        stat_ranges (dict[str, tuple[float, float]]): Dict of canonical stat
            name → (min, max) range.

    Returns:
        dict[str, float]: Dict of canonical stat name → normalized float in [0, 1].

    """
    result: dict[str, float] = {}
    for name, (lo, hi) in stat_ranges.items():
        if name not in raw:
            continue
        value = float(raw[name])
        clipped = max(lo, min(hi, value))
        result[name] = (clipped - lo) / (hi - lo) if hi > lo else 0.0
    return result


# ---------------------------------------------------------------------------
# Aggregated format (original behaviour)
# ---------------------------------------------------------------------------


def _parse_aggregated(
    header: list[str],
    data_rows: list[list[str]],
) -> tuple[dict[str, float], int, list[str]]:
    """Parse a pre-aggregated herd stats CSV (one row per profile).

    Multiple rows are averaged column-wise — preserves the original
    ``individual`` → column-mean behaviour but no longer distinguishes that
    sub-case in the response format.

    Args:
        header (list[str]): Column headers.
        data_rows (list[list[str]]): Remaining data rows.

    Returns:
        tuple: (raw_stats dict, row count, warnings).

    """
    warnings: list[str] = []
    col_map: dict[int, str] = {}
    for idx, name in enumerate(header):
        canonical = _resolve_aggregated_column(name)
        if canonical is not None:
            col_map[idx] = canonical

    if not col_map:
        raise ValueError(
            "No recognised herd stat columns found. Expected one or more of: "
            + ", ".join(CANONICAL_NAMES)
        )

    found = set(col_map.values())
    missing = [n for n in CANONICAL_NAMES if n not in found]
    if missing:
        warnings.append(f"Missing columns (absent from result): {', '.join(missing)}")

    column_values: dict[str, list[float]] = defaultdict(list)
    nan_counts: dict[str, int] = defaultdict(int)

    for row in data_rows:
        for idx, canonical in col_map.items():
            cell = row[idx] if idx < len(row) else ""
            parsed = _parse_number(cell)
            if parsed is None:
                nan_counts[canonical] += 1
            else:
                column_values[canonical].append(parsed)

    for canonical, count in nan_counts.items():
        if count > 0:
            warnings.append(
                f"Column '{canonical}' had {count} unparseable value(s); excluded from mean."
            )

    raw_stats = {
        canonical: sum(values) / len(values)
        for canonical, values in column_values.items()
        if values
    }
    return raw_stats, len(data_rows), warnings


# ---------------------------------------------------------------------------
# Test-day parsers
# ---------------------------------------------------------------------------


def _header_index(header: list[str], *candidates: str) -> int | None:
    """Return the index of the first header matching any candidate (case-insensitive)."""
    wanted = {c.lower() for c in candidates}
    for idx, name in enumerate(header):
        if _normalise_header(name) in wanted:
            return idx
    return None


def _parse_icar_test_day(
    header: list[str],
    data_rows: list[list[str]],
) -> tuple[list[_CowLactation], list[str]]:
    warnings: list[str] = []
    idx_id = _header_index(header, "testid")
    idx_dim = _header_index(header, "daysinmilk", "dim")
    idx_milk = _header_index(header, "dailymilkingyield", "milk")
    idx_parity = _header_index(header, "parity")
    idx_event = _header_index(header, "eventtype")

    if idx_id is None or idx_dim is None or idx_milk is None:
        raise ValueError(
            "ICAR test-day file is missing required columns "
            "(TestId, DaysInMilk, DailyMilkingYield)."
        )

    by_cow: dict[str, list[tuple[int, float]]] = defaultdict(list)
    parity_by_cow: dict[str, int] = {}
    skipped_event = 0
    skipped_parse = 0

    for row in data_rows:
        if idx_event is not None and idx_event < len(row):
            event = row[idx_event].strip().lower()
            if event and event != "milkrecording":
                skipped_event += 1
                continue

        if idx_id >= len(row) or idx_dim >= len(row) or idx_milk >= len(row):
            skipped_parse += 1
            continue

        cow_id = row[idx_id].strip()
        dim_val = _parse_number(row[idx_dim])
        milk_val = _parse_number(row[idx_milk])
        if not cow_id or dim_val is None or milk_val is None or milk_val < 0:
            skipped_parse += 1
            continue

        by_cow[cow_id].append((int(dim_val), float(milk_val)))

        if idx_parity is not None and idx_parity < len(row) and cow_id not in parity_by_cow:
            parity_val = _parse_number(row[idx_parity])
            if parity_val is not None and parity_val > 0:
                parity_by_cow[cow_id] = int(parity_val)

    if skipped_event:
        warnings.append(f"Ignored {skipped_event} row(s) with EventType != 'MilkRecording'.")
    if skipped_parse:
        warnings.append(f"Skipped {skipped_parse} row(s) with missing/unparseable values.")

    cows = [
        _CowLactation(
            cow_id=cow_id,
            parity=parity_by_cow.get(cow_id),
            test_days=sorted(records),
            milk_305d=None,
        )
        for cow_id, records in by_cow.items()
    ]
    return cows, warnings


def _parse_dairycom_test_day(
    header: list[str],
    data_rows: list[list[str]],
) -> tuple[list[_CowLactation], list[str]]:
    warnings: list[str] = []
    idx_id = _header_index(header, "id", "cowid")
    idx_dim = _header_index(header, "dim", "daysinmilk")
    idx_milk = _header_index(header, "milk")
    idx_305 = _header_index(header, "305me", "305ME")

    if idx_id is None or idx_dim is None or idx_milk is None:
        raise ValueError("DairyCom export is missing required columns (ID, DIM, MILK).")

    by_cow: dict[str, list[tuple[int, float]]] = defaultdict(list)
    latest_305: dict[str, float] = {}
    skipped = 0
    zero_milk = 0

    for row in data_rows:
        if idx_id >= len(row) or idx_dim >= len(row) or idx_milk >= len(row):
            skipped += 1
            continue
        cow_id = row[idx_id].strip()
        dim_val = _parse_number(row[idx_dim])
        milk_val = _parse_number(row[idx_milk])
        if not cow_id or dim_val is None or milk_val is None:
            skipped += 1
            continue
        if milk_val <= 0:
            zero_milk += 1
            continue

        by_cow[cow_id].append((int(dim_val), milk_val * LBS_TO_KG))

        if idx_305 is not None and idx_305 < len(row):
            val_305 = _parse_number(row[idx_305])
            if val_305 is not None and val_305 > 0:
                latest_305[cow_id] = val_305 * LBS_TO_KG

    if skipped:
        warnings.append(f"Skipped {skipped} row(s) with missing/unparseable values.")
    if zero_milk:
        warnings.append(f"Excluded {zero_milk} row(s) with zero or negative milk yield.")

    cows = [
        _CowLactation(
            cow_id=cow_id,
            parity=None,
            test_days=sorted(records),
            milk_305d=latest_305.get(cow_id),
        )
        for cow_id, records in by_cow.items()
    ]
    return cows, warnings


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _trapezoid_cumulative(
    test_days: list[tuple[int, float]],
    max_dim: int | None = None,
) -> float | None:
    """Trapezoidal cumulative milk yield from test-day records.

    Integration starts at DIM 0 with yield 0 (calving baseline) and trapezoids
    between consecutive observations. If ``max_dim`` is provided, integration
    stops there (with linear interpolation for observations that straddle the
    cap, and extrapolation holding the last yield if observations end earlier).
    If ``max_dim`` is None, the integration covers only the observed range —
    no extrapolation past the last observation.

    Args:
        test_days (list[tuple[int, float]]): Sorted (DIM, milk_kg) pairs.
        max_dim (int | None): Integration cap in days. ``305`` for the ICAR
            305-d yield; ``None`` for total observed lactation yield.

    Returns:
        float | None: Estimated cumulative yield in kg, or None if there are
        fewer than two usable observations.

    """
    usable = [(d, y) for d, y in test_days if d > 0]
    if len(usable) < 2:
        return None

    total = 0.0
    prev_d, prev_y = 0, 0.0
    for d, y in usable:
        if max_dim is not None and d > max_dim:
            if prev_d >= max_dim:
                break
            frac = (max_dim - prev_d) / (d - prev_d)
            y_cap = prev_y + frac * (y - prev_y)
            total += (prev_y + y_cap) / 2 * (max_dim - prev_d)
            return total
        total += (prev_y + y) / 2 * (d - prev_d)
        prev_d, prev_y = d, y

    if max_dim is not None and prev_d < max_dim:
        total += prev_y * (max_dim - prev_d)
    return total


def _mean_near_dim(
    cows: list[_CowLactation],
    target_dim: int,
    window: int,
) -> float | None:
    """Mean milk yield across cows at ``target_dim`` (±window), per-cow nearest fallback.

    For each cow we first look for test-days inside the window. If the window
    is empty for that cow but the cow has any observation close enough
    (within ``window * 3``), we fall back to the nearest. Cows with no nearby
    observation are skipped.
    """
    per_cow_means: list[float] = []
    for cow in cows:
        if not cow.test_days:
            continue
        in_window = [y for d, y in cow.test_days if abs(d - target_dim) <= window]
        if in_window:
            per_cow_means.append(sum(in_window) / len(in_window))
            continue
        closest = min(cow.test_days, key=lambda dy: abs(dy[0] - target_dim))
        if abs(closest[0] - target_dim) <= window * 3:
            per_cow_means.append(closest[1])

    if not per_cow_means:
        return None
    return statistics.fmean(per_cow_means)


def _aggregate_test_days(
    cows: list[_CowLactation],
) -> tuple[dict[str, float], list[str]]:
    warnings: list[str] = []
    stats: dict[str, float] = {}

    # AchievedMilk = herd-average of cumulative lactation yield per cow (kg).
    # Integrated across each cow's full observed range, no 305-d cap.
    cumulative_per_cow = [
        y for y in (_trapezoid_cumulative(c.test_days, max_dim=None) for c in cows) if y is not None
    ]
    if cumulative_per_cow:
        stats["AchievedMilk"] = statistics.fmean(cumulative_per_cow)

    near_21 = _mean_near_dim(cows, 21, _DIM_21_WINDOW)
    if near_21 is not None:
        stats["Achieved21Milk"] = near_21
    else:
        warnings.append("No test-day records near DIM 21 — Achieved21Milk left to slider default.")

    near_75 = _mean_near_dim(cows, 75, _DIM_75_WINDOW)
    if near_75 is not None:
        stats["Achieved75Milk"] = near_75
    else:
        warnings.append("No test-day records near DIM 75 — Achieved75Milk left to slider default.")

    max_dims = [max(d for d, _ in cow.test_days) for cow in cows if cow.test_days]
    if max_dims:
        stats["DaysInMilk"] = statistics.fmean(max_dims)

    direct_305 = [c.milk_305d for c in cows if c.milk_305d is not None]
    if len(direct_305) >= max(1, int(0.8 * len(cows))):
        stats["Achieved305Milk"] = statistics.fmean(direct_305)
    else:
        estimates = [
            y
            for y in (_trapezoid_cumulative(c.test_days, max_dim=305) for c in cows)
            if y is not None
        ]
        dropped = len(cows) - len(estimates)
        if estimates:
            stats["Achieved305Milk"] = statistics.fmean(estimates)
            if dropped / max(len(cows), 1) > 0.2:
                warnings.append(
                    f"Dropped {dropped} cow(s) from 305-d estimate (too few test-day records)."
                )
        else:
            warnings.append("Could not estimate Achieved305Milk — left to slider default.")

    return stats, warnings
