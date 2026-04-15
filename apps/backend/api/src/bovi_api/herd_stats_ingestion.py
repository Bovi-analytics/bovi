"""CSV ingestion and normalization utility for herd stats upload.

Two public functions:
  - parse_csv: reads CSV bytes → raw dict of domain values
  - normalize_herd_stats: maps raw domain values → 0-1 using config ranges

Normalization is inlined here (not imported from lactation_autoencoder) to keep
the central API free of ML framework dependencies.
"""

import csv
import io
from typing import Literal

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

# Case-insensitive column header → canonical stat name
_ALIASES: dict[str, str] = {name.lower(): name for name in CANONICAL_NAMES}
_ALIASES.update(
    {
        "dim": "DaysInMilk",
        "days_in_milk": "DaysInMilk",
        "daysinmilk": "DaysInMilk",
        "305milk": "Achieved305Milk",
        "21milk": "Achieved21Milk",
        "75milk": "Achieved75Milk",
        "total_milk": "AchievedMilk",
        "totalmilk": "AchievedMilk",
        "calvinginterval": "HistoricCalvingInterval",
        "calving_interval": "HistoricCalvingInterval",
        "qualitysequence": "QualitySequence",
        "quality": "QualitySequence",
        "daysdry": "DaysDry",
        "days_dry": "DaysDry",
        "daysopen": "DaysOpen",
        "days_open": "DaysOpen",
        "dayspregnant": "DaysPregnant",
        "days_pregnant": "DaysPregnant",
    }
)


def _resolve_column(header: str) -> str | None:
    """Map a CSV column header to a canonical stat name, or None if unrecognised."""
    return _ALIASES.get(header.strip().lower())


def _is_binary_content(content: bytes) -> bool:
    """Return True if content looks like non-text binary data."""
    if b"\x00" in content:
        return True
    # Count non-printable, non-whitespace bytes
    non_printable = sum(1 for b in content if b < 0x09 or (0x0E <= b <= 0x1F) or b == 0x7F)
    return len(content) > 0 and non_printable / len(content) > 0.3


def parse_csv(
    content: bytes,
    max_rows: int = 100_000,
) -> tuple[dict[str, float], Literal["aggregated", "individual"], int, list[str]]:
    """Parse uploaded CSV bytes into raw herd stats.

    Args:
        content (bytes): Raw CSV bytes from the uploaded file.
        max_rows (int): Maximum rows to process; excess rows are truncated
            with a warning.

    Returns:
        tuple: (raw_stats, format_detected, row_count, warnings)

    Raises:
        ValueError: If bytes are not parseable as CSV, or no recognised
            columns are found.

    """
    if _is_binary_content(content):
        raise ValueError("Not a valid CSV file — binary content detected")

    try:
        text = content.decode("utf-8", errors="replace")
        reader = csv.DictReader(io.StringIO(text))
        rows = list(reader)
    except Exception as exc:
        raise ValueError("Not a valid CSV file") from exc

    if not rows:
        raise ValueError("Not a valid CSV file — no data rows found")

    # Map headers to canonical names
    col_map: dict[str, str] = {}  # original header → canonical name
    for header in rows[0]:
        canonical = _resolve_column(header)
        if canonical is not None:
            col_map[header] = canonical

    if not col_map:
        expected = ", ".join(CANONICAL_NAMES[:5]) + ", ..."
        raise ValueError(
            f"No recognised herd stat columns found. Expected one or more of: {expected}"
        )

    warnings: list[str] = []

    # Report missing canonical columns
    found_canonical = set(col_map.values())
    missing = [n for n in CANONICAL_NAMES if n not in found_canonical]
    if missing:
        warnings.append(f"Missing columns (absent from result): {', '.join(missing)}")

    # Truncate if over max_rows
    if len(rows) > max_rows:
        rows = rows[:max_rows]
        warnings.append(
            f"File had more rows than the limit of {max_rows}; only the first {max_rows} were used."
        )

    row_count = len(rows)
    format_detected: Literal["aggregated", "individual"] = (
        "aggregated" if row_count == 1 else "individual"
    )

    # Collect values per canonical column
    column_values: dict[str, list[float]] = {c: [] for c in col_map.values()}
    nan_counts: dict[str, int] = {c: 0 for c in col_map.values()}

    for row in rows:
        for header, canonical in col_map.items():
            raw_val = (row.get(header) or "").strip()
            try:
                column_values[canonical].append(float(raw_val))
            except (ValueError, TypeError):
                nan_counts[canonical] += 1

    for canonical, count in nan_counts.items():
        if count > 0:
            warnings.append(
                f"Column '{canonical}' had {count} unparseable value(s); excluded from mean."
            )

    raw_stats: dict[str, float] = {
        canonical: sum(values) / len(values)
        for canonical, values in column_values.items()
        if values
    }

    return raw_stats, format_detected, row_count, warnings


def normalize_herd_stats(
    raw: dict[str, float],
    stat_ranges: dict[str, tuple[float, float]],
) -> dict[str, float]:
    """Map raw domain values to 0–1 using provided ranges.

    Values outside the range are clamped. Only keys present in both raw and
    stat_ranges are normalized; missing keys are omitted from the output.

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
