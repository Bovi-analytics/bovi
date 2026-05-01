"""CSV ingestion for Pad B benchmark submissions."""

from __future__ import annotations

import csv
import io


def parse_submission_csv(
    content: bytes,
    return_failed: bool = False,
) -> dict[str, float] | tuple[dict[str, float], list[str]]:
    """Parse a Pad B CSV upload into {cow_id: yield_305day}.

    Args:
        content: Raw CSV bytes. Required columns: cow_id, yield_305day.
        return_failed: If True, return (yields, failed_cow_ids) tuple.

    Returns:
        dict[str, float] or (dict[str, float], list[str]) if return_failed=True.

    Raises:
        ValueError: If file is empty, missing required columns, or all rows invalid.

    """
    if not content.strip():
        raise ValueError("CSV file is empty.")

    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("CSV file is not valid UTF-8.") from exc

    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames is None:
        raise ValueError("CSV file has no headers.")

    normalised = {h.strip().lower(): h for h in reader.fieldnames}
    if "cow_id" not in normalised or "yield_305day" not in normalised:
        missing = [c for c in ("cow_id", "yield_305day") if c not in normalised]
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    cow_col = normalised["cow_id"]
    yield_col = normalised["yield_305day"]

    results: dict[str, float] = {}
    failed: list[str] = []

    for row in reader:
        cow_id = row[cow_col].strip()
        try:
            value = float(row[yield_col])
        except (ValueError, KeyError):
            failed.append(cow_id)
            continue
        if value < 0:
            raise ValueError(f"negative yield for cow '{cow_id}': {value}")
        results[cow_id] = value

    if return_failed:
        return results, failed
    return results
