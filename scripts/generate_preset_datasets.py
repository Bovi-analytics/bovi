#!/usr/bin/env python3
"""Generate preset cow-dataset JSON blobs for the Curves tab.

Reads Aurora and Sunnyside CSVs from Azure Blob Storage (icarwebsite container,
dataset/ folder), generates 9 JSON samples per dataset (3 sizes × 3 periods),
and uploads them back as preset-datasets/{aurora|sunnyside}/{size}_{period}.json.

Usage:
    uv run python scripts/generate_preset_datasets.py [--dry-run]

Requires CONNECTION_STRING env var (full Azure Storage connection string).
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
from datetime import datetime, timezone
from typing import cast

import pandas as pd
from azure.storage.blob import BlobServiceClient

LBS_TO_KG = 0.45359237
CONTAINER = "icarwebsite"

SIZES: dict[str, int] = {"small": 200, "medium": 1000, "large": 5000}

DATASET_CONFIGS: dict[str, dict] = {
    "aurora": {
        "blob_path": "dataset/AuroraTDM23_26.csv",
        "id_col": "ID",
        "parity_col": "LACT",
        "dim_col": "DIM",
        "milk_col": "MILK",
        "date_col": "TestDate",
        "bdat_col": "BDAT",
        "periods": {
            "recent": ("2025-01-01", None),
            "old": (None, "2023-12-31"),
            "mixed": (None, None),
        },
    },
    "sunnyside": {
        "blob_path": "dataset/MilkRecordingsSunnyside.csv",
        "id_col": "ID",
        "parity_col": "LACT",
        "dim_col": "DIM",
        "milk_col": "MILK",
        "date_col": "TestDate",
        "bdat_col": None,
        "periods": {
            "recent": ("2020-01-01", None),
            "old": (None, "2009-12-31"),
            "mixed": (None, None),
        },
    },
}


def _read_blob_csv(client: BlobServiceClient, blob_path: str) -> pd.DataFrame:
    """Download a CSV blob and return as a DataFrame. Auto-detects separator."""
    print(f"  Downloading {blob_path} …", end=" ", flush=True)
    data = client.get_blob_client(CONTAINER, blob_path).download_blob().readall()
    print(f"{len(data) / 1_048_576:.1f} MB")
    # Sniff separator from the first line
    first_line = data[: data.index(b"\n")].decode("utf-8", errors="replace")
    sep = ";" if first_line.count(";") > first_line.count(",") else ","
    df = pd.read_csv(io.BytesIO(data), sep=sep, quotechar='"', low_memory=False)
    # Normalise column names (strip whitespace)
    df.columns = [c.strip() for c in df.columns]
    print(f"  → {len(df):,} rows, columns: {list(df.columns)}")
    return df


def _stratified_sample(lactations: list[dict], n: int) -> list[dict]:
    """Return n records sampled proportionally across parity values."""
    if len(lactations) <= n:
        return lactations
    # Group by parity
    by_parity: dict[int | None, list[dict]] = {}
    for lac in lactations:
        p = lac["parity"]
        by_parity.setdefault(p, []).append(lac)
    total = len(lactations)
    selected: list[dict] = []
    for parity_key, group in by_parity.items():
        quota = max(1, round(n * len(group) / total))
        # Sample without replacement, capped to group size
        import random

        selected.extend(random.sample(group, min(quota, len(group))))
    # Trim or top-up to exactly n
    random.shuffle(selected)
    if len(selected) > n:
        selected = selected[:n]
    elif len(selected) < n:
        remainder = [lac for lac in lactations if lac not in selected]
        import random

        selected.extend(random.sample(remainder, min(n - len(selected), len(remainder))))
    return selected


def _build_lactations(
    df: pd.DataFrame,
    config: dict,
    period: str,
) -> list[dict]:
    """Parse the DataFrame into a list of lactation dicts filtered by period."""
    id_col = config["id_col"]
    parity_col = config["parity_col"]
    dim_col = config["dim_col"]
    milk_col = config["milk_col"]
    date_col = config["date_col"]
    bdat_col = config["bdat_col"]
    period_bounds: tuple[str | None, str | None] = config["periods"][period]

    # Parse and clean required columns
    df[dim_col] = pd.to_numeric(df[dim_col], errors="coerce")
    # Milk: strip whitespace, handle European decimal comma
    milk_series = df[milk_col].astype(str).str.strip().str.replace(",", ".", regex=False)
    df["_milk_kg"] = cast(pd.Series, pd.to_numeric(milk_series, errors="coerce")) * LBS_TO_KG
    df[parity_col] = pd.to_numeric(df[parity_col], errors="coerce")

    # Drop rows without usable DIM or milk
    df = df.dropna(subset=[id_col, dim_col, "_milk_kg"])
    df = cast(pd.DataFrame, df[df[dim_col] >= 0])
    df = cast(pd.DataFrame, df[df["_milk_kg"] > 0])

    # Parse TestDate for period filtering
    df["_date"] = pd.to_datetime(df[date_col], errors="coerce", dayfirst=False)

    # Parse BDAT (birth date) for display label in Aurora
    if bdat_col and bdat_col in df.columns:
        df["_bdat"] = pd.to_datetime(df[bdat_col], errors="coerce", dayfirst=False)
    else:
        df["_bdat"] = pd.NaT

    # Group by (ID, parity) → one dict per unique lactation
    # Cast ID and parity to string/int for JSON serialisability
    df["_id"] = df[id_col].astype(str).str.strip()
    df["_parity"] = df[parity_col].astype("Int64")  # nullable int

    lactations: list[dict] = []

    for group_key, group in df.groupby(["_id", "_parity"], sort=False):
        cow_id, parity_val = cast(tuple[str, object], group_key)
        group_sorted = group.sort_values(dim_col)
        latest_date = group_sorted["_date"].max()
        # Period filter
        low, high = period_bounds
        if low and not bool(pd.isna(latest_date)) and latest_date < pd.Timestamp(low):
            continue
        if high and not bool(pd.isna(latest_date)) and latest_date > pd.Timestamp(high):
            continue

        parity_int: int | None = (
            int(cast(int, parity_val)) if not bool(pd.isna(parity_val)) else None
        )
        dim_list = group_sorted[dim_col].astype(int).tolist()
        milk_list = [round(v, 2) for v in group_sorted["_milk_kg"].tolist()]

        # Build display name
        bdat_year: int | None = None
        if bdat_col:
            bdat_vals = group_sorted["_bdat"].dropna()
            if not bdat_vals.empty:
                bdat_year = int(bdat_vals.iloc[0].year)

        if bdat_year is not None:
            display_name = f"Cow {cow_id} (b. {bdat_year}) — parity {parity_int}"
        else:
            display_name = f"Cow {cow_id} — parity {parity_int}"

        lactations.append(
            {
                "cow_id": f"{cow_id}_{parity_int}",
                "display_name": display_name,
                "parity": parity_int,
                "dim": dim_list,
                "milk_kg": milk_list,
            }
        )

    print(f"    period={period}: {len(lactations):,} lactations before size sampling")
    return lactations


def _generate_blob(
    lactations: list[dict],
    dataset_key: str,
    size_key: str,
    period: str,
) -> bytes:
    """Serialize a preset dataset blob as JSON bytes."""
    size_limit = SIZES[size_key]
    sampled = _stratified_sample(lactations, size_limit)
    payload = {
        "dataset": dataset_key,
        "size": size_key,
        "period": period,
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "cow_count": len(sampled),
        "cows": sampled,
    }
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


def main(dry_run: bool = False) -> None:
    conn_str = os.environ.get("CONNECTION_STRING")
    if not conn_str:
        print("ERROR: CONNECTION_STRING environment variable not set.", file=sys.stderr)
        sys.exit(1)

    client = BlobServiceClient.from_connection_string(conn_str)

    for dataset_key, config in DATASET_CONFIGS.items():
        print(f"\n=== {dataset_key.upper()} ===")
        df = _read_blob_csv(client, config["blob_path"])

        for period in ["recent", "old", "mixed"]:
            print(f"  Period: {period}")
            lactations = _build_lactations(df.copy(), config, period)

            for size_key in ["small", "medium", "large"]:
                blob_data = _generate_blob(lactations, dataset_key, size_key, period)
                dest_path = f"preset-datasets/{dataset_key}/{size_key}_{period}.json"
                cow_count = json.loads(blob_data)["cow_count"]
                kb = len(blob_data) / 1024
                print(f"    {size_key:6s}: {cow_count:5,} cows, {kb:.0f} KB → {dest_path}")
                if not dry_run:
                    client.get_blob_client(CONTAINER, dest_path).upload_blob(
                        blob_data, overwrite=True, content_type="application/json"
                    )

    if dry_run:
        print("\n[dry-run] No blobs were uploaded.")
    else:
        print("\nDone. All blobs uploaded.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Skip blob upload, print stats only")
    args = parser.parse_args()
    main(dry_run=args.dry_run)
