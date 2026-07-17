"""Compatibility writer for source-14/source-24 `pcdart.bpo` output."""

from __future__ import annotations

from pathlib import Path

from bestpred.models import DcrResultRow, Format4Record


def _to_fortran_int(value: float) -> int:
    return int(value)


def _cow_id7(record: Format4Record) -> str:
    return record.cow_id.strip()[-7:]


def _daily_line(
    *,
    record: Format4Record,
    row: DcrResultRow,
    projected_actual: tuple[float, float, float, float],
    dim: int,
) -> str:
    td_flag = 1 if any(segment.dim == dim for segment in record.segments) else 0
    values = row.numeric_values
    me_305 = tuple(_to_fortran_int(values[index]) for index in (3, 4, 5, 6))
    projected = tuple(_to_fortran_int(value) for value in projected_actual)
    dcr = tuple(_to_fortran_int(values[index]) for index in (0, 1, 2))
    persistency = tuple(values[index] for index in (19, 20, 21, 22))
    reliability = tuple(_to_fortran_int(values[index] * 100.0) for index in range(23, 31))

    return (
        f"{_cow_id7(record):<7} {dim:3d} {td_flag:1d} "
        f"{0.0:5.1f} {0.0:3.1f} {0.0:3.1f} {0.0:3.1f} "
        f"{me_305[0]:5d} {me_305[1]:4d} {me_305[2]:4d} {me_305[3]:4d} "
        f"{projected[0]:5d} {projected[1]:4d} {projected[2]:4d} {projected[3]:4d} "
        f"{me_305[0]:5d} {me_305[1]:4d} {me_305[2]:4d} {me_305[3]:4d} "
        f"{dcr[0]:3d} {dcr[1]:3d} {dcr[2]:3d} "
        f"{persistency[0]:5.2f} {persistency[1]:5.2f} "
        f"{persistency[2]:5.2f} {persistency[3]:5.2f} "
        f"{reliability[0]:2d} {reliability[1]:2d} {reliability[2]:2d} {reliability[3]:2d} "
        f"{reliability[4]:2d} {reliability[5]:2d} {reliability[6]:2d} {reliability[7]:2d} "
        f"{0.0:5.1f} {0.0:3.1f} {0.0:3.1f} {0.0:3.1f}"
    )


def write_pcdart_output(
    path: Path,
    *,
    records: list[Format4Record],
    rows: list[DcrResultRow],
    projected_actuals: list[tuple[float, float, float, float]],
    include_compatibility_rows: bool,
) -> None:
    """Write a Fortran-compatible source-14/source-24 `pcdart.bpo` file."""

    lines: list[str] = []
    for record, row, projected_actual in zip(records, rows, projected_actuals, strict=True):
        if record.compatibility_tag is not None and not include_compatibility_rows:
            continue
        for dim in range(1, 306):
            lines.append(
                _daily_line(
                    record=record,
                    row=row,
                    projected_actual=projected_actual,
                    dim=dim,
                )
            )
    path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")
