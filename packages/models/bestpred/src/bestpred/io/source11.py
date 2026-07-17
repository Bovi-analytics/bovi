"""Parser for source-11 `DCRexample.txt` files."""

from __future__ import annotations

from pathlib import Path

from bestpred.models import Source11Example, Source11PlanLine


def _parse_plan_line(line: str) -> Source11PlanLine:
    parts = line.split(maxsplit=9)
    if len(parts) < 9:
        raise ValueError(f"Invalid DCRexample plan line: {line!r}")
    numbers = [int(part) for part in parts[:9]]
    name = parts[9] if len(parts) == 10 else ""
    return Source11PlanLine(
        supervised=numbers[0],
        times_milked=numbers[1],
        times_weighed=numbers[2],
        times_sampled=numbers[3],
        ler_days=numbers[4],
        test_interval=numbers[5],
        first_test=numbers[6],
        last_test=numbers[7],
        parity=numbers[8],
        name=name.strip(),
    )


def read_source11_examples(path: Path) -> list[Source11Example]:
    """Read source-11 examples separated by blank lines."""

    examples: list[Source11Example] = []
    current: list[Source11PlanLine] = []

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("_"):
            if current:
                examples.append(
                    Source11Example(number=len(examples) + 1, plan_lines=tuple(current))
                )
                current = []
            continue
        current.append(_parse_plan_line(line))

    if current:
        examples.append(Source11Example(number=len(examples) + 1, plan_lines=tuple(current)))

    return examples
