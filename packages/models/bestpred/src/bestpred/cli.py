"""Command-line interface for the Python BESTPRED port."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from bestpred.compare_bovi import (
    DEFAULT_BOVI_PATH,
    compare_records_with_bovi,
    format_comparison_summary,
    format_comparison_table,
    read_records_for_source,
    run_bovi_best_predict,
    summarize_comparison_rows,
)
from bestpred.core.kernel import predict_pcdart_projected_actual_305, predict_records
from bestpred.core.source11 import simulate_source11_records
from bestpred.io.dcr import read_dcr_results, write_compatibility_dcr
from bestpred.io.parameters import read_parameters
from bestpred.io.pcdart import write_pcdart_output
from bestpred.io.source10 import read_source10_records
from bestpred.io.source11 import read_source11_examples
from bestpred.io.source14 import read_source14_records, read_source24_records
from bestpred.io.source15 import read_source15_records
from bestpred.models import BestpredRunRequest, BestpredSource


def _run_source11(request: BestpredRunRequest) -> int:
    parameters = read_parameters(request.parameter_path)
    examples = read_source11_examples(request.input_path)
    records = simulate_source11_records(examples, parameters)

    if request.oracle_output_path is not None:
        # Compatibility mode for the first porting milestone: preserve the
        # validated Fortran oracle while the pure Python kernel is being ported.
        rows = read_dcr_results(request.oracle_output_path)
        write_compatibility_dcr(request.output_path, rows)
        return 0

    rows = predict_records(records, parameters)
    write_compatibility_dcr(request.output_path, rows)
    return 0


def _run_source10(request: BestpredRunRequest) -> int:
    parameters = read_parameters(request.parameter_path)
    records = read_source10_records(request.input_path)

    if request.oracle_output_path is not None:
        rows = read_dcr_results(request.oracle_output_path)
        write_compatibility_dcr(request.output_path, rows)
        return 0

    rows = predict_records(records, parameters, source11_compat=False)
    write_compatibility_dcr(request.output_path, rows)
    return 0


def _run_source15(request: BestpredRunRequest) -> int:
    parameters = read_parameters(request.parameter_path)
    means_path = request.input_path.with_suffix(".means")
    records = read_source15_records(request.input_path, means_path)

    if request.oracle_output_path is not None:
        rows = read_dcr_results(request.oracle_output_path)
        write_compatibility_dcr(request.output_path, rows)
        return 0

    rows = predict_records(records, parameters, source11_compat=False)
    write_compatibility_dcr(request.output_path, rows)
    return 0


def _run_source14(request: BestpredRunRequest) -> int:
    parameters = read_parameters(request.parameter_path)
    records = read_source14_records(request.input_path)

    if request.oracle_output_path is not None:
        rows = read_dcr_results(request.oracle_output_path)
        write_compatibility_dcr(request.output_path, rows)
        return 0

    rows = predict_records(records, parameters, source11_compat=False)
    write_compatibility_dcr(request.output_path, rows)
    if request.pcdart_output_path is not None:
        projected_actuals = predict_pcdart_projected_actual_305(records, parameters)
        write_pcdart_output(
            request.pcdart_output_path,
            records=records,
            rows=rows,
            projected_actuals=projected_actuals,
            include_compatibility_rows=True,
        )
    return 0


def _run_source24(request: BestpredRunRequest) -> int:
    parameters = read_parameters(request.parameter_path)
    records = read_source24_records(request.input_path)

    if request.oracle_output_path is not None:
        rows = read_dcr_results(request.oracle_output_path)
        write_compatibility_dcr(request.output_path, rows)
        return 0

    rows = predict_records(records, parameters, source11_compat=False)
    write_compatibility_dcr(request.output_path, rows)
    if request.pcdart_output_path is not None:
        projected_actuals = predict_pcdart_projected_actual_305(records, parameters)
        write_pcdart_output(
            request.pcdart_output_path,
            records=records,
            rows=rows,
            projected_actuals=projected_actuals,
            include_compatibility_rows=False,
        )
    return 0


def run(request: BestpredRunRequest) -> int:
    """Run BESTPRED for the requested source."""

    if request.source == BestpredSource.FORMAT4:
        return _run_source10(request)
    if request.source == BestpredSource.FORMAT4_WITH_MEANS:
        return _run_source15(request)
    if request.source == BestpredSource.DCR_EXAMPLE:
        return _run_source11(request)
    if request.source == BestpredSource.DRMS:
        return _run_source14(request)
    if request.source == BestpredSource.DRMS_FILE_LIST:
        return _run_source24(request)
    raise NotImplementedError(f"Source {request.source.value} is not implemented yet.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bestpred")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run BESTPRED")
    run_parser.add_argument("--source", type=int, required=True)
    run_parser.add_argument("--input", type=Path, required=True)
    run_parser.add_argument("--par", type=Path, required=True)
    run_parser.add_argument("--output", type=Path, required=True)
    run_parser.add_argument("--oracle-output", type=Path)
    run_parser.add_argument("--pcdart-output", type=Path)

    compare_parser = subparsers.add_parser(
        "compare-bovi",
        help="Compare bestpred-py 305-day milk with Bovi's current dataframe best-predict",
    )
    compare_parser.add_argument("--source", type=int, required=True)
    compare_parser.add_argument("--input", type=Path, required=True)
    compare_parser.add_argument("--par", type=Path, required=True)
    compare_parser.add_argument("--bovi-path", type=Path, default=DEFAULT_BOVI_PATH)
    compare_parser.add_argument("--bovi-python", type=Path)
    compare_parser.add_argument("--limit", type=int, default=20)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "compare-bovi":
        try:
            source = BestpredSource(args.source)
            records, parameters = read_records_for_source(
                source,
                input_path=args.input,
                parameter_path=args.par,
            )
            rows = compare_records_with_bovi(
                records,
                parameters,
                bovi_runner=lambda bovi_rows: run_bovi_best_predict(
                    bovi_rows,
                    bovi_path=args.bovi_path,
                    bovi_python=args.bovi_python,
                ),
                source11_compat=source == BestpredSource.DCR_EXAMPLE,
            )
            print(format_comparison_table(rows, limit=args.limit))
            print(format_comparison_summary(summarize_comparison_rows(rows)))
            return 0
        except (FileNotFoundError, ModuleNotFoundError, subprocess.CalledProcessError) as exc:
            print(f"Bovi comparison failed: {exc}", file=sys.stderr)
            return 1
        except NotImplementedError as exc:
            print(str(exc), file=sys.stderr)
            return 2

    if args.command != "run":
        raise AssertionError("argparse should only dispatch known subcommands")

    request = BestpredRunRequest(
        source=BestpredSource(args.source),
        input_path=args.input,
        output_path=args.output,
        parameter_path=args.par,
        oracle_output_path=args.oracle_output,
        pcdart_output_path=args.pcdart_output,
    )

    try:
        return run(request)
    except NotImplementedError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except FileNotFoundError as exc:
        print(f"Missing file: {exc}", file=sys.stderr)
        return 1


def copy_oracle_output(source: Path, destination: Path) -> None:
    """Copy a Fortran oracle output file.

    Kept as a tiny explicit helper for tests that need byte-identical
    compatibility output during the kernel port.
    """

    shutil.copyfile(source, destination)


if __name__ == "__main__":
    raise SystemExit(main())
