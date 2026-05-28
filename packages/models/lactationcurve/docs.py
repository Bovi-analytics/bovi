from pathlib import Path

import pdoc

pdoc.pdoc(
    "src/lactationcurve",
    "lactationcurve.characteristics.best_predict",
    "lactationcurve.characteristics.lactation_curve_characteristics",
    "lactationcurve.characteristics.method_test_interval",
    output_directory=Path("docs"),
)
