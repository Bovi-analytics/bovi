from pathlib import Path

import pdoc

pdoc.pdoc(
    "src/lactationcurve",
    "lactationcurve.characteristics.best_predict",
    output_directory=Path("docs"),
)
