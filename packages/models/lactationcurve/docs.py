from pathlib import Path

import pdoc

pdoc.pdoc(
    "src/lactationcurve",
    output_directory=Path("docs"),
)
