from pathlib import Path

import pdoc

pdoc.pdoc(
    "lactationcurve",
    output_directory=Path("docs"),
)
