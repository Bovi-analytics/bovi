"""Smoke test run by the release workflow against the built wheel.

Checks that the bundled data files made it into the wheel and that the main
entry points run through the public API.
"""

import sys
from importlib import resources

import lactationcurve
from lactationcurve.fitting import fit_lactation_curve

print(f"lactationcurve {lactationcurve.__version__} on Python {sys.version.split()[0]}")

data = resources.files("lactationcurve") / "characteristics" / "data"
for name in [
    "corr_matrix.pkl",
    "covariance_matrix_best_predict.npy",
    "standard_lc_grid.npy",
    "standard_lc_wood.npy",
    "std_per_grid_day.npy",
]:
    assert (data / name).is_file(), f"{name} missing from the wheel"

curve = fit_lactation_curve(
    [10, 40, 70, 100, 150, 200, 250, 300], [30, 38, 36, 33, 29, 25, 21, 18], model="wood"
)
assert len(curve) == 305, len(curve)
print("ok")
