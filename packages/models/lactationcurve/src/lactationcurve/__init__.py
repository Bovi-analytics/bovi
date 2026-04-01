"""

A package for fitting **dairy animal lactation curves**, evaluating
**lactation curve characteristics (LCCs)** (time to peak, peak yield,
cumulative yield, persistency), and computing **305-day milk yield**
using the **ICAR guideline**.

> **Contact:** Meike van Leerdam, mbv32@cornell.edu
>
> **Authors:** Meike van Leerdam,Judith Osei-Tete, Douwe de Kok, Lucia Trapanese

> **Initial authored:** 2025‑08‑12

> **Updated:** 2026‑02‑24

---

## Main Lactation curve models implemented:

MilkBot: Flexible four-parameter model describing rise, peak,
and decline. (Both frequentist and Bayesian fitting available)

Wood: Incomplete gamma function; most popular due to its simplicity,
its stability in the presence of missing data, and its
computational ease.

Wilmink: Linear–exponential hybrid model, with fixed or estimated decay rate.

Ali & Schaeffer: Polynomial-logarithmic model with a linear
regression component for more complex curve shapes.

Fischer: Simple exponential decay model.

Additional models available for a.o. symbolic LCC derivations:
**Brody**, **Sikka**, **Nelder**, **Dhanoa**, **Emmans**,
**Hayashi**, **Rook**, **Dijkstra**, **Prasad**.

---

## Model Formulas

* **Wood** : `y(t) = a * t^b * exp(-c * t)`
* **Wilmink** : `y(t) = a + b * t + c * exp(k * t)` with default `k = -0.05`
* **Ali & Schaeffer** :  `t_scaled = t / 305`, `L = ln(305 / t)`

  `y(t) = a + b*t_scaled + c*t_scaled^2 + d*L + k*L^2`
* **Fischer** : `y(t) = a - b*t - a*exp(-c*t)`
* **MilkBot** : `y(t) = a * (1 - exp((c - t)/b) / 2) * exp(-d*t)`

---

## Features

- **Frequentist fitting** (numeric optimization & least squares):
  - Wood, Wilmink, Ali & Schaeffer, Fischer, MilkBot
- **Bayesian fitting via MilkBot API**:
  - MilkBot
- **Lactation Curve Characteristics** — symbolic + numeric:
  - time_to_peak, peak_yield, cumulative_milk_yield, persistency
- **ICAR procedures cumulative milk yield:**
  - Test Interval Method
- Input validation/normalization via `validate_and_prepare_inputs`
- Caching of symbolic expressions for performance

---

## API Overview

The package is organized into three main modules:

1. `lactationcurve.fitting`
2. `lactationcurve.characteristics`
3. `lactationcurve.preprocessing`

---

## Output Types Summary Of Most Important Functions

| Function | Output |

|---------|--------|

| `fit_lactation_curve` | Predicted yields (np.ndarray) |

| `get_lc_parameters` | Tuple of numerical parameters |

| `bayesian_fit_milkbot_single_lactation` | Dict of MilkBot parameters |

| `lactation_curve_characteristic_function` | (expr, params, func) |

| `calculate_characteristic` | float (LCC value) |

| `test_interval_method` | DataFrame with 305‑day totals |

---

## Bayesian Fitting (MilkBot API)

* Set `fitting="bayesian"` and `model="milkbot"` in
  `fit_lactation_curve` or `calculate_characteristic`.
* Provide an **API key** via .env
* Choose priors via `continent="USA" | "EU" | "CHEN"`
  ([CHEN](https://github.com/Bovi-analytics/Chen-et-al-2023b)
  supplies published priors from literature).
* The helper `bayesian_fit_milkbot_single_lactation(...)`
  normalizes differing API responses.
* The key can be requested by sending an email to Jim Ehrlich
  [jehrlich@MilkBot.com](mailto:jehrlich@MilkBot.com).
* More information about the API can be found in the
  [API documentation](https://api.milkbot.com/), or in the
  corresponding
  [paper](https://peerj.com/articles/54/#MainContent).

---

## Citing the lactationcurve package

If you use the `lactationcurve` package in your research, please consider citing it as follows:

*van Leerdam, M. B., de Kok, D., Osei-Tete, J. A., &
Hostens, M. (2026). Bovi-analytics/bovi:
v.1.0.2. (v.1.0.2). Zenodo.
https://doi.org/10.5281/zenodo.18715145*


If you also use the Bayesian fitting functionality that relies
on the MilkBot API, please also cite the following paper:

*Ehrlich, J.L., 2013. Quantifying inter-group variability
in lactation curve shape and magnitude with the MilkBot
lactation model. PeerJ 1, e54.
https://doi.org/10.7717/peerj.54*

---

## License

[MIT License](https://github.com/Bovi-analytics/bovi/blob/master/LICENSE)


---

## Current version of the package

"""


# import submodules to make them available at the package level

from . import characteristics, fitting, preprocessing

__all__ = ["fitting", "characteristics", "preprocessing"]
# Re-exports are intentionally commented out to avoid duplicate entries
# in pdoc-generated documentation. Use submodule imports instead, e.g.:
#   from lactationcurve.fitting import fit_lactation_curve
#   from lactationcurve.characteristics import calculate_characteristic
# from .characteristics import (
#     calculate_characteristic,
#     lactation_curve_characteristic_function,
#     numeric_cumulative_yield,
#     numeric_peak_yield,
#     numeric_time_to_peak,
#     persistency_fitted_curve,
#     persistency_milkbot,
#     persistency_wood,
#     test_interval_method,
# )
# from .fitting import (
#     ali_schaeffer_model,
#     bayesian_fit_milkbot_single_lactation,
#     brody_model,
#     dhanoa_model,
#     dijkstra_model,
#     emmans_model,
#     fischer_model,
#     fit_lactation_curve,
#     get_chen_priors,
#     get_lc_parameters,
#     get_lc_parameters_least_squares,
#     hayashi_model,
#     milkbot_model,
#     nelder_model,
#     prasad_model,
#     rook_model,
#     sikka_model,
#     wilmink_model,
#     wood_model,
#     build_prior,
# )
# from .preprocessing import (
#     PreparedInputs,
#     standardize_lactation_columns,
#     validate_and_prepare_inputs,
# )

# Expose package version (try metadata, fall back to a sensible dev string)
try:
    from importlib.metadata import PackageNotFoundError, version
except Exception:
    try:
        from importlib_metadata import PackageNotFoundError, version  # type: ignore
    except Exception:
        version = None
        PackageNotFoundError = Exception

if version:
    try:
        __version__ = version("lactationcurve")
    except PackageNotFoundError:
        __version__ = "0+dev"
else:
    __version__ = "0+dev"

__all__.append("__version__")
