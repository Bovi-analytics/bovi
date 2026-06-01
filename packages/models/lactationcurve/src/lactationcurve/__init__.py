"""

A package for fitting **dairy animal lactation curves**, evaluating
**lactation curve characteristics (LCCs)** (time to peak, peak yield,
cumulative yield, persistency), and computing **305-day milk yield**
using the **ICAR guideline**.

> **Contact:** Meike van Leerdam, mbv32@cornell.edu
>
> **Authors:** Meike van Leerdam, Douwe de Kok, Judith Osei-Tete, Lucia Trapanese

> **Initial authored:** 2025‑08‑12

> **Updated:** 2026-05-29

---

  The 305 day yield for milk, fat, and protein is a widely used metric
  in dairy production, and the International Committee for Animal
  Recording (ICAR) provides guidelines outlining approved methods for
  its calculation. However, a global survey of milk recording
  organizations revealed substantial variation in how these methods are
  implemented. The Test Interval Method is used by 74% of the
  organizations, reflecting a preference for methodological simplicity,
  but it comes with trade-offs in estimation accuracy. The use of the
  other approved methods showed wide variation in correction factors,
  standard lactation curves, test day definitions, minimum sample
  requirements, and exclusion criteria. Such inconsistencies can
  introduce yield variability that complicates comparisons, for
  example in international breeding value evaluation, and limit the
  metric’s usefulness in universal models, such as decision support
  tools. Thus, the objective of this work was to reformulate the ICAR
  guideline section 2, procedure 2, into a unified, transparent, and
  accessible software implementation to improve standardization,
  enhance documentation, support continuous development, and increase
  the accuracy of 305 day yield estimation.

  To achieve this, the ICAR guideline was translated into an
  open-source Python package that serves as a reference implementation
  for 305-day yield calculation, with lactation curve modelling at its
  core. In addition to the methods described in the original guideline,
  the package incorporates 14 lactation curve models, including both
  traditional and Bayesian fitting approaches as well as AI-based
  models. It also provides tools to derive biologically relevant
  characteristics such as time to peak, peak yield, cumulative yield,
  and persistency.

  The framework can be directly integrated into analytical workflows,
  allowing users to calculate 305-day yields, fit and compare
  lactation curves, and derive key lactation characteristics through a
  single function call. These functionalities are further supported
  through an interactive website and an openly available [GitHub
  repository](https://github.com/Bovi-analytics/bovi).
  In addition, the project includes an online validation
  platform that enables users to use standardized lactation data to
  compare self-estimated 305-day yields against both reference
  calculations (from the package) and observed daily milk yields.

  We encourage everyone to use, test, and contribute to the package,
  which is available under the MIT license. We welcome feedback and
  suggestions for improvement, and we are committed to maintaining and
  updating the package to ensure it remains a valuable resource
  for the dairy industry and research community.
  For bug reports or feature requests, please submit them on the
  [GitHub issues page](https://github.com/Bovi-analytics/bovi/issues)
  or contact us directly via [email](mbv32@cornell.edu).


## Main Lactation curve models implemented:

**Fischer**:
A combination of an exponential and a linear component, characterized by an
early exponential phase followed by an approximately linear decline.

**Wood**:
An incomplete gamma-type function combining a power-law growth term t^b with
an exponential decay term e^(-ct).

**Ali & Schaeffer**:
A linear regression model based on quadratic polynomials in standardized time
(t/305) and in the log-transformed inverse time log⁡(305/t), where the log
term increases flexibility in modelling the ascending phase and peak of
lactation.

**Wilmink**:
A combination of an exponential and a linear component. In contrast to the
Fischer model, the additional parameter scaling the exponential term increases
flexibility in describing early-lactation dynamics and peak formation.

**MilkBot**:
An empirical, mechanistically motivated four-parameter model describing
lactation as the development and decay of udder capacity. It consists of a
ramp-up phase and exponential decline, with parameters controlling scale,
onset, growth rate, and decay.
Both Bayesian and frequentist fitting approaches are implemented for the
MilkBot model, allowing users to choose between traditional optimization
methods and a probabilistic framework that incorporates prior knowledge.

Additional models available for a.o. symbolic lactation curve
characteristics (LCC) derivations:
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

- **Frequentist fitting** (numeric optimization):
  - Wood, Wilmink, Ali & Schaeffer, Fischer, MilkBot
- **Frequentist fitting** (algebraic least squares):
  - MilkBot
- **Bayesian fitting via MilkBot API**:
  - MilkBot
- **Lactation Curve Characteristics** — symbolic + numeric:
  - time_to_peak, peak_yield, cumulative_milk_yield, persistency
- **ICAR procedures cumulative milk yield:**
  - Test Interval Method
  - Interpolation Standard Lactation Curve (ISLC) Method
  - Best Predict Method
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

| `test_interval_method` | DataFrame with 305‑day totals per TestId |

| `ISLC_method` | DataFrame with 305‑day totals per TestId  |

| `best_predict_method` | DataFrame with 305‑day totals per TestId |

---

## The meaning of a TestId


The `TestId` is an identifier for a lactation,
which can be used to group records belonging to the same lactation together.
It is not the same as a cow ID, as a cow can have multiple lactations
(e.g., across different calvings).
If a `TestId` column is not provided,
the package will assume all records belong to a single lactation
and will create a `TestId` column with all values set to 0.

---

## Often used abreviations

- **DIM**: Days in Milk (the days since calving in the current lactation)
- **LC**: Lactation Curve
- **LCC**: Lactation Curve Characteristic
- **ISLC**: Interpolation using the Standard Lactation Curve
- **ICAR**: International Committee for Animal Recording
- **API**: Application Programming Interface


---

## Bayesian Fitting (MilkBot API)

* Set `fitting="bayesian"` and `model="milkbot"` in
  `fit_lactation_curve` or `calculate_characteristic`.
* Provide an **API key** via .env
* Choose priors via custom_priors:
    - "[CHEN](https://github.com/Bovi-analytics/Chen-et-al-2023b)" → Chen et al. 2023
      published priors
    - dict    → Custom priors in MilkBot format (overrides `continent`)
      Custom priors have a specific format, to help you build them,
      use the `build_prior` helper function. To make your own prior you need
      for each MilkBot parameter
      (scale,ramp,decay,onset) to specify a mean and a standard deviation (std).
      Also provide a standard deviation for milk yield through seMilk to
      specify the expected noise in the data.
      This seMilk is default set to 4 kg to reflect typical day-to-day variation in milk yield,
      but you can adjust it based on the expected variability in your data.

* if no custom priors are provided, the default MilkBot priors will be used:
    In that case set cutsom_priors to None and specify the desired continent and breed.
    continent options:
    - "USA"   → MilkBot USA priors (default)
    - "EU"    → MilkBot EU priors > mainly estimates lower milk production

    breed options:
    - H → Holstein (default)
    - J → Jersey

    If also parity is provided, the continent-specific priors will be
    further refined by parity-specific priors.
    The priors are sensitive to the used metric. The default is kg, but if you use lb,
    specify milk_unit="lb" to use the appropriate priors.

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
v.1.1.4. (v.1.1.4). Zenodo.
https://doi.org/10.5281/zenodo.18715145*


If you also use the Bayesian fitting functionality that relies
on the MilkBot API, please also cite the following paper:

*Ehrlich, J.L., 2013. Quantifying inter-group variability
in lactation curve shape and magnitude with the MilkBot
lactation model. PeerJ 1, e54.
https://doi.org/10.7717/peerj.54*

If you use the 305-day yield calculation methods based on the ICAR guideline,
please also cite the following paper:
Best Predict method:
*VanRaden, P. M. (1997). Lactation yields and accuracies computed from test
day yields and (co) variances by best prediction.
Journal of dairy science, 80(11), 3015-3022.*

ISLC:
*Wilmink, J. B. M. (1987).
Comparison of different methods of predicting 305-day milk yield using means
calculated from within-herd lactation curves. Livestock Production Science, 17, 1-17.*

Test Interval Method:
*Sargent, F. D., V. H. Lyton, and 0. G. Wall, J r . 1968.
Test interval method of calculating Dairy Herd Improvement Association records.
Journal of dairy science, 51-170.*

---

## License

[MIT License](https://github.com/Bovi-analytics/lactation_curve_core/blob/master/LICENSE)


---

## Version v.1.1.4

"""


# import submodules to make them available at the package level

from . import characteristics, fitting, preprocessing

__all__ = ["fitting", "characteristics", "preprocessing"]
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

# __all__ = [
#     # Preprocessing
#     "PreparedInputs",
#     "standardize_lactation_columns",
#     "validate_and_prepare_inputs",
#     # Fitting
#     "ali_schaeffer_model",
#     "bayesian_fit_milkbot_single_lactation",
#     "brody_model",
#     "dhanoa_model",
#     "dijkstra_model",
#     "emmans_model",
#     "fischer_model",
#     "fit_lactation_curve",
#     "get_chen_priors",
#     "get_lc_parameters",
#     "get_lc_parameters_least_squares",
#     "hayashi_model",
#     "milkbot_model",
#     "nelder_model",
#     "prasad_model",
#     "rook_model",
#     "sikka_model",
#     "wilmink_model",
#     "wood_model",
#     # Characteristics
#     "calculate_characteristic",
#     "lactation_curve_characteristic_function",
#     "numeric_cumulative_yield",
#     "numeric_peak_yield",
#     "numeric_time_to_peak",
#     "persistency_fitted_curve",
#     "persistency_milkbot",
#     "persistency_wood",
#     "test_interval_method",
#     "build_prior",
# ]

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
