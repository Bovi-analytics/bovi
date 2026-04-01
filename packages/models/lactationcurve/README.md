# lactationcurve (Python)

A toolkit for fitting **dairy cow lactation curves**, evaluating **lactation curve characteristics (LCCs)** (time to peak, peak yield, cumulative yield, persistency), and computing **305‑day milk yield** using the **ICAR guideline**.

> **Contact:** Meike van Leerdam, mbv32@cornell.edu
>
> **Authors:** Judith Osei-Tete, Douwe de Kok, Lucia Trapanese & Meike van Leerdam

> **Initial authored:** 2025‑08‑12

> **Updated:** 2026‑02‑12

---

## Main Lactation curve models implemented:

_MilkBot_ – Flexible four-parameter model describing rise, peak, and decline. (Both frequentist and Bayesian fitting
available)

_Wood_ – Classic three-parameter gamma function model.

_Wilmink_ – Linear–exponential hybrid model, with fixed or estimated decay rate.

_Ali & Schaeffer_ – Polynomial–logarithmic model for complex curve shapes.

_Fischer_ – Simplified exponential decay model.

Additional models available for a.o. symbolic LCC derivations:
**Brody** ,  **Sikka** ,  **Nelder** ,  **Dhanoa** ,  **Emmans** ,  **Hayashi** ,  **Rook** ,  **Dijkstra** ,  **Prasad** .

## Model Formulas

* **Wood** : `y(t) = a * t^b * exp(-c * t)`
* **Wilmink** : `y(t) = a + b * t + c * exp(k * t)` with default `k = -0.05`
* **Ali & Schaeffer** :  `t_scaled = t / 340`, `L = ln(340 / t)`

  `y(t) = a + b*t_scaled + c*t_scaled^2 + d*L + k*L^2`
* **Fischer** : `y(t) = a - b*t - a*exp(-c*t)`
* **MilkBot** : `y(t) = a * (1 - exp((c - t)/b) / 2) * exp(-d*t)`

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

## API Overview

The package is organized into three main modules:

1. `lactationcurve.fitting`
2. `lactationcurve.characteristics`
3. `lactationcurve.preprocessing`


## Output Types Summary

| Function | Output |

|---------|--------|

| `fit_lactation_curve` | Predicted yields (np.ndarray) |

| `get_lc_parameters` | Tuple of numerical parameters |

| `bayesian_fit_milkbot_single_lactation` | Dict of MilkBot parameters |

| `lactation_curve_characteristic_function` | (expr, params, func) |

| `calculate_characteristic` | float (LCC value) |

| `test_interval_method` | DataFrame with 305‑day totals |

## Bayesian (MilkBot API)

* Set `fitting="bayesian"` and `model="milkbot"` in `fit_lactation_curve` or `calculate_characteristic`.
* Provide an **API key** via .env
* Choose priors via `continent="USA" | "EU" | "CHEN"` ([CHEN](https://github.com/Bovi-analytics/Chen-et-al-2023b) supplies published priors from literature).
* The helper `bayesian_fit_milkbot_single_lactation(...)` normalizes differing API responses.
* The key can be requested by sending an email to Jim Ehrlich [jehrlich@MilkBot.com](mailto:jehrlich@MilkBot.com).
* More information about the API can be found [here](https://api.milkbot.com/).

# Citation

**Citing the lactationcurve package**

If you use the `lactationcurve` package in your research, please consider citing it as follows:

*van Leerdam, M. B., de Kok, D., Osei-Tete, J. A., & Hostens, M. (2026). Bovi-analytics/bovi: v.1.1.1. (v.1.0.6).*

*Zenodo. https://doi.org/10.5281/zenodo.18715145*


BibTex:

@software{van_leerdam_2026_lactationcurve,

  author       = {van Leerdam, Meike Beatrijs and de Kok, D. and Osei-Tete, J. A. and Hostens, M.},

  title        = {Bovi-analytics/lactation\_curve\_core: v.1.1.1},

  version      = {1.0.6},

  year         = {2026},

  publisher    = {Zenodo},

  doi          = {10.5281/zenodo.18715145},

  url          = {https://doi.org/10.5281/zenodo.18715145}

}

``


A machine-readable citation is included in `CITATION.cff


If you also use the Bayesian fitting functionality that relies on the MilkBot API, please also cite the following paper:

*Ehrlich, J.L., 2013. Quantifying inter-group variability in lactation curve shape and magnitude with the MilkBot® lactation model. PeerJ 1, e54.*

*https://doi.org/10.7717/peerj.54*


# License

[MIT License](https://github.com/Bovi-analytics/bovi/blob/master/LICENSE)
