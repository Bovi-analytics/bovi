# lactationcurve (Python)

A package for fitting **dairy animal lactation curves**, evaluating
**lactation curve characteristics (LCCs)** (time to peak, peak yield,
cumulative yield, persistency), and computing **305-day milk yield**
using the **ICAR guideline**.

> **Contact:** Meike van Leerdam, mbv32@cornell.edu
>
> **Authors:** Meike van Leerdam, Douwe de Kok, Judith Osei-Tete, Lucia Trapanese
>
> **Initial authored:** 2025-08-12
>
> **Updated:** 2026-10-26

---

Lactation curves are widely used in dairy science to model milk production
over time. They provide insights into the lactation performance of individual
animals and herds, enabling better management decisions and genetic evaluations.
The 305-day yield is a standard metric for assessing milk production, and
accurate estimation of this yield is crucial for dairy farmers, researchers,
and breeding programs.

The International Committee for Animal Recording (ICAR) provides guidelines
outlining approved methods for calculating the 305-day yield of milk, fat, and
protein. This package provides an open-source Python implementation of these
methods, with lactation curve modelling at its core.

In addition to the methods described in the original guideline, the package
incorporates 14 lactation curve models, including traditional and Bayesian
fitting approaches as well as AI-based models. It also provides tools to
derive biologically relevant characteristics such as time to peak, peak yield,
cumulative yield, and persistency.

The package can be integrated into analytical workflows to calculate 305-day
yields, fit and compare lactation curves, and derive key lactation
characteristics through a single function call. These functionalities are
supported through an [interactive website](https://bovi-analytics.github.io/bovi/)
and an openly available [GitHub repository](https://github.com/Bovi-analytics/bovi).

We encourage everyone to use, test, and contribute to the package, which is
available under the AGPL-3.0-or-later license. For bug reports or feature
requests, please submit them on the
[GitHub issues page](https://github.com/Bovi-analytics/bovi/issues) or contact
us directly via [email](mailto:mbv32@cornell.edu).

---

## What can this package do?

- **Frequentist fitting of lactation curve models** (numeric optimization):
  - Wood, Wilmink, Ali & Schaeffer, Fischer, MilkBot
- **Frequentist fitting of lactation curve models** (algebraic least squares):
  - MilkBot
- **Bayesian fitting of the MilkBot lactation curve model via MilkBot API**:
  - MilkBot
- **Calculate Lactation Curve Characteristics** — symbolic + numeric:
  - time_to_peak, peak_yield, cumulative_milk_yield, persistency
- **Calculate cumulative milk yield over a lactation period**:
  - Test Interval Method (also called the Fleischmann method)
  - Centering Date Method
  - Interpolation Standard Lactation Curve (ISLC) Method
  - Best Predict Method
- **Data quality control for lactation curves**:
  - Input validation/normalization via `validate_and_prepare_inputs`

---

## Main Lactation curve models implemented

**Fischer**: A combination of an exponential and a linear component,
characterized by an early exponential phase followed by an approximately linear
decline.

**Wood**: An incomplete gamma-type function combining a power-law growth term
`t^b` with an exponential decay term `e^(-ct)`.

**Ali & Schaeffer**: A linear regression model based on quadratic polynomials
in standardized time (`t/305`) and in the log-transformed inverse time
`log(305/t)`.

**Wilmink**: A combination of an exponential and a linear component. An
additional parameter scaling the exponential term increases flexibility in
describing early-lactation dynamics and peak formation.

**MilkBot**: An empirical, mechanistically motivated four-parameter model
describing lactation as the development and decay of udder capacity. Both
Bayesian and frequentist fitting approaches are implemented.

Additional models available for symbolic lactation curve characteristics:
**Brody**, **Sikka**, **Nelder**, **Dhanoa**, **Emmans**, **Hayashi**,
**Rook**, **Dijkstra**, **Prasad**.

---

## Model formulas

- **Wood**: `y(t) = a * t^b * exp(-c * t)`
- **Wilmink**: `y(t) = a + b * t + c * exp(k * t)` with default `k = -0.05`
- **Ali & Schaeffer**: `t_scaled = t / 305`, `L = ln(305 / t)`,
  `y(t) = a + b*t_scaled + c*t_scaled^2 + d*L + k*L^2`
- **Fischer**: `y(t) = a - b*t - a*exp(-c*t)`
- **MilkBot**: `y(t) = a * (1 - exp((c - t)/b) / 2) * exp(-d*t)`

---

## Package structure

The package is organized into three main modules:

1. `lactationcurve.fitting`
2. `lactationcurve.characteristics`
3. `lactationcurve.preprocessing`

Generated API documentation is available at
[bovi-analytics.github.io/bovi/lactationcurve.html](https://bovi-analytics.github.io/bovi/lactationcurve.html).

---

## Core functions

The functions below are the main tools most users need. A **lactation** means
the milk records belonging to one cow after one calving. A **TestId** is the
label used to keep each lactation separate.

| Function | What it does | What it returns |
|---|---|---|
| `fit_lactation_curve` | Fits a mathematical curve to milk records and predicts milk yield for each day of the lactation. | A NumPy array of predicted daily yields. |
| `get_lc_parameters` | Finds the numerical parameters that describe a fitted lactation-curve model. | A tuple of model parameters. |
| `bayesian_fit_milkbot_single_lactation` | Fits the MilkBot model to one lactation using Bayesian statistics and prior knowledge. | A dictionary containing the fitted MilkBot parameters. |
| `lactation_curve_characteristic_function` | Prepares a function for calculating a lactation-curve characteristic, such as peak yield or time to peak. | A symbolic expression, parameter names, and a callable function. |
| `calculate_characteristic` | Calculates one lactation-curve characteristic from milk records or a fitted curve. | One number, such as peak yield, time to peak, or cumulative yield. |
| `test_interval_method` | Estimates total milk yield by calculating the area between consecutive test days. Use this for one or more lactations in a dataframe. | A dataframe with one cumulative milk-yield total per `TestId`. |
| `centering_date_method` | Estimates total milk yield by treating each monthly test-day yield as representative of a centered part of that calendar month. | A dataframe with one cumulative milk-yield total per `TestId`. |
| `ISLC_method` | Estimates the total yield by using a standard lactation curve to fill in missing test days from a predefined grid. | A dataframe with one cumulative milk-yield total per `TestId`. |
| `best_predict_method` | Estimates the total yield by using a standard lactation curve to fill in all missing test days. | A dataframe with one cumulative milk-yield total per `TestId`. |

For the dataframe-based methods, the result usually has two columns:
`TestId` identifies the lactation, and `LactationMilkYield` contains the
estimated cumulative milk yield. The yield uses the same unit as the input
records, such as kilograms or pounds.

---

## The meaning of a `TestId`

The `TestId` is an identifier for a lactation, which can be used to group
records belonging to the same lactation together. It is not the same as a cow
ID, as a cow can have multiple lactations across different calvings.

If a `TestId` column is not provided, the package assumes all records belong to
a single lactation and creates a `TestId` column with all values set to `0`.

---

## Minimum number of records per lactation needed per method

The minimum number of records needed to technically calculate the 305-day
yield depends on the method:

- **Test Interval Method**: At least 2 records.
- **Centering Date Method**: At least 2 records.
- **ISLC Method**: Can be applied to lactations without measurements; the
  result is then the population mean from the standard curve.
- **Best Predict Method**: Can be applied to lactations without measurements;
  the result is then the population mean from the standard curve.
- **Lactation curve models**: In general, at least one record more than the
  number of model parameters is needed. Wood and Fischer (3 parameters) need
  at least 4 records, Wilmink and MilkBot (4 parameters) at least 5 records,
  and Ali & Schaeffer (5 parameters) at least 6 records.

In practice, more records are recommended to ensure a reliable fit and avoid
overfitting. Ideally, records should be well distributed across the lactation
period to capture the curve's shape accurately.

---

## Often used abbreviations

- **DIM**: Days in Milk (the days since calving in the current lactation)
- **LC**: Lactation Curve
- **LCC**: Lactation Curve Characteristic
- **ISLC**: Interpolation using the Standard Lactation Curve
- **ICAR**: International Committee for Animal Recording
- **API**: Application Programming Interface

---

## Bayesian fitting (MilkBot API)

- Set `fitting="bayesian"` and `model="milkbot"` in `fit_lactation_curve` or
  `calculate_characteristic`.
- Provide an **API key** via `.env`.
- Choose priors with `custom_priors`:
  - [`CHEN`](https://github.com/Bovi-analytics/Chen-et-al-2023b) for
    published priors from Chen et al. (2023).
  - A dictionary containing custom MilkBot priors. Use the `build_prior`
    helper to create one. Each MilkBot parameter (`scale`, `ramp`, `decay`,
    `onset`) needs a mean and standard deviation, and `seMilk` specifies the
    expected noise in milk yield. The default `seMilk` is 4 kg.
- If no custom priors are provided, set `custom_priors=None` and specify the
  desired continent and breed:
  - `continent="USA"` (default) or `continent="EU"`.
  - `breed="H"` for Holstein (default) or `breed="J"` for Jersey.
  - If parity is provided, continent-specific priors are further refined by
    parity-specific priors.
- The priors are sensitive to the unit. The default is kilograms; specify
  `milk_unit="lb"` when using pounds.
- The helper `bayesian_fit_milkbot_single_lactation(...)` normalizes differing
  API responses.
- Request an API key by emailing
  [Jim Ehrlich](mailto:jehrlich@MilkBot.com).
- More information is available in the
  [MilkBot API documentation](https://api.milkbot.com/) and the
  [corresponding paper](https://peerj.com/articles/54/#MainContent).

---

## Tutorials

Hands-on programming example notebooks are available in the
[GitHub tutorials and example notebooks](https://github.com/Bovi-analytics/bovi/tree/main/packages/models/lactationcurve/notebooks).
They demonstrate how to import the package, fit lactation curves, calculate
305-day yields, and derive lactation curve characteristics.

---

## Citing the lactationcurve package

If you use the `lactationcurve` package in your research, please consider
citing it as follows:

*van Leerdam, M. B., de Kok, D., Osei-Tete, J. A., & Hostens, M. (2026).
Bovi-analytics/bovi: v.1.1.6. Zenodo.*

https://doi.org/10.5281/zenodo.18715145

If you use the Bayesian fitting functionality that relies on the MilkBot API,
please also cite:

*Ehrlich, J. L. (2013). Quantifying inter-group variability in lactation curve
shape and magnitude with the MilkBot lactation model. PeerJ 1, e54.*

https://doi.org/10.7717/peerj.54

If you use the 305-day yield calculation methods based on the ICAR guideline,
please also cite:

**Best Predict method:** VanRaden, P. M. (1997). Lactation yields and
accuracies computed from test day yields and (co)variances by best prediction.
*Journal of Dairy Science, 80*(11), 3015-3022.

**ISLC:** Wilmink, J. B. M. (1987). Comparison of different methods of
predicting 305-day milk yield using means calculated from within-herd
lactation curves. *Livestock Production Science, 17*, 1-17.

**Test Interval Method:** Sargent, F. D., Lyton, V. H., and Wall, O. G.
(1968). Test interval method of calculating Dairy Herd Improvement Association
records. *Journal of Dairy Science, 51*, 170.

**Centering Date Method:** U.S. Department of Agriculture, Bureau of
Dairying. (1925). *Cow Testing Association Letter No. 1*. Washington, DC.

---

## License

[GNU AGPLv3 License](https://github.com/Bovi-analytics/bovi/blob/main/LICENSE)
