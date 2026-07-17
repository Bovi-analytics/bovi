"""Numerical BESTPRED kernel and Fortran-compatible output entrypoints."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from math import sqrt
from typing import Final, cast

import numpy as np
import numpy.typing as npt

from bestpred.core.adjustments import adjust_3x, expected_daily_yield
from bestpred.core.age import format4_age_factors
from bestpred.core.covariance import observation_covariance
from bestpred.core.curves import InterpolatedCurve, interpolate_curve
from bestpred.core.prediction import solve_prediction_system
from bestpred.core.scs import format4_scs_age_factor
from bestpred.models import BestpredParameters, DcrResultRow, Format4Record, TestDaySegment, Trait

FloatArray = npt.NDArray[np.float64]

LB_PER_KG = 2.205
RESULTS_V2_NUMERIC_FIELD_COUNT: Final = 43
RMONTH: Final[tuple[tuple[float, float, float, float], ...]] = (
    (0.962, 0.960, 0.962, 0.943),
    (0.958, 0.956, 0.958, 0.960),
)
BREED_SD: dict[int, tuple[float, float, float, float]] = {
    1: (2300.0, 115.0, 92.0, 1.28),
    2: (2530.0, 115.0, 92.0, 1.21),
    3: (2324.0, 112.0, 92.0, 1.35),
    4: (2946.0, 119.0, 97.0, 1.34),
    5: (2128.0, 109.0, 89.0, 1.18),
    6: (2332.0, 112.0, 92.0, 1.30),
}
BREED_MEAN: dict[int, tuple[tuple[float, float, float, float], ...]] = {
    1: ((15080.0, 589.0, 505.0, 3.16), (18532.0, 713.0, 582.0, 2.95)),
    2: ((16616.0, 668.0, 586.0, 3.22), (21577.0, 867.0, 714.0, 2.93)),
    3: ((13864.0, 625.0, 481.0, 3.35), (16850.0, 745.0, 551.0, 3.29)),
    4: ((20845.0, 763.0, 656.0, 3.20), (25658.0, 935.0, 771.0, 3.08)),
    5: ((14120.0, 662.0, 531.0, 3.31), (18161.0, 833.0, 644.0, 3.32)),
    6: ((14472.0, 518.0, 476.0, 2.87), (17475.0, 624.0, 544.0, 3.06)),
}


@dataclass(frozen=True)
class Milk305DebugPrediction:
    """Minimal source-11/ST milk prediction state for parity with Fortran."""

    observation_covariance: FloatArray
    covariance_to_305: FloatArray
    deviations: FloatArray
    predicted_deviation_305: float
    herd_305_internal: float
    milk_305_internal: float
    milk_305_actual_output: float
    standard_305_variance: float
    variance_factor: float
    reliability_305: float
    dcr_305: float
    herd_ratio: float
    lactation_3x_factor: float
    age_factor: float
    used_segments: tuple[TestDaySegment, ...]


@dataclass(frozen=True)
class SingleTrait305Prediction:
    """Detailed state for one single-trait BESTPRED calculation."""

    trait: int
    observation_covariance: FloatArray
    covariance_to_305: FloatArray
    covariance_to_365: FloatArray
    covariance_to_laclen: FloatArray
    covariance_to_partial: FloatArray
    covariance_to_persistency: FloatArray
    deviations: FloatArray
    predicted_deviation_305: float
    predicted_deviation_365: float
    predicted_deviation_laclen: float
    predicted_deviation_partial: float
    predicted_persistency: float
    herd_305_internal: float
    herd_365_internal: float
    herd_laclen_internal: float
    herd_partial_internal: float
    yld_305_internal: float
    yld_365_internal: float
    yld_laclen_internal: float
    yld_partial_internal: float
    yld_305_actual_output: float
    yld_365_actual_output: float
    yld_laclen_actual_output: float
    yld_partial_actual_output: float
    standard_305_variance: float
    variance_factor: float
    reliability_305: float
    persistency_reliability: float
    dcr_305: float
    expanded_yield_output: float
    herd_305_output: float
    bumpiness: float
    herd_ratio: float
    lactation_3x_factor: float
    age_factor: float
    used_segments: tuple[TestDaySegment, ...]


@dataclass(frozen=True)
class TestDayObservation:
    """One trait-specific observation used in MT matrix assembly."""

    segment_index: int
    segment: TestDaySegment
    trait: int


@dataclass(frozen=True)
class MultiTraitMfpPrediction:
    """Multi-trait M/F/P prediction state for source-11 `mtrait=3`."""

    observation_covariance: FloatArray
    deviations: FloatArray
    used_observations: tuple[TestDayObservation, ...]
    yld_305_outputs: tuple[float, float, float]
    yld_305_actual_outputs: tuple[float, float, float]
    yld_365_outputs: tuple[float, float, float]
    yld_laclen_outputs: tuple[float, float, float]
    yld_partial_outputs: tuple[float, float, float]
    persistencies: tuple[float, float, float]
    reliability_305: tuple[float, float, float]
    persistency_reliability: tuple[float, float, float]
    dcr_305: tuple[float, float, float]
    expanded_yield_outputs: tuple[float, float, float]
    herd_305_outputs: tuple[float, float, float]


def predict_records(
    records: list[Format4Record],
    parameters: BestpredParameters,
    *,
    source11_compat: bool = True,
) -> list[DcrResultRow]:
    """Predict Fortran-compatible BESTPRED rows for prepared Format-4 records."""

    return [
        _predict_source11_partial_row(record, parameters, source11_compat=source11_compat)
        for record in records
    ]


def predict_pcdart_projected_actual_305(
    records: list[Format4Record],
    parameters: BestpredParameters,
) -> list[tuple[float, float, float, float]]:
    """Predict the source-14/24 `PROJact` 305-day values used by `pcdart.bpo`."""

    projected_actuals: list[tuple[float, float, float, float]] = []
    for record in records:
        prediction_record = (
            record.model_copy(update={"length": 1})
            if record.compatibility_tag == "source14_eof_zero"
            else record
        )
        effective_mtrait = _effective_mtrait(parameters)
        if effective_mtrait == 3:
            mfp_prediction = predict_source11_mfp_multi_trait_debug(
                prediction_record,
                parameters,
            )
            values = (*mfp_prediction.yld_305_actual_outputs, 0.0)
        else:
            values = tuple(
                predict_source11_trait_305_debug(
                    prediction_record,
                    parameters,
                    trait=trait,
                ).yld_305_actual_output
                for trait in range(1, 5)
            )
        projected_actuals.append(cast(tuple[float, float, float, float], values))
    return projected_actuals


def predict_source11_mfp_multi_trait_debug(
    record: Format4Record,
    parameters: BestpredParameters,
) -> MultiTraitMfpPrediction:
    """Run the source-11 multi-trait M/F/P route used when `mtrait=3`."""

    maxlen = parameters.maxlen
    laclen = min(parameters.laclen, maxlen)
    partial_length = min(record.length, maxlen)
    parity_group = min(max(record.parity, 1), 2)
    breed = parameters.breed11
    curves = _build_standard_curves(parity_group=parity_group, breed=breed, parameters=parameters)
    daily_yield = np.vstack([curve.daily_yield for curve in curves])
    daily_sd = np.vstack([curve.daily_sd for curve in curves])
    cumulative_yield = np.vstack([curve.cumulative_yield for curve in curves])
    adjustment = adjust_3x(
        dims=tuple(segment.dim for segment in record.segments),
        length=partial_length,
        fresh_year=int(record.fresh_date[:4]),
        parity=record.parity,
        milkings=tuple(segment.times_milked for segment in record.segments),
        cumulative_yield=cumulative_yield,
        use_3x=parameters.use3x,
        maxlen=maxlen,
    )
    age_factors = tuple(_age_factor_for_trait(record, trait) for trait in range(1, 4))
    herd_305 = np.zeros(3, dtype=np.float64)
    herd_365 = np.zeros(3, dtype=np.float64)
    herd_laclen = np.zeros(3, dtype=np.float64)
    herd_partial = np.zeros(3, dtype=np.float64)
    herd_ratios = np.ones(4, dtype=np.float64)
    standard_305_variances = np.zeros(3, dtype=np.float64)
    variance_factors = np.zeros(3, dtype=np.float64)
    persistency_variances: list[PersistencyVariance] = []

    for trait in range(1, 4):
        herd_average = _herd_average_for_trait(
            record,
            trait,
            breed=breed,
            parity_group=parity_group,
        )
        if parameters.units_in == "P":
            herd_average /= LB_PER_KG
        herd_305[trait - 1] = herd_average * adjustment.lactation_factors[trait - 1]
        herd_365[trait - 1] = herd_305[trait - 1] * (
            curves[trait - 1].cumulative_yield[364] / curves[trait - 1].cumulative_yield[304]
        )
        herd_laclen[trait - 1] = herd_305[trait - 1] * (
            curves[trait - 1].cumulative_yield[laclen - 1] / curves[trait - 1].cumulative_yield[304]
        )
        herd_partial[trait - 1] = herd_305[trait - 1] * (
            curves[trait - 1].cumulative_yield[partial_length - 1]
            / curves[trait - 1].cumulative_yield[304]
        )
        herd_ratios[trait - 1] = herd_305[trait - 1] / curves[trait - 1].cumulative_yield[304]
        standard_305_variances[trait - 1] = _cached_standard_305_variance(
            trait=trait,
            parity_group=parity_group,
            breed=breed,
            maxlen=maxlen,
            int_method=parameters.int_method,
            int_method_scs=parameters.int_method_scs,
            region=parameters.region,
            season=parameters.season,
        )
        breed_sd = BREED_SD[breed][trait - 1]
        if parameters.units_in == "P":
            breed_sd /= LB_PER_KG
        variance_factors[trait - 1] = (
            breed_sd * 305.0 / sqrt(float(standard_305_variances[trait - 1]))
        )
        persistency_variances.append(
            _cached_persistency_variance(
                trait=trait,
                parity_group=parity_group,
                breed=breed,
                maxlen=maxlen,
                standard_305_variance=float(standard_305_variances[trait - 1]),
                dim0=_persistency_dim0(parameters, parity_group, trait),
                estimate_dim0=parameters.dim0flag == 1,
                int_method=parameters.int_method,
                int_method_scs=parameters.int_method_scs,
                region=parameters.region,
                season=parameters.season,
            )
        )

    used_observations = tuple(
        TestDayObservation(segment_index=index, segment=segment, trait=trait)
        for index, segment in enumerate(record.segments)
        if 1 <= segment.dim <= maxlen
        for trait in range(1, 4)
        if _observed_trait_value(segment, trait) > 0
    )
    observation_matrix = np.zeros(
        (len(used_observations), len(used_observations)),
        dtype=np.float64,
    )
    target_count = 15
    covariance_to_targets = np.zeros((target_count, len(used_observations)), dtype=np.float64)
    deviations = np.zeros((len(used_observations), 1), dtype=np.float64)

    for row, observation in enumerate(used_observations):
        observed_trait = observation.trait
        observed = _observed_trait_value(observation.segment, observed_trait)
        if parameters.units_in == "P":
            observed /= LB_PER_KG
        expected = expected_daily_yield(
            trait=observed_trait,
            dim=observation.segment.dim,
            mrd=observation.segment.ler_days,
            daily_yield=daily_yield,
            herd_ratio=tuple(float(value) for value in herd_ratios),
        )
        deviations[row, 0] = (
            observed * adjustment.test_factors[observed_trait - 1, observation.segment_index]
            - expected / age_factors[observed_trait - 1]
        )

        for target_trait in range(1, 4):
            target_index = target_trait - 1
            for target_row, target_length in enumerate((305, 365, laclen, partial_length)):
                covariance = (
                    _covariance_to_partial_for_observation(
                        segment=observation.segment,
                        target_trait=target_trait,
                        observed_trait=observed_trait,
                        partial_length=target_length,
                        daily_sd=daily_sd,
                        parity_group=parity_group,
                        maxlen=maxlen,
                    )
                    if target_row == 3
                    else _covariance_to_lactation_for_observation(
                        segment=observation.segment,
                        target_trait=target_trait,
                        observed_trait=observed_trait,
                        lactation_length=target_length,
                        daily_sd=daily_sd,
                        parity_group=parity_group,
                        maxlen=maxlen,
                    )
                )
                covariance_to_targets[target_index * 5 + target_row, row] = (
                    covariance
                    * variance_factors[observed_trait - 1]
                    * variance_factors[target_trait - 1]
                    / age_factors[observed_trait - 1]
                )
            covariance_to_targets[target_index * 5 + 4, row] = (
                _covariance_to_persistency_for_observation(
                    segment=observation.segment,
                    target_trait=target_trait,
                    observed_trait=observed_trait,
                    daily_sd=daily_sd,
                    parity_group=parity_group,
                    maxlen=maxlen,
                    dim0=persistency_variances[target_trait - 1].dim0,
                    variance_scale=persistency_variances[target_trait - 1].variance_scale,
                )
                * variance_factors[observed_trait - 1]
                * variance_factors[target_trait - 1]
                / age_factors[observed_trait - 1]
            )

        for column, other in enumerate(used_observations[: row + 1]):
            covariance = (
                _observation_covariance_for_traits(
                    observation.segment,
                    other.segment,
                    trait_left=observation.trait,
                    trait_right=other.trait,
                    daily_sd=daily_sd,
                    parity_group=parity_group,
                    maxlen=maxlen,
                )
                * variance_factors[observation.trait - 1]
                * variance_factors[other.trait - 1]
                / (age_factors[observation.trait - 1] * age_factors[other.trait - 1])
            )
            observation_matrix[row, column] = covariance
            observation_matrix[column, row] = covariance

    solved = solve_prediction_system(
        covariance_to_targets=covariance_to_targets,
        observation_covariance=observation_matrix,
        deviations=deviations,
    )
    yld_305_outputs: list[float] = []
    yld_305_actual_outputs: list[float] = []
    yld_365_outputs: list[float] = []
    yld_laclen_outputs: list[float] = []
    yld_partial_outputs: list[float] = []
    persistencies: list[float] = []
    reliabilities: list[float] = []
    persistency_reliabilities: list[float] = []
    dcrs: list[float] = []
    expanded_yields: list[float] = []
    herd_outputs: list[float] = []

    for trait in range(1, 4):
        base_index = (trait - 1) * 5
        yld_305_internal = float(solved.predictions[base_index, 0] + herd_305[trait - 1])
        yld_365_internal = float(solved.predictions[base_index + 1, 0] + herd_365[trait - 1])
        yld_laclen_internal = float(solved.predictions[base_index + 2, 0] + herd_laclen[trait - 1])
        yld_partial_internal = float(
            solved.predictions[base_index + 3, 0] + herd_partial[trait - 1]
        )
        reliability = float(
            solved.reliability_covariance[base_index, base_index]
            / (standard_305_variances[trait - 1] * variance_factors[trait - 1] ** 2)
        )
        persistency_reliability = float(
            solved.reliability_covariance[base_index + 4, base_index + 4]
            / variance_factors[trait - 1] ** 2
        )
        yld_305_outputs.append(
            _internal_output(
                value=yld_305_internal,
                trait=trait,
                target_length=305,
                parameters=parameters,
            )
        )
        yld_305_actual_outputs.append(
            _actual_output_from_internal(
                value=yld_305_internal,
                trait=trait,
                target_length=305,
                age_factor=age_factors[trait - 1],
                factor_3x=adjustment.lactation_factors[trait - 1],
                parameters=parameters,
            )
        )
        yld_365_outputs.append(
            _internal_output(
                value=yld_365_internal,
                trait=trait,
                target_length=365,
                parameters=parameters,
            )
        )
        yld_laclen_outputs.append(
            _internal_output(
                value=yld_laclen_internal,
                trait=trait,
                target_length=laclen,
                parameters=parameters,
            )
        )
        yld_partial_outputs.append(
            _internal_output(
                value=yld_partial_internal,
                trait=trait,
                target_length=partial_length,
                parameters=parameters,
            )
        )
        persistencies.append(float(solved.predictions[base_index + 4, 0]))
        reliabilities.append(reliability)
        persistency_reliabilities.append(persistency_reliability)
        dcrs.append(100.0 * reliability / RMONTH[parity_group - 1][trait - 1])
        expanded_yields.append(
            _expanded_yield_output(
                prediction_internal=yld_305_internal,
                herd_internal=float(herd_305[trait - 1]),
                reliability=reliability,
                trait=trait,
                parameters=parameters,
            )
        )
        herd_outputs.append(
            _internal_output(
                value=float(herd_305[trait - 1]),
                trait=trait,
                target_length=305,
                parameters=parameters,
            )
        )

    return MultiTraitMfpPrediction(
        observation_covariance=observation_matrix,
        deviations=deviations,
        used_observations=used_observations,
        yld_305_outputs=cast(tuple[float, float, float], tuple(yld_305_outputs)),
        yld_305_actual_outputs=cast(
            tuple[float, float, float],
            tuple(yld_305_actual_outputs),
        ),
        yld_365_outputs=cast(tuple[float, float, float], tuple(yld_365_outputs)),
        yld_laclen_outputs=cast(tuple[float, float, float], tuple(yld_laclen_outputs)),
        yld_partial_outputs=cast(tuple[float, float, float], tuple(yld_partial_outputs)),
        persistencies=cast(tuple[float, float, float], tuple(persistencies)),
        reliability_305=cast(tuple[float, float, float], tuple(reliabilities)),
        persistency_reliability=cast(
            tuple[float, float, float],
            tuple(persistency_reliabilities),
        ),
        dcr_305=cast(tuple[float, float, float], tuple(dcrs)),
        expanded_yield_outputs=cast(tuple[float, float, float], tuple(expanded_yields)),
        herd_305_outputs=cast(tuple[float, float, float], tuple(herd_outputs)),
    )


def predict_source11_milk_305_debug(
    record: Format4Record,
    parameters: BestpredParameters,
    *,
    milk_age_factor: float | None = None,
) -> Milk305DebugPrediction:
    """Run the first real Python kernel path: source-11 single-trait milk 305.

    `milk_age_factor=None` uses the ported `aiplage` milk factor from the
    prepared Format 4 record. Passing a float keeps debug experiments explicit.
    """

    prediction = predict_source11_trait_305_debug(
        record,
        parameters,
        trait=1,
        age_factor=milk_age_factor,
    )

    return Milk305DebugPrediction(
        observation_covariance=prediction.observation_covariance,
        covariance_to_305=prediction.covariance_to_305,
        deviations=prediction.deviations,
        predicted_deviation_305=prediction.predicted_deviation_305,
        herd_305_internal=prediction.herd_305_internal,
        milk_305_internal=prediction.yld_305_internal,
        milk_305_actual_output=prediction.yld_305_actual_output,
        standard_305_variance=prediction.standard_305_variance,
        variance_factor=prediction.variance_factor,
        reliability_305=prediction.reliability_305,
        dcr_305=prediction.dcr_305,
        herd_ratio=prediction.herd_ratio,
        lactation_3x_factor=prediction.lactation_3x_factor,
        age_factor=prediction.age_factor,
        used_segments=prediction.used_segments,
    )


def predict_source11_trait_305_debug(
    record: Format4Record,
    parameters: BestpredParameters,
    *,
    trait: int,
    age_factor: float | None = None,
) -> SingleTrait305Prediction:
    """Return detailed single-trait state for parity diagnostics."""

    if not 1 <= trait <= 4:
        raise ValueError(f"Unsupported trait: {trait!r}")

    maxlen = parameters.maxlen
    laclen = min(parameters.laclen, maxlen)
    partial_length = min(record.length, maxlen)
    resolved_age_factor = _age_factor_for_trait(record, trait) if age_factor is None else age_factor
    parity_group = min(max(record.parity, 1), 2)
    breed = parameters.breed11
    curves = _build_standard_curves(parity_group=parity_group, breed=breed, parameters=parameters)
    daily_yield = np.vstack([curve.daily_yield for curve in curves])
    daily_sd = np.vstack([curve.daily_sd for curve in curves])
    cumulative_yield = np.vstack([curve.cumulative_yield for curve in curves])

    adjustment = adjust_3x(
        dims=tuple(segment.dim for segment in record.segments),
        length=partial_length,
        fresh_year=int(record.fresh_date[:4]),
        parity=record.parity,
        milkings=tuple(segment.times_milked for segment in record.segments),
        cumulative_yield=cumulative_yield,
        use_3x=parameters.use3x,
        maxlen=maxlen,
    )

    herd_average = _herd_average_for_trait(
        record,
        trait,
        breed=breed,
        parity_group=parity_group,
    )
    if trait < 4 and parameters.units_in == "P":
        herd_average /= LB_PER_KG
    if trait == 4:
        herd_average *= 305.0
    herd_305_internal = herd_average * adjustment.lactation_factors[trait - 1]
    herd_365_internal = herd_305_internal * (
        curves[trait - 1].cumulative_yield[364] / curves[trait - 1].cumulative_yield[304]
    )
    herd_laclen_internal = herd_305_internal * (
        curves[trait - 1].cumulative_yield[laclen - 1] / curves[trait - 1].cumulative_yield[304]
    )
    herd_partial_internal = herd_305_internal * (
        curves[trait - 1].cumulative_yield[partial_length - 1]
        / curves[trait - 1].cumulative_yield[304]
    )
    herd_ratio = herd_305_internal / curves[trait - 1].cumulative_yield[304]
    standard_305_variance = _cached_standard_305_variance(
        trait=trait,
        parity_group=parity_group,
        breed=breed,
        maxlen=maxlen,
        int_method=parameters.int_method,
        int_method_scs=parameters.int_method_scs,
        region=parameters.region,
        season=parameters.season,
    )
    breed_sd = BREED_SD[breed][trait - 1]
    if trait < 4 and parameters.units_in == "P":
        breed_sd /= LB_PER_KG
    variance_factor = breed_sd * 305.0 / sqrt(standard_305_variance)

    used_segments = tuple(
        segment
        for segment in record.segments
        if 1 <= segment.dim <= maxlen and _observed_trait_value(segment, trait) > 0
    )
    observation_matrix = np.zeros((len(used_segments), len(used_segments)), dtype=np.float64)
    covariance_to_targets = np.zeros((5, len(used_segments)), dtype=np.float64)
    deviations = np.zeros((len(used_segments), 1), dtype=np.float64)
    persistency_variance = _cached_persistency_variance(
        trait=trait,
        parity_group=parity_group,
        breed=breed,
        maxlen=maxlen,
        standard_305_variance=standard_305_variance,
        dim0=_persistency_dim0(parameters, parity_group, trait),
        estimate_dim0=parameters.dim0flag == 1,
        int_method=parameters.int_method,
        int_method_scs=parameters.int_method_scs,
        region=parameters.region,
        season=parameters.season,
    )

    for row, segment in enumerate(used_segments):
        observed = _observed_trait_value(segment, trait)
        if trait < 4 and parameters.units_in == "P":
            observed /= LB_PER_KG
        expected = expected_daily_yield(
            trait=trait,
            dim=segment.dim,
            mrd=segment.ler_days,
            daily_yield=daily_yield,
            herd_ratio=_herd_ratio_vector(trait, herd_ratio),
        )
        deviations[row, 0] = (
            observed * adjustment.test_factors[trait - 1, row] - expected / resolved_age_factor
        )
        for target_row, target_length in enumerate((305, 365, laclen, partial_length)):
            covariance = (
                _covariance_to_partial_for_observation(
                    segment=segment,
                    target_trait=trait,
                    observed_trait=trait,
                    partial_length=target_length,
                    daily_sd=daily_sd,
                    parity_group=parity_group,
                    maxlen=maxlen,
                )
                if target_row == 3
                else _covariance_to_lactation_for_segment(
                    segment=segment,
                    trait=trait,
                    lactation_length=target_length,
                    daily_sd=daily_sd,
                    parity_group=parity_group,
                    maxlen=maxlen,
                )
            )
            covariance_to_targets[target_row, row] = (
                covariance * variance_factor**2 / resolved_age_factor
            )
        covariance_to_targets[4, row] = (
            _covariance_to_persistency_for_segment(
                segment=segment,
                trait=trait,
                daily_sd=daily_sd,
                parity_group=parity_group,
                maxlen=maxlen,
                dim0=persistency_variance.dim0,
                variance_scale=persistency_variance.variance_scale,
            )
            * variance_factor**2
            / resolved_age_factor
        )

        for column, other in enumerate(used_segments[: row + 1]):
            covariance = (
                _segment_observation_covariance(
                    segment,
                    other,
                    trait=trait,
                    daily_sd=daily_sd,
                    parity_group=parity_group,
                    maxlen=maxlen,
                )
                * variance_factor**2
                / resolved_age_factor**2
            )
            observation_matrix[row, column] = covariance
            observation_matrix[column, row] = covariance

    solved = solve_prediction_system(
        covariance_to_targets=covariance_to_targets,
        observation_covariance=observation_matrix,
        deviations=deviations,
    )
    predicted_deviation_305 = float(solved.predictions[0, 0])
    predicted_deviation_365 = float(solved.predictions[1, 0])
    predicted_deviation_laclen = float(solved.predictions[2, 0])
    predicted_deviation_partial = float(solved.predictions[3, 0])
    predicted_persistency = float(solved.predictions[4, 0])
    yld_305_internal = predicted_deviation_305 + herd_305_internal
    yld_365_internal = predicted_deviation_365 + herd_365_internal
    yld_laclen_internal = predicted_deviation_laclen + herd_laclen_internal
    yld_partial_internal = predicted_deviation_partial + herd_partial_internal
    yld_305_actual = _actual_output_from_internal(
        value=yld_305_internal,
        trait=trait,
        target_length=305,
        age_factor=resolved_age_factor,
        factor_3x=adjustment.lactation_factors[trait - 1],
        parameters=parameters,
    )
    yld_365_actual = _actual_output_from_internal(
        value=yld_365_internal,
        trait=trait,
        target_length=365,
        age_factor=resolved_age_factor,
        factor_3x=adjustment.lactation_factors[trait - 1],
        parameters=parameters,
    )
    yld_laclen_actual = _actual_output_from_internal(
        value=yld_laclen_internal,
        trait=trait,
        target_length=laclen,
        age_factor=resolved_age_factor,
        factor_3x=adjustment.lactation_factors[trait - 1],
        parameters=parameters,
    )
    yld_partial_actual = _actual_output_from_internal(
        value=yld_partial_internal,
        trait=trait,
        target_length=partial_length,
        age_factor=resolved_age_factor,
        factor_3x=adjustment.partial_factors[trait - 1],
        parameters=parameters,
    )
    reliability_305 = float(
        solved.reliability_covariance[0, 0] / (standard_305_variance * variance_factor**2)
    )
    persistency_reliability = float(solved.reliability_covariance[4, 4] / variance_factor**2)
    dcr_305 = 100.0 * reliability_305 / RMONTH[parity_group - 1][trait - 1]
    expanded_yield = _expanded_yield_output(
        prediction_internal=yld_305_internal,
        herd_internal=herd_305_internal,
        reliability=reliability_305,
        trait=trait,
        parameters=parameters,
    )
    herd_305_output = _internal_305_output(
        value=herd_305_internal,
        trait=trait,
        parameters=parameters,
    )
    bumpiness = _bumpiness(
        deviations=deviations,
        observation_covariance=observation_matrix,
        segments=used_segments,
        trait=trait,
    )

    return SingleTrait305Prediction(
        trait=trait,
        observation_covariance=observation_matrix,
        covariance_to_305=covariance_to_targets[0:1, :],
        covariance_to_365=covariance_to_targets[1:2, :],
        covariance_to_laclen=covariance_to_targets[2:3, :],
        covariance_to_partial=covariance_to_targets[3:4, :],
        covariance_to_persistency=covariance_to_targets[4:5, :],
        deviations=deviations,
        predicted_deviation_305=predicted_deviation_305,
        predicted_deviation_365=predicted_deviation_365,
        predicted_deviation_laclen=predicted_deviation_laclen,
        predicted_deviation_partial=predicted_deviation_partial,
        predicted_persistency=predicted_persistency,
        herd_305_internal=float(herd_305_internal),
        herd_365_internal=float(herd_365_internal),
        herd_laclen_internal=float(herd_laclen_internal),
        herd_partial_internal=float(herd_partial_internal),
        yld_305_internal=yld_305_internal,
        yld_365_internal=yld_365_internal,
        yld_laclen_internal=yld_laclen_internal,
        yld_partial_internal=yld_partial_internal,
        yld_305_actual_output=yld_305_actual,
        yld_365_actual_output=yld_365_actual,
        yld_laclen_actual_output=yld_laclen_actual,
        yld_partial_actual_output=yld_partial_actual,
        standard_305_variance=standard_305_variance,
        variance_factor=variance_factor,
        reliability_305=reliability_305,
        persistency_reliability=persistency_reliability,
        dcr_305=dcr_305,
        expanded_yield_output=expanded_yield,
        herd_305_output=herd_305_output,
        bumpiness=bumpiness,
        herd_ratio=float(herd_ratio),
        lactation_3x_factor=float(adjustment.lactation_factors[trait - 1]),
        age_factor=resolved_age_factor,
        used_segments=used_segments,
    )


def _predict_source11_partial_row(
    record: Format4Record,
    parameters: BestpredParameters,
    *,
    source11_compat: bool,
) -> DcrResultRow:
    if record.compatibility_tag == "source14_eof_zero":
        return _predict_source14_eof_zero_row(record, parameters)

    numeric_values = [float("nan")] * RESULTS_V2_NUMERIC_FIELD_COUNT
    if record.length >= 1:
        effective_mtrait = _effective_mtrait(parameters)
        if effective_mtrait == 3:
            mfp_prediction = predict_source11_mfp_multi_trait_debug(record, parameters)
            numeric_values[0] = mfp_prediction.dcr_305[0]
            numeric_values[1] = _component_dcr_output(
                fat_dcr=mfp_prediction.dcr_305[1],
                protein_dcr=mfp_prediction.dcr_305[2],
                has_protein_tests=_has_observed_trait(record, 3),
            )
            for trait in range(1, 4):
                _fill_multi_trait_mfp_output(
                    numeric_values=numeric_values,
                    trait=trait,
                    prediction=mfp_prediction,
                )
        else:
            single_predictions = [
                predict_source11_trait_305_debug(record, parameters, trait=trait)
                for trait in range(1, 4)
            ]
            numeric_values[0] = single_predictions[0].dcr_305
            numeric_values[1] = _component_dcr_output(
                fat_dcr=single_predictions[1].dcr_305,
                protein_dcr=single_predictions[2].dcr_305,
                has_protein_tests=_has_observed_trait(record, 3),
            )
            for trait in range(1, 4):
                prediction = single_predictions[trait - 1]
                _fill_single_trait_output(
                    numeric_values=numeric_values,
                    trait=trait,
                    prediction=prediction,
                    parameters=parameters,
                    partial_length=min(record.length, parameters.maxlen),
                    effective_mtrait=effective_mtrait,
                )

        if _has_usable_observed_trait(record, 4, maxlen=parameters.maxlen) or not record.segments:
            scs_prediction = predict_source11_trait_305_debug(record, parameters, trait=4)
            numeric_values[2] = scs_prediction.dcr_305
            _fill_single_trait_output(
                numeric_values=numeric_values,
                trait=4,
                prediction=scs_prediction,
                parameters=parameters,
                partial_length=min(record.length, parameters.maxlen),
                effective_mtrait=effective_mtrait,
            )
        else:
            _fill_missing_scs_output(numeric_values)
        if source11_compat:
            _fill_source11_herd_mean_output(numeric_values=numeric_values, record=record)

    raw_line = _format_partial_results_v2_row(
        animal_id=record.cow_id,
        fresh_date=record.fresh_date,
        dim=record.length,
        numeric_values=tuple(numeric_values),
    )
    return DcrResultRow(
        animal_id=record.cow_id,
        fresh_date=record.fresh_date,
        dim=record.length,
        numeric_values=tuple(numeric_values),
        raw_line=raw_line,
    )


def _predict_source14_eof_zero_row(
    record: Format4Record,
    parameters: BestpredParameters,
) -> DcrResultRow:
    synthetic = record.model_copy(update={"length": 1})
    effective_mtrait = _effective_mtrait(parameters)
    numeric_values = [float("nan")] * RESULTS_V2_NUMERIC_FIELD_COUNT

    if effective_mtrait == 3:
        mfp_prediction = predict_source11_mfp_multi_trait_debug(synthetic, parameters)
        for trait in range(1, 4):
            _fill_multi_trait_mfp_output(
                numeric_values=numeric_values,
                trait=trait,
                prediction=mfp_prediction,
            )
    else:
        single_predictions = [
            predict_source11_trait_305_debug(synthetic, parameters, trait=trait)
            for trait in range(1, 4)
        ]
        for trait in range(1, 4):
            _fill_single_trait_output(
                numeric_values=numeric_values,
                trait=trait,
                prediction=single_predictions[trait - 1],
                parameters=parameters,
                partial_length=1,
                effective_mtrait=effective_mtrait,
            )

    for index in (
        0,
        1,
        2,
        6,
        10,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        27,
        28,
        29,
        30,
    ):
        numeric_values[index] = 0.0
    for index in (31, 32, 33, 34):
        numeric_values[index] = float("nan")
    numeric_values[38] = 0.0
    numeric_values[39] = 0.0
    numeric_values[40] = 0.0
    numeric_values[41] = 0.0
    numeric_values[42] = 0.0

    raw_line = _format_partial_results_v2_row(
        animal_id=record.cow_id,
        fresh_date=record.fresh_date,
        dim=0,
        numeric_values=tuple(numeric_values),
    )
    return DcrResultRow(
        animal_id=record.cow_id,
        fresh_date=record.fresh_date,
        dim=0,
        numeric_values=tuple(numeric_values),
        raw_line=raw_line,
    )


def _format_partial_results_v2_row(
    *,
    animal_id: str,
    fresh_date: str,
    dim: int,
    numeric_values: tuple[float, ...],
) -> str:
    values = " ".join(_format_partial_number(value) for value in numeric_values)
    return f"{animal_id:<17} {fresh_date} {dim:3d} {values}"


def _format_partial_number(value: float) -> str:
    if np.isnan(value):
        return "nan"
    if abs(value) >= 1000:
        return f"{value:.0f}"
    return f"{value:.2f}"


def _effective_mtrait(parameters: BestpredParameters) -> int:
    return parameters.global_mtrait if parameters.global_mtrait > 0 else parameters.mtrait


def _component_dcr_output(
    *,
    fat_dcr: float,
    protein_dcr: float,
    has_protein_tests: bool,
) -> float:
    if not has_protein_tests:
        return fat_dcr
    return (fat_dcr + protein_dcr) / 2.0


def _fill_missing_scs_output(numeric_values: list[float]) -> None:
    for index in (2, 6, 10, 14, 18, 22, 26, 30, 38, 42):
        numeric_values[index] = 0.0


def _fill_source11_herd_mean_output(
    *,
    numeric_values: list[float],
    record: Format4Record,
) -> None:
    # `results_v2.dcr` is written by `bestpred_main.f90`, one level above the
    # core `bestpred()` routine. For source 11, the M/F/P herd columns come from
    # the caller's synthetic source herd means, not from the 3X-adjusted internal
    # `herd305` values used inside the solver.
    numeric_values[35] = float(record.herd_me_milk)
    numeric_values[36] = float(record.herd_me_fat)
    numeric_values[37] = float(record.herd_me_protein)


def _fill_single_trait_output(
    *,
    numeric_values: list[float],
    trait: int,
    prediction: SingleTrait305Prediction,
    parameters: BestpredParameters,
    partial_length: int,
    effective_mtrait: int,
) -> None:
    numeric_values[2 + trait] = _internal_output(
        value=prediction.yld_305_internal,
        trait=trait,
        target_length=305,
        parameters=parameters,
    )
    numeric_values[6 + trait] = _internal_output(
        value=prediction.yld_365_internal,
        trait=trait,
        target_length=365,
        parameters=parameters,
    )
    numeric_values[10 + trait] = _internal_output(
        value=prediction.yld_laclen_internal,
        trait=trait,
        target_length=min(parameters.laclen, parameters.maxlen),
        parameters=parameters,
    )
    numeric_values[14 + trait] = _internal_output(
        value=prediction.yld_partial_internal,
        trait=trait,
        target_length=partial_length,
        parameters=parameters,
    )
    numeric_values[18 + trait] = prediction.predicted_persistency
    numeric_values[22 + trait] = prediction.reliability_305
    numeric_values[26 + trait] = prediction.persistency_reliability
    numeric_values[30 + trait] = _single_trait_yvec_output(
        prediction=prediction,
        parameters=parameters,
        effective_mtrait=effective_mtrait,
    )
    numeric_values[34 + trait] = prediction.herd_305_output
    numeric_values[38 + trait] = prediction.bumpiness


def _fill_multi_trait_mfp_output(
    *,
    numeric_values: list[float],
    trait: int,
    prediction: MultiTraitMfpPrediction,
) -> None:
    index = trait - 1
    numeric_values[2 + trait] = prediction.yld_305_outputs[index]
    numeric_values[6 + trait] = prediction.yld_365_outputs[index]
    numeric_values[10 + trait] = prediction.yld_laclen_outputs[index]
    numeric_values[14 + trait] = prediction.yld_partial_outputs[index]
    numeric_values[18 + trait] = prediction.persistencies[index]
    numeric_values[22 + trait] = prediction.reliability_305[index]
    numeric_values[26 + trait] = prediction.persistency_reliability[index]
    numeric_values[30 + trait] = prediction.expanded_yield_outputs[index]
    numeric_values[34 + trait] = prediction.herd_305_outputs[index]
    numeric_values[38 + trait] = 0.0


def _expanded_yield_output(
    *,
    prediction_internal: float,
    herd_internal: float,
    reliability: float,
    trait: int,
    parameters: BestpredParameters,
) -> float:
    if reliability == 0.0:
        expanded = float("nan")
    else:
        expanded = herd_internal + (prediction_internal - herd_internal) / reliability
    return _internal_305_output(value=expanded, trait=trait, parameters=parameters)


def _single_trait_yvec_output(
    *,
    prediction: SingleTrait305Prediction,
    parameters: BestpredParameters,
    effective_mtrait: int,
) -> float:
    if (
        prediction.trait == 4
        and effective_mtrait == 3
        and prediction.persistency_reliability == 0.0
    ):
        return float("nan")
    if prediction.persistency_reliability == 0.0:
        expanded = prediction.predicted_persistency
    else:
        expanded = prediction.predicted_persistency / prediction.persistency_reliability
    return _internal_305_output(
        value=expanded,
        trait=prediction.trait,
        parameters=parameters,
    )


def _internal_305_output(*, value: float, trait: int, parameters: BestpredParameters) -> float:
    return _internal_output(value=value, trait=trait, target_length=305, parameters=parameters)


def _internal_output(
    *,
    value: float,
    trait: int,
    target_length: int,
    parameters: BestpredParameters,
) -> float:
    output = value
    if trait < 4 and parameters.units_out == "P":
        output *= LB_PER_KG
    if trait == 4:
        output /= target_length
    return output


def _bumpiness(
    *,
    deviations: FloatArray,
    observation_covariance: FloatArray,
    segments: tuple[TestDaySegment, ...],
    trait: int,
) -> float:
    if len(segments) <= 1:
        return 0.0

    x305 = 305.0 if trait == 4 else 1.0
    z_deviations = deviations[:, 0] * 305.0 / (x305 * np.sqrt(np.diag(observation_covariance)))
    sorted_indices = sorted(range(len(segments)), key=lambda index: segments[index].dim)

    bump_sum = 0.0
    variance_sum = 0.0
    for current_index, previous_index in zip(
        sorted_indices[1:],
        sorted_indices[:-1],
        strict=False,
    ):
        difference = z_deviations[current_index] - z_deviations[previous_index]
        bump_sum += float(difference**2)
        covariance = observation_covariance[current_index, previous_index]
        denominator = sqrt(
            float(
                observation_covariance[current_index, current_index]
                * observation_covariance[previous_index, previous_index]
            )
        )
        variance_sum += float(2.0 * (1.0 - covariance / denominator))

    if variance_sum == 0.0:
        return 0.0
    return min(max(bump_sum / variance_sum, 0.0), 99.0)


def _build_standard_curves(
    *,
    parity_group: int,
    breed: int,
    parameters: BestpredParameters,
) -> tuple[InterpolatedCurve, InterpolatedCurve, InterpolatedCurve, InterpolatedCurve]:
    return _cached_standard_curves(
        parity_group=parity_group,
        breed=breed,
        maxlen=parameters.maxlen,
        int_method=parameters.int_method,
        int_method_scs=parameters.int_method_scs,
        region=parameters.region,
        season=parameters.season,
    )


@cache
def _cached_standard_curves(
    *,
    parity_group: int,
    breed: int,
    maxlen: int,
    int_method: str,
    int_method_scs: str,
    region: int,
    season: int,
) -> tuple[InterpolatedCurve, InterpolatedCurve, InterpolatedCurve, InterpolatedCurve]:
    return (
        interpolate_curve(
            trait=Trait.MILK,
            parity_group=parity_group,
            breed=breed,
            method=int_method,
            maxlen=maxlen,
            region=region,
            season=season,
        ),
        interpolate_curve(
            trait=Trait.FAT,
            parity_group=parity_group,
            breed=breed,
            method=int_method,
            maxlen=maxlen,
            region=region,
            season=season,
        ),
        interpolate_curve(
            trait=Trait.PROTEIN,
            parity_group=parity_group,
            breed=breed,
            method=int_method,
            maxlen=maxlen,
            region=region,
            season=season,
        ),
        interpolate_curve(
            trait=Trait.SCS,
            parity_group=parity_group,
            breed=breed,
            method=int_method_scs,
            maxlen=maxlen,
            region=region,
            season=season,
        ),
    )


@cache
def _cached_standard_daily_sd(
    *,
    parity_group: int,
    breed: int,
    maxlen: int,
    int_method: str,
    int_method_scs: str,
    region: int,
    season: int,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    curves = _cached_standard_curves(
        parity_group=parity_group,
        breed=breed,
        maxlen=maxlen,
        int_method=int_method,
        int_method_scs=int_method_scs,
        region=region,
        season=season,
    )
    return (
        curves[0].daily_sd,
        curves[1].daily_sd,
        curves[2].daily_sd,
        curves[3].daily_sd,
    )


def _cached_daily_sd_matrix(
    *,
    parity_group: int,
    breed: int,
    maxlen: int,
    int_method: str,
    int_method_scs: str,
    region: int,
    season: int,
) -> FloatArray:
    return np.vstack(
        _cached_standard_daily_sd(
            parity_group=parity_group,
            breed=breed,
            maxlen=maxlen,
            int_method=int_method,
            int_method_scs=int_method_scs,
            region=region,
            season=season,
        )
    )


@cache
def _cached_standard_305_variance(
    *,
    trait: int,
    parity_group: int,
    breed: int,
    maxlen: int,
    int_method: str,
    int_method_scs: str,
    region: int,
    season: int,
) -> float:
    return _standard_305_variance(
        daily_sd=_cached_daily_sd_matrix(
            parity_group=parity_group,
            breed=breed,
            maxlen=maxlen,
            int_method=int_method,
            int_method_scs=int_method_scs,
            region=region,
            season=season,
        ),
        trait=trait,
        parity_group=parity_group,
        maxlen=maxlen,
    )


def _standard_305_variance(
    *, daily_sd: FloatArray, trait: int, parity_group: int, maxlen: int
) -> float:
    variance = 0.0
    for day_i in range(1, 306):
        for day_j in range(1, 306):
            variance += observation_covariance(
                dim1=day_i,
                trait1=trait,
                supervision1=1,
                milkings1=2,
                samples1=2,
                mrd1=1,
                dim2=day_j,
                trait2=trait,
                supervision2=1,
                milkings2=2,
                samples2=2,
                mrd2=1,
                daily_sd=daily_sd,
                parity_group=parity_group,
                maxlen=maxlen,
            )
    return variance


@dataclass(frozen=True)
class PersistencyVariance:
    """Fortran `varp` setup for one trait/parity persistency target."""

    dim0: float
    variance_scale: float


@cache
def _cached_persistency_variance(
    *,
    trait: int,
    parity_group: int,
    breed: int,
    maxlen: int,
    standard_305_variance: float,
    dim0: int,
    estimate_dim0: bool,
    int_method: str,
    int_method_scs: str,
    region: int,
    season: int,
) -> PersistencyVariance:
    return _persistency_variance(
        daily_sd=_cached_daily_sd_matrix(
            parity_group=parity_group,
            breed=breed,
            maxlen=maxlen,
            int_method=int_method,
            int_method_scs=int_method_scs,
            region=region,
            season=season,
        ),
        trait=trait,
        parity_group=parity_group,
        maxlen=maxlen,
        standard_305_variance=standard_305_variance,
        dim0=dim0,
        estimate_dim0=estimate_dim0,
    )


def _persistency_variance(
    *,
    daily_sd: FloatArray,
    trait: int,
    parity_group: int,
    maxlen: int,
    standard_305_variance: float,
    dim0: int,
    estimate_dim0: bool,
) -> PersistencyVariance:
    d_v1 = 0.0
    d_vd = 0.0
    for day_i in range(1, 306):
        for day_j in range(1, 306):
            covariance = observation_covariance(
                dim1=day_i,
                trait1=trait,
                supervision1=1,
                milkings1=2,
                samples1=2,
                mrd1=1,
                dim2=day_j,
                trait2=trait,
                supervision2=1,
                milkings2=2,
                samples2=2,
                mrd2=1,
                daily_sd=daily_sd,
                parity_group=parity_group,
                maxlen=maxlen,
            )
            d_v1 += day_i * covariance
            d_vd += day_i * day_j * covariance

    resolved_dim0 = d_v1 / standard_305_variance if estimate_dim0 else float(dim0)
    variance = d_vd - 2.0 * d_v1 * resolved_dim0 + standard_305_variance * resolved_dim0**2
    return PersistencyVariance(dim0=resolved_dim0, variance_scale=sqrt(variance))


def _covariance_to_persistency_for_segment(
    *,
    segment: TestDaySegment,
    trait: int,
    daily_sd: FloatArray,
    parity_group: int,
    maxlen: int,
    dim0: float,
    variance_scale: float,
) -> float:
    raw_yield_covariance = 0.0
    raw_dim_covariance = 0.0
    begin = segment.dim - segment.ler_days + 1
    end = segment.dim
    if trait > 1:
        begin = segment.dim - (segment.ler_days - 1) // 2
        end = begin

    for measurement_day in range(begin, end + 1):
        for lactation_day in range(1, 306):
            covariance = observation_covariance(
                dim1=lactation_day,
                trait1=trait,
                supervision1=1,
                milkings1=2,
                samples1=2,
                mrd1=1,
                dim2=measurement_day,
                trait2=trait,
                supervision2=1,
                milkings2=2,
                samples2=2,
                mrd2=1,
                daily_sd=daily_sd,
                parity_group=parity_group,
                maxlen=maxlen,
            )
            raw_yield_covariance += covariance
            raw_dim_covariance += lactation_day * covariance

    averaged_yield_covariance = raw_yield_covariance / (end - begin + 1)
    averaged_dim_covariance = raw_dim_covariance / (end - begin + 1)
    return (averaged_dim_covariance - averaged_yield_covariance * dim0) / variance_scale


def _covariance_to_persistency_for_observation(
    *,
    segment: TestDaySegment,
    target_trait: int,
    observed_trait: int,
    daily_sd: FloatArray,
    parity_group: int,
    maxlen: int,
    dim0: float,
    variance_scale: float,
) -> float:
    raw_yield_covariance = 0.0
    raw_dim_covariance = 0.0
    begin, end = _measurement_range_for_observation(
        dim=segment.dim,
        trait=observed_trait,
        mrd=segment.ler_days,
    )

    for measurement_day in range(begin, end + 1):
        for lactation_day in range(1, 306):
            covariance = observation_covariance(
                dim1=lactation_day,
                trait1=target_trait,
                supervision1=1,
                milkings1=2,
                samples1=2,
                mrd1=1,
                dim2=measurement_day,
                trait2=observed_trait,
                supervision2=1,
                milkings2=2,
                samples2=2,
                mrd2=1,
                daily_sd=daily_sd,
                parity_group=parity_group,
                maxlen=maxlen,
            )
            raw_yield_covariance += covariance
            raw_dim_covariance += lactation_day * covariance

    averaged_yield_covariance = raw_yield_covariance / (end - begin + 1)
    averaged_dim_covariance = raw_dim_covariance / (end - begin + 1)
    return (averaged_dim_covariance - averaged_yield_covariance * dim0) / variance_scale


def _persistency_dim0(parameters: BestpredParameters, parity_group: int, trait: int) -> int:
    return parameters.dim0[(parity_group - 1) * 4 + trait - 1]


def _covariance_to_lactation_for_segment(
    *,
    segment: TestDaySegment,
    trait: int,
    lactation_length: int,
    daily_sd: FloatArray,
    parity_group: int,
    maxlen: int,
) -> float:
    if not 1 <= lactation_length <= maxlen:
        raise ValueError(f"lactation_length must be between 1 and {maxlen}")

    begin = segment.dim - segment.ler_days + 1
    end = segment.dim
    if trait > 1:
        begin = segment.dim - (segment.ler_days - 1) // 2
        end = begin
    covariance = 0.0
    for measurement_day in range(begin, end + 1):
        for lactation_day in range(1, lactation_length + 1):
            covariance += observation_covariance(
                dim1=lactation_day,
                trait1=trait,
                supervision1=1,
                milkings1=2,
                samples1=2,
                mrd1=1,
                dim2=measurement_day,
                trait2=trait,
                supervision2=1,
                milkings2=2,
                samples2=2,
                mrd2=1,
                daily_sd=daily_sd,
                parity_group=parity_group,
                maxlen=maxlen,
            )
    return covariance / (end - begin + 1)


def _covariance_to_lactation_for_observation(
    *,
    segment: TestDaySegment,
    target_trait: int,
    observed_trait: int,
    lactation_length: int,
    daily_sd: FloatArray,
    parity_group: int,
    maxlen: int,
) -> float:
    if not 1 <= lactation_length <= maxlen:
        raise ValueError(f"lactation_length must be between 1 and {maxlen}")

    begin, end = _measurement_range_for_observation(
        dim=segment.dim,
        trait=observed_trait,
        mrd=segment.ler_days,
    )
    covariance = 0.0
    for measurement_day in range(begin, end + 1):
        for lactation_day in range(1, lactation_length + 1):
            covariance += observation_covariance(
                dim1=lactation_day,
                trait1=target_trait,
                supervision1=1,
                milkings1=2,
                samples1=2,
                mrd1=1,
                dim2=measurement_day,
                trait2=observed_trait,
                supervision2=1,
                milkings2=2,
                samples2=2,
                mrd2=1,
                daily_sd=daily_sd,
                parity_group=parity_group,
                maxlen=maxlen,
            )
    return covariance / (end - begin + 1)


def _covariance_to_partial_for_observation(
    *,
    segment: TestDaySegment,
    target_trait: int,
    observed_trait: int,
    partial_length: int,
    daily_sd: FloatArray,
    parity_group: int,
    maxlen: int,
) -> float:
    if not 1 <= partial_length <= maxlen:
        raise ValueError(f"partial_length must be between 1 and {maxlen}")

    covariance = 0.0
    for lactation_day in range(1, partial_length + 1):
        covariance += observation_covariance(
            dim1=segment.dim,
            trait1=observed_trait,
            supervision1=segment.supervised,
            milkings1=segment.times_milked,
            samples1=_samples_for_trait(segment, observed_trait),
            mrd1=segment.ler_days,
            dim2=lactation_day,
            trait2=target_trait,
            supervision2=1,
            milkings2=2,
            samples2=2,
            mrd2=1,
            daily_sd=daily_sd,
            parity_group=parity_group,
            maxlen=maxlen,
        )
    return covariance


def _actual_output_from_internal(
    *,
    value: float,
    trait: int,
    target_length: int,
    age_factor: float,
    factor_3x: float,
    parameters: BestpredParameters,
) -> float:
    actual = value / (age_factor * factor_3x)
    if trait < 4 and parameters.units_out == "P":
        actual *= LB_PER_KG
    if trait == 4:
        actual /= target_length
    return actual


def _segment_observation_covariance(
    left: TestDaySegment,
    right: TestDaySegment,
    *,
    trait: int,
    daily_sd: FloatArray,
    parity_group: int,
    maxlen: int,
) -> float:
    left_samples = left.times_weighed if trait == 1 else left.times_sampled
    right_samples = right.times_weighed if trait == 1 else right.times_sampled
    if trait == 1 and left.ler_days > 1:
        left_samples = left.times_milked
    if trait == 1 and right.ler_days > 1:
        right_samples = right.times_milked

    return observation_covariance(
        dim1=left.dim,
        trait1=trait,
        supervision1=left.supervised,
        milkings1=left.times_milked,
        samples1=left_samples,
        mrd1=left.ler_days,
        dim2=right.dim,
        trait2=trait,
        supervision2=right.supervised,
        milkings2=right.times_milked,
        samples2=right_samples,
        mrd2=right.ler_days,
        daily_sd=daily_sd,
        parity_group=parity_group,
        maxlen=maxlen,
    )


def _observation_covariance_for_traits(
    left: TestDaySegment,
    right: TestDaySegment,
    *,
    trait_left: int,
    trait_right: int,
    daily_sd: FloatArray,
    parity_group: int,
    maxlen: int,
) -> float:
    return observation_covariance(
        dim1=left.dim,
        trait1=trait_left,
        supervision1=left.supervised,
        milkings1=left.times_milked,
        samples1=_samples_for_trait(left, trait_left),
        mrd1=left.ler_days,
        dim2=right.dim,
        trait2=trait_right,
        supervision2=right.supervised,
        milkings2=right.times_milked,
        samples2=_samples_for_trait(right, trait_right),
        mrd2=right.ler_days,
        daily_sd=daily_sd,
        parity_group=parity_group,
        maxlen=maxlen,
    )


def _samples_for_trait(segment: TestDaySegment, trait: int) -> int:
    if trait == 1 and segment.ler_days > 1:
        return segment.times_milked
    return segment.times_weighed if trait == 1 else segment.times_sampled


def _measurement_range_for_observation(*, dim: int, trait: int, mrd: int) -> tuple[int, int]:
    begin = dim - mrd + 1
    end = dim
    if trait > 1:
        begin = dim - (mrd - 1) // 2
        end = begin
    return begin, end


def _age_factor_for_trait(record: Format4Record, trait: int) -> float:
    if trait == 4:
        return format4_scs_age_factor(record)

    age_factors = format4_age_factors(record)
    return (age_factors.milk, age_factors.fat, age_factors.protein)[trait - 1]


def _herd_average_for_trait(
    record: Format4Record,
    trait: int,
    *,
    breed: int,
    parity_group: int,
) -> float:
    if trait == 1:
        if record.herd_me_milk > 0:
            return float(record.herd_me_milk)
        return BREED_MEAN[breed][parity_group - 1][trait - 1]
    if trait == 2:
        if record.herd_me_fat > 0:
            return float(record.herd_me_fat)
        return BREED_MEAN[breed][parity_group - 1][trait - 1]
    if trait == 3:
        if record.herd_me_protein > 0:
            return float(record.herd_me_protein)
        return BREED_MEAN[breed][parity_group - 1][trait - 1]
    if record.herd_me_scs is not None and record.herd_me_scs > 0:
        return record.herd_me_scs
    return BREED_MEAN[breed][parity_group - 1][trait - 1]


def _observed_trait_value(segment: TestDaySegment, trait: int) -> float:
    milk = segment.milk_yield / 10.0
    if trait == 1:
        return milk
    if trait == 2:
        return milk * (segment.fat_percent / 10.0) / 100.0
    if trait == 3:
        return milk * (segment.protein_percent / 10.0) / 100.0
    return segment.scs / 10.0


def _has_observed_trait(record: Format4Record, trait: int) -> bool:
    return any(_observed_trait_value(segment, trait) > 0.0 for segment in record.segments)


def _has_usable_observed_trait(record: Format4Record, trait: int, *, maxlen: int) -> bool:
    return any(
        1 <= segment.dim <= maxlen and _observed_trait_value(segment, trait) > 0.0
        for segment in record.segments
    )


def _herd_ratio_vector(trait: int, herd_ratio: float) -> tuple[float, float, float, float]:
    if trait == 1:
        return herd_ratio, 1.0, 1.0, 1.0
    if trait == 2:
        return 1.0, herd_ratio, 1.0, 1.0
    if trait == 3:
        return 1.0, 1.0, herd_ratio, 1.0
    return 1.0, 1.0, 1.0, herd_ratio
