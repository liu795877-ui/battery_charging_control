"""Numerically equivalent closed-form thermal supervisor for R3T."""

from __future__ import annotations

from typing import Any

import numpy as np


def optimized_thermal_peak(
    current_a: float,
    temperature_c: float,
    ambient_temperature_c: float,
    r1: Any,
    braking: bool,
) -> float:
    thermal = r1.thermal
    coefficients = thermal["surrogate"]
    q = 1.0 + float(coefficients["coefficient_temperature_c_per_step_c"])
    horizon = int(thermal["prediction_horizon_steps"])
    floor = float(thermal["braking_floor_current_a"])
    relative_temperature = float(temperature_c - ambient_temperature_c)

    def heat(current: float) -> float:
        return (
            float(coefficients["coefficient_i2_c_per_step_a2"])
            * current**2
            + float(coefficients["coefficient_i_c_per_step_a"]) * current
            + float(coefficients["intercept_c_per_step"])
        )

    if not braking:
        forcing = heat(float(current_a))
        first = q * relative_temperature + forcing
        q_horizon = q**horizon
        final = (
            q_horizon * relative_temperature
            + forcing * (1.0 - q_horizon) / (1.0 - q)
        )
        return float(ambient_temperature_c + max(first, final))

    maximum = -np.inf
    remaining = horizon
    predicted_current = float(current_a)
    while remaining and predicted_current > floor:
        relative_temperature = q * relative_temperature + heat(predicted_current)
        maximum = max(maximum, relative_temperature)
        remaining -= 1
        predicted_current = max(floor, predicted_current - 2.0)
    if remaining:
        forcing = heat(predicted_current)
        first = q * relative_temperature + forcing
        q_remaining = q**remaining
        final = (
            q_remaining * relative_temperature
            + forcing * (1.0 - q_remaining) / (1.0 - q)
        )
        maximum = max(maximum, first, final)
    return float(ambient_temperature_c + maximum)


def optimized_thermal_current_limit(
    temperature_c: float,
    ambient_temperature_c: float,
    search_upper_a: float,
    r1: Any,
    braking: bool,
) -> tuple[float, float]:
    limit = (
        float(r1.thermal["maximum_average_temperature_c"])
        - float(r1.thermal["temperature_guard_c"])
    )

    def peak(current: float) -> float:
        return optimized_thermal_peak(
            current,
            temperature_c,
            ambient_temperature_c,
            r1,
            braking,
        )

    peak_zero = peak(0.0)
    if search_upper_a < 0.0 or peak_zero > limit:
        return -1.0, peak_zero
    peak_upper = peak(float(search_upper_a))
    if peak_upper <= limit:
        return float(search_upper_a), peak_upper
    lower, upper = 0.0, float(search_upper_a)
    tolerance = float(r1.thermal["current_search_tolerance_a"])
    while upper - lower > tolerance:
        current = 0.5 * (lower + upper)
        if peak(current) <= limit:
            lower = current
        else:
            upper = current
    return lower, peak(lower)
