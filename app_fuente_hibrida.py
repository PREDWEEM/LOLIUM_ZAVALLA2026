from __future__ import annotations

from typing import Any

import pandas as pd

import app_detalle_1pct as base


principal = base.operational.base.principal
_ORIGINAL_RESOLVE_WEATHER = principal.resolve_weather


def _hybrid_source_label(weather: Any) -> str:
    prefix = (
        "Serie híbrida Zavalla: SMN Rosario Aero 87480 (observado) → "
        "NOAA NCEI GSOD (respaldo) → Open-Meteo ECMWF IFS "
        "(faltantes y pronóstico)"
    )
    if weather is None or not isinstance(weather, pd.DataFrame):
        return prefix
    if "Fuente" not in weather or "Fecha" not in weather:
        return prefix

    source = weather["Fuente"].astype("string").fillna("")
    observed = source.str.contains("SMN_WIS2", regex=False) | source.str.contains(
        "NOAA_NCEI_GSOD", regex=False
    )
    provisional = source.str.contains(
        "OPEN_METEO_ECMWF_IFS_ARCHIVE_FALLBACK",
        regex=False,
    )
    forecast = source.str.contains(
        "OPEN_METEO_ECMWF_IFS_FORECAST",
        regex=False,
    )

    return (
        f"{prefix}. Días con variables observadas: {int(observed.sum())}; "
        f"completados por archivo ECMWF: {int(provisional.sum())}; "
        f"pronosticados: {int(forecast.sum())}."
    )


def _resolve_weather_with_hybrid_label(site: Any, uploaded: Any):
    weather, source = _ORIGINAL_RESOLVE_WEATHER(site, uploaded)
    if uploaded is None and getattr(site, "slug", None) == "zavalla":
        source = _hybrid_source_label(weather)
    return weather, source


def run() -> None:
    """Ejecuta PREDWEEM mostrando la procedencia híbrida de Zavalla."""
    original_resolve_weather = principal.resolve_weather
    principal.resolve_weather = _resolve_weather_with_hybrid_label
    try:
        base.run()
    finally:
        principal.resolve_weather = original_resolve_weather
