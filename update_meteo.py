from __future__ import annotations

"""Actualizador meteorológico público y seguro para PREDWEEM Zavalla.

El motor histórico se conserva en ``update_meteo_core.py``. Este módulo aplica
los controles de calidad antes de ejecutar la actualización para que tanto
``python update_meteo.py`` como los workflows programados utilicen exactamente
la misma lógica.
"""

from typing import Any

import pandas as pd

import update_meteo_core as core

# Reexporta la interfaz pública existente para conservar compatibilidad con
# pruebas y módulos que importan constantes o funciones desde update_meteo.
for _name in dir(core):
    if not _name.startswith("__"):
        globals()[_name] = getattr(core, _name)

_ORIGINAL_PRECIPITATION_MM = core._precipitation_mm
_ORIGINAL_FETCH_SMN = core.fetch_smn_rosario_daily
_ORIGINAL_MERGE = core.merge_observed_priority_history


def _safe_precipitation_mm(value: float, units: str) -> float:
    """Descarta códigos negativos de precipitación comunicados por el origen."""
    converted = float(_ORIGINAL_PRECIPITATION_MM(value, units))
    return float("nan") if converted < 0 else converted


def sanitize_observed_frame(frame: pd.DataFrame | None) -> pd.DataFrame:
    """Convierte valores físicos inválidos en faltantes reemplazables.

    Esos faltantes son completados posteriormente respetando la prioridad
    SMN → NOAA → Open-Meteo ECMWF IFS Archive.
    """
    if frame is None:
        return core._empty_weather()

    cleaned = frame.copy()
    if cleaned.empty:
        return cleaned

    if "Prec" in cleaned:
        precipitation = pd.to_numeric(cleaned["Prec"], errors="coerce")
        invalid = precipitation < 0
        if bool(invalid.any()):
            dates = pd.to_datetime(
                cleaned.loc[invalid, "Fecha"], errors="coerce"
            ).dt.strftime("%Y-%m-%d")
            print(
                "Precipitación observada negativa descartada en: "
                + ", ".join(dates.dropna().astype(str).tolist()[:20])
            )
            cleaned.loc[invalid, "Prec"] = pd.NA
            if "Fuente_Prec" in cleaned:
                cleaned.loc[invalid, "Fuente_Prec"] = pd.NA

    for variable, provenance in (
        ("TMAX", "Fuente_TMAX"),
        ("TMIN", "Fuente_TMIN"),
    ):
        if variable not in cleaned:
            continue
        temperature = pd.to_numeric(cleaned[variable], errors="coerce")
        invalid = (temperature < -60) | (temperature > 60)
        if bool(invalid.any()):
            cleaned.loc[invalid, variable] = pd.NA
            if provenance in cleaned:
                cleaned.loc[invalid, provenance] = pd.NA

    return cleaned


def fetch_smn_rosario_daily(*args: Any, **kwargs: Any) -> pd.DataFrame:
    """Obtiene SMN y elimina sentinelas antes de combinar fuentes."""
    return sanitize_observed_frame(_ORIGINAL_FETCH_SMN(*args, **kwargs))


def merge_observed_priority_history(
    smn: pd.DataFrame,
    noaa: pd.DataFrame,
    archive: pd.DataFrame,
    **kwargs: Any,
) -> pd.DataFrame:
    """Sanea nuevamente cada fuente antes de seleccionar cada variable."""
    return _ORIGINAL_MERGE(
        sanitize_observed_frame(smn),
        sanitize_observed_frame(noaa),
        archive,
        **kwargs,
    )


def install_runtime_corrections() -> None:
    """Instala los controles en el módulo donde se ejecutan las funciones."""
    core._precipitation_mm = _safe_precipitation_mm
    core.fetch_smn_rosario_daily = fetch_smn_rosario_daily
    core.merge_observed_priority_history = merge_observed_priority_history


# Las correcciones se instalan también al importar el módulo, para que las
# pruebas y cualquier llamada externa usen la misma política que el CLI.
install_runtime_corrections()


def main() -> None:
    core.main()


if __name__ == "__main__":
    main()
