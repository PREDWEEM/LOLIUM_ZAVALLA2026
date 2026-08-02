from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from io import StringIO
from pathlib import Path
from zoneinfo import ZoneInfo
import hashlib
import json
import os
import tempfile
import time

import pandas as pd
import requests

from sitios_lolium import DEFAULT_SITE_SLUG, LoliumSite, get_site, ordered_sites


LEGACY_OUTPUT = Path("meteo_daily.csv")
STATE = Path("data/estado_actualizacion_meteo.json")
START_DATE = date(2026, 1, 1)
FORECAST_DAYS = 8
FORECAST_PAST_DAYS = 2
TIMEOUT = 90
MAX_ATTEMPTS = 3

CORE_COLUMNS = [
    "Fecha",
    "TMAX",
    "TMIN",
    "Prec",
    "Fuente",
    "TipoDato",
    "CalidadDato",
    "Emision",
]
PROVENANCE_COLUMNS = ["Fuente_TMAX", "Fuente_TMIN", "Fuente_Prec"]
COLUMNS = CORE_COLUMNS + PROVENANCE_COLUMNS

SMN_ITEMS_URL = (
    "https://w2b.smn.gov.ar/oapi/collections/"
    "urn%3Awmo%3Amd%3Aar-smn%3Aslt0ci/items"
)
SMN_WIGOS_ID = "0-20000-0-87480"
SMN_SOURCE = "SMN_WIS2_ROSARIO_AERO_87480"
SMN_PAGE_LIMIT = 10_000
SMN_CHUNK_DAYS = 31

NOAA_SERVICE_URL = "https://www.ncei.noaa.gov/access/services/data/v1"
NOAA_BULK_ROOT = (
    "https://www.ncei.noaa.gov/data/global-summary-of-the-day/access"
)
NOAA_STATION_ID = "87480099999"
NOAA_SOURCE = "NOAA_NCEI_GSOD_ROSARIO_AERO_87480"

ARCHIVE_SOURCE = "OPEN_METEO_ECMWF_IFS_ARCHIVE_FALLBACK"
FORECAST_SOURCE = "OPEN_METEO_ECMWF_IFS_FORECAST"

PUBLIC_HEADERS = {
    "User-Agent": "PREDWEEM-LOLIUM-multisitio/2.0",
    "Accept": "application/json,text/csv,*/*",
}


def _request(
    url: str,
    *,
    params: dict | None = None,
    accept: str | None = None,
) -> requests.Response:
    """Realiza una solicitud pública con reintentos y diagnóstico."""
    last_error: Exception | None = None
    headers = dict(PUBLIC_HEADERS)
    if accept:
        headers["Accept"] = accept

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = requests.get(
                url,
                params=params,
                headers=headers,
                timeout=TIMEOUT,
            )
            print(f"HTTP {response.status_code}: {response.url}")
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            last_error = exc
            if attempt < MAX_ATTEMPTS:
                time.sleep(3 * attempt)

    raise RuntimeError(f"No fue posible consultar {url}") from last_error


def get_json(url: str, params: dict | None = None) -> dict | list:
    response = _request(url, params=params, accept="application/json")
    try:
        payload = response.json()
    except ValueError as exc:
        raise RuntimeError(f"{url} no devolvió JSON válido.") from exc

    if isinstance(payload, dict) and payload.get("error"):
        raise RuntimeError(str(payload.get("reason") or payload))
    return payload


def _numeric(series: pd.Series) -> pd.Series:
    """Convierte números aun cuando vengan acompañados por atributos de calidad."""
    extracted = (
        series.astype("string")
        .str.replace(",", ".", regex=False)
        .str.extract(r"([-+]?\d+(?:\.\d+)?)", expand=False)
    )
    return pd.to_numeric(extracted, errors="coerce")


def _empty_weather() -> pd.DataFrame:
    return pd.DataFrame(columns=COLUMNS)


def _add_provenance(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    out = frame.copy()
    for column in PROVENANCE_COLUMNS:
        out[column] = source
    return out


def daily_frame(
    payload: dict,
    source: str,
    data_type: str,
    quality: str,
    emission: str,
) -> pd.DataFrame:
    """Convierte una respuesta diaria de Open-Meteo al esquema PREDWEEM."""
    daily = payload.get("daily") or {}
    required = [
        "time",
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
    ]
    missing = [key for key in required if key not in daily]
    if missing:
        raise RuntimeError("Respuesta incompleta: " + ", ".join(missing))

    frame = pd.DataFrame(
        {
            "Fecha": pd.to_datetime(daily["time"], errors="coerce"),
            "TMAX": pd.to_numeric(
                daily["temperature_2m_max"], errors="coerce"
            ),
            "TMIN": pd.to_numeric(
                daily["temperature_2m_min"], errors="coerce"
            ),
            "Prec": pd.to_numeric(
                daily["precipitation_sum"], errors="coerce"
            ),
        }
    )
    frame["Fuente"] = source
    frame["TipoDato"] = data_type
    frame["CalidadDato"] = quality
    frame["Emision"] = emission
    frame = _add_provenance(frame, source)

    if frame[["Fecha", "TMAX", "TMIN", "Prec"]].isna().any().any():
        raise RuntimeError(f"{source} devolvió valores críticos nulos.")
    return frame[COLUMNS]


def validate(frame: pd.DataFrame) -> pd.DataFrame:
    """Valida continuidad diaria y variables meteorológicas críticas."""
    missing_columns = [column for column in COLUMNS if column not in frame]
    if missing_columns:
        raise RuntimeError(
            "Faltan columnas meteorológicas: " + ", ".join(missing_columns)
        )

    data = frame[COLUMNS].copy()
    data["Fecha"] = pd.to_datetime(data["Fecha"], errors="coerce").dt.normalize()
    for column in ("TMAX", "TMIN", "Prec"):
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data = (
        data.dropna(subset=["Fecha"])
        .sort_values("Fecha")
        .drop_duplicates("Fecha", keep="last")
        .reset_index(drop=True)
    )
    if data.empty:
        raise RuntimeError("La serie meteorológica quedó vacía.")
    if data[["TMAX", "TMIN", "Prec"]].isna().any().any():
        raise RuntimeError("La serie conserva valores meteorológicos críticos nulos.")
    if (data["TMAX"] < data["TMIN"]).any():
        bad = data.loc[data["TMAX"] < data["TMIN"], "Fecha"].iloc[0]
        raise RuntimeError(f"TMAX menor que TMIN el {bad:%Y-%m-%d}.")
    if (data["Prec"] < 0).any():
        bad = data.loc[data["Prec"] < 0, "Fecha"].iloc[0]
        raise RuntimeError(f"Precipitación negativa el {bad:%Y-%m-%d}.")

    expected = pd.date_range(data["Fecha"].min(), data["Fecha"].max(), freq="D")
    missing = expected.difference(pd.DatetimeIndex(data["Fecha"]))
    if len(missing):
        raise RuntimeError(
            "Fechas faltantes: "
            + ", ".join(ts.strftime("%Y-%m-%d") for ts in missing[:10])
        )

    data["TMAX"] = data["TMAX"].round(2)
    data["TMIN"] = data["TMIN"].round(2)
    data["Prec"] = data["Prec"].round(2)
    return data


def _iso_utc(local_day: date, timezone_name: str) -> str:
    local_midnight = datetime.combine(
        local_day,
        datetime.min.time(),
        tzinfo=ZoneInfo(timezone_name),
    )
    utc_value = local_midnight.astimezone(timezone.utc)
    return utc_value.isoformat().replace("+00:00", "Z")


def _smn_feature_pages(
    start_date: date,
    end_date: date,
    timezone_name: str,
) -> list[dict]:
    """Descarga observaciones SYNOP de Rosario Aero desde el OGC API del SMN."""
    features: list[dict] = []
    chunk_start = start_date

    while chunk_start <= end_date:
        chunk_end = min(
            end_date,
            chunk_start + timedelta(days=SMN_CHUNK_DAYS - 1),
        )
        offset = 0

        while True:
            params = {
                "f": "json",
                "limit": SMN_PAGE_LIMIT,
                "offset": offset,
                "datetime": (
                    f"{_iso_utc(chunk_start, timezone_name)}/"
                    f"{_iso_utc(chunk_end + timedelta(days=1), timezone_name)}"
                ),
                "wigos_station_identifier": SMN_WIGOS_ID,
                "sortby": "reportTime",
            }
            payload = get_json(SMN_ITEMS_URL, params)
            if not isinstance(payload, dict):
                raise RuntimeError("El SMN devolvió una estructura inesperada.")

            page = payload.get("features") or []
            accepted = [
                feature
                for feature in page
                if (
                    feature.get("properties", {}).get(
                        "wigos_station_identifier"
                    )
                    == SMN_WIGOS_ID
                )
            ]
            features.extend(accepted)

            returned = int(payload.get("numberReturned") or len(page))
            if returned < SMN_PAGE_LIMIT or not page:
                break
            offset += returned

        chunk_start = chunk_end + timedelta(days=1)

    return features


def _temperature_celsius(value: float, units: str) -> float:
    normalized = units.strip().lower()
    if normalized in {"k", "kelvin"}:
        return value - 273.15
    if normalized in {"f", "°f", "degf", "fahrenheit"}:
        return (value - 32.0) * 5.0 / 9.0
    return value


def _precipitation_mm(value: float, units: str) -> float:
    normalized = units.strip().lower()
    if normalized in {"in", "inch", "inches"}:
        return value * 25.4
    # 1 kg m-2 de agua equivale a 1 mm.
    return value


def fetch_smn_rosario_daily(
    start_date: date,
    end_date: date,
    timezone_name: str,
    emission: str,
) -> pd.DataFrame:
    """Agrega observaciones SYNOP del SMN por día local.

    Para precipitación se conserva el máximo acumulado comunicado en el día.
    Esta decisión evita sumar reportes SYNOP con períodos de acumulación
    superpuestos. NOAA y Open-Meteo completan los valores no disponibles.
    """
    if end_date < start_date:
        return _empty_weather()

    features = _smn_feature_pages(start_date, end_date, timezone_name)
    temperature_rows: list[tuple[date, float]] = []
    precipitation_rows: list[tuple[date, float]] = []

    for feature in features:
        properties = feature.get("properties") or {}
        value = pd.to_numeric(properties.get("value"), errors="coerce")
        if pd.isna(value):
            continue

        timestamp = pd.to_datetime(
            properties.get("reportTime") or properties.get("phenomenonTime"),
            errors="coerce",
            utc=True,
        )
        if pd.isna(timestamp):
            continue

        local_date = timestamp.tz_convert(timezone_name).date()
        if not (start_date <= local_date <= end_date):
            continue

        name = str(properties.get("name") or "").strip().lower()
        units = str(properties.get("units") or "").strip()

        is_temperature = (
            name == "air_temperature"
            or name.startswith("maximum_temperature")
            or name.startswith("minimum_temperature")
        )
        if is_temperature:
            temperature_rows.append(
                (local_date, _temperature_celsius(float(value), units))
            )
        elif "precipitation" in name:
            precipitation_rows.append(
                (local_date, _precipitation_mm(float(value), units))
            )

    dates = pd.date_range(start_date, end_date, freq="D")
    result = pd.DataFrame({"Fecha": dates})

    if temperature_rows:
        temperature = pd.DataFrame(
            temperature_rows,
            columns=["Fecha_local", "Temperatura"],
        )
        temperature["Fecha"] = pd.to_datetime(temperature["Fecha_local"])
        grouped = temperature.groupby("Fecha")["Temperatura"]
        result = result.merge(
            grouped.agg(TMAX="max", TMIN="min").reset_index(),
            on="Fecha",
            how="left",
        )
    else:
        result["TMAX"] = pd.NA
        result["TMIN"] = pd.NA

    if precipitation_rows:
        precipitation = pd.DataFrame(
            precipitation_rows,
            columns=["Fecha_local", "Prec"],
        )
        precipitation["Fecha"] = pd.to_datetime(
            precipitation["Fecha_local"]
        )
        daily_precip = (
            precipitation.groupby("Fecha")["Prec"].max().reset_index()
        )
        result = result.merge(daily_precip, on="Fecha", how="left")
    else:
        result["Prec"] = pd.NA

    result["Fuente"] = SMN_SOURCE
    result["TipoDato"] = "Observado"
    result["CalidadDato"] = "Observado_SINOP_SMN"
    result["Emision"] = emission
    return _add_provenance(result, SMN_SOURCE)[COLUMNS]


def _normalize_noaa_frame(
    raw: pd.DataFrame,
    start_date: date,
    end_date: date,
    emission: str,
) -> pd.DataFrame:
    if raw.empty:
        return _empty_weather()

    normalized = raw.copy()
    normalized.columns = [str(column).upper() for column in normalized.columns]
    if "DATE" not in normalized:
        raise RuntimeError("NOAA GSOD no incluyó la columna DATE.")

    result = pd.DataFrame(
        {
            "Fecha": pd.to_datetime(normalized["DATE"], errors="coerce"),
            "TMAX": _numeric(
                normalized["MAX"]
                if "MAX" in normalized
                else pd.Series(index=normalized.index, dtype="object")
            ),
            "TMIN": _numeric(
                normalized["MIN"]
                if "MIN" in normalized
                else pd.Series(index=normalized.index, dtype="object")
            ),
            "Prec": _numeric(
                normalized["PRCP"]
                if "PRCP" in normalized
                else pd.Series(index=normalized.index, dtype="object")
            ),
        }
    )
    result = result.loc[
        result["Fecha"].dt.date.between(start_date, end_date)
    ].copy()

    # Sentinelas habituales de GSOD.
    result.loc[result["TMAX"].abs() >= 90, "TMAX"] = pd.NA
    result.loc[result["TMIN"].abs() >= 90, "TMIN"] = pd.NA
    result.loc[result["Prec"] >= 900, "Prec"] = pd.NA
    result.loc[result["Prec"] < 0, "Prec"] = pd.NA

    result["Fuente"] = NOAA_SOURCE
    result["TipoDato"] = "Observado_respaldo"
    result["CalidadDato"] = "Observado_NOAA_ISD_GSOD"
    result["Emision"] = emission
    return _add_provenance(result, NOAA_SOURCE)[COLUMNS]


def fetch_noaa_gsod_daily(
    start_date: date,
    end_date: date,
    emission: str,
) -> pd.DataFrame:
    """Descarga GSOD de Rosario Aero, derivado de observaciones ISD."""
    if end_date < start_date:
        return _empty_weather()

    service_error: Exception | None = None
    try:
        payload = get_json(
            NOAA_SERVICE_URL,
            {
                "dataset": "global-summary-of-the-day",
                "dataTypes": "MAX,MIN,PRCP",
                "stations": NOAA_STATION_ID,
                "startDate": start_date.isoformat(),
                "endDate": end_date.isoformat(),
                "format": "json",
                "units": "metric",
                "includeAttributes": "false",
            },
        )
        if isinstance(payload, list) and payload:
            return _normalize_noaa_frame(
                pd.DataFrame(payload),
                start_date,
                end_date,
                emission,
            )
    except Exception as exc:
        service_error = exc
        print(f"Servicio NOAA subset no disponible: {exc}")

    frames: list[pd.DataFrame] = []
    for year in range(start_date.year, end_date.year + 1):
        url = f"{NOAA_BULK_ROOT}/{year}/{NOAA_STATION_ID}.csv"
        try:
            response = _request(url, accept="text/csv")
            frames.append(pd.read_csv(StringIO(response.text)))
        except Exception as exc:
            print(f"Archivo NOAA GSOD {year} no disponible: {exc}")

    if frames:
        return _normalize_noaa_frame(
            pd.concat(frames, ignore_index=True),
            start_date,
            end_date,
            emission,
        )

    raise RuntimeError(
        "NOAA no devolvió GSOD para Rosario Aero."
    ) from service_error


def fetch_open_meteo_history(
    site: LoliumSite,
    end_date: date,
    emission: str,
) -> pd.DataFrame:
    if end_date < START_DATE:
        return _empty_weather()

    payload = get_json(
        "https://archive-api.open-meteo.com/v1/archive",
        {
            "latitude": site.latitud,
            "longitude": site.longitud,
            "start_date": START_DATE.isoformat(),
            "end_date": end_date.isoformat(),
            "daily": (
                "temperature_2m_max,temperature_2m_min,"
                "precipitation_sum"
            ),
            "models": "ecmwf_ifs",
            "timezone": site.timezone,
            "temperature_unit": "celsius",
            "precipitation_unit": "mm",
            "cell_selection": "land",
        },
    )
    if not isinstance(payload, dict):
        raise RuntimeError("Open-Meteo Archive devolvió una estructura inválida.")
    return daily_frame(
        payload,
        ARCHIVE_SOURCE,
        "Provisional",
        "Provisional_hueco_observaciones",
        emission,
    )


def fetch_open_meteo_forecast(
    site: LoliumSite,
    emission: str,
) -> pd.DataFrame:
    payload = get_json(
        "https://api.open-meteo.com/v1/forecast",
        {
            "latitude": site.latitud,
            "longitude": site.longitud,
            "daily": (
                "temperature_2m_max,temperature_2m_min,"
                "precipitation_sum"
            ),
            "models": "ecmwf_ifs",
            "forecast_days": FORECAST_DAYS,
            "past_days": FORECAST_PAST_DAYS,
            "timezone": site.timezone,
            "temperature_unit": "celsius",
            "precipitation_unit": "mm",
            "cell_selection": "land",
        },
    )
    if not isinstance(payload, dict):
        raise RuntimeError("Open-Meteo Forecast devolvió una estructura inválida.")
    return daily_frame(
        payload,
        FORECAST_SOURCE,
        "Pronostico",
        "Pronostico_operativo",
        emission,
    )


def _row_source(row: pd.Series) -> str:
    values = [
        str(row[column])
        for column in PROVENANCE_COLUMNS
        if pd.notna(row[column]) and str(row[column]).strip()
    ]
    return "+".join(dict.fromkeys(values))


def _row_classification(source: str) -> tuple[str, str]:
    has_smn = SMN_SOURCE in source
    has_noaa = NOAA_SOURCE in source
    has_archive = ARCHIVE_SOURCE in source

    if has_archive:
        return "Provisional", "Provisional_hueco_observaciones"
    if has_smn and has_noaa:
        return "Observado_compuesto", "Observado_SMN_con_respaldo_NOAA"
    if has_smn:
        return "Observado", "Observado_SINOP_SMN"
    if has_noaa:
        return "Observado_respaldo", "Observado_NOAA_ISD_GSOD"
    return "Provisional", "Calidad_fuente_no_clasificada"


def merge_observed_priority_history(
    smn: pd.DataFrame,
    noaa: pd.DataFrame,
    archive: pd.DataFrame,
    *,
    start_date: date,
    end_date: date,
    emission: str,
) -> pd.DataFrame:
    """Prioridad por variable: SMN → NOAA → Open-Meteo Archive."""
    dates = pd.date_range(start_date, end_date, freq="D")
    merged = pd.DataFrame(index=dates)
    merged.index.name = "Fecha"

    for variable, provenance_column in (
        ("TMAX", "Fuente_TMAX"),
        ("TMIN", "Fuente_TMIN"),
        ("Prec", "Fuente_Prec"),
    ):
        merged[variable] = pd.NA
        merged[provenance_column] = pd.NA

        for source_frame in (smn, noaa, archive):
            if source_frame is None or source_frame.empty:
                continue
            available = source_frame.copy()
            available["Fecha"] = pd.to_datetime(
                available["Fecha"], errors="coerce"
            ).dt.normalize()
            available = (
                available.dropna(subset=["Fecha"])
                .drop_duplicates("Fecha", keep="last")
                .set_index("Fecha")
            )
            if variable not in available:
                continue

            candidate = pd.to_numeric(
                available[variable], errors="coerce"
            ).reindex(merged.index)
            source_values = (
                available["Fuente"].astype("string").reindex(merged.index)
            )
            fill = merged[variable].isna() & candidate.notna()
            merged.loc[fill, variable] = candidate.loc[fill]
            merged.loc[fill, provenance_column] = source_values.loc[fill]

    merged = merged.reset_index()
    missing = merged[["TMAX", "TMIN", "Prec"]].isna()
    if missing.any().any():
        failures = merged.loc[
            missing.any(axis=1), ["Fecha", "TMAX", "TMIN", "Prec"]
        ]
        raise RuntimeError(
            "No fue posible completar la meteorología de Zavalla. "
            f"Primeros faltantes:\n{failures.head(10).to_string(index=False)}"
        )

    merged["Fuente"] = merged.apply(_row_source, axis=1)
    classifications = merged["Fuente"].map(_row_classification)
    merged["TipoDato"] = classifications.map(lambda item: item[0])
    merged["CalidadDato"] = classifications.map(lambda item: item[1])
    merged["Emision"] = emission
    return validate(merged[COLUMNS])


def combine_historical_and_forecast(
    historical: pd.DataFrame,
    forecast_with_recent_days: pd.DataFrame,
    today: date,
) -> pd.DataFrame:
    """Conserva observaciones hasta ayer y ECMWF IFS desde el día actual."""
    historical_block = historical.loc[
        pd.to_datetime(historical["Fecha"]).dt.date < today
    ].copy()
    forecast_block = forecast_with_recent_days.loc[
        pd.to_datetime(forecast_with_recent_days["Fecha"]).dt.date >= today
    ].copy()

    available_dates = set(pd.to_datetime(forecast_block["Fecha"]).dt.date)
    if today not in available_dates:
        first_available = (
            pd.to_datetime(forecast_block["Fecha"]).min().date().isoformat()
            if not forecast_block.empty
            else "sin fechas"
        )
        raise RuntimeError(
            "Open-Meteo no devolvió el día actual "
            f"{today.isoformat()} ni siquiera usando past_days="
            f"{FORECAST_PAST_DAYS}. Primera fecha disponible: "
            f"{first_available}."
        )

    return validate(
        pd.concat([historical_block, forecast_block], ignore_index=True)
    )


def build_zavalla_weather(
    site: LoliumSite,
) -> tuple[pd.DataFrame, str, dict[str, object]]:
    now = datetime.now(ZoneInfo(site.timezone))
    today = now.date()
    yesterday = today - timedelta(days=1)
    emission = now.isoformat(timespec="seconds")
    diagnostics: dict[str, object] = {}

    archive = fetch_open_meteo_history(site, yesterday, emission)

    try:
        smn = fetch_smn_rosario_daily(
            START_DATE,
            yesterday,
            site.timezone,
            emission,
        )
        diagnostics["smn"] = {
            "estado": "disponible",
            "filas_con_temperatura": int(
                smn[["TMAX", "TMIN"]].notna().all(axis=1).sum()
            ),
            "filas_con_precipitacion": int(smn["Prec"].notna().sum()),
        }
    except Exception as exc:
        print(f"SMN no disponible; se continúa con respaldos: {exc}")
        smn = _empty_weather()
        diagnostics["smn"] = {
            "estado": "no_disponible",
            "error": str(exc),
        }

    try:
        noaa = fetch_noaa_gsod_daily(
            START_DATE,
            yesterday,
            emission,
        )
        diagnostics["noaa"] = {
            "estado": "disponible",
            "filas": int(len(noaa)),
        }
    except Exception as exc:
        print(f"NOAA no disponible; se continúa con ECMWF: {exc}")
        noaa = _empty_weather()
        diagnostics["noaa"] = {
            "estado": "no_disponible",
            "error": str(exc),
        }

    historical = merge_observed_priority_history(
        smn,
        noaa,
        archive,
        start_date=START_DATE,
        end_date=yesterday,
        emission=emission,
    )
    forecast = fetch_open_meteo_forecast(site, emission)
    combined = combine_historical_and_forecast(
        historical,
        forecast,
        today,
    )
    diagnostics["fuentes_finales"] = source_summary(combined)
    return combined, emission, diagnostics


def download_exact_repository_file(
    site: LoliumSite,
) -> tuple[bytes, str]:
    """Copia sin transformar el meteo_daily.csv de los otros sitios."""
    response = _request(site.raw_meteo_url, accept="text/csv,*/*")
    if not response.content:
        raise RuntimeError(f"{site.nombre}: el archivo descargado está vacío.")
    return response.content, response.url


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def write_exact_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary_path = Path(handle.name)
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())

    try:
        temporary_path.replace(path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    if path.read_bytes() != content:
        raise RuntimeError(
            f"La verificación byte a byte falló para {path.as_posix()}."
        )


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        suffix=".csv",
        delete=False,
        dir=path.parent,
        encoding="utf-8",
        newline="",
    ) as handle:
        temporary = Path(handle.name)
        frame.to_csv(handle, index=False, date_format="%Y-%m-%d")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        suffix=".json",
        delete=False,
        dir=path.parent,
        encoding="utf-8",
    ) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def source_summary(frame: pd.DataFrame) -> list[dict[str, object]]:
    grouped = frame.groupby(["Fuente", "TipoDato"]).size()
    return [
        {
            "fuente": source,
            "tipo": data_type,
            "filas": int(count),
        }
        for (source, data_type), count in grouped.items()
    ]


def main() -> None:
    # Primero se obtienen todas las fuentes. Los archivos se escriben únicamente
    # cuando Zavalla y las copias de las otras localidades fueron resueltos.
    exact_downloads: dict[str, tuple[LoliumSite, bytes, str]] = {}
    for site in ordered_sites():
        if site.slug == DEFAULT_SITE_SLUG:
            continue
        print(f"\n=== Copia exacta: {site.etiqueta} ===")
        content, requested_url = download_exact_repository_file(site)
        exact_downloads[site.slug] = (site, content, requested_url)

    zavalla = get_site(DEFAULT_SITE_SLUG)
    print(f"\n=== Serie híbrida observada/pronosticada: {zavalla.etiqueta} ===")
    zavalla_frame, emission, diagnostics = build_zavalla_weather(zavalla)

    updated_at = datetime.now(
        ZoneInfo("America/Argentina/Buenos_Aires")
    ).isoformat(timespec="seconds")
    state: dict[str, object] = {
        "actualizado_en": updated_at,
        "modo": "HIBRIDO_ZAVALLA_INDEPENDIENTE",
        "descripcion": (
            "Zavalla prioriza SMN Rosario Aero, completa con NOAA NCEI GSOD "
            "y utiliza Open-Meteo ECMWF IFS para faltantes y pronóstico. "
            "Los demás sitios conservan copia exacta de sus repositorios."
        ),
        "sitios": {},
    }

    for slug, (site, content, requested_url) in exact_downloads.items():
        destination = site.meteo_path(".")
        write_exact_bytes(destination, content)
        digest = sha256_bytes(content)
        state["sitios"][slug] = {
            "modo": "copia_exacta",
            "repositorio": site.repositorio,
            "url_origen": site.raw_meteo_url,
            "url_solicitada": requested_url,
            "archivo_destino": destination.as_posix(),
            "bytes": len(content),
            "sha256": digest,
            "copia_exacta": True,
        }

    zavalla_destination = zavalla.meteo_path(".")
    atomic_csv(zavalla_frame, zavalla_destination)
    atomic_csv(zavalla_frame, LEGACY_OUTPUT)
    if zavalla_destination.read_bytes() != LEGACY_OUTPUT.read_bytes():
        raise RuntimeError(
            "meteo_daily.csv raíz no coincide con la serie híbrida de Zavalla."
        )

    state["sitios"][zavalla.slug] = {
        "modo": "hibrido_observado_pronosticado",
        "archivo_destino": zavalla_destination.as_posix(),
        "archivo_raiz": LEGACY_OUTPUT.as_posix(),
        "actualizado": emission,
        "inicio": zavalla_frame["Fecha"].min().date().isoformat(),
        "fin": zavalla_frame["Fecha"].max().date().isoformat(),
        "filas": int(len(zavalla_frame)),
        "prioridad": [
            SMN_SOURCE,
            NOAA_SOURCE,
            ARCHIVE_SOURCE,
            FORECAST_SOURCE,
        ],
        "diagnostico": diagnostics,
        "fuentes": source_summary(zavalla_frame),
    }

    atomic_json(state, STATE)
    print(
        "\nActualización terminada: Zavalla híbrido independiente; "
        f"{len(exact_downloads)} copias exactas."
    )


if __name__ == "__main__":
    main()
