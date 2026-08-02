from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from config_zavalla import CONFIG, ZavallaConfig


REQUIRED_MODEL_FILES = ("IW.npy", "bias_IW.npy", "LW.npy", "bias_out.npy")


class PracticalANNModel:
    def __init__(
        self,
        iw: np.ndarray,
        bias_iw: np.ndarray,
        lw: np.ndarray,
        bias_out: np.ndarray,
    ):
        self.iw = np.asarray(iw, dtype=float)
        self.bias_iw = np.asarray(bias_iw, dtype=float)
        self.lw = np.asarray(lw, dtype=float)
        self.bias_out = np.asarray(bias_out, dtype=float).reshape(-1)
        self.input_min = np.array([1.0, 0.0, -7.0, 0.0])
        self.input_max = np.array([300.0, 41.0, 25.5, 84.0])

    def predict(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        normalized = (
            2.0
            * (values - self.input_min)
            / (self.input_max - self.input_min)
            - 1.0
        )
        hidden = np.tanh(normalized @ self.iw + self.bias_iw)
        linear = (hidden @ self.lw.T).reshape(-1)
        bias = float(self.bias_out[0]) if self.bias_out.size else 0.0
        return np.clip((np.tanh(linear + bias) + 1.0) / 2.0, 0.0, 1.0)


def load_ann(base: str | Path = ".") -> PracticalANNModel:
    base_path = Path(base)
    missing = [
        name for name in REQUIRED_MODEL_FILES if not (base_path / name).is_file()
    ]
    if missing:
        raise FileNotFoundError("Faltan activos ANN: " + ", ".join(missing))
    return PracticalANNModel(
        np.load(base_path / "IW.npy"),
        np.load(base_path / "bias_IW.npy"),
        np.load(base_path / "LW.npy"),
        np.load(base_path / "bias_out.npy"),
    )


def canonicalize_weather(raw: pd.DataFrame) -> pd.DataFrame:
    data = raw.copy()
    data.columns = [str(column).upper().strip() for column in data.columns]
    data = data.rename(
        columns={
            "FECHA": "Fecha",
            "DATE": "Fecha",
            "DATETIME": "Fecha",
            "PREC": "Prec",
            "PRECIPITACION": "Prec",
            "PRECIPITACIÓN": "Prec",
            "LLUVIA": "Prec",
        }
    )
    required = ["Fecha", "TMAX", "TMIN", "Prec"]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError("Faltan columnas meteorológicas: " + ", ".join(missing))

    data["Fecha"] = pd.to_datetime(data["Fecha"], errors="coerce").dt.normalize()
    for column in ("TMAX", "TMIN", "Prec"):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = (
        data.dropna(subset=required)
        .sort_values("Fecha")
        .drop_duplicates("Fecha", keep="last")
        .reset_index(drop=True)
    )
    if data.empty:
        raise ValueError("No hay datos meteorológicos válidos.")
    if (data["TMAX"] < data["TMIN"]).any():
        raise ValueError("Se detectó TMAX menor que TMIN.")
    if (data["Prec"] < 0).any():
        raise ValueError("Se detectó precipitación negativa.")

    expected = pd.date_range(data["Fecha"].min(), data["Fecha"].max(), freq="D")
    missing_dates = expected.difference(pd.DatetimeIndex(data["Fecha"]))
    if len(missing_dates):
        preview = ", ".join(ts.strftime("%Y-%m-%d") for ts in missing_dates[:8])
        raise ValueError(f"La meteorología no es continua. Fechas faltantes: {preview}")
    return data


def surface_parameters(coverage_percent: float) -> tuple[float, float]:
    coverage = float(np.clip(coverage_percent, 0.0, 100.0))
    points = [0.0, 30.0, 70.0, 100.0]
    ke = float(np.interp(coverage, points, [0.85, 0.50, 0.25, 0.10]))
    thermal_modulator = float(
        np.interp(coverage, points, [0.95, 0.90, 0.85, 0.80])
    )
    return ke, thermal_modulator


def calculate_et0_hargreaves(
    julian_day: Iterable[float],
    tmax: Iterable[float],
    tmin: Iterable[float],
    latitude: float,
) -> np.ndarray:
    jd = np.asarray(julian_day, dtype=float)
    tx = np.asarray(tmax, dtype=float)
    tn = np.asarray(tmin, dtype=float)
    lat_rad = np.radians(float(latitude))
    dr = 1.0 + 0.033 * np.cos(2.0 * np.pi / 365.0 * jd)
    dec = 0.409 * np.sin(2.0 * np.pi / 365.0 * jd - 1.39)
    ws = np.arccos(np.clip(-np.tan(lat_rad) * np.tan(dec), -1.0, 1.0))
    ra = (
        (24.0 * 60.0 / np.pi)
        * 0.0820
        * dr
        * (
            ws * np.sin(lat_rad) * np.sin(dec)
            + np.cos(lat_rad) * np.cos(dec) * np.sin(ws)
        )
    )
    ra_mm = ra / 2.45
    tmean = (tx + tn) / 2.0
    trange = np.maximum(tx - tn, 0.0)
    return np.maximum(
        0.0023 * ra_mm * (tmean + 17.8) * np.sqrt(trange),
        0.0,
    )


def surface_water_balance(
    precipitation: Iterable[float],
    et0: Iterable[float],
    wmax: float,
    ke: float,
    kr_exponent: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    prec = np.asarray(precipitation, dtype=float)
    et = np.asarray(et0, dtype=float)
    if float(wmax) <= 0:
        raise ValueError("Wmax debe ser mayor que cero.")

    water = np.zeros(len(prec), dtype=float)
    kr_daily = np.ones(len(prec), dtype=float)
    if len(water) == 0:
        return water, kr_daily

    water[0] = float(wmax) / 2.0
    exponent = max(float(kr_exponent), 0.0)
    for index in range(1, len(water)):
        relative = float(np.clip(water[index - 1] / float(wmax), 0.0, 1.0))
        kr = 1.0 if exponent == 0.0 else relative**exponent
        kr_daily[index] = kr
        evaporation = et[index] * float(ke) * kr
        water[index] = np.clip(
            water[index - 1] + prec[index] - evaporation,
            0.0,
            float(wmax),
        )
    return water, kr_daily


def thermal_time_scalar(t: float, base: float, optimum: float, critical: float) -> float:
    if t <= base:
        return 0.0
    if t <= optimum:
        return t - base
    if t < critical:
        return (t - base) * ((critical - t) / (critical - optimum))
    return 0.0


def cumulative_thermal_time_from_peak(
    daily_thermal_time: Iterable[float],
    peak_index: int | None,
) -> np.ndarray:
    daily = np.asarray(daily_thermal_time, dtype=float)
    cumulative = np.full(len(daily), np.nan, dtype=float)
    if peak_index is None:
        return cumulative
    start = int(peak_index)
    if start < 0 or start >= len(daily):
        raise IndexError("El índice del pico está fuera de la serie meteorológica.")
    cumulative[start:] = np.cumsum(np.nan_to_num(daily[start:], nan=0.0))
    return cumulative


def phenology_window_dates(
    dates: Iterable,
    cumulative_thermal_time: Iterable[float],
    control_cd: float = 600.0,
    limit_cd: float = 800.0,
) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    control = float(control_cd)
    limit = float(limit_cd)
    if control < 0 or limit <= control:
        raise ValueError("La ventana fenológica requiere 0 <= control < límite.")

    date_values = pd.to_datetime(pd.Series(list(dates)), errors="coerce")
    tt_values = np.asarray(list(cumulative_thermal_time), dtype=float)
    if len(date_values) != len(tt_values):
        raise ValueError("Fechas y tiempo térmico deben tener la misma longitud.")

    valid = date_values.notna().to_numpy() & np.isfinite(tt_values)
    control_candidates = np.flatnonzero(valid & (tt_values >= control))
    limit_candidates = np.flatnonzero(valid & (tt_values >= limit))
    control_date = (
        pd.Timestamp(date_values.iloc[control_candidates[0]]).normalize()
        if control_candidates.size
        else None
    )
    limit_date = (
        pd.Timestamp(date_values.iloc[limit_candidates[0]]).normalize()
        if limit_candidates.size
        else None
    )
    return control_date, limit_date


def shift_signal(values: np.ndarray, lag_days: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    shifted = np.zeros_like(values)
    if lag_days == 0:
        return values.copy()
    if lag_days > 0:
        if lag_days < len(values):
            shifted[lag_days:] = values[:-lag_days]
    else:
        offset = abs(lag_days)
        if offset < len(values):
            shifted[:-offset] = values[offset:]
    return shifted


def first_peak_index(values: np.ndarray, threshold: float) -> int | None:
    candidates = np.flatnonzero(np.asarray(values, dtype=float) > float(threshold))
    return int(candidates[0]) if candidates.size else None


def apply_termoinhibition_and_peak_filter(
    base_signal: np.ndarray,
    *,
    thermoinhibited: np.ndarray,
    julian_days: np.ndarray,
    latency_jd: int,
    peak_threshold: float,
) -> tuple[np.ndarray, int | None]:
    filtered = np.asarray(base_signal, dtype=float).copy()
    filtered[np.asarray(thermoinhibited, dtype=bool)] = 0.0
    filtered[np.asarray(julian_days, dtype=float) <= int(latency_jd)] = 0.0
    filtered = np.clip(filtered, 0.0, 1.0)
    peak_index = first_peak_index(filtered, peak_threshold)
    if peak_index is None:
        filtered[:] = 0.0
    else:
        filtered[:peak_index] = 0.0
    return filtered, peak_index


def apply_cohort_decay_weibull(
    values: Iterable[float],
    peak_index: int | None,
    *,
    tau_days: float,
    beta: float,
    intensity: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aplica D(t)=exp[-(t/tau)^beta] desde el primer pico.

    La mezcla aplicada es 1-intensidad*(1-D). El día del pico conserva factor
    uno y los días previos no se modifican, reproduciendo el motor de Balcarce.
    """

    signal = np.asarray(values, dtype=float).copy()
    factor = np.ones(len(signal), dtype=float)
    days_since_peak = np.zeros(len(signal), dtype=float)
    if peak_index is None or len(signal) == 0:
        return signal, factor, days_since_peak

    peak = int(peak_index)
    if peak < 0 or peak >= len(signal):
        raise IndexError("El índice del pico está fuera de la señal.")

    tau = max(float(tau_days), 0.01)
    shape = max(float(beta), 0.01)
    mixing = float(np.clip(intensity, 0.0, 1.0))
    days_since_peak = np.maximum(np.arange(len(signal), dtype=float) - peak, 0.0)
    base_factor = np.exp(-np.power(days_since_peak / tau, shape))
    base_factor[:peak] = 1.0
    factor = 1.0 - mixing * (1.0 - base_factor)
    return np.clip(signal * factor, 0.0, 1.0), factor, days_since_peak


@dataclass
class SimulationResult:
    data: pd.DataFrame
    first_peak_no_lag: pd.Timestamp | None
    first_peak_lag: pd.Timestamp | None
    ke: float
    thermal_modulator: float


def simulate_dual(
    raw_weather: pd.DataFrame,
    ann: PracticalANNModel,
    *,
    coverage_percent: float,
    wmax: float,
    lag_days: int,
    kr_exponent: float = 0.0,
    config: ZavallaConfig = CONFIG,
) -> SimulationResult:
    data = canonicalize_weather(raw_weather)
    data["Julian_days"] = data["Fecha"].dt.dayofyear
    data["Tmedia_aire"] = (data["TMAX"] + data["TMIN"]) / 2.0
    ke, thermal_modulator = surface_parameters(coverage_percent)
    data["Cobertura_Rastrojo"] = float(coverage_percent)
    data["Ke_Suelo"] = ke
    data["Exponente_Kr"] = float(kr_exponent)

    inputs = data[["Julian_days", "TMAX", "TMIN", "Prec"]].to_numpy(float)
    raw_ann = ann.predict(inputs)
    data["EMERREL_RAW_ANN"] = raw_ann

    emer_base = raw_ann.copy()
    data["Prec_3d"] = data["Prec"].rolling(
        config.ventana_lluvia_dias,
        min_periods=1,
    ).sum()
    hydric_shock = (
        (data["Julian_days"] > config.latencia_jd)
        & (data["Julian_days"] <= config.fin_choque_hidrico_jd)
        & (data["Prec_3d"] >= config.umbral_choque_hidrico_mm)
    )
    shock_values = hydric_shock.to_numpy()
    emer_base[shock_values] = np.maximum(
        emer_base[shock_values],
        config.techo_choque_hidrico,
    )
    data["Choque_Hidrico"] = hydric_shock

    data["ET0"] = calculate_et0_hargreaves(
        data["Julian_days"],
        data["TMAX"],
        data["TMIN"],
        config.latitud,
    )
    water, kr_daily = surface_water_balance(
        data["Prec"],
        data["ET0"],
        wmax,
        ke,
        kr_exponent,
    )
    data["W_superficial"] = water
    data["Kr_Diario"] = kr_daily
    relative_water = water / max(float(wmax), 1e-12)
    data["Humedad_Relativa"] = relative_water
    hydric_factor = 1.0 / (
        1.0
        + np.exp(
            -config.pendiente_hidrica
            * (relative_water - config.p50_hidrico)
        )
    )
    data["Hydric_Factor"] = hydric_factor
    emer_base *= hydric_factor
    emer_base[relative_water < config.corte_hidrico] = 0.0
    recharge = (data["Prec"] >= float(wmax)).cummax().to_numpy()
    data["Lluvia_Recarga"] = recharge
    emer_base[~recharge] = 0.0

    thermal_column = f"Tmedia_{int(config.ventana_termica_dias)}d"
    data[thermal_column] = data["Tmedia_aire"].rolling(
        config.ventana_termica_dias,
        min_periods=1,
    ).mean()
    # Compatibilidad con las exportaciones previas, cuyo nombre era fijo.
    data["Tmedia_5d"] = data[thermal_column]
    thermoinhibited_no_lag = (
        data[thermal_column] >= config.umbral_termoinhibicion_c
    )
    thermoinhibited_lag = (
        data[thermal_column] >= config.umbral_termoinhibicion_con_lag_c
    )
    data["Termoinhibida"] = thermoinhibited_no_lag
    data["Termoinhibida_SIN_LAG"] = thermoinhibited_no_lag
    data["Termoinhibida_CON_LAG"] = thermoinhibited_lag
    data["Umbral_Termoinhibicion_SIN_LAG_C"] = float(
        config.umbral_termoinhibicion_c
    )
    data["Umbral_Termoinhibicion_CON_LAG_C"] = float(
        config.umbral_termoinhibicion_con_lag_c
    )

    julian = data["Julian_days"].to_numpy(float)
    emer_no_lag, idx0 = apply_termoinhibition_and_peak_filter(
        emer_base,
        thermoinhibited=thermoinhibited_no_lag.to_numpy(),
        julian_days=julian,
        latency_jd=config.latencia_jd,
        peak_threshold=config.umbral_primer_pico,
    )
    no_lag_before_decay = emer_no_lag.copy()

    lag_unshifted, _ = apply_termoinhibition_and_peak_filter(
        emer_base,
        thermoinhibited=thermoinhibited_lag.to_numpy(),
        julian_days=julian,
        latency_jd=config.latencia_jd,
        peak_threshold=config.umbral_primer_pico,
    )
    emer_lag = shift_signal(lag_unshifted, int(lag_days))
    idx_lag = first_peak_index(emer_lag, config.umbral_primer_pico)
    if idx_lag is None:
        emer_lag[:] = 0.0
    else:
        emer_lag[:idx_lag] = 0.0
    lag_before_decay = emer_lag.copy()

    decay_active = bool(config.decaimiento_activo)
    if decay_active:
        emer_no_lag, factor_no_lag, days_no_lag = apply_cohort_decay_weibull(
            emer_no_lag,
            idx0,
            tau_days=config.decaimiento_tau_dias,
            beta=config.decaimiento_beta,
            intensity=config.decaimiento_intensidad,
        )
        emer_lag, factor_lag, days_lag = apply_cohort_decay_weibull(
            emer_lag,
            idx_lag,
            tau_days=config.decaimiento_tau_dias,
            beta=config.decaimiento_beta,
            intensity=config.decaimiento_intensidad,
        )
    else:
        factor_no_lag = np.ones(len(data), dtype=float)
        factor_lag = np.ones(len(data), dtype=float)
        days_no_lag = np.zeros(len(data), dtype=float)
        days_lag = np.zeros(len(data), dtype=float)

    data["EMERREL_SIN_LAG_ANTES_DECAIMIENTO"] = no_lag_before_decay
    data["FACTOR_DECAIMIENTO_SIN_LAG"] = factor_no_lag
    data["DIAS_DESDE_PICO_SIN_LAG"] = days_no_lag
    data["EMERREL_SIN_LAG"] = emer_no_lag

    data["EMERREL_CON_LAG_ANTES_DESPLAZAMIENTO"] = lag_unshifted
    data["EMERREL_CON_LAG_ANTES_DECAIMIENTO"] = lag_before_decay
    data["FACTOR_DECAIMIENTO_CON_LAG"] = factor_lag
    data["DIAS_DESDE_PICO_CON_LAG"] = days_lag
    data["EMERREL_CON_LAG"] = emer_lag

    data["Decaimiento_Activo"] = decay_active
    data["Decaimiento_Tau_Dias"] = float(config.decaimiento_tau_dias)
    data["Decaimiento_Beta"] = float(config.decaimiento_beta)
    data["Decaimiento_Intensidad"] = float(config.decaimiento_intensidad)
    data["Modelo_Referencia_Local"] = str(config.modelo_referencia_local)

    data["EMERAC_SIN_LAG"] = np.cumsum(emer_no_lag)
    data["EMERAC_CON_LAG"] = np.cumsum(emer_lag)

    data["GD_Tb2"] = [
        thermal_time_scalar(
            temperature,
            config.t_base_c,
            config.t_optima_c,
            config.t_critica_c,
        )
        for temperature in data["Tmedia_aire"]
    ]
    data["TT_ACUM"] = data["GD_Tb2"].cumsum()
    data["TT_DESDE_PICO_SIN_LAG"] = cumulative_thermal_time_from_peak(
        data["GD_Tb2"],
        idx0,
    )
    data["TT_DESDE_PICO_CON_LAG"] = cumulative_thermal_time_from_peak(
        data["GD_Tb2"],
        idx_lag,
    )

    peak0 = pd.Timestamp(data.loc[idx0, "Fecha"]) if idx0 is not None else None
    peak_lag = (
        pd.Timestamp(data.loc[idx_lag, "Fecha"])
        if idx_lag is not None
        else None
    )
    return SimulationResult(data, peak0, peak_lag, ke, thermal_modulator)
