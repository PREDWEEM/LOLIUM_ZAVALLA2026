from __future__ import annotations

import numpy as np
import pandas as pd


def _serie_canonica(
    fechas: pd.Series | list,
    valores: pd.Series | list,
) -> pd.DataFrame:
    """Normaliza y ordena una serie temporal diaria para su representación."""
    data = pd.DataFrame({"Fecha": fechas, "Valor": valores})
    data["Fecha"] = pd.to_datetime(data["Fecha"], errors="coerce").dt.normalize()
    data["Valor"] = (
        pd.to_numeric(data["Valor"], errors="coerce")
        .fillna(0.0)
        .clip(lower=0.0)
    )
    data = data.dropna(subset=["Fecha"])
    if data.empty:
        return pd.DataFrame(columns=["Fecha", "Valor"])
    return (
        data.groupby("Fecha", as_index=False, sort=True)["Valor"]
        .max()
        .reset_index(drop=True)
    )


def agrupar_pulsos(
    fechas: pd.Series | list,
    valores: pd.Series | list,
    *,
    umbral: float = 0.01,
    max_dias_sin_flujo: int = 3,
) -> list[tuple[int, int]]:
    """Agrupa activaciones cercanas en pulsos operativos.

    Dos activaciones pertenecen al mismo pulso cuando entre ellas existen como
    máximo ``max_dias_sin_flujo`` días sin señal superior al umbral. Los índices
    devueltos corresponden a la serie canónica ordenada por fecha.
    """
    data = _serie_canonica(fechas, valores)
    if data.empty:
        return []

    active = np.flatnonzero(data["Valor"].to_numpy(dtype=float) > float(umbral))
    if active.size == 0:
        return []

    groups: list[tuple[int, int]] = []
    start = previous = int(active[0])
    dates = data["Fecha"].to_numpy(dtype="datetime64[D]")

    for current_raw in active[1:]:
        current = int(current_raw)
        days_between = int((dates[current] - dates[previous]).astype(int)) - 1
        if days_between <= int(max_dias_sin_flujo):
            previous = current
            continue
        groups.append((start, previous))
        start = previous = current

    groups.append((start, previous))
    return groups


def construir_campanas_agrupadas(
    fechas: pd.Series | list,
    valores: pd.Series | list,
    *,
    umbral: float = 0.01,
    max_dias_sin_flujo: int = 3,
    puntos_por_dia: int = 6,
    sigma_min_dias: float = 2.0,
) -> pd.DataFrame:
    """Convierte pulsos diarios en una envolvente de campanas gaussianas.

    Cada grupo conserva la fecha central ponderada y la altura máxima observada.
    Las campanas se combinan por máximo para evitar que la superposición genere
    valores artificialmente superiores a la escala normalizada de EMERREL.
    """
    data = _serie_canonica(fechas, valores)
    if data.empty:
        return pd.DataFrame(columns=["Fecha", "EMERREL_CAMPANA"])

    first_date = pd.Timestamp(data["Fecha"].iloc[0])
    x_days = (data["Fecha"] - first_date).dt.days.to_numpy(dtype=float)
    y = data["Valor"].to_numpy(dtype=float)

    if len(data) == 1:
        return pd.DataFrame(
            {
                "Fecha": data["Fecha"],
                "EMERREL_CAMPANA": np.maximum(y, 0.0),
            }
        )

    resolution = max(int(puntos_por_dia), 1)
    number_points = max(int(round((x_days[-1] - x_days[0]) * resolution)) + 1, 2)
    x_fine = np.linspace(x_days[0], x_days[-1], number_points)
    envelope = np.zeros_like(x_fine, dtype=float)

    groups = agrupar_pulsos(
        data["Fecha"],
        data["Valor"],
        umbral=umbral,
        max_dias_sin_flujo=max_dias_sin_flujo,
    )

    for start, end in groups:
        group_x = x_days[start : end + 1]
        group_y = y[start : end + 1]
        total_weight = float(group_y.sum())
        if total_weight <= 0.0:
            continue

        center = float(np.average(group_x, weights=group_y))
        variance = float(np.average((group_x - center) ** 2, weights=group_y))
        span = max(float(group_x[-1] - group_x[0]), 1.0)
        sigma = max(float(sigma_min_dias), np.sqrt(max(variance, 0.0)), span / 3.0)
        amplitude = float(np.max(group_y))

        bell = amplitude * np.exp(-0.5 * ((x_fine - center) / sigma) ** 2)
        support = np.abs(x_fine - center) <= 3.5 * sigma
        bell = np.where(support, bell, 0.0)
        envelope = np.maximum(envelope, bell)

    dates_fine = first_date + pd.to_timedelta(x_fine, unit="D")
    return pd.DataFrame(
        {
            "Fecha": dates_fine,
            "EMERREL_CAMPANA": np.clip(envelope, 0.0, max(float(y.max()), 0.0)),
        }
    )
