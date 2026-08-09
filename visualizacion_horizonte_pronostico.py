from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

FORECAST_FILL = "rgba(125, 211, 252, 0.20)"
FORECAST_LINE = "#0284C7"


def _normalizar_serie(data: Any) -> pd.DataFrame:
    if not isinstance(data, pd.DataFrame) or "Fecha" not in data.columns or "EMERREL" not in data.columns:
        return pd.DataFrame()
    frame = data.copy()
    frame["Fecha"] = pd.to_datetime(frame["Fecha"], errors="coerce").dt.normalize()
    frame["EMERREL"] = pd.to_numeric(frame["EMERREL"], errors="coerce")
    frame = frame.dropna(subset=["Fecha", "EMERREL"])
    return frame.sort_values("Fecha").drop_duplicates("Fecha", keep="last")


def _mascara_pronostico(frame: pd.DataFrame) -> pd.Series:
    today = pd.Timestamp.now().normalize()
    mask = pd.Series(False, index=frame.index, dtype=bool)
    for column in ("TipoDato", "TIPODATO", "TIPO", "Fuente", "FUENTE", "CalidadDato", "CALIDADDATO", "Calidad", "CALIDAD"):
        if column in frame.columns:
            text = frame[column].astype("string").fillna("").str.lower()
            mask |= text.str.contains("pronost", regex=False)
    mask &= frame["Fecha"] >= today
    return mask if bool(mask.any()) else frame["Fecha"] >= today


def obtener_horizonte_pronostico(data: Any) -> pd.DataFrame:
    frame = _normalizar_serie(data)
    if frame.empty:
        return frame
    forecast = frame.loc[_mascara_pronostico(frame), ["Fecha", "EMERREL"]].copy()
    if forecast.empty:
        return forecast
    return forecast.groupby("Fecha", as_index=False, sort=True)["EMERREL"].max().reset_index(drop=True)


def es_figura_emergencia_principal(figure: Any) -> bool:
    try:
        title = str(figure.layout.title.text or "").lower()
        names = " ".join(str(getattr(trace, "name", "") or "").lower() for trace in figure.data)
    except Exception:
        return False
    return "emerg" in title and (
        "dinám" in title or "fisiol" in title or "tasa diaria" in names or "simulad" in names
    )


def _rango_temporal_figura(figure: Any) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    fechas: list[pd.Timestamp] = []
    try:
        for trace in figure.data:
            values = getattr(trace, "x", None)
            if values is None:
                continue
            parsed = pd.to_datetime(pd.Series(list(values)), errors="coerce").dropna()
            fechas.extend(pd.Timestamp(value).normalize() for value in parsed)
    except Exception:
        return None
    if not fechas:
        return None
    return min(fechas), max(fechas)


def aplicar_area_pronostico(figure: Any, data: Any = None) -> Any:
    if not es_figura_emergencia_principal(figure):
        return figure
    start = end = None
    forecast = obtener_horizonte_pronostico(data)
    if not forecast.empty:
        start = pd.Timestamp(forecast["Fecha"].min())
        end = pd.Timestamp(forecast["Fecha"].max())
    else:
        rango = _rango_temporal_figura(figure)
        if rango is not None:
            today = pd.Timestamp.now().normalize()
            _, maximum = rango
            if maximum >= today:
                start, end = today, maximum
    if start is None or end is None:
        return figure
    figure.add_vrect(x0=start, x1=end + pd.Timedelta(days=1), fillcolor=FORECAST_FILL, layer="below", line_width=0)
    return figure


def construir_figura_horizonte(data: Any, site_name: str) -> tuple[pd.DataFrame, go.Figure] | None:
    forecast = obtener_horizonte_pronostico(data)
    if forecast.empty:
        return None
    start = pd.Timestamp(forecast["Fecha"].min())
    end = pd.Timestamp(forecast["Fecha"].max())
    labels = [f"{value:.4f}" for value in forecast["EMERREL"]]
    figure = go.Figure(go.Scatter(x=forecast["Fecha"], y=forecast["EMERREL"], mode="lines+markers+text", text=labels, textposition="top center", cliponaxis=False, line={"color": FORECAST_LINE, "width": 3.0}, marker={"size": 9, "color": FORECAST_LINE}, hovertemplate="<b>%{x|%d-%m-%Y}</b><br>EMERREL: %{y:.4f}<extra></extra>"))
    figure.update_layout(template="plotly_white", title={"text": f"Detalle del pronóstico de emergencia · {site_name}", "x": 0.0, "xanchor": "left"}, xaxis={"title": "Fecha", "tickmode": "array", "tickvals": forecast["Fecha"], "ticktext": forecast["Fecha"].dt.strftime("%d-%m"), "range": [start - pd.Timedelta(hours=12), end + pd.Timedelta(hours=12)], "showgrid": False, "fixedrange": False}, yaxis={"title": "EMERREL", "range": [0.0, 1.0], "tickmode": "array", "tickvals": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0], "showgrid": True, "gridcolor": "rgba(148,163,184,0.25)", "fixedrange": False}, hovermode="x unified", height=390, margin={"l": 72, "r": 24, "t": 70, "b": 62}, showlegend=False)
    return forecast, figure


def mostrar_horizonte_pronostico(data: Any, site_name: str) -> None:
    result = construir_figura_horizonte(data, site_name)
    if result is None:
        return
    forecast, figure = result
    start = pd.Timestamp(forecast["Fecha"].min())
    end = pd.Timestamp(forecast["Fecha"].max())
    st.markdown("##### 🔭 Pronóstico de emergencia — detalle diario")
    st.caption(f"{site_name}: horizonte disponible del {start.strftime('%d-%m-%Y')} al {end.strftime('%d-%m-%Y')}. Eje Y en escala lineal EMERREL 0–1; no se agregan ni extrapolan días.")
    st.plotly_chart(figure, width="stretch", config={"displaylogo": False, "responsive": True, "scrollZoom": True, "modeBarButtonsToRemove": ["lasso2d", "select2d"]})


if not getattr(st.plotly_chart, "_predweem_forecast_area", False):
    _ORIGINAL_ST_PLOTLY_CHART = st.plotly_chart

    def _plotly_chart_con_area_pronostico(figure: Any, *args: Any, **kwargs: Any):
        aplicar_area_pronostico(figure)
        return _ORIGINAL_ST_PLOTLY_CHART(figure, *args, **kwargs)

    _plotly_chart_con_area_pronostico._predweem_forecast_area = True
    st.plotly_chart = _plotly_chart_con_area_pronostico
