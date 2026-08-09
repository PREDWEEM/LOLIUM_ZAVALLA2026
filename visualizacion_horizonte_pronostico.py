from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


def _normalizar_serie(data: Any) -> pd.DataFrame:
    if not isinstance(data, pd.DataFrame) or "Fecha" not in data.columns or "EMERREL" not in data.columns:
        return pd.DataFrame()
    frame = data.copy()
    frame["Fecha"] = pd.to_datetime(frame["Fecha"], errors="coerce").dt.normalize()
    frame["EMERREL"] = pd.to_numeric(frame["EMERREL"], errors="coerce")
    frame = frame.dropna(subset=["Fecha", "EMERREL"])
    return frame.sort_values("Fecha").drop_duplicates("Fecha", keep="last") if not frame.empty else frame


def _mascara_pronostico(frame: pd.DataFrame, today: pd.Timestamp) -> pd.Series:
    mask = pd.Series(False, index=frame.index, dtype=bool)
    for column in ("TipoDato", "TIPODATO", "TIPO", "Fuente", "FUENTE", "CalidadDato", "CALIDADDATO", "Calidad", "CALIDAD"):
        if column in frame.columns:
            text = frame[column].astype("string").fillna("").str.lower()
            mask = mask | text.str.contains("pronost", regex=False)
    mask = mask & (frame["Fecha"] >= today)
    return mask if bool(mask.any()) else frame["Fecha"] >= today


def construir_figura_horizonte(data: Any, site_name: str) -> tuple[pd.DataFrame, go.Figure] | None:
    frame = _normalizar_serie(data)
    if frame.empty:
        return None
    today = pd.Timestamp.now().normalize()
    forecast = frame.loc[_mascara_pronostico(frame, today), ["Fecha", "EMERREL"]].copy()
    if forecast.empty:
        return None
    forecast = forecast.groupby("Fecha", as_index=False, sort=True)["EMERREL"].max().reset_index(drop=True)
    start, end = pd.Timestamp(forecast["Fecha"].min()), pd.Timestamp(forecast["Fecha"].max())
    labels = [f"{v:.5f}" if abs(float(v)) < 0.001 else f"{v:.3f}" for v in forecast["EMERREL"]]
    figure = go.Figure(go.Scatter(x=forecast["Fecha"], y=forecast["EMERREL"], mode="lines+markers+text", text=labels, textposition="top center", cliponaxis=False, line={"color":"#2563EB","width":3.0}, marker={"size":9,"color":"#2563EB"}, hovertemplate="<b>%{x|%d-%m-%Y}</b><br>EMERREL: %{y:.5f}<extra></extra>"))
    figure.update_layout(template="plotly_white", title={"text":f"Detalle diario del pronóstico de emergencia · {site_name}","x":0.0,"xanchor":"left"}, xaxis={"title":"Fecha","tickmode":"array","tickvals":forecast["Fecha"],"ticktext":forecast["Fecha"].dt.strftime("%d-%m"),"range":[start-pd.Timedelta(hours=12),end+pd.Timedelta(hours=12)],"showgrid":False,"fixedrange":False}, yaxis={"title":"EMERREL","rangemode":"tozero","showgrid":True,"gridcolor":"rgba(148,163,184,0.25)","fixedrange":False}, hovermode="x unified", height=390, margin={"l":72,"r":24,"t":70,"b":62}, showlegend=False)
    return forecast, figure


def mostrar_horizonte_pronostico(data: Any, site_name: str) -> None:
    result = construir_figura_horizonte(data, site_name)
    if result is None:
        return
    forecast, figure = result
    start, end = pd.Timestamp(forecast["Fecha"].min()), pd.Timestamp(forecast["Fecha"].max())
    st.markdown("##### 🔭 Pronóstico de emergencia — horizonte completo")
    st.caption(f"{site_name}: {len(forecast)} días meteorológicos disponibles, del {start.strftime('%d-%m-%Y')} al {end.strftime('%d-%m-%Y')}. Se muestran únicamente las fechas presentes en la serie operativa; no se agregan ni extrapolan días.")
    st.plotly_chart(figure, width="stretch", config={"displaylogo":False,"responsive":True,"scrollZoom":True,"modeBarButtonsToRemove":["lasso2d","select2d"]})
