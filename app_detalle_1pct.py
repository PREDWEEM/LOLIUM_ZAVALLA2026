from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import app_umbral_operativo as operational


LOW_EMERGENCE_MAX_PCT = 2.0
LOW_EMERGENCE_MAX = 0.02
CAMPAIGN_EPSILON = 1e-12

_ORIGINAL_LOW_EMERGENCE_FIGURE = operational._low_emergence_figure
_ORIGINAL_PLOTLY_WITH_LOW_PANEL = operational._plotly_chart_with_low_panel
_ORIGINAL_TOGGLE = st.toggle


def _canonical_daily(data: Any) -> pd.DataFrame:
    """Normaliza la serie diaria y conserva el máximo EMERREL por fecha."""
    daily = data.loc[:, ["Fecha", "EMERREL"]].copy()
    daily["Fecha"] = pd.to_datetime(
        daily["Fecha"], errors="coerce"
    ).dt.normalize()
    daily["EMERREL"] = pd.to_numeric(
        daily["EMERREL"], errors="coerce"
    )
    daily = daily.dropna(subset=["Fecha", "EMERREL"])
    if daily.empty:
        return daily
    return (
        daily.groupby("Fecha", as_index=False, sort=True)["EMERREL"]
        .max()
        .reset_index(drop=True)
    )


def _campaign_values_at_dates(
    smooth: Any,
    dates: pd.Series,
) -> np.ndarray:
    """Interpola la envolvente pintada en las fechas de los candidatos."""
    trend = smooth.loc[:, ["Fecha", "EMERREL_CAMPANA"]].copy()
    trend["Fecha"] = pd.to_datetime(
        trend["Fecha"], errors="coerce"
    )
    trend["EMERREL_CAMPANA"] = pd.to_numeric(
        trend["EMERREL_CAMPANA"], errors="coerce"
    )
    trend = (
        trend.dropna(subset=["Fecha", "EMERREL_CAMPANA"])
        .sort_values("Fecha")
        .drop_duplicates(subset=["Fecha"], keep="last")
    )
    if trend.empty or dates.empty:
        return np.zeros(len(dates), dtype=float)

    trend_x = trend["Fecha"].astype("int64").to_numpy(dtype=np.int64)
    trend_y = trend["EMERREL_CAMPANA"].to_numpy(dtype=float)
    candidate_x = pd.to_datetime(dates).astype("int64").to_numpy(dtype=np.int64)

    return np.interp(
        candidate_x.astype(float),
        trend_x.astype(float),
        trend_y,
        left=0.0,
        right=0.0,
    )


def _low_emergence_figure_2pct(
    data: Any,
    smooth: Any,
    x_range: Any,
    site_name: str,
    model_name: str,
    today: Any,
) -> go.Figure:
    """Amplía 0–2 % y destaca puntos que están fuera de las campanas."""
    figure = _ORIGINAL_LOW_EMERGENCE_FIGURE(
        data,
        smooth,
        x_range,
        site_name,
        model_name,
        today,
    )

    daily = _canonical_daily(data)
    highlighted = daily.loc[
        (daily["EMERREL"] >= operational.EMERGENCE_THRESHOLD)
        & (daily["EMERREL"] <= LOW_EMERGENCE_MAX)
    ].copy()

    if not highlighted.empty:
        highlighted["EMERREL_CAMPANA"] = _campaign_values_at_dates(
            smooth,
            highlighted["Fecha"],
        )
        highlighted = highlighted.loc[
            highlighted["EMERREL_CAMPANA"] <= CAMPAIGN_EPSILON
        ].copy()

    highlighted["EMERREL_PCT"] = highlighted["EMERREL"] * 100.0

    if not highlighted.empty:
        # La traza se agrega al final para que los círculos queden por encima de
        # barras y líneas. Solo se incluyen fechas sin área de campana pintada.
        figure.add_trace(
            go.Scatter(
                x=highlighted["Fecha"],
                y=highlighted["EMERREL_PCT"],
                customdata=highlighted["EMERREL"],
                mode="markers",
                name="Puntos fuera de las campanas",
                marker={
                    "symbol": "circle",
                    "size": 11,
                    "color": "rgba(255,255,255,0.98)",
                    "line": {"color": "#dc2626", "width": 2.4},
                    "opacity": 1.0,
                },
                hovertemplate=(
                    "<b>Punto fuera de las campanas</b><br>"
                    "Fecha: %{x|%d-%m-%Y}<br>"
                    "Intensidad relativa: %{y:.3f}%<br>"
                    "EMERREL: %{customdata:.4f}<extra></extra>"
                ),
                showlegend=False,
                cliponaxis=False,
            )
        )

    title_text = str(figure.layout.title.text or "")
    for previous in ("EMERREL 0–0,01", "EMERREL 0–0,05"):
        title_text = title_text.replace(previous, "EMERREL 0–0,02")
    figure.update_layout(title={"text": title_text})
    figure.update_yaxes(
        range=[0.0, LOW_EMERGENCE_MAX_PCT],
        tickmode="array",
        tickvals=[0.0, 0.01, 0.1, 0.5, 1.0, 1.5, 2.0],
        ticktext=["0", "0,01", "0,1", "0,5", "1", "1,5", "2"],
    )
    return figure


def _plotly_chart_with_2pct_caption(*args: Any, **kwargs: Any):
    """Actualiza el texto del panel 0–2 % durante su renderizado."""
    original_caption = st.caption

    def caption_2pct(body: Any, *caption_args: Any, **caption_kwargs: Any):
        if isinstance(body, str):
            replacement = (
                "Ampliación de EMERREL 0–0,02 (0–2 %). "
                "Los círculos con borde rojo identifican los puntos con "
                "EMERREL ≥ 0,0001 y ≤ 0,02 que no están incluidos bajo "
                "las campanas pintadas."
            )
            for previous in (
                "Ampliación de EMERREL 0–0,05 (0–5 %).",
                "Ampliación de EMERREL 0–0,01 (0–1 %).",
                "Ampliación de EMERREL 0–0,02 (0–2 %).",
            ):
                if previous in body:
                    body = body.replace(previous, replacement)
                    break
        return original_caption(body, *caption_args, **caption_kwargs)

    st.caption = caption_2pct
    try:
        return _ORIGINAL_PLOTLY_WITH_LOW_PANEL(*args, **kwargs)
    finally:
        st.caption = original_caption


def _toggle_2pct(*args: Any, **kwargs: Any):
    """Actualiza la ayuda del control del panel ampliado."""
    help_text = kwargs.get("help")
    if isinstance(help_text, str):
        help_text = help_text.replace("0 a 5 %", "0 a 2 %")
        help_text = help_text.replace("0 a 1 %", "0 a 2 %")
        help_text = help_text.replace("hasta 0,05", "hasta 0,02")
        help_text = help_text.replace("hasta 0,01", "hasta 0,02")
        kwargs["help"] = help_text
    return _ORIGINAL_TOGGLE(*args, **kwargs)


def run() -> None:
    """Ejecuta PREDWEEM con detalle 0–2 % y puntos fuera de las campanas."""
    original_max_pct = operational.LOW_EMERGENCE_MAX_PCT
    original_low_figure = operational._low_emergence_figure
    original_plotly = operational._plotly_chart_with_low_panel
    original_toggle = st.toggle

    operational.LOW_EMERGENCE_MAX_PCT = LOW_EMERGENCE_MAX_PCT
    operational._low_emergence_figure = _low_emergence_figure_2pct
    operational._plotly_chart_with_low_panel = _plotly_chart_with_2pct_caption
    st.toggle = _toggle_2pct

    try:
        operational.run()
    finally:
        operational.LOW_EMERGENCE_MAX_PCT = original_max_pct
        operational._low_emergence_figure = original_low_figure
        operational._plotly_chart_with_low_panel = original_plotly
        st.toggle = original_toggle
