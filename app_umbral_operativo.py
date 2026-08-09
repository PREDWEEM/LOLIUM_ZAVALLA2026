from __future__ import annotations

import re
from typing import Any, Mapping

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import app_zoom_operativo as base
from visualizacion_horizonte_pronostico import construir_figura_horizonte


EMERGENCE_THRESHOLD = 0.0001
LOW_EMERGENCE_MAX_PCT = 2.0
LOW_EMERGENCE_THRESHOLD_PCT = EMERGENCE_THRESHOLD * 100.0

_ORIGINAL_RENDER_EMERGENCE_SEMAPHORE = base._render_emergence_semaphore
_ORIGINAL_EMERGENCE_FIGURE_WITH_PHENOLOGY = base._emergence_figure_with_phenology
_LOW_PANEL_FIGURES: dict[int, go.Figure] = {}
_FORECAST_PANEL_FIGURES: dict[int, tuple[pd.DataFrame, go.Figure]] = {}
_SHOW_LOW_PANEL = True


def _seven_day_emergence_forecast(data: Any, today: Any) -> dict[str, Any]:
    """Evalúa emergencia futura con el criterio EMERREL >= 0,0001."""
    today_date = pd.Timestamp(today).normalize()
    start_date = today_date + pd.Timedelta(days=1)
    end_date = today_date + pd.Timedelta(days=base._FORECAST_DAYS)

    status: dict[str, Any] = {
        "start_label": start_date.strftime("%d/%m/%Y"),
        "end_label": end_date.strftime("%d/%m/%Y"),
        "available_days": 0,
        "has_emergence": False,
        "positive_days": 0,
        "max_intensity_pct": 0.0,
        "first_emergence_label": None,
    }

    if data is None or "Fecha" not in data or "EMERREL" not in data:
        return status

    forecast = data.loc[:, ["Fecha", "EMERREL"]].copy()
    forecast["Fecha"] = pd.to_datetime(
        forecast["Fecha"], errors="coerce"
    ).dt.normalize()
    forecast["EMERREL"] = pd.to_numeric(
        forecast["EMERREL"], errors="coerce"
    )
    forecast = forecast.dropna(subset=["Fecha", "EMERREL"])
    forecast = forecast.loc[
        (forecast["Fecha"] >= start_date)
        & (forecast["Fecha"] <= end_date)
    ].sort_values("Fecha")

    if forecast.empty:
        return status

    positive = forecast["EMERREL"] >= EMERGENCE_THRESHOLD
    status["available_days"] = int(forecast["Fecha"].nunique())
    status["has_emergence"] = bool(positive.any())
    status["positive_days"] = int(positive.sum())
    status["max_intensity_pct"] = float(
        forecast["EMERREL"].clip(lower=0.0).max() * 100.0
    )

    if bool(positive.any()):
        first_date = pd.Timestamp(forecast.loc[positive, "Fecha"].iloc[0])
        status["first_emergence_label"] = first_date.strftime("%d/%m/%Y")

    return status


def _render_emergence_semaphore(status: Mapping[str, Any]) -> None:
    """Conserva el diseño y muestra correctamente el nuevo criterio operativo."""
    original_markdown = st.markdown

    def markdown_with_threshold(body: Any, *args: Any, **kwargs: Any):
        if isinstance(body, str):
            body = re.sub(
                r"EMERREL &gt;=? [0-9.]+",
                "EMERREL &gt;= 0.0001",
                body,
            )
        return original_markdown(body, *args, **kwargs)

    st.markdown = markdown_with_threshold
    try:
        _ORIGINAL_RENDER_EMERGENCE_SEMAPHORE(status)
    finally:
        st.markdown = original_markdown


def _low_emergence_figure(
    data: Any,
    smooth: Any,
    x_range: Any,
    site_name: str,
    model_name: str,
    today: Any,
) -> go.Figure:
    """Amplía la emergencia relativa comprendida entre 0 y 2 %."""
    daily = data.loc[:, ["Fecha", "EMERREL"]].copy()
    daily["Fecha"] = pd.to_datetime(daily["Fecha"], errors="coerce")
    daily["EMERREL"] = pd.to_numeric(daily["EMERREL"], errors="coerce")
    daily = daily.dropna(subset=["Fecha", "EMERREL"])
    daily["EMERREL_PCT"] = daily["EMERREL"].clip(lower=0.0) * 100.0

    trend = smooth.loc[:, ["Fecha", "EMERREL_CAMPANA"]].copy()
    trend["Fecha"] = pd.to_datetime(trend["Fecha"], errors="coerce")
    trend["EMERREL_CAMPANA"] = pd.to_numeric(
        trend["EMERREL_CAMPANA"], errors="coerce"
    )
    trend = trend.dropna(subset=["Fecha", "EMERREL_CAMPANA"])
    trend["EMERREL_CAMPANA_PCT"] = (
        trend["EMERREL_CAMPANA"].clip(lower=0.0) * 100.0
    )

    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            x=daily["Fecha"],
            y=daily["EMERREL_PCT"],
            customdata=daily["EMERREL"],
            marker={
                "color": "rgba(37,99,235,0.62)",
                "line": {"color": "rgba(29,78,216,0.72)", "width": 0.3},
            },
            hovertemplate=(
                "<b>%{x|%d-%m-%Y}</b><br>"
                "Intensidad relativa: %{y:.3f}%<br>"
                "EMERREL: %{customdata:.4f}<extra></extra>"
            ),
            showlegend=False,
        )
    )
    if not trend.empty:
        figure.add_trace(
            go.Scatter(
                x=trend["Fecha"],
                y=trend["EMERREL_CAMPANA_PCT"],
                customdata=trend["EMERREL_CAMPANA"],
                mode="lines",
                line={"color": "#64748b", "width": 1.7, "shape": "spline"},
                fill="tozeroy",
                fillcolor="rgba(96,165,250,0.11)",
                opacity=0.8,
                hovertemplate=(
                    "<b>%{x|%d-%m-%Y}</b><br>"
                    "Tendencia: %{y:.3f}%<br>"
                    "EMERREL agrupada: %{customdata:.4f}<extra></extra>"
                ),
                showlegend=False,
            )
        )

    figure.add_hrect(
        y0=0.0,
        y1=LOW_EMERGENCE_THRESHOLD_PCT,
        fillcolor="rgba(148,163,184,0.08)",
        line_width=0,
        layer="below",
    )
    figure.add_hline(
        y=LOW_EMERGENCE_THRESHOLD_PCT,
        line_width=1.6,
        line_dash="dot",
        line_color="#dc2626",
        annotation_text="<b>Umbral operativo</b> · EMERREL ≥ 0,0001 (0,01 %)",
        annotation_position="top left",
        annotation_font={"size": 10, "color": "#991b1b"},
    )

    today_value = pd.Timestamp(today).normalize()
    if not daily.empty:
        minimum = pd.Timestamp(daily["Fecha"].min()).normalize()
        maximum = pd.Timestamp(daily["Fecha"].max()).normalize()
        if minimum <= today_value <= maximum:
            figure.add_vline(
                x=today_value,
                line_width=1.5,
                line_dash="dash",
                line_color="#111827",
            )

    figure.update_layout(
        template="plotly_white",
        title={
            "text": (
                "<b>Detalle operativo de baja emergencia</b><br>"
                "<span style='font-size:12px;color:#64748b'>"
                f"{site_name} · {model_name} · EMERREL 0–0,02"
                "</span>"
            ),
            "x": 0,
            "xanchor": "left",
            "font": {"size": 17, "color": "#0f172a"},
        },
        xaxis={
            "title": {"text": "Fecha", "standoff": 10},
            "range": x_range,
            "showgrid": False,
            "showline": True,
            "linecolor": "#94a3b8",
            "ticks": "outside",
            "tickfont": {"size": 10, "color": "#475569"},
            "automargin": True,
        },
        yaxis={
            "title": {"text": "Intensidad relativa (%)", "standoff": 10},
            "range": [0.0, LOW_EMERGENCE_MAX_PCT],
            "tickmode": "array",
            "tickvals": [0.0, 0.01, 0.1, 0.5, 1.0, 1.5, 2.0],
            "ticktext": ["0", "0,01", "0,1", "0,5", "1", "1,5", "2"],
            "ticksuffix": "%",
            "showgrid": True,
            "gridcolor": "rgba(148,163,184,0.24)",
            "griddash": "dash",
            "zeroline": False,
            "automargin": True,
        },
        barmode="overlay",
        bargap=0.15,
        hovermode="x unified",
        hoverlabel={
            "bgcolor": "#ffffff",
            "bordercolor": "#cbd5e1",
            "font": {"size": 11, "color": "#0f172a"},
        },
        height=315,
        margin={"l": 82, "r": 28, "t": 78, "b": 62},
        showlegend=False,
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font={"family": "Arial, sans-serif", "color": "#334155"},
        dragmode="zoom",
    )
    figure.update_xaxes(fixedrange=False)
    figure.update_yaxes(fixedrange=False)
    return figure


def _emergence_figure_with_low_panel(*args: Any, **kwargs: Any):
    """Añade detalle bajo y el horizonte futuro completo al gráfico principal."""
    figure, x_range = _ORIGINAL_EMERGENCE_FIGURE_WITH_PHENOLOGY(*args, **kwargs)

    data = args[0] if args else kwargs.get("data")
    smooth = args[1] if len(args) > 1 else kwargs.get("smooth")
    site_name = args[2] if len(args) > 2 else kwargs.get("site_name", "")
    model_name = args[3] if len(args) > 3 else kwargs.get("model_name", "")
    scale_mode = args[8] if len(args) > 8 else kwargs.get("scale_mode")
    today = args[9] if len(args) > 9 else kwargs.get("today")

    if (
        _SHOW_LOW_PANEL
        and scale_mode == "Operativa (%)"
        and data is not None
        and smooth is not None
    ):
        _LOW_PANEL_FIGURES[id(figure)] = _low_emergence_figure(
            data,
            smooth,
            x_range,
            str(site_name),
            str(model_name),
            today,
        )

    if data is not None:
        forecast_result = construir_figura_horizonte(data, str(site_name))
        if forecast_result is not None:
            _FORECAST_PANEL_FIGURES[id(figure)] = forecast_result

    return figure, x_range


def _plotly_chart_with_low_panel(*args: Any, **kwargs: Any):
    """Renderiza detalle 0–2 % y horizonte futuro debajo del gráfico principal."""
    config = kwargs.get("config")
    if isinstance(config, Mapping) and config.get("scrollZoom"):
        kwargs["config"] = base._config_with_zoom(config)

    result = base._ORIGINAL_PLOTLY_CHART(*args, **kwargs)
    figure = args[0] if args else kwargs.get("figure_or_data")

    low_figure = _LOW_PANEL_FIGURES.pop(id(figure), None)
    if low_figure is not None:
        low_config = base._config_with_zoom(
            {
                "displaylogo": False,
                "responsive": True,
                "scrollZoom": True,
                "modeBarButtonsToRemove": ["lasso2d", "select2d"],
                "toImageButtonOptions": {
                    "format": "png",
                    "filename": "PREDWEEM_detalle_baja_emergencia",
                    "height": 700,
                    "width": 2200,
                    "scale": 2,
                },
            }
        )
        base._ORIGINAL_PLOTLY_CHART(low_figure, width="stretch", config=low_config)
        st.caption(
            "Ampliación de EMERREL 0–0,02 (0–2 %). "
            "La línea roja punteada indica el umbral operativo "
            "EMERREL ≥ 0,0001 (0,01 %)."
        )

    forecast_panel = _FORECAST_PANEL_FIGURES.pop(id(figure), None)
    if forecast_panel is not None:
        forecast, forecast_figure = forecast_panel
        start = pd.Timestamp(forecast["Fecha"].min())
        end = pd.Timestamp(forecast["Fecha"].max())
        st.markdown("##### 🔭 Pronóstico de emergencia — horizonte completo")
        st.caption(
            f"Zavalla: {len(forecast)} días meteorológicos disponibles, "
            f"del {start.strftime('%d-%m-%Y')} al {end.strftime('%d-%m-%Y')}. "
            "Se muestran únicamente las fechas presentes en la serie operativa; "
            "no se agregan ni extrapolan días."
        )
        forecast_config = base._config_with_zoom(
            {
                "displaylogo": False,
                "responsive": True,
                "scrollZoom": True,
                "modeBarButtonsToRemove": ["lasso2d", "select2d"],
            }
        )
        base._ORIGINAL_PLOTLY_CHART(
            forecast_figure,
            width="stretch",
            config=forecast_config,
        )

    semaphore = base._figure_meta(figure).get(base._SEMAPHORE_META_KEY)
    if isinstance(semaphore, Mapping):
        base._render_emergence_semaphore(semaphore)

    return result


def run() -> None:
    """Ejecuta PREDWEEM con campaña completa, detalle bajo y umbral actualizado."""
    global _SHOW_LOW_PANEL

    original_threshold = base._EMERGENCE_THRESHOLD
    original_forecast = base._seven_day_emergence_forecast
    original_renderer = base._render_emergence_semaphore
    original_plotly_wrapper = base._plotly_chart_with_zoom
    original_emergence_wrapper = base._emergence_figure_with_phenology
    original_toggle = st.toggle

    _LOW_PANEL_FIGURES.clear()
    _FORECAST_PANEL_FIGURES.clear()
    _SHOW_LOW_PANEL = True

    def toggle_with_operational_defaults(*args: Any, **kwargs: Any):
        global _SHOW_LOW_PANEL

        key = str(kwargs.get("key", ""))
        if key.startswith("full_campaign_"):
            kwargs["value"] = True

        result = original_toggle(*args, **kwargs)

        if key.startswith("thermal_panel_"):
            site_slug = key.removeprefix("thermal_panel_")
            _SHOW_LOW_PANEL = bool(
                original_toggle(
                    "Mostrar detalle baja emergencia",
                    value=True,
                    key=f"low_emergence_panel_{site_slug}",
                    help=(
                        "Muestra un panel sincronizado con eje Y de 0 a 2 % "
                        "para distinguir valores EMERREL inferiores a 0,02. "
                        "Se aplica únicamente en escala Operativa (%)."
                    ),
                )
            )

        return result

    base._EMERGENCE_THRESHOLD = EMERGENCE_THRESHOLD
    base._seven_day_emergence_forecast = _seven_day_emergence_forecast
    base._render_emergence_semaphore = _render_emergence_semaphore
    base._plotly_chart_with_zoom = _plotly_chart_with_low_panel
    base._emergence_figure_with_phenology = _emergence_figure_with_low_panel
    st.toggle = toggle_with_operational_defaults

    try:
        base.run()
    finally:
        base._EMERGENCE_THRESHOLD = original_threshold
        base._seven_day_emergence_forecast = original_forecast
        base._render_emergence_semaphore = original_renderer
        base._plotly_chart_with_zoom = original_plotly_wrapper
        base._emergence_figure_with_phenology = original_emergence_wrapper
        st.toggle = original_toggle
        _LOW_PANEL_FIGURES.clear()
        _FORECAST_PANEL_FIGURES.clear()
