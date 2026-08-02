from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd
import streamlit as st

from app_multisitio import get_site, ordered_sites
import app_multisitio_principal as principal
from mapa_sitios import render_site_map


_ORIGINAL_PLOTLY_CHART = st.plotly_chart
_ORIGINAL_CAPTION = st.caption
_ORIGINAL_EMERGENCE_FIGURE = principal.emergence_figure
_ORIGINAL_THERMAL_FIGURE = principal.thermal_figure
_ZOOM_BUTTONS = (
    "zoom2d",
    "pan2d",
    "autoScale2d",
    "resetScale2d",
)
_FORECAST_DAYS = 7
_EMERGENCE_THRESHOLD = 0.01
_SEMAPHORE_META_KEY = "predweem_emergence_semaphore"


def _config_with_zoom(config: Mapping[str, Any]) -> dict[str, Any]:
    """Devuelve una copia de la configuración con controles de ampliación."""
    enhanced = dict(config)
    current_buttons = list(enhanced.get("modeBarButtonsToAdd", []))

    for button in _ZOOM_BUTTONS:
        if button not in current_buttons:
            current_buttons.append(button)

    enhanced["modeBarButtonsToAdd"] = current_buttons
    enhanced["displayModeBar"] = True
    enhanced["scrollZoom"] = True
    enhanced["doubleClick"] = "reset+autosize"
    return enhanced


def _figure_meta(figure: Any) -> Mapping[str, Any]:
    layout = getattr(figure, "layout", None)
    meta = getattr(layout, "meta", None)
    return meta if isinstance(meta, Mapping) else {}


def _render_emergence_semaphore(status: Mapping[str, Any]) -> None:
    """Muestra el semáforo inmediatamente debajo del gráfico de emergencia."""
    available_days = int(status.get("available_days", 0))
    start_label = str(status.get("start_label", "—"))
    end_label = str(status.get("end_label", "—"))

    if available_days == 0:
        color = "#64748b"
        background = "#f8fafc"
        border = "#cbd5e1"
        glow = "rgba(100,116,139,0.26)"
        headline = "PRONÓSTICO NO DISPONIBLE"
        detail = "No hay datos meteorológicos/modelados para la ventana solicitada."
    elif bool(status.get("has_emergence")):
        color = "#dc2626"
        background = "#fef2f2"
        border = "#fecaca"
        glow = "rgba(220,38,38,0.30)"
        headline = "EMERGENCIA PRONOSTICADA"
        first_date = str(status.get("first_emergence_label", "—"))
        max_intensity = float(status.get("max_intensity_pct", 0.0))
        positive_days = int(status.get("positive_days", 0))
        detail = (
            f"Primer día previsto: <b>{first_date}</b> · "
            f"Máxima intensidad relativa: <b>{max_intensity:.1f}%</b> · "
            f"{positive_days} día(s) sobre el umbral."
        )
    else:
        color = "#16a34a"
        background = "#f0fdf4"
        border = "#bbf7d0"
        glow = "rgba(22,163,74,0.28)"
        headline = "SIN EMERGENCIA PRONOSTICADA"
        max_intensity = float(status.get("max_intensity_pct", 0.0))
        detail = (
            "Ningún día supera el umbral operativo "
            f"EMERREL &gt; {_EMERGENCE_THRESHOLD:.2f}. "
            f"Máxima intensidad relativa prevista: <b>{max_intensity:.1f}%</b>."
        )

    st.markdown(
        f"""
        <div style="
            display:flex;
            align-items:center;
            gap:18px;
            margin:8px 0 18px 0;
            padding:16px 20px;
            border:1px solid {border};
            border-radius:15px;
            background:linear-gradient(90deg,{background},#ffffff);
            box-shadow:0 5px 18px rgba(15,23,42,0.06);">
            <div style="
                width:58px;
                height:58px;
                flex:0 0 58px;
                border-radius:50%;
                background:{color};
                border:5px solid #ffffff;
                box-shadow:0 0 0 2px {border},0 0 20px {glow};">
            </div>
            <div style="line-height:1.35;">
                <div style="font-size:0.78rem;font-weight:700;color:#64748b;
                    letter-spacing:0.06em;text-transform:uppercase;">
                    Pronóstico de emergencia · próximos {_FORECAST_DAYS} días
                </div>
                <div style="font-size:1.22rem;font-weight:800;color:{color};
                    margin:2px 0 4px 0;">
                    {headline}
                </div>
                <div style="font-size:0.91rem;color:#475569;">
                    Ventana evaluada: <b>{start_label}</b> a <b>{end_label}</b> ·
                    Datos disponibles: <b>{available_days}/{_FORECAST_DAYS} días</b>.<br>
                    {detail}
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _plotly_chart_with_zoom(*args: Any, **kwargs: Any):
    """Añade la lupa y muestra el semáforo bajo el gráfico que lo contiene."""
    config = kwargs.get("config")
    if isinstance(config, Mapping) and config.get("scrollZoom"):
        kwargs["config"] = _config_with_zoom(config)

    result = _ORIGINAL_PLOTLY_CHART(*args, **kwargs)
    figure = args[0] if args else kwargs.get("figure_or_data")
    semaphore = _figure_meta(figure).get(_SEMAPHORE_META_KEY)
    if isinstance(semaphore, Mapping):
        _render_emergence_semaphore(semaphore)
    return result


def _date_from_annotation(text: str) -> str:
    """Recupera la fecha ya formateada desde una anotación de Plotly."""
    return text.split("<br>", 1)[1] if "<br>" in text else ""


def _seven_day_emergence_forecast(data: Any, today: Any) -> dict[str, Any]:
    """Evalúa si EMERREL supera el umbral operativo durante los próximos 7 días."""
    today_date = pd.Timestamp(today).normalize()
    start_date = today_date + pd.Timedelta(days=1)
    end_date = today_date + pd.Timedelta(days=_FORECAST_DAYS)

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
    forecast["Fecha"] = pd.to_datetime(forecast["Fecha"], errors="coerce").dt.normalize()
    forecast["EMERREL"] = pd.to_numeric(forecast["EMERREL"], errors="coerce")
    forecast = forecast.dropna(subset=["Fecha", "EMERREL"])
    forecast = forecast.loc[
        (forecast["Fecha"] >= start_date) & (forecast["Fecha"] <= end_date)
    ].sort_values("Fecha")

    if forecast.empty:
        return status

    positive = forecast["EMERREL"] > _EMERGENCE_THRESHOLD
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


def _emergence_figure_with_phenology(*args: Any, **kwargs: Any):
    """Mantiene los límites fenológicos y añade el semáforo al gráfico principal."""
    figure, x_range = _ORIGINAL_EMERGENCE_FIGURE(*args, **kwargs)
    kept_annotations = []

    for annotation in figure.layout.annotations or ():
        text = str(annotation.text or "")

        if text == "<b>Ventana recomendada de intervención</b>":
            continue
        if "<b>Primer pico válido</b>" in text:
            continue

        if "<b>600 °Cd</b>" in text:
            date_label = _date_from_annotation(text)
            annotation.update(
                text=(
                    "<b>2–3 macollos</b><br>"
                    f"600 °Cd · {date_label}"
                )
            )
        elif "<b>800 °Cd</b>" in text:
            date_label = _date_from_annotation(text)
            annotation.update(
                text=(
                    "<b>6 macollos</b><br>"
                    f"800 °Cd · {date_label}"
                )
            )

        kept_annotations.append(annotation)

    data = args[0] if args else kwargs.get("data")
    today = args[9] if len(args) > 9 else kwargs.get("today")
    metadata = dict(_figure_meta(figure))
    metadata[_SEMAPHORE_META_KEY] = _seven_day_emergence_forecast(data, today)

    figure.update_layout(
        annotations=kept_annotations,
        showlegend=False,
        meta=metadata,
    )
    figure.update_traces(showlegend=False)
    return figure, x_range


def _thermal_figure_with_phenology(*args: Any, **kwargs: Any):
    """Relaciona los umbrales térmicos con los estados de macollaje."""
    figure = _ORIGINAL_THERMAL_FIGURE(*args, **kwargs)

    for annotation in figure.layout.annotations or ():
        text = str(annotation.text or "")
        if "600 °Cd · inicio de ventana" in text:
            annotation.update(text="2–3 macollos · 600 °Cd · inicio de ventana")
        elif "800 °Cd · fin de ventana" in text:
            annotation.update(text="6 macollos · 800 °Cd · fin de ventana")

    return figure


def _caption_with_phenology(body: Any, *args: Any, **kwargs: Any):
    """Aclara la interpretación fenológica debajo del gráfico principal."""
    if body == (
        "Barras azules: emergencia diaria. Línea gris: tendencia de pulsos. "
        "Banda ámbar: ventana recomendada. Línea negra: fecha actual."
    ):
        body = (
            "Barras azules: emergencia diaria. Línea gris: tendencia de pulsos. "
            "Banda ámbar: ventana recomendada entre 2–3 macollos (600 °Cd) "
            "y 6 macollos (800 °Cd). Línea negra: fecha actual."
        )
    return _ORIGINAL_CAPTION(body, *args, **kwargs)


class _SidebarWithSiteMap:
    """Inserta el mapa en el cuerpo principal antes de abrir la barra lateral."""

    def __init__(self, sidebar: Any, runtime_state: dict[str, Any]) -> None:
        self._sidebar = sidebar
        self._runtime_state = runtime_state

    def _render_map_once(self) -> None:
        if self._runtime_state.get("map_rendered"):
            return

        slug = self._runtime_state.get("selected_site_slug")
        if not slug:
            return

        site = get_site(str(slug))
        st.subheader("🗺️ Red PREDWEEM y sitio seleccionado")
        st.caption(
            "Los marcadores azules representan la red de localidades. "
            "El sitio activo se destaca en rojo y se actualiza al cambiar el selector."
        )
        try:
            with st.container(border=True):
                render_site_map(site, ordered_sites(), height=455)
        except Exception as exc:
            st.warning(f"No se pudo representar el mapa de sitios: {exc}")

        self._runtime_state["map_rendered"] = True

    def __enter__(self):
        self._render_map_once()
        return self._sidebar.__enter__()

    def __exit__(self, exc_type, exc_value, traceback):
        return self._sidebar.__exit__(exc_type, exc_value, traceback)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._sidebar, name)


def run() -> None:
    """Ejecuta PREDWEEM con mapa, zoom, fenología y semáforo a 7 días."""
    original_selectbox = st.selectbox
    original_sidebar = st.sidebar
    runtime_state: dict[str, Any] = {
        "selected_site_slug": None,
        "map_rendered": False,
    }

    def selectbox_with_site_capture(*args: Any, **kwargs: Any):
        value = original_selectbox(*args, **kwargs)
        if kwargs.get("key") == "selected_lolium_site":
            runtime_state["selected_site_slug"] = value
        return value

    st.plotly_chart = _plotly_chart_with_zoom
    st.caption = _caption_with_phenology
    st.selectbox = selectbox_with_site_capture
    st.sidebar = _SidebarWithSiteMap(original_sidebar, runtime_state)
    principal.emergence_figure = _emergence_figure_with_phenology
    principal.thermal_figure = _thermal_figure_with_phenology

    try:
        principal.run()
    finally:
        st.plotly_chart = _ORIGINAL_PLOTLY_CHART
        st.caption = _ORIGINAL_CAPTION
        st.selectbox = original_selectbox
        st.sidebar = original_sidebar
        principal.emergence_figure = _ORIGINAL_EMERGENCE_FIGURE
        principal.thermal_figure = _ORIGINAL_THERMAL_FIGURE
