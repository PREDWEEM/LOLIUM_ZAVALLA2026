from __future__ import annotations

from html import escape
from typing import Iterable, Protocol

import folium
import streamlit.components.v1 as components


class SiteLike(Protocol):
    slug: str
    nombre: str
    etiqueta: str
    latitud: float
    longitud: float
    provincia: str
    modelo_operativo_etiqueta: str


_LABEL_TRANSFORMS = (
    "translate(-50%, -58px)",
    "translate(-50%, 18px)",
    "translate(-108%, -20px)",
    "translate(8%, -20px)",
)

_MAP_CSS = """
<style>
.predweem-map-label {
    display: inline-block;
    width: max-content;
    max-width: 180px;
    padding: 5px 10px;
    border: 1px solid rgba(203, 213, 225, 0.92);
    border-radius: 10px;
    background: rgba(255, 255, 255, 0.94);
    color: #166534;
    font: 700 12px/1.15 Arial, sans-serif;
    text-align: center;
    white-space: nowrap;
    box-shadow: 0 3px 10px rgba(15, 23, 42, 0.16);
    pointer-events: none;
}
.predweem-map-label.selected {
    border: 2px solid rgba(220, 38, 38, 0.82);
    background: rgba(255, 255, 255, 0.98);
    color: #991b1b;
    font-size: 13px;
    box-shadow: 0 4px 14px rgba(220, 38, 38, 0.24);
}
.predweem-map-legend {
    position: fixed;
    right: 14px;
    top: 14px;
    z-index: 9999;
    padding: 8px 11px;
    border: 1px solid rgba(203, 213, 225, 0.95);
    border-radius: 10px;
    background: rgba(255, 255, 255, 0.94);
    color: #334155;
    font: 600 11px/1.55 Arial, sans-serif;
    box-shadow: 0 3px 10px rgba(15, 23, 42, 0.14);
}
.predweem-map-dot {
    display: inline-block;
    width: 9px;
    height: 9px;
    margin-right: 5px;
    border-radius: 50%;
}
</style>
"""


def _safe(value: object) -> str:
    return escape(str(value), quote=True)


def _popup_html(site: SiteLike, selected: bool) -> str:
    status = "Sitio seleccionado" if selected else "Sitio disponible"
    return f"""
    <div style="min-width:220px;font-family:Arial,sans-serif;color:#334155;">
        <div style="font-size:15px;font-weight:700;color:#166534;margin-bottom:5px;">
            {_safe(site.etiqueta)}
        </div>
        <div style="font-size:12px;margin-bottom:3px;">
            <b>Estado:</b> {_safe(status)}
        </div>
        <div style="font-size:12px;margin-bottom:3px;">
            <b>Modelo:</b> {_safe(site.modelo_operativo_etiqueta)}
        </div>
        <div style="font-size:12px;margin-bottom:3px;">
            <b>Provincia:</b> {_safe(site.provincia)}
        </div>
        <div style="font-size:12px;">
            <b>Coordenadas:</b> {site.latitud:.5f}, {site.longitud:.5f}
        </div>
    </div>
    """


def build_site_map(selected_site: SiteLike, sites: Iterable[SiteLike]) -> folium.Map:
    """Construye un mapa de la red y destaca la localidad activa."""
    site_list = list(sites)
    if not site_list:
        raise ValueError("No existen sitios configurados para representar.")

    center_lat = sum(float(site.latitud) for site in site_list) / len(site_list)
    center_lon = sum(float(site.longitud) for site in site_list) / len(site_list)

    map_object = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=6,
        tiles="CartoDB positron",
        control_scale=True,
        zoom_control=True,
        prefer_canvas=True,
        scrollWheelZoom=True,
    )
    map_object.get_root().header.add_child(folium.Element(_MAP_CSS))

    for index, site in enumerate(site_list):
        is_selected = site.slug == selected_site.slug
        location = [float(site.latitud), float(site.longitud)]

        if is_selected:
            folium.CircleMarker(
                location=location,
                radius=18,
                color="#dc2626",
                weight=2,
                fill=True,
                fill_color="#fecaca",
                fill_opacity=0.42,
                interactive=False,
            ).add_to(map_object)

        folium.Marker(
            location=location,
            tooltip=(
                f"{site.etiqueta} · sitio seleccionado"
                if is_selected
                else site.etiqueta
            ),
            popup=folium.Popup(
                _popup_html(site, is_selected),
                max_width=280,
            ),
            icon=folium.Icon(
                color="red" if is_selected else "blue",
                icon="map-marker",
                prefix="fa",
            ),
            z_index_offset=1000 if is_selected else 0,
        ).add_to(map_object)

        transform = (
            "translate(-50%, -62px)"
            if is_selected
            else _LABEL_TRANSFORMS[index % len(_LABEL_TRANSFORMS)]
        )
        label_class = "predweem-map-label selected" if is_selected else "predweem-map-label"
        label_html = (
            f'<div class="{label_class}" style="transform:{transform};">'
            f"{_safe(site.nombre)}</div>"
        )
        folium.Marker(
            location=location,
            icon=folium.DivIcon(
                html=label_html,
                icon_size=(1, 1),
                icon_anchor=(0, 0),
            ),
            interactive=False,
            z_index_offset=1100 if is_selected else 100,
        ).add_to(map_object)

    bounds = [
        [
            min(float(site.latitud) for site in site_list),
            min(float(site.longitud) for site in site_list),
        ],
        [
            max(float(site.latitud) for site in site_list),
            max(float(site.longitud) for site in site_list),
        ],
    ]
    map_object.fit_bounds(bounds, padding=(38, 38))

    legend = """
    <div class="predweem-map-legend">
        <div><span class="predweem-map-dot" style="background:#dc2626;"></span>Sitio seleccionado</div>
        <div><span class="predweem-map-dot" style="background:#2563eb;"></span>Otros sitios</div>
    </div>
    """
    map_object.get_root().html.add_child(folium.Element(legend))
    return map_object


def render_site_map(
    selected_site: SiteLike,
    sites: Iterable[SiteLike],
    height: int = 455,
) -> None:
    """Renderiza el mapa Folium dentro de la página principal de Streamlit."""
    map_object = build_site_map(selected_site, sites)
    components.html(
        map_object.get_root().render(),
        height=height,
        scrolling=False,
    )
