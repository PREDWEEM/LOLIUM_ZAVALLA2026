from __future__ import annotations

import io
from dataclasses import replace

import numpy as np
import pandas as pd
import streamlit as st

from app_multisitio import (
    BASE,
    CHART_STYLES,
    CONFIG,
    DEFAULT_SITE_SLUG,
    SITES,
    build_operational_data,
    clock_figure,
    clock_state,
    format_window,
    get_ann,
    get_site,
    ordered_sites,
    phenology_window_dates,
    resolve_weather,
    simulate_dual,
    smooth_pulses,
)

from visualizacion_operativa import (
    SCALE_MODES,
    compact_date,
    emergence_figure,
    operational_status,
    thermal_figure,
)


def run() -> None:
    st.set_page_config(
        page_title="PREDWEEM LOLIUM Multisitio",
        page_icon="🌾",
        layout="wide",
    )
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.15rem;
            padding-bottom: 2.25rem;
            max-width: 1540px;
        }
        [data-testid='stSidebar'] {
            background: linear-gradient(180deg,#f0fdf4 0%,#ecfdf5 52%,#f8fafc 100%);
            border-right: 1px solid #d1fae5;
        }
        [data-testid='stMetric'] {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 13px;
            padding: .82rem .95rem;
            box-shadow: 0 4px 14px rgba(15,23,42,.055);
            min-height: 112px;
        }
        div[data-testid='stPlotlyChart'] {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 15px;
            padding: .4rem;
            box-shadow: 0 6px 20px rgba(15,23,42,.065);
        }
        .site-panel {
            padding: 15px 18px;
            border-radius: 14px;
            border: 1px solid #bbf7d0;
            background: linear-gradient(90deg,#f0fdf4,#ffffff);
            box-shadow: 0 4px 14px rgba(15,23,42,.05);
            min-height: 116px;
        }
        .upload-panel {
            padding: 12px 15px;
            border-radius: 12px;
            border: 1px solid #cbd5e1;
            background: #f8fafc;
            color: #475569;
        }
        .coverage-panel {
            padding: 15px 18px;
            border-radius: 14px;
            border: 1px solid #bfdbfe;
            background: linear-gradient(90deg,#eff6ff,#ffffff);
            box-shadow: 0 4px 14px rgba(15,23,42,.05);
            min-height: 112px;
        }
        .operation-panel {
            padding: 13px 17px;
            border-radius: 14px;
            border: 1px solid #fde68a;
            background: linear-gradient(90deg,#fffbeb,#ffffff);
            box-shadow: 0 4px 14px rgba(15,23,42,.045);
        }
        h1,h2,h3 {letter-spacing: -.02em;}
        #MainMenu,footer {visibility: hidden;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    sites = ordered_sites()
    slugs = [site.slug for site in sites]

    st.title("🌾 PREDWEEM LOLIUM — Plataforma multisitio")
    st.caption(
        "Predicción operativa de emergencia de Lolium con selección automática "
        "del modelo según la localidad."
    )

    st.subheader("📍 Sitio y datos meteorológicos")
    site_column, upload_column = st.columns([1.25, 1.75])

    with site_column:
        slug = st.selectbox(
            "Seleccionar sitio",
            slugs,
            index=slugs.index(DEFAULT_SITE_SLUG),
            format_func=lambda value: SITES[value].etiqueta,
            key="selected_lolium_site",
            help="Define la localidad, las coordenadas y el modelo operativo.",
        )
        site = get_site(slug)
        st.markdown(
            f"""
            <div class="site-panel">
                <b style="color:#166534;font-size:1.05rem;">{site.etiqueta}</b><br>
                <span style="color:#475569;">
                    Latitud {site.latitud:.5f} · Longitud {site.longitud:.5f}
                </span><br>
                <span style="color:#166534;">
                    Modelo automático: <b>{site.modelo_operativo_etiqueta}</b>
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with upload_column:
        uploaded = st.file_uploader(
            f"Cargar meteorología diaria de {site.nombre}",
            type=["csv", "xlsx", "xls"],
            key=f"weather_upload_{site.slug}",
            help=(
                "Archivo diario con las columnas Fecha, TMAX, TMIN y Prec. "
                "Cuando no se carga un archivo, se usa la serie del repositorio."
            ),
        )
        if uploaded is not None:
            st.success(f"Archivo cargado: **{uploaded.name}**")
        else:
            st.markdown(
                """
                <div class="upload-panel">
                    Sin archivo manual: PREDWEEM utilizará automáticamente la
                    serie meteorológica disponible en el repositorio del sitio.
                </div>
                """,
                unsafe_allow_html=True,
            )

    with st.sidebar:
        st.header("Información y visualización")
        st.write(f"**{site.etiqueta}**")
        st.caption(f"Lat {site.latitud:.5f} · Lon {site.longitud:.5f}")
        st.markdown(
            f"Repositorio de referencia: [`{site.repositorio}`]"
            f"({site.repository_url})"
        )
        st.success(f"Modelo automático: **{site.modelo_operativo_etiqueta}**")
        style = st.selectbox(
            "Estilo de visualización",
            CHART_STYLES,
            index=0,
            key=f"chart_style_{site.slug}",
            help=(
                "Operativo: lectura rápida. Minimalista: máxima simplicidad. "
                "Académico: informes y publicaciones."
            ),
        )
        st.caption(
            "El sitio, la meteorología, la cobertura, la escala y la vista "
            "temporal se ajustan en la página principal."
        )

    config = replace(
        CONFIG,
        nombre_sitio=site.etiqueta,
        latitud=site.latitud,
        longitud=site.longitud,
        timezone=site.timezone,
    )

    st.subheader("🌱 Ajuste de cobertura superficial")
    coverage_column, explanation_column = st.columns([1.65, 2.35])
    with coverage_column:
        coverage = st.slider(
            "Cobertura de rastrojo (%)",
            0,
            100,
            CONFIG.cobertura_predeterminada_pct,
            5,
            key=f"coverage_{site.slug}",
            help=(
                "Única variable agronómica de ajuste visible. Modifica el "
                "coeficiente de evaporación y el microclima superficial."
            ),
        )
    with explanation_column:
        st.markdown(
            f"""
            <div class="coverage-panel">
                <b style="color:#1d4ed8;font-size:1.02rem;">
                    Cobertura seleccionada: {coverage}%
                </b><br>
                <span style="color:#475569;">
                    Este valor actualiza inmediatamente la simulación. Los demás
                    parámetros biofísicos permanecen fijos según la calibración
                    operativa de {site.nombre}.
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.subheader("📊 Visualización operativa")
    scale_column, campaign_column, thermal_column = st.columns([1.7, 1.15, 1.15])
    with scale_column:
        scale_mode = st.radio(
            "Escala del eje Y",
            SCALE_MODES,
            index=0,
            horizontal=True,
            key=f"scale_mode_{site.slug}",
            help=(
                "La escala operativa expresa la intensidad relativa de 0 a 100 %. "
                "La alternativa logarítmica usa "
                "Log10(Intensidad relativa de emergencia (%) + 1)."
            ),
        )
    with campaign_column:
        show_full_campaign = st.toggle(
            "Mostrar campaña completa",
            value=False,
            key=f"full_campaign_{site.slug}",
            help=(
                "Desactivado: recorta la vista alrededor del período "
                "agronómicamente relevante."
            ),
        )
    with thermal_column:
        show_thermal_panel = st.toggle(
            "Mostrar panel térmico",
            value=False,
            key=f"thermal_panel_{site.slug}",
            help="Agrega el tiempo térmico acumulado con límites en 600 y 800 °Cd.",
        )

    st.markdown(
        """
        <div class="operation-panel">
            <b style="color:#92400e;">Lectura predeterminada:</b>
            barras azules para la emergencia diaria, línea gris para la tendencia,
            banda ámbar para la ventana recomendada y marcador negro para hoy.
        </div>
        """,
        unsafe_allow_html=True,
    )

    weather, source = resolve_weather(site, uploaded)
    st.caption(f"**Fuente meteorológica activa:** {source}")
    if weather is None or weather.empty:
        st.warning(
            f"No existe una serie meteorológica operativa para {site.etiqueta}. "
            "Ejecute el workflow diario o cargue un archivo con Fecha, TMAX, "
            "TMIN y Prec."
        )
        st.stop()

    try:
        result = simulate_dual(
            weather,
            get_ann(),
            coverage_percent=coverage,
            wmax=float(config.wmax_predeterminado_mm),
            lag_days=int(site.lag_operativo_dias),
            kr_exponent=float(config.exponente_kr_predeterminado),
            config=config,
        )
        data, model_name, peak = build_operational_data(
            result.data,
            site.modelo_operativo,
            site.lag_operativo_dias,
        )
    except Exception as exc:
        st.error(f"No se pudo ejecutar PREDWEEM para {site.nombre}: {exc}")
        st.stop()

    data["Fecha"] = pd.to_datetime(data["Fecha"])
    data["EMERREL_PCT"] = data["EMERREL"].clip(lower=0.0) * 100.0
    data = data.drop(columns=["EMERREL_LOG"], errors="ignore")
    data["LOG10_INTENSIDAD_PCT_MAS_1"] = np.log10(data["EMERREL_PCT"] + 1.0)
    data["Sitio"] = site.nombre
    data["Latitud"] = site.latitud
    data["Longitud"] = site.longitud

    control, limit = phenology_window_dates(
        data["Fecha"],
        data["TT_DESDE_PICO"],
        config.tt_control_cd,
        config.tt_limite_cd,
    )
    today_local = pd.Timestamp.now(tz=site.timezone).tz_localize(None).normalize()
    state = clock_state(data, peak, today_local)
    current_status = operational_status(today_local, peak, control, limit)

    st.write("")
    indicators = st.columns(4)
    indicators[0].metric(
        "Primer pico válido",
        compact_date(peak) if peak is not None else "Pendiente",
        help="Primer día con EMERREL superior al umbral operativo.",
    )
    indicators[1].metric(
        "Inicio de ventana",
        compact_date(control) if control is not None else "Pendiente",
        delta="600 °Cd" if control is not None else None,
        delta_color="off",
    )
    indicators[2].metric(
        "Fin de ventana",
        compact_date(limit) if limit is not None else "Pendiente",
        delta="800 °Cd" if limit is not None else None,
        delta_color="off",
    )
    indicators[3].metric(
        "Estado actual",
        current_status,
        delta=today_local.strftime("%d/%m/%Y"),
        delta_color="off",
    )

    st.markdown(
        f"""
        <div style="padding:13px 17px;border-radius:14px;border:1px solid #bbf7d0;
        background:linear-gradient(90deg,#f0fdf4,#ffffff);">
            <b style="color:#166534">Sitio:</b> {site.etiqueta} ·
            <b style="color:#166534">Modelo:</b> {model_name} ·
            <b style="color:#166534">Cobertura:</b> {coverage}% ·
            <b style="color:#166534">Escala:</b> {scale_mode}
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")

    smooth = smooth_pulses(data)
    smooth = smooth.drop(columns=["EMERREL_CAMPANA_LOG"], errors="ignore")
    smooth["EMERREL_CAMPANA_PCT"] = (
        smooth["EMERREL_CAMPANA"].clip(lower=0.0) * 100.0
    )
    smooth["LOG10_INTENSIDAD_CAMPANA_PCT_MAS_1"] = np.log10(
        smooth["EMERREL_CAMPANA_PCT"] + 1.0
    )
    smooth["Modelo"] = model_name
    smooth["Sitio"] = site.nombre
    figure, x_range = emergence_figure(
        data,
        smooth,
        site.nombre,
        model_name,
        style,
        peak,
        control,
        limit,
        scale_mode,
        today_local,
        show_full_campaign,
    )

    main_column, gauge_column = st.columns([3.4, 1])
    with main_column:
        st.plotly_chart(
            figure,
            width="stretch",
            config={
                "displaylogo": False,
                "responsive": True,
                "scrollZoom": True,
                "modeBarButtonsToRemove": ["lasso2d", "select2d"],
                "toImageButtonOptions": {
                    "format": "png",
                    "filename": (
                        f"PREDWEEM_{site.slug}_"
                        f"{scale_mode.lower().replace(' ', '_')}"
                    ),
                    "height": 1150,
                    "width": 2200,
                    "scale": 2,
                },
            },
        )
        st.caption(
            "Barras azules: emergencia diaria. Línea gris: tendencia de pulsos. "
            "Banda ámbar: ventana recomendada. Línea negra: fecha actual."
        )
        st.caption(f"Ventana fenológica: {format_window(control, limit)}")

    with gauge_column:
        st.plotly_chart(clock_figure(state), width="stretch")
        st.caption(str(state["estado"]))

    if show_thermal_panel:
        st.plotly_chart(
            thermal_figure(data, x_range, today_local),
            width="stretch",
            config={
                "displaylogo": False,
                "responsive": True,
                "scrollZoom": True,
                "modeBarButtonsToRemove": ["lasso2d", "select2d"],
            },
        )

    with st.expander("Resultados diarios del modelo operativo"):
        st.dataframe(data, width="stretch", hide_index=True)
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            pd.DataFrame(
                [
                    {
                        "Sitio": site.nombre,
                        "Provincia": site.provincia,
                        "Latitud": site.latitud,
                        "Longitud": site.longitud,
                        "Repositorio_origen": site.repositorio,
                        "Archivo_meteorologico": str(
                            site.meteo_path(BASE).relative_to(BASE)
                        ),
                        "Fuente_meteorologica_activa": source,
                        "Modelo_operativo": model_name,
                        "Lag_operativo_dias": (
                            site.lag_operativo_dias
                            if site.modelo_operativo == "con_lag"
                            else 0
                        ),
                        "Cobertura_rastrojo_pct": coverage,
                        "Estilo_grafico": style,
                        "Escala_grafica": scale_mode,
                        "Transformacion_logaritmica": (
                            "log10(Intensidad relativa de emergencia (%) + 1)"
                        ),
                        "Vista_temporal": (
                            "Campaña completa"
                            if show_full_campaign
                            else "Vista operativa recortada"
                        ),
                        "Panel_termico_visible": show_thermal_panel,
                        "Serie_diaria": "Barras azules",
                        "Tendencia": "Línea gris de pulsos agrupados",
                        "Color_ventana_fenologica": "ámbar",
                        "Estado_actual": current_status,
                        "Seleccion_automatica": True,
                        "Usa_recuento_campo": False,
                    }
                ]
            ).to_excel(writer, sheet_name="Sitio", index=False)
            data.to_excel(
                writer,
                sheet_name="Simulacion_Operativa",
                index=False,
            )
            pd.DataFrame(
                [
                    {
                        "Modelo": model_name,
                        "Fecha_primer_pico": peak,
                        "Fecha_reloj": state["fecha_hoy"],
                        "TT_actual_desde_pico_Cd": state["dga_hoy"],
                        "Fecha_pronostico_7d": state["fecha_pronostico"],
                        "TT_pronostico_7d_Cd": state["dga_7dias"],
                        "Estado_fenologico": state["estado"],
                        "Fecha_600_Cd": control,
                        "Fecha_800_Cd": limit,
                        "Cobertura_rastrojo_pct": coverage,
                    }
                ]
            ).to_excel(writer, sheet_name="Reloj_Fenologico", index=False)
            smooth.to_excel(
                writer,
                sheet_name="Pulsos_Agrupados",
                index=False,
            )

        st.download_button(
            "Descargar resultados completos",
            data=buffer.getvalue(),
            file_name=f"PREDWEEM_{site.slug}_Resultados.xlsx",
            mime=(
                "application/vnd.openxmlformats-officedocument."
                "spreadsheetml.sheet"
            ),
        )

    st.caption(
        "Política automática: Pergamino y Zavalla usan solamente el modelo "
        "con lag fijo; las demás localidades usan solamente el modelo sin lag."
    )
