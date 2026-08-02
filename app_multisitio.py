from __future__ import annotations

import io
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config_zavalla import CONFIG
from predweem_core import load_ann, phenology_window_dates, simulate_dual
from sitios_lolium import DEFAULT_SITE_SLUG, SITES, get_site, ordered_sites
from visualizacion_pulsos import construir_campanas_agrupadas

BASE = Path(__file__).resolve().parent
LEGACY_METEO_PATH = BASE / "meteo_daily.csv"
LOG_OFFSET = 0.01
LOG_Y_RANGE = [-2.18, 0.12]
LOG_Y_TICKS = [-2.0, -1.5, -1.0, -0.5, 0.0]
CHART_STYLES = ("Operativo mejorado", "Minimalista", "Académico")
MONTH_NAMES = {1:"Ene",2:"Feb",3:"Mar",4:"Abr",5:"May",6:"Jun",7:"Jul",8:"Ago",9:"Sep",10:"Oct",11:"Nov",12:"Dic"}
_ANN = None


def read_table(source):
    name = str(getattr(source, "name", source)).lower()
    return pd.read_excel(source) if name.endswith((".xlsx", ".xls")) else pd.read_csv(source)


def get_ann():
    global _ANN
    if _ANN is None:
        _ANN = load_ann(BASE)
    return _ANN


def resolve_weather(site, uploaded):
    if uploaded is not None:
        return read_table(uploaded), "Archivo meteorológico cargado por el usuario"
    path = site.meteo_path(BASE)
    if path.is_file() and path.stat().st_size > 40:
        return pd.read_csv(path), f"Copia exacta del repositorio: {path.relative_to(BASE)}"
    if site.slug == DEFAULT_SITE_SLUG and LEGACY_METEO_PATH.is_file() and LEGACY_METEO_PATH.stat().st_size > 40:
        return pd.read_csv(LEGACY_METEO_PATH), "Copia exacta heredada: meteo_daily.csv"
    return None, "Sin serie meteorológica disponible"


def build_operational_data(data, model_mode, lag_days):
    out = data.copy()
    if model_mode == "con_lag":
        model_name = f"Con lag fijo de {lag_days} días"
        cols = ("EMERREL_CON_LAG","EMERAC_CON_LAG","TT_DESDE_PICO_CON_LAG","Termoinhibida_CON_LAG","Umbral_Termoinhibicion_CON_LAG_C","FACTOR_DECAIMIENTO_CON_LAG","DIAS_DESDE_PICO_CON_LAG")
    elif model_mode == "sin_lag":
        model_name = "Sin lag"
        cols = ("EMERREL_SIN_LAG","EMERAC_SIN_LAG","TT_DESDE_PICO_SIN_LAG","Termoinhibida_SIN_LAG","Umbral_Termoinhibicion_SIN_LAG_C","FACTOR_DECAIMIENTO_SIN_LAG","DIAS_DESDE_PICO_SIN_LAG")
    else:
        raise ValueError(f"Modelo operativo desconocido: {model_mode}")
    out["EMERREL"] = out[cols[0]]
    out["EMERREL_LOG"] = np.log10(out["EMERREL"].clip(lower=0) + LOG_OFFSET)
    out["EMERAC"], out["TT_DESDE_PICO"] = out[cols[1]], out[cols[2]]
    out["Termoinhibida_Operativa"] = out[cols[3]]
    out["Umbral_Termoinhibicion_Operativo_C"] = out[cols[4]]
    out["FACTOR_DECAIMIENTO_OPERATIVO"] = out[cols[5]]
    out["DIAS_DESDE_PICO_OPERATIVO"] = out[cols[6]]
    out["Modelo_Operativo"] = model_name
    out["Lag_Operativo_Dias"] = int(lag_days if model_mode == "con_lag" else 0)
    out["Modelo_Referencia_Local"] = model_mode
    candidates = out.index[out["EMERREL"] > float(CONFIG.umbral_primer_pico)]
    peak = pd.Timestamp(out.loc[candidates[0], "Fecha"]) if len(candidates) else None
    out = out.drop(columns=[c for c in out.columns if "SIN_LAG" in c or "CON_LAG" in c], errors="ignore")
    return out, model_name, peak


def clock_state(data, peak, today):
    ordered = data.sort_values("Fecha").reset_index(drop=True)
    dates = pd.to_datetime(ordered["Fecha"]).dt.normalize()
    current = ordered.index[dates <= today].tolist()
    idx_now = current[-1] if current else 0
    future = ordered.index[dates <= today + pd.Timedelta(days=7)].tolist()
    idx_7 = max(future[-1] if future else idx_now, idx_now)
    dga_now = ordered.loc[idx_now, "TT_DESDE_PICO"]
    dga_7 = ordered.loc[idx_7, "TT_DESDE_PICO"]
    dga_now = max(float(dga_now) if pd.notna(dga_now) else 0.0, 0.0)
    dga_7 = max(float(dga_7) if pd.notna(dga_7) else dga_now, dga_now)
    if peak is None:
        message, status = "Esperando pico de emergencia...", "Reloj térmico aún no iniciado"
    else:
        message = f"Pico validado > {float(CONFIG.umbral_primer_pico):.2f} el {pd.Timestamp(peak).strftime('%d/%m')}"
        status = "Acumulación térmica previa al control" if dga_now < CONFIG.tt_control_cd else ("Ventana fenológica de máxima susceptibilidad" if dga_now < CONFIG.tt_limite_cd else "Ventana fenológica de 600–800 °Cd superada")
    return {"fecha_hoy":pd.Timestamp(ordered.loc[idx_now,"Fecha"]).normalize(),"fecha_pronostico":pd.Timestamp(ordered.loc[idx_7,"Fecha"]).normalize(),"dga_hoy":dga_now,"dga_7dias":dga_7,"mensaje":message,"estado":status}


def clock_figure(state):
    max_axis = float(CONFIG.tt_limite_cd) * 1.2
    marker = min(max(float(state["dga_7dias"]), 0.0), max_axis)
    fig = go.Figure(go.Indicator(mode="gauge+number",value=float(state["dga_hoy"]),domain={"x":[0,1],"y":[0,1]},title={"text":"<b>TT POST-EMERGENCIA (°Cd)</b>","font":{"size":18}},gauge={"axis":{"range":[None,max_axis]},"bar":{"color":"#1e293b","thickness":0.3},"steps":[{"range":[0,float(CONFIG.tt_control_cd)],"color":"#4ade80"},{"range":[float(CONFIG.tt_control_cd),float(CONFIG.tt_limite_cd)],"color":"#facc15"},{"range":[float(CONFIG.tt_limite_cd),max_axis],"color":"#f87171"}],"threshold":{"line":{"color":"#2563eb","width":6},"thickness":0.8,"value":marker}}))
    fig.add_annotation(x=.5,y=-.1,text=f"{state['mensaje']}<br>Pronóstico +7d: <b>{float(state['dga_7dias']):.1f} °Cd</b>",showarrow=False,font={"size":14,"color":"#1e3a8a"},align="center")
    fig.update_layout(height=350,margin={"t":80,"b":50,"l":30,"r":30},paper_bgcolor="#fff",font={"family":"Arial, sans-serif","color":"#334155"})
    return fig


def smooth_pulses(data):
    smooth = construir_campanas_agrupadas(data["Fecha"], data["EMERREL"], umbral=.01, max_dias_sin_flujo=3, puntos_por_dia=6, sigma_min_dias=2.0)
    smooth["EMERREL_CAMPANA_LOG"] = np.log10(smooth["EMERREL_CAMPANA"].clip(lower=0) + LOG_OFFSET)
    return smooth


def monthly_ticks(dates):
    start = pd.Timestamp(dates.min()).to_period("M").to_timestamp()
    end = pd.Timestamp(dates.max()).to_period("M").to_timestamp()
    ticks = pd.date_range(start, end, freq="MS")
    return ticks, [f"{MONTH_NAMES[d.month]} {d.year}" for d in ticks]


def add_window(fig, control, limit, final, style):
    if control is None:
        return
    end = limit if limit is not None else final
    opacity = .26 if style == "Operativo mejorado" else .18
    fig.add_vrect(x0=control,x1=end,fillcolor=f"rgba(250,204,21,{opacity})",layer="below",line_width=0)
    for date in (control, limit):
        if date is not None:
            fig.add_vline(x=date,line_width=1.8,line_dash="dot",line_color="#a16207")
    if style != "Minimalista":
        label = "Ventana fenológica 600–800 °Cd" if limit is not None else "Ventana fenológica · 800 °Cd pendiente"
        fig.add_annotation(x=control+(end-control)/2,y=.985,xref="x",yref="paper",text=f"<b>{label}</b>",showarrow=False,xanchor="center",yanchor="top",bgcolor="rgba(254,249,195,.96)",bordercolor="rgba(202,138,4,.55)",borderwidth=1,borderpad=5,font={"size":11,"color":"#713f12"})
        for date, text in ((control,"600 °Cd"),(limit,"800 °Cd")):
            if date is not None:
                fig.add_annotation(x=date,y=1.04,xref="x",yref="paper",text=text,showarrow=False,bgcolor="rgba(255,255,255,.95)",bordercolor="rgba(161,98,7,.45)",borderwidth=1,borderpad=4,font={"size":11,"color":"#713f12"})


def add_traces(fig, data, smooth, style, peak):
    if style != "Minimalista":
        academic = style == "Académico"
        fig.add_trace(go.Scatter(x=smooth["Fecha"],y=smooth["EMERREL_CAMPANA_LOG"],customdata=smooth["EMERREL_CAMPANA"],name="Envolvente de pulsos" if academic else "Pulsos agrupados",mode="lines",line={"color":"#475569" if academic else "#334155","width":1.8 if academic else 2.0,"dash":"dash" if academic else "solid","shape":"spline"},opacity=.86 if academic else .72,hovertemplate="<b>%{x|%d-%m-%Y}</b><br>Pulsos (log): %{y:.3f}<br>EMERREL agrupada: %{customdata:.3f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=data["Fecha"],y=data["EMERREL_LOG"],customdata=data["EMERREL"],name="Tasa diaria simulada",mode="lines",line={"color":"#1d4ed8" if style=="Académico" else "#075FCF","width":2.25 if style=="Académico" else 2.8},hovertemplate="<b>%{x|%d-%m-%Y}</b><br>Log10(EMERREL + 0,01): %{y:.3f}<br>EMERREL: %{customdata:.3f}<extra></extra>"))
    if peak is not None and style != "Minimalista":
        row = data[data["Fecha"] == pd.Timestamp(peak)]
        if not row.empty:
            row = row.iloc[0]
            fig.add_trace(go.Scatter(x=[peak],y=[float(row["EMERREL_LOG"])],customdata=[float(row["EMERREL"])],name="Primer pico válido",mode="markers",marker={"size":10,"color":"#dc2626","line":{"width":2,"color":"#fff"}},hovertemplate="<b>Primer pico válido</b><br>Fecha: %{x|%d-%m-%Y}<br>EMERREL: %{customdata:.3f}<extra></extra>"))
            fig.add_vline(x=peak,line_dash="dot",line_width=1.3,line_color="rgba(220,38,38,.60)")


def emergence_figure(data, smooth, site_name, model_name, style, peak, control, limit):
    fig = go.Figure()
    add_traces(fig, data, smooth, style, peak)
    add_window(fig, control, limit, pd.Timestamp(data["Fecha"].max()), style)
    ticks, labels = monthly_ticks(data["Fecha"])
    title = {"Operativo mejorado":"Dinámica operativa de emergencia","Minimalista":"Emergencia simulada","Académico":"Dinámica temporal de emergencia simulada"}[style]
    subtitle = f"{site_name} · {model_name} · Escala Log10(EMERREL + 0,01)"
    fig.update_layout(template="plotly_white",title={"text":f"<b>{title}</b><br><span style='font-size:13px;color:#64748b'>{subtitle}</span>","x":0,"xanchor":"left","font":{"size":21,"color":"#0f172a"}},xaxis={"title":{"text":"Fecha","standoff":14},"tickmode":"array","tickvals":ticks,"ticktext":labels,"tickfont":{"size":11,"color":"#475569"},"showgrid":False,"showline":True,"linecolor":"#94a3b8","ticks":"outside","ticklen":5,"zeroline":False,"automargin":True,"rangeslider":{"visible":False}},yaxis={"title":{"text":"Log10(EMERREL + 0,01)","standoff":13},"range":LOG_Y_RANGE,"tickmode":"array","tickvals":LOG_Y_TICKS,"tickfont":{"size":11,"color":"#475569"},"showgrid":True,"gridcolor":"rgba(148,163,184,.24)","griddash":"dash","showline":True,"linecolor":"#94a3b8","zeroline":False,"automargin":True},hovermode="x unified",hoverlabel={"bgcolor":"#fff","bordercolor":"#cbd5e1","font":{"size":12,"color":"#0f172a"}},height=550 if style=="Minimalista" else 620,margin={"l":82,"r":28,"t":112,"b":78},showlegend=style!="Minimalista",legend={"orientation":"h","yanchor":"bottom","y":1.10,"xanchor":"right","x":1,"bgcolor":"rgba(255,255,255,.92)","bordercolor":"rgba(148,163,184,.38)","borderwidth":1,"font":{"size":11,"color":"#334155"}},paper_bgcolor="#fff",plot_bgcolor="#fff",font={"family":"Arial, sans-serif","color":"#334155"},dragmode="zoom")
    fig.update_xaxes(fixedrange=False)
    fig.update_yaxes(fixedrange=False)
    return fig


def format_window(control, limit):
    if control is None:
        return "600 °Cd todavía no alcanzados"
    return f"600 °Cd = {control.strftime('%d/%m/%Y')}; 800 °Cd = {limit.strftime('%d/%m/%Y') if limit is not None else 'pendiente'}"


def run():
    st.set_page_config(page_title="PREDWEEM LOLIUM Multisitio",page_icon="🌾",layout="wide")
    st.markdown("""<style>.block-container{padding-top:1.25rem;padding-bottom:2.25rem;max-width:1520px}[data-testid='stSidebar']{background:linear-gradient(180deg,#f0fdf4 0%,#ecfdf5 52%,#f8fafc 100%);border-right:1px solid #d1fae5}[data-testid='stMetric']{background:#fff;border:1px solid #e2e8f0;border-radius:12px;padding:.75rem .9rem;box-shadow:0 3px 12px rgba(15,23,42,.055)}div[data-testid='stPlotlyChart']{background:#fff;border:1px solid #e2e8f0;border-radius:14px;padding:.35rem;box-shadow:0 5px 18px rgba(15,23,42,.065)}h1,h2,h3{letter-spacing:-.02em}#MainMenu,footer{visibility:hidden}</style>""",unsafe_allow_html=True)
    sites = ordered_sites(); slugs = [s.slug for s in sites]
    with st.sidebar:
        st.header("Configuración geográfica")
        slug = st.selectbox("Sitio específico",slugs,index=slugs.index(DEFAULT_SITE_SLUG),format_func=lambda x:SITES[x].etiqueta,key="selected_lolium_site")
        site = get_site(slug)
        st.write(f"**{site.etiqueta}**"); st.caption(f"Lat {site.latitud:.5f} · Lon {site.longitud:.5f}")
        st.markdown(f"Repositorio de referencia: [`{site.repositorio}`]({site.repository_url})")
        st.success(f"Modelo automático: **{site.modelo_operativo_etiqueta}**")
        uploaded = st.file_uploader(f"Meteorología diaria de {site.nombre}",type=["csv","xlsx","xls"],key=f"weather_upload_{site.slug}")
        coverage = st.slider("Cobertura de rastrojo (%)",0,100,CONFIG.cobertura_predeterminada_pct,5,key=f"coverage_{site.slug}",help="Única variable agronómica de ajuste visible.")
        style = st.selectbox("Estilo de visualización",CHART_STYLES,index=0,key=f"chart_style_{site.slug}",help="Operativo: lectura rápida. Minimalista: máxima simplicidad. Académico: informes y publicaciones.")
        st.caption("La cobertura es la única variable agronómica ajustable. El estilo solo modifica la presentación.")
    config = replace(CONFIG,nombre_sitio=site.etiqueta,latitud=site.latitud,longitud=site.longitud,timezone=site.timezone)
    st.title(f"🌾 PREDWEEM {site.nombre} — {site.modelo_operativo_etiqueta}")
    st.caption("Predicción operativa de emergencia de Lolium con selección automática del modelo según la localidad.")
    weather, source = resolve_weather(site, uploaded); st.caption(f"**Fuente meteorológica activa:** {source}")
    if weather is None or weather.empty:
        st.warning(f"No existe una serie meteorológica operativa para {site.etiqueta}. Ejecute el workflow diario o cargue un archivo con Fecha, TMAX, TMIN y Prec."); st.stop()
    try:
        result = simulate_dual(weather,get_ann(),coverage_percent=coverage,wmax=float(config.wmax_predeterminado_mm),lag_days=int(site.lag_operativo_dias),kr_exponent=float(config.exponente_kr_predeterminado),config=config)
        data, model_name, peak = build_operational_data(result.data,site.modelo_operativo,site.lag_operativo_dias)
    except Exception as exc:
        st.error(f"No se pudo ejecutar PREDWEEM para {site.nombre}: {exc}"); st.stop()
    data["Fecha"] = pd.to_datetime(data["Fecha"]); data["Sitio"],data["Latitud"],data["Longitud"] = site.nombre,site.latitud,site.longitud
    control, limit = phenology_window_dates(data["Fecha"],data["TT_DESDE_PICO"],config.tt_control_cd,config.tt_limite_cd)
    state = clock_state(data,peak,pd.Timestamp.now(tz=site.timezone).tz_localize(None).normalize())
    st.markdown(f"<div style='padding:16px 18px;border-radius:14px;border:1px solid #bbf7d0;background:linear-gradient(90deg,#f0fdf4,#fff);box-shadow:0 4px 14px rgba(15,23,42,.055)'><b style='color:#166534'>Sitio:</b> {site.etiqueta}<br><b style='color:#166534'>Modelo operativo:</b> {model_name}<br><b style='color:#166534'>Cobertura:</b> {coverage}% · <b style='color:#166534'>Visualización:</b> {style}<br><span style='color:#64748b'>Selección fija por localidad; sin recuentos de campo.</span></div>",unsafe_allow_html=True)
    st.write("")
    m = st.columns(3); m[0].metric("Cobertura de rastrojo",f"{coverage}%"); m[1].metric("Primer pico",peak.strftime("%d/%m/%Y") if peak is not None else "—"); m[2].metric("Ventana fenológica",f"{control.strftime('%d/%m')}–{limit.strftime('%d/%m')}" if control is not None and limit is not None else "Pendiente")
    smooth = smooth_pulses(data); smooth["Modelo"],smooth["Sitio"] = model_name,site.nombre
    fig = emergence_figure(data,smooth,site.nombre,model_name,style,peak,control,limit)
    main_col, gauge_col = st.columns([3.4,1])
    with main_col:
        st.plotly_chart(fig,width="stretch",config={"displaylogo":False,"responsive":True,"scrollZoom":True,"modeBarButtonsToRemove":["lasso2d","select2d"],"toImageButtonOptions":{"format":"png","filename":f"PREDWEEM_{site.slug}_{style.lower().replace(' ','_')}","height":1100,"width":2100,"scale":2}})
        st.caption("La franja amarilla identifica la ventana fenológica de 600–800 °Cd. El eje Y usa Log10(EMERREL + 0,01); el cursor conserva EMERREL original.")
        st.caption(f"Ventana fenológica: {format_window(control,limit)}")
    with gauge_col:
        st.plotly_chart(clock_figure(state),width="stretch"); st.caption(str(state["estado"]))
    with st.expander("Resultados diarios del modelo operativo"):
        st.dataframe(data,width="stretch",hide_index=True)
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer,engine="openpyxl") as writer:
            pd.DataFrame([{"Sitio":site.nombre,"Provincia":site.provincia,"Latitud":site.latitud,"Longitud":site.longitud,"Repositorio_origen":site.repositorio,"Archivo_meteorologico":str(site.meteo_path(BASE).relative_to(BASE)),"Fuente_meteorologica_activa":source,"Modelo_operativo":model_name,"Lag_operativo_dias":site.lag_operativo_dias if site.modelo_operativo=="con_lag" else 0,"Cobertura_rastrojo_pct":coverage,"Estilo_grafico":style,"Color_ventana_fenologica":"amarillo","Transformacion_grafica_Y":"log10(EMERREL + 0.01)","Seleccion_automatica":True,"Usa_recuento_campo":False}]).to_excel(writer,sheet_name="Sitio",index=False)
            data.to_excel(writer,sheet_name="Simulacion_Operativa",index=False)
            pd.DataFrame([{"Modelo":model_name,"Fecha_primer_pico":peak,"Fecha_reloj":state["fecha_hoy"],"TT_actual_desde_pico_Cd":state["dga_hoy"],"Fecha_pronostico_7d":state["fecha_pronostico"],"TT_pronostico_7d_Cd":state["dga_7dias"],"Estado_fenologico":state["estado"],"Fecha_600_Cd":control,"Fecha_800_Cd":limit,"Cobertura_rastrojo_pct":coverage}]).to_excel(writer,sheet_name="Reloj_Fenologico",index=False)
            smooth.to_excel(writer,sheet_name="Pulsos_Agrupados",index=False)
        st.download_button("Descargar resultados completos",data=buffer.getvalue(),file_name=f"PREDWEEM_{site.slug}_Resultados.xlsx",mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    st.caption("Política automática: Pergamino y Zavalla usan solamente el modelo con lag fijo; las demás localidades usan solamente el modelo sin lag.")
