from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from config_zavalla import CONFIG

LOG_Y_RANGE = [0.0, 2.05]
LOG_Y_TICKS = [0.0, 0.5, 1.0, 1.5, 2.0]
SCIENTIFIC_SCALE = "Log10(Intensidad % + 1)"
SCALE_MODES = ("Operativa (%)", SCIENTIFIC_SCALE)
MONTH_NAMES = {
    1: "Ene",
    2: "Feb",
    3: "Mar",
    4: "Abr",
    5: "May",
    6: "Jun",
    7: "Jul",
    8: "Ago",
    9: "Sep",
    10: "Oct",
    11: "Nov",
    12: "Dic",
}
MONTH_NAMES_FULL = {
    1: "enero",
    2: "febrero",
    3: "marzo",
    4: "abril",
    5: "mayo",
    6: "junio",
    7: "julio",
    8: "agosto",
    9: "septiembre",
    10: "octubre",
    11: "noviembre",
    12: "diciembre",
}


def compact_date(date) -> str:
    if date is None:
        return "Pendiente"
    value = pd.Timestamp(date)
    return f"{value.day} {MONTH_NAMES_FULL[value.month]}"


def operational_status(today, peak, control, limit) -> str:
    today = pd.Timestamp(today).normalize()
    if peak is None:
        return "Esperando primer pico"
    if control is None or today < pd.Timestamp(control).normalize():
        return "Antes de la ventana"
    if limit is None or today <= pd.Timestamp(limit).normalize():
        return "En ventana"
    return "Postventana"


def _monthly_ticks(dates, start, end):
    first_month = pd.Timestamp(start).to_period("M").to_timestamp()
    last_month = pd.Timestamp(end).to_period("M").to_timestamp()
    ticks = pd.date_range(first_month, last_month, freq="MS")
    labels = [f"{MONTH_NAMES[date.month]} {date.year}" for date in ticks]
    return ticks, labels


def _operational_x_range(data, peak, limit, today, show_full_campaign):
    dates = pd.to_datetime(data["Fecha"])
    minimum = pd.Timestamp(dates.min()).normalize()
    maximum = pd.Timestamp(dates.max()).normalize()

    if show_full_campaign:
        return [minimum, maximum]

    events = data.loc[data["EMERREL"] > 0.01, "Fecha"]
    if not events.empty:
        first_event = pd.Timestamp(events.min()).normalize()
    elif peak is not None:
        first_event = pd.Timestamp(peak).normalize()
    else:
        first_event = minimum

    start = max(minimum, first_event - pd.Timedelta(days=15))
    end_reference = (
        pd.Timestamp(limit).normalize()
        if limit is not None
        else maximum
    )
    end = end_reference + pd.Timedelta(days=15)

    today = pd.Timestamp(today).normalize()
    if minimum <= today <= maximum:
        end = max(end, today + pd.Timedelta(days=3))

    return [start, min(maximum, end)]


def _add_intervention_window(figure, control, limit, final_date):
    if control is None:
        return

    control = pd.Timestamp(control)
    end = pd.Timestamp(limit) if limit is not None else pd.Timestamp(final_date)
    figure.add_vrect(
        x0=control,
        x1=end,
        fillcolor="rgba(245,158,11,0.28)",
        layer="below",
        line_width=0,
    )
    figure.add_annotation(
        x=control + (end - control) / 2,
        y=0.97,
        xref="x",
        yref="paper",
        text="<b>Ventana recomendada de intervención</b>",
        showarrow=False,
        xanchor="center",
        yanchor="top",
        bgcolor="rgba(255,251,235,0.95)",
        bordercolor="rgba(217,119,6,0.48)",
        borderwidth=1,
        borderpad=5,
        font={"size": 11, "color": "#92400e"},
    )

    limits = (
        (control, f"<b>600 °Cd</b><br>{compact_date(control)}"),
        (
            pd.Timestamp(limit) if limit is not None else None,
            f"<b>800 °Cd</b><br>{compact_date(limit)}"
            if limit is not None
            else None,
        ),
    )
    for date, label in limits:
        if date is None or label is None:
            continue
        figure.add_vline(
            x=date,
            line_width=1.7,
            line_dash="dot",
            line_color="#b45309",
        )
        figure.add_annotation(
            x=date,
            y=1.035,
            xref="x",
            yref="paper",
            text=label,
            showarrow=False,
            xanchor="center",
            yanchor="bottom",
            bgcolor="rgba(255,255,255,0.96)",
            bordercolor="rgba(180,83,9,0.42)",
            borderwidth=1,
            borderpad=4,
            font={"size": 10, "color": "#78350f"},
        )


def _add_today_marker(figure, today, minimum, maximum):
    today = pd.Timestamp(today).normalize()
    minimum = pd.Timestamp(minimum).normalize()
    maximum = pd.Timestamp(maximum).normalize()
    if not minimum <= today <= maximum:
        return

    figure.add_vrect(
        x0=today,
        x1=maximum,
        fillcolor="rgba(148,163,184,0.055)",
        layer="below",
        line_width=0,
    )
    figure.add_vline(
        x=today,
        line_width=1.7,
        line_dash="dash",
        line_color="#111827",
    )
    figure.add_annotation(
        x=today,
        y=1.035,
        xref="x",
        yref="paper",
        text="<b>Hoy</b>",
        showarrow=False,
        xanchor="center",
        yanchor="bottom",
        bgcolor="rgba(255,255,255,0.96)",
        bordercolor="rgba(17,24,39,0.38)",
        borderwidth=1,
        borderpad=4,
        font={"size": 10, "color": "#111827"},
    )


def _add_emergence_traces(figure, data, smooth, scale_mode, style, peak):
    scientific = scale_mode == SCIENTIFIC_SCALE
    data = data.copy()
    smooth = smooth.copy()

    data["EMERREL_PCT"] = data["EMERREL"].clip(lower=0.0) * 100.0
    data["EMERREL_LOG_PCT"] = np.log10(data["EMERREL_PCT"] + 1.0)
    smooth["EMERREL_CAMPANA_PCT"] = (
        smooth["EMERREL_CAMPANA"].clip(lower=0.0) * 100.0
    )
    smooth["EMERREL_CAMPANA_LOG_PCT"] = np.log10(
        smooth["EMERREL_CAMPANA_PCT"] + 1.0
    )

    if scientific:
        bar_values = data["EMERREL_LOG_PCT"]
        bar_base = 0.0
        smooth_values = smooth["EMERREL_CAMPANA_LOG_PCT"]
        customdata = np.column_stack(
            [
                data["EMERREL_LOG_PCT"],
                data["EMERREL_PCT"],
                data["EMERREL"],
            ]
        )
        bar_hover = (
            "<b>%{x|%d-%m-%Y}</b><br>"
            "Log10(Intensidad % + 1): %{customdata[0]:.3f}<br>"
            "Intensidad relativa: %{customdata[1]:.1f}%<br>"
            "EMERREL: %{customdata[2]:.3f}<extra></extra>"
        )
        smooth_customdata = np.column_stack(
            [
                smooth["EMERREL_CAMPANA_PCT"],
                smooth["EMERREL_CAMPANA"],
            ]
        )
        smooth_hover = (
            "<b>%{x|%d-%m-%Y}</b><br>"
            "Tendencia log10: %{y:.3f}<br>"
            "Intensidad agrupada: %{customdata[0]:.1f}%<br>"
            "EMERREL agrupada: %{customdata[1]:.3f}<extra></extra>"
        )
    else:
        bar_values = data["EMERREL_PCT"]
        bar_base = 0.0
        smooth_values = smooth["EMERREL_CAMPANA_PCT"]
        customdata = np.column_stack([data["EMERREL_PCT"], data["EMERREL"]])
        bar_hover = (
            "<b>%{x|%d-%m-%Y}</b><br>"
            "Intensidad relativa de emergencia: %{customdata[0]:.1f}%<br>"
            "EMERREL: %{customdata[1]:.3f}<extra></extra>"
        )
        smooth_customdata = smooth["EMERREL_CAMPANA"]
        smooth_hover = (
            "<b>%{x|%d-%m-%Y}</b><br>"
            "Pulsos agrupados: %{y:.1f}%<br>"
            "EMERREL agrupada: %{customdata:.3f}<extra></extra>"
        )

    figure.add_trace(
        go.Bar(
            x=data["Fecha"],
            y=bar_values,
            base=bar_base,
            customdata=customdata,
            name="Emergencia diaria simulada",
            marker={
                "color": "rgba(37,99,235,0.58)",
                "line": {"color": "rgba(29,78,216,0.68)", "width": 0.35},
            },
            hovertemplate=bar_hover,
        )
    )

    if style != "Minimalista":
        figure.add_trace(
            go.Scatter(
                x=smooth["Fecha"],
                y=smooth_values,
                customdata=smooth_customdata,
                name="Tendencia · pulsos agrupados",
                mode="lines",
                line={
                    "color": "#64748b",
                    "width": 1.55 if style == "Académico" else 1.75,
                    "dash": "dash" if style == "Académico" else "solid",
                    "shape": "spline",
                },
                fill="tozeroy",
                fillcolor="rgba(96, 165, 250, 0.12)",
                opacity=0.75,
                hovertemplate=smooth_hover,
            )
        )

    if peak is None:
        return
    peak_rows = data[data["Fecha"] == pd.Timestamp(peak)]
    if peak_rows.empty:
        return
    peak_row = peak_rows.iloc[0]
    peak_y = (
        float(peak_row["EMERREL_LOG_PCT"])
        if scientific
        else float(peak_row["EMERREL_PCT"])
    )
    peak_emergence = float(peak_row["EMERREL"])
    peak_intensity = float(peak_row["EMERREL_PCT"])
    figure.add_trace(
        go.Scatter(
            x=[peak],
            y=[peak_y],
            customdata=[[peak_emergence, peak_intensity]],
            name="Primer pico válido",
            mode="markers",
            marker={
                "size": 10,
                "color": "#dc2626",
                "line": {"width": 2, "color": "#ffffff"},
            },
            hovertemplate=(
                "<b>Primer pico válido</b><br>"
                "Fecha: %{x|%d-%m-%Y}<br>"
                "Intensidad relativa: %{customdata[1]:.1f}%<br>"
                "EMERREL: %{customdata[0]:.3f}<extra></extra>"
            ),
        )
    )
    figure.add_vline(
        x=peak,
        line_dash="dot",
        line_width=1.0,
        line_color="rgba(220,38,38,0.20)",
    )
    figure.add_annotation(
        x=peak,
        y=peak_y,
        xref="x",
        yref="y",
        text=(
            "<b>Primer pico válido</b><br>"
            f"{compact_date(peak)} · EMERREL {peak_emergence:.2f}"
        ),
        showarrow=True,
        arrowhead=2,
        arrowsize=0.8,
        arrowwidth=1.1,
        arrowcolor="#dc2626",
        ax=72,
        ay=-48,
        bgcolor="rgba(255,255,255,0.96)",
        bordercolor="rgba(220,38,38,0.36)",
        borderwidth=1,
        borderpad=5,
        font={"size": 10, "color": "#7f1d1d"},
        align="left",
    )


def emergence_figure(
    data,
    smooth,
    site_name,
    model_name,
    style,
    peak,
    control,
    limit,
    scale_mode,
    today,
    show_full_campaign,
):
    figure = go.Figure()
    _add_emergence_traces(figure, data, smooth, scale_mode, style, peak)
    _add_intervention_window(
        figure,
        control,
        limit,
        pd.Timestamp(data["Fecha"].max()),
    )
    _add_today_marker(
        figure,
        today,
        data["Fecha"].min(),
        data["Fecha"].max(),
    )

    x_range = _operational_x_range(
        data,
        peak,
        limit,
        today,
        show_full_campaign,
    )
    ticks, labels = _monthly_ticks(data["Fecha"], x_range[0], x_range[1])

    scientific = scale_mode == SCIENTIFIC_SCALE
    yaxis = {
        "title": {
            "text": (
                "Log10(Intensidad relativa de emergencia (%) + 1)"
                if scientific
                else "Intensidad relativa de emergencia (%)"
            ),
            "standoff": 13,
        },
        "range": LOG_Y_RANGE if scientific else [0, 105],
        "tickmode": "array",
        "tickvals": LOG_Y_TICKS if scientific else [0, 20, 40, 60, 80, 100],
        "ticksuffix": "" if scientific else "%",
        "tickfont": {"size": 11, "color": "#475569"},
        "showgrid": True,
        "gridcolor": "rgba(148,163,184,0.24)",
        "griddash": "dash",
        "showline": True,
        "linecolor": "#94a3b8",
        "zeroline": False,
        "automargin": True,
    }

    title = {
        "Operativo mejorado": "Emergencia simulada y ventana de decisión",
        "Minimalista": "Emergencia simulada",
        "Académico": "Dinámica temporal de emergencia simulada",
    }[style]
    scale_label = (
        "Log10(Intensidad relativa de emergencia (%) + 1)"
        if scientific
        else "Intensidad relativa de emergencia 0–100 %"
    )
    subtitle = f"{site_name} · {model_name} · {scale_label}"

    figure.update_layout(
        template="plotly_white",
        title={
            "text": (
                f"<b>{title}</b><br>"
                f"<span style='font-size:13px;color:#64748b'>{subtitle}</span>"
            ),
            "x": 0,
            "xanchor": "left",
            "font": {"size": 21, "color": "#0f172a"},
        },
        xaxis={
            "title": {"text": "Fecha", "standoff": 14},
            "range": x_range,
            "tickmode": "array",
            "tickvals": ticks,
            "ticktext": labels,
            "tickfont": {"size": 11, "color": "#475569"},
            "showgrid": False,
            "showline": True,
            "linecolor": "#94a3b8",
            "ticks": "outside",
            "ticklen": 5,
            "zeroline": False,
            "automargin": True,
            "rangeslider": {"visible": False},
        },
        yaxis=yaxis,
        barmode="overlay",
        bargap=0.15,
        hovermode="x unified",
        hoverlabel={
            "bgcolor": "#ffffff",
            "bordercolor": "#cbd5e1",
            "font": {"size": 12, "color": "#0f172a"},
        },
        height=570 if style == "Minimalista" else 630,
        margin={"l": 98 if scientific else 82, "r": 28, "t": 126, "b": 78},
        showlegend=style != "Minimalista",
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.12,
            "xanchor": "right",
            "x": 1,
            "bgcolor": "rgba(255,255,255,0.92)",
            "bordercolor": "rgba(148,163,184,0.38)",
            "borderwidth": 1,
            "font": {"size": 11, "color": "#334155"},
        },
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font={"family": "Arial, sans-serif", "color": "#334155"},
        dragmode="zoom",
    )
    figure.update_xaxes(fixedrange=False)
    figure.update_yaxes(fixedrange=False)
    return figure, x_range


def thermal_figure(data, x_range, today):
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=data["Fecha"],
            y=data["TT_DESDE_PICO"],
            name="Tiempo térmico acumulado",
            mode="lines",
            line={"color": "#92400e", "width": 2.2},
            fill="tozeroy",
            fillcolor="rgba(245,158,11,0.08)",
            hovertemplate=(
                "<b>%{x|%d-%m-%Y}</b><br>"
                "Tiempo térmico: %{y:.1f} °Cd<extra></extra>"
            ),
        )
    )
    for threshold, label in (
        (float(CONFIG.tt_control_cd), "600 °Cd · inicio de ventana"),
        (float(CONFIG.tt_limite_cd), "800 °Cd · fin de ventana"),
    ):
        figure.add_hline(
            y=threshold,
            line_dash="dot",
            line_width=1.5,
            line_color="#b45309",
            annotation_text=label,
            annotation_position="top left",
            annotation_font={"size": 10, "color": "#78350f"},
        )

    _add_today_marker(
        figure,
        today,
        data["Fecha"].min(),
        data["Fecha"].max(),
    )
    ticks, labels = _monthly_ticks(data["Fecha"], x_range[0], x_range[1])
    figure.update_layout(
        template="plotly_white",
        title={
            "text": "<b>Tiempo térmico acumulado desde el primer pico</b>",
            "x": 0,
            "xanchor": "left",
            "font": {"size": 16, "color": "#0f172a"},
        },
        xaxis={
            "range": x_range,
            "tickmode": "array",
            "tickvals": ticks,
            "ticktext": labels,
            "showgrid": False,
            "showline": True,
            "linecolor": "#94a3b8",
            "ticks": "outside",
            "tickfont": {"size": 10, "color": "#475569"},
        },
        yaxis={
            "title": "°Cd acumulados",
            "range": [0, max(float(CONFIG.tt_limite_cd) * 1.12, 900)],
            "showgrid": True,
            "gridcolor": "rgba(148,163,184,0.22)",
            "griddash": "dash",
            "zeroline": False,
        },
        height=285,
        margin={"l": 74, "r": 28, "t": 58, "b": 55},
        showlegend=False,
        hovermode="x unified",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
    )
    return figure
