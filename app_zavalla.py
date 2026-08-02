from __future__ import annotations

from typing import Any

import streamlit as st

import app_fuente_hibrida as base
from sitios_lolium import DEFAULT_SITE_SLUG


def run() -> None:
    original_page_config = st.set_page_config
    original_title = st.title
    original_caption = st.caption
    original_selectbox = st.selectbox

    def local_page_config(*args: Any, **kwargs: Any):
        kwargs["page_title"] = "PREDWEEM LOLIUM — Zavalla 2026"
        kwargs.setdefault("page_icon", "🌾")
        kwargs.setdefault("layout", "wide")
        return original_page_config(*args, **kwargs)

    def local_title(body: Any, *args: Any, **kwargs: Any):
        if isinstance(body, str) and "Plataforma multisitio" in body:
            body = "🌾 PREDWEEM LOLIUM — Zavalla 2026"
        return original_title(body, *args, **kwargs)

    def local_caption(body: Any, *args: Any, **kwargs: Any):
        if isinstance(body, str):
            if body.startswith("Predicción operativa de emergencia"):
                body = (
                    "Predicción operativa de emergencia de Lolium para "
                    "Zavalla, Santa Fe, con meteorología híbrida y modelo "
                    "local con lag fijo de 15 días."
                )
            elif body.startswith("Política automática:"):
                body = (
                    "Modelo local de Zavalla: lag fijo de 15 días, "
                    "calibración específica y sin selección multisitio."
                )
        return original_caption(body, *args, **kwargs)

    def local_selectbox(label: Any, options: Any, *args: Any, **kwargs: Any):
        if str(label) == "Seleccionar sitio":
            key = str(kwargs.get("key", "selected_lolium_site"))
            st.session_state[key] = DEFAULT_SITE_SLUG
            return DEFAULT_SITE_SLUG
        return original_selectbox(label, options, *args, **kwargs)

    st.set_page_config = local_page_config
    st.title = local_title
    st.caption = local_caption
    st.selectbox = local_selectbox
    try:
        base.run()
    finally:
        st.set_page_config = original_page_config
        st.title = original_title
        st.caption = original_caption
        st.selectbox = original_selectbox
