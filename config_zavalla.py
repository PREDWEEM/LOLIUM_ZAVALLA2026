from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SiteCalibration:
    slug: str = "zavalla"
    cobertura_predeterminada_pct: int = 30
    wmax_predeterminado_mm: float = 20.0
    exponente_kr_predeterminado: float = 0.0
    latencia_jd: int = 25
    ventana_termica_dias: int = 5
    umbral_termoinhibicion_c: float = 24.0
    umbral_termoinhibicion_con_lag_c: float = 20.0
    ventana_lluvia_dias: int = 3
    umbral_choque_hidrico_mm: float = 45.0
    fin_choque_hidrico_jd: int = 110
    techo_choque_hidrico: float = 0.75
    umbral_primer_pico: float = 0.20
    lag_candidato_dias: int = 15
    decaimiento_activo: bool = False
    decaimiento_tau_dias: float = 1.0
    decaimiento_beta: float = 1.0
    decaimiento_intensidad: float = 0.0
    modelo_referencia_local: str = "con_lag"
    repositorio_referencia: str = "PREDWEEM/LOLIUM_ZAVALLA2026"
    archivo_motor_referencia: str = "predweem_core.py"


ZAVALLA_CALIBRATION = SiteCalibration()
SITE_CALIBRATIONS = {"zavalla": ZAVALLA_CALIBRATION}


def get_site_calibration(slug: str | None = None) -> SiteCalibration:
    normalized = str(slug or "zavalla").strip().lower()
    if normalized != "zavalla":
        raise KeyError(f"Este repositorio solo admite Zavalla: {slug}")
    return ZAVALLA_CALIBRATION


@dataclass(frozen=True)
class ZavallaConfig:
    nombre_sitio: str = "Zavalla, Santa Fe"
    latitud: float = -33.02157
    longitud: float = -60.87930
    timezone: str = "America/Argentina/Cordoba"

    latencia_jd: int = 25
    ventana_termica_dias: int = 5
    umbral_termoinhibicion_c: float = 24.0
    umbral_termoinhibicion_con_lag_c: float = 20.0
    ventana_lluvia_dias: int = 3
    umbral_choque_hidrico_mm: float = 45.0
    fin_choque_hidrico_jd: int = 110
    techo_choque_hidrico: float = 0.75

    cobertura_predeterminada_pct: int = 30
    wmax_predeterminado_mm: float = 20.0
    exponente_kr_predeterminado: float = 0.0
    p50_hidrico: float = 0.30
    pendiente_hidrica: float = 10.0
    corte_hidrico: float = 0.20

    umbral_primer_pico: float = 0.20
    lag_candidato_dias: int = 15
    tolerancia_sin_lag_dias: int = 4
    tolerancia_lag_dias: int = 5
    densidad_confirmacion_pl_m2: float = 0.5

    decaimiento_activo: bool = False
    decaimiento_tau_dias: float = 1.0
    decaimiento_beta: float = 1.0
    decaimiento_intensidad: float = 0.0
    modelo_referencia_local: str = "con_lag"

    t_base_c: float = 2.0
    t_optima_c: float = 20.0
    t_critica_c: float = 30.0
    tt_control_cd: float = 600.0
    tt_limite_cd: float = 800.0


CONFIG = ZavallaConfig()
