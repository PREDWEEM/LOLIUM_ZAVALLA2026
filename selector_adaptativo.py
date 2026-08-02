from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json

import pandas as pd

from config_zavalla import CONFIG, ZavallaConfig


INSPECTION_COLUMNS = [
    "Fecha",
    "Operario",
    "Plantas_m2",
    "Cuadros_positivos",
    "Cuadros_totales",
    "Emergencia_confirmada",
    "Observaciones",
]


@dataclass
class SelectorDecision:
    estado: str
    modelo_activo: str
    lag_operativo_dias: int | None
    proxima_accion: str
    proxima_inspeccion: str | None
    fecha_pico_sin_lag: str | None
    fecha_pico_con_lag: str | None
    negativas_validas: int
    positivas_validas: int
    confianza: str
    motivo: str

    def to_dict(self) -> dict:
        return asdict(self)


def empty_inspections() -> pd.DataFrame:
    return pd.DataFrame(columns=INSPECTION_COLUMNS)


def normalize_inspections(raw: pd.DataFrame | None) -> pd.DataFrame:
    if raw is None or raw.empty:
        return empty_inspections()
    data = raw.copy()
    for column in INSPECTION_COLUMNS:
        if column not in data.columns:
            data[column] = None
    data = data[INSPECTION_COLUMNS]
    data["Fecha"] = pd.to_datetime(data["Fecha"], errors="coerce").dt.normalize()
    for column in ("Plantas_m2", "Cuadros_positivos", "Cuadros_totales"):
        data[column] = pd.to_numeric(data[column], errors="coerce").fillna(0)
    explicit = data["Emergencia_confirmada"].astype(str).str.lower().isin(
        {"true", "1", "si", "sí", "yes"}
    )
    data["Emergencia_confirmada"] = explicit
    return data.dropna(subset=["Fecha"]).sort_values("Fecha").reset_index(drop=True)


def classify_inspections(
    data: pd.DataFrame | None,
    config: ZavallaConfig = CONFIG,
) -> pd.DataFrame:
    out = normalize_inspections(data)
    if out.empty:
        out["Positiva"] = pd.Series(dtype=bool)
        return out
    out["Positiva"] = (
        out["Emergencia_confirmada"]
        | (out["Plantas_m2"] >= config.densidad_confirmacion_pl_m2)
        | (out["Cuadros_positivos"] >= 2)
    )
    return out


def visible_models(decision: SelectorDecision) -> tuple[bool, bool]:
    """Devuelve visibilidad (sin_lag, con_lag) para la gráfica.

    Mientras no hay una selección concluyente se muestran ambas hipótesis. Una
    vez seleccionado un modelo, la alternativa se oculta.
    """
    if decision.modelo_activo == "sin_lag" and decision.estado == "SIN_LAG_CONFIRMADO":
        return True, False
    if decision.modelo_activo == "con_lag" and decision.estado in {
        "CON_LAG_PROVISIONAL",
        "CON_LAG_CONFIRMADO",
    }:
        return False, True
    return True, True


def evaluate_selector(
    peak_no_lag: pd.Timestamp | None,
    peak_lag: pd.Timestamp | None,
    inspections: pd.DataFrame | None,
    *,
    today: pd.Timestamp | None = None,
    lag_candidate_days: int = CONFIG.lag_candidato_dias,
    config: ZavallaConfig = CONFIG,
) -> SelectorDecision:
    """Selecciona entre lag 0 y un lag fijo mediante verificación de campo.

    La secuencia operativa es:

    - una inspección positiva antes de acumular dos negativas confirma sin lag;
    - una primera inspección negativa mantiene ambos modelos en evaluación;
    - si los dos primeros conteos válidos son negativos, el modelo con lag se
      selecciona automáticamente, sin depender de la fecha actual;
    - una observación positiva posterior en la ventana desplazada confirma el lag;
    - para rechazar ambos modelos se exige una verificación negativa adicional
      en la ventana con lag.

    No se estima ningún lag local intermedio.
    """
    now = pd.Timestamp(today or pd.Timestamp.today()).normalize()
    if peak_no_lag is None:
        return SelectorDecision(
            "SIN_PICO_SIMULADO",
            "pendiente",
            None,
            "Continuar actualizando la meteorología; aún no existe un primer pico habilitado.",
            None,
            None,
            None,
            0,
            0,
            "baja",
            "La curva sin lag todavía no supera el umbral del primer pico.",
        )

    peak0 = pd.Timestamp(peak_no_lag).normalize()
    peak_fixed_lag = (
        pd.Timestamp(peak_lag).normalize()
        if peak_lag is not None
        else peak0 + pd.Timedelta(days=int(lag_candidate_days))
    )
    classified = classify_inspections(inspections, config)
    valid = classified[
        classified["Fecha"] >= peak0 - pd.Timedelta(days=2)
    ].copy()
    positives = valid[valid["Positiva"]]
    negatives = valid[~valid["Positiva"]]

    common = {
        "fecha_pico_sin_lag": peak0.date().isoformat(),
        "fecha_pico_con_lag": peak_fixed_lag.date().isoformat(),
        "negativas_validas": len(negatives),
        "positivas_validas": len(positives),
    }

    if now < peak0 - pd.Timedelta(days=2):
        return SelectorDecision(
            "SIN_LAG_EN_EVALUACION",
            "pendiente",
            None,
            "Esperar la ventana del primer pico sin lag y organizar la inspección de campo.",
            peak0.date().isoformat(),
            confianza="baja",
            motivo="El primer pico sin lag todavía es futuro.",
            **common,
        )

    if valid.empty:
        return SelectorDecision(
            "VERIFICACION_1_PENDIENTE",
            "pendiente",
            None,
            "Realizar la primera inspección en el pico sin lag y registrar presencia o ausencia de emergencia.",
            now.date().isoformat(),
            confianza="baja",
            motivo="El pico sin lag fue alcanzado, pero aún no existe una inspección válida.",
            **common,
        )

    initial_rows = valid.iloc[:2]
    first_positive = initial_rows[initial_rows["Positiva"]]

    if not first_positive.empty:
        positive_date = pd.Timestamp(first_positive.iloc[0]["Fecha"]).normalize()
        return SelectorDecision(
            "SIN_LAG_CONFIRMADO",
            "sin_lag",
            0,
            "Mantener únicamente el modelo sin lag y continuar la validación Event-to-Event.",
            None,
            confianza="alta",
            motivo=(
                "La emergencia fue confirmada dentro de los dos conteos iniciales "
                f"({positive_date.date().isoformat()})."
            ),
            **common,
        )

    if len(initial_rows) == 1:
        first_negative_date = pd.Timestamp(initial_rows.iloc[0]["Fecha"]).normalize()
        suggested_date = min(
            first_negative_date + pd.Timedelta(days=3),
            peak_fixed_lag,
        )
        if suggested_date < now:
            suggested_date = now
        return SelectorDecision(
            "VERIFICACION_2_PENDIENTE",
            "pendiente",
            None,
            "Realizar una segunda inspección. Si la emergencia vuelve a ser 0, el modelo con lag se seleccionará automáticamente.",
            suggested_date.date().isoformat(),
            confianza="media",
            motivo=(
                "El primer conteo inicial fue negativo. Falta un segundo conteo "
                "válido para completar la selección automática."
            ),
            **common,
        )

    # Al llegar aquí, los dos primeros conteos válidos fueron negativos.
    second_negative_date = pd.Timestamp(initial_rows.iloc[1]["Fecha"]).normalize()
    later = valid[valid["Fecha"] > second_negative_date].copy()
    later_positives = later[later["Positiva"]]

    if not later_positives.empty:
        first_lag_positive = pd.Timestamp(later_positives.iloc[0]["Fecha"]).normalize()
        distance_to_fixed_lag = abs(int((first_lag_positive - peak_fixed_lag).days))
        if distance_to_fixed_lag <= config.tolerancia_lag_dias:
            return SelectorDecision(
                "CON_LAG_CONFIRMADO",
                "con_lag",
                int(lag_candidate_days),
                "Mantener únicamente el modelo con lag fijo y validar su desempeño Event-to-Event.",
                None,
                confianza="alta",
                motivo=(
                    "Los dos conteos iniciales fueron negativos y la emergencia "
                    f"posterior coincidió con la ventana del lag fijo de {lag_candidate_days} días."
                ),
                **common,
            )
        return SelectorDecision(
            "CON_LAG_PROVISIONAL",
            "con_lag",
            int(lag_candidate_days),
            "Mantener el modelo con lag fijo; la emergencia observada quedó fuera de su ventana y requiere revisión biofísica.",
            None,
            confianza="media",
            motivo=(
                "Los dos conteos iniciales negativos seleccionaron automáticamente "
                "el modelo con lag. La emergencia posterior quedó fuera de la ventana fija."
            ),
            **common,
        )

    lag_window_start = peak_fixed_lag - pd.Timedelta(days=config.tolerancia_lag_dias)
    later_lag_negatives = later[
        (~later["Positiva"]) & (later["Fecha"] >= lag_window_start)
    ]
    lag_window_end = peak_fixed_lag + pd.Timedelta(days=config.tolerancia_lag_dias)

    if not later_lag_negatives.empty and now > lag_window_end:
        return SelectorDecision(
            "NINGUNO_CONFIRMADO",
            "ninguno",
            None,
            "Revisar meteorología, banco de semillas, dormición, Wmax, cobertura y termoinhibición.",
            None,
            confianza="baja",
            motivo=(
                "Los dos conteos iniciales activaron el modelo con lag, pero una "
                "verificación negativa adicional en su ventana no confirmó emergencia."
            ),
            **common,
        )

    return SelectorDecision(
        "CON_LAG_PROVISIONAL",
        "con_lag",
        int(lag_candidate_days),
        "Usar únicamente el modelo con lag fijo y realizar una inspección en su ventana prevista.",
        peak_fixed_lag.date().isoformat(),
        confianza="media",
        motivo=(
            "Los dos primeros conteos válidos registraron emergencia igual a 0; "
            "el modelo con lag fijo fue seleccionado automáticamente."
        ),
        **common,
    )


def load_inspections(path: str | Path) -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.is_file() or file_path.stat().st_size == 0:
        return empty_inspections()
    return normalize_inspections(pd.read_csv(file_path))


def save_inspections(data: pd.DataFrame, path: str | Path) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    normalize_inspections(data).to_csv(file_path, index=False)


def save_decision(decision: SelectorDecision, path: str | Path) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(
        json.dumps(decision.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
