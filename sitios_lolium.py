from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class LoliumSite:
    slug: str
    nombre: str
    provincia: str
    latitud: float
    longitud: float
    repositorio: str
    timezone: str = "America/Argentina/Cordoba"
    usa_siga_historico: bool = False
    rama_meteo: str = "main"
    archivo_meteo: str = "meteo_daily.csv"
    modelo_operativo: str = "con_lag"
    lag_operativo_dias: int = 15

    @property
    def etiqueta(self) -> str:
        return f"{self.nombre} ({self.provincia})"

    @property
    def repository_url(self) -> str:
        return f"https://github.com/{self.repositorio}"

    @property
    def raw_meteo_url(self) -> str:
        return (
            f"https://raw.githubusercontent.com/{self.repositorio}/"
            f"{self.rama_meteo}/{self.archivo_meteo}"
        )

    @property
    def modelo_operativo_etiqueta(self) -> str:
        return f"Con lag fijo de {self.lag_operativo_dias} días"

    def meteo_path(self, base: str | Path = ".") -> Path:
        return Path(base) / "data" / "meteo_sitios" / "zavalla.csv"

    def inspections_path(self, base: str | Path = ".") -> Path:
        return Path(base) / "data" / "inspecciones" / "zavalla.csv"

    def selector_state_path(self, base: str | Path = ".") -> Path:
        return Path(base) / "data" / "selector" / "zavalla.json"

    def to_dict(self) -> dict:
        return asdict(self)


ZAVALLA = LoliumSite(
    slug="zavalla",
    nombre="Zavalla",
    provincia="Santa Fe",
    latitud=-33.02157,
    longitud=-60.87930,
    repositorio="PREDWEEM/LOLIUM_ZAVALLA2026",
    timezone="America/Argentina/Cordoba",
    modelo_operativo="con_lag",
    lag_operativo_dias=15,
)

SITES: dict[str, LoliumSite] = {"zavalla": ZAVALLA}
DEFAULT_SITE_SLUG = "zavalla"


def get_site(slug: str = DEFAULT_SITE_SLUG) -> LoliumSite:
    normalized = str(slug or DEFAULT_SITE_SLUG).strip().lower()
    if normalized != DEFAULT_SITE_SLUG:
        raise KeyError(f"Este repositorio solo admite Zavalla: {slug}")
    return ZAVALLA


def ordered_sites() -> list[LoliumSite]:
    return [ZAVALLA]
