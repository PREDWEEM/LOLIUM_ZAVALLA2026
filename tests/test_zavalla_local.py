import math

import update_meteo as meteo
from sitios_lolium import DEFAULT_SITE_SLUG, SITES, get_site, ordered_sites


def test_repositorio_contiene_un_solo_sitio():
    assert DEFAULT_SITE_SLUG == "zavalla"
    assert list(SITES) == ["zavalla"]
    assert ordered_sites() == [get_site()]
    assert get_site().repositorio == "PREDWEEM/LOLIUM_ZAVALLA2026"
    assert get_site().lag_operativo_dias == 15


def test_precipitacion_negativa_se_descarta():
    assert math.isnan(meteo._safe_precipitation_mm(-1.0, "mm"))
