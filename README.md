# PREDWEEM LOLIUM — Zavalla 2026

Repositorio independiente del modelo operativo PREDWEEM para **Zavalla, Santa Fe**.

- Coordenadas: `-33.02157, -60.87930`.
- Modelo operativo: lag fijo de 15 días.
- Cobertura de rastrojo ajustable.
- Umbral operativo: `EMERREL >= 0.0001`.
- Ventana fenológica: 600–800 °Cd.
- Meteorología: SMN Rosario Aero → NOAA → ECMWF Archive → ECMWF Forecast.
- Actualización automática diaria: 07:30 de Argentina.

```bash
python -m pip install -r requirements.txt
python update_meteo.py
streamlit run app.py
```

La versión multisitio previa se conserva en la rama
`multisitio-legacy-20260802` de `PREDWEEM/LOLIUMZAVALLA`.

**PREDWEEM by Guillermo R. Chantre**
