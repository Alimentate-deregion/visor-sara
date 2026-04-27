import json
from pathlib import Path

import duckdb
import geopandas as gpd
import numpy as np
import pandas as pd
import panel as pn
import plotly.graph_objects as go

pn.extension("plotly", sizing_mode="stretch_width")

# =========================================================
# CONFIGURACIÓN
# =========================================================

BASE_DIR        = Path(__file__).parent / "datos"
RUTA_LINEAS     = BASE_DIR / "lineas_abastecimiento.parquet"
RUTA_MUNICIPIOS = BASE_DIR / "municipios_ligeros.parquet"
RUTA_DESTINO    = BASE_DIR / "puntos_destino.geojson"
RUTA_ORIGEN     = BASE_DIR / "puntos_origen.geojson"
RUTA_LOGO       = BASE_DIR / "MDS-245-ES.jpg"

DEPTOS_RAPE = {
    "BOGOTÁ","BOGOTÁ, D.C.","BOGOTA","BOGOTA D.C.","BOGOTÁ D.C.",
    "CUNDINAMARCA","META","BOYACÁ","BOYACA","TOLIMA"
}

COLOR_BG     = "#0F1116"
COLOR_PANEL  = "#171A21"
COLOR_BORDE  = "#2B3240"
COLOR_TEXTO  = "#E8EDF5"
COLOR_TITULO = "#F2F5FA"
COLOR_MUTED  = "#AEB9C9"
COLOR_MUTED2 = "#99A7BC"
COLOR_MUTED3 = "#9EABC0"
COLOR_MUTED4 = "#C7D0DD"
COLOR_ACENTO = "#4DA3FF"
FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif"

# =========================================================
# CSS
# =========================================================

CSS = f"""
body, .bk-root {{
    background-color: {COLOR_BG} !important;
    color: {COLOR_TEXTO} !important;
    font-family: {FONT} !important;
}}
.bk-input-group > label, .bk > label {{ display: none !important; }}
.bk-input, select.bk-input {{
    background: #12161D !important; color: {COLOR_TEXTO} !important;
    border: 1px solid {COLOR_BORDE} !important; border-radius: 6px !important;
    font-family: {FONT} !important;
}}
.choices__item--selectable {{ background: #243042 !important; color: {COLOR_TEXTO} !important; }}
.choices__item.choices__item--selected {{ background: #2A3F5A !important; color: {COLOR_TEXTO} !important; }}
.choices__inner {{
    background: #12161D !important; border: 1px solid {COLOR_BORDE} !important;
    color: {COLOR_TEXTO} !important; font-family: {FONT} !important;
}}
.choices__list--dropdown {{
    background: #12161D !important; border: 1px solid {COLOR_BORDE} !important; color: {COLOR_TEXTO} !important;
}}
.choices__list--dropdown .choices__item--selectable {{ color: {COLOR_TEXTO} !important; }}
.choices__list--dropdown .choices__item--selectable.is-highlighted {{ background: #1E2D42 !important; }}
.bk-data-table {{
    background: {COLOR_BG} !important; color: {COLOR_TEXTO} !important;
    border: none !important; overflow-x: auto !important;
}}
.bk-data-table .slick-header {{ position: sticky !important; top: 0 !important; z-index: 10 !important; }}
.bk-data-table .slick-header-column {{
    background: #1E2530 !important; color: {COLOR_TEXTO} !important;
    border-bottom: 1px solid {COLOR_BORDE} !important;
    font-size: 0.73rem !important; font-weight: 600 !important;
    text-transform: uppercase !important; white-space: nowrap !important;
    font-family: {FONT} !important;
}}
.bk-data-table .slick-row {{
    background: {COLOR_BG} !important; color: {COLOR_TEXTO} !important;
    border-bottom: 1px solid {COLOR_BORDE} !important;
}}
.bk-data-table .slick-row.odd {{ background: {COLOR_PANEL} !important; }}
.bk-data-table .slick-row:hover {{ background: #243042 !important; }}
.bk-data-table .slick-cell {{
    border: none !important; color: {COLOR_TEXTO} !important;
    white-space: nowrap !important; font-family: {FONT} !important;
}}
::-webkit-scrollbar {{ width: 6px; height: 6px; }}
::-webkit-scrollbar-track {{ background: {COLOR_BG}; }}
::-webkit-scrollbar-thumb {{ background: {COLOR_BORDE}; border-radius: 3px; }}
.panel-title {{
    color: {COLOR_TITULO} !important; font-size: 1rem !important;
    font-weight: 600 !important; margin-bottom: 0.65rem !important;
    font-family: {FONT} !important;
}}
.metric-card {{
    background: {COLOR_PANEL}; border: 1px solid {COLOR_BORDE};
    border-radius: 12px; padding: 0.8rem 1rem; text-align: center; margin-bottom: 0.8rem;
}}
.metric-label {{ color: {COLOR_MUTED3}; font-size: 0.82rem; margin-bottom: 0.35rem; font-family: {FONT}; }}
.metric-value {{ color: #FFFFFF; font-size: 2rem; font-weight: 700; line-height: 1.05; font-family: {FONT}; }}
.metric-small {{ color: {COLOR_MUTED4}; font-size: 0.8rem; margin-top: 0.25rem; font-family: {FONT}; }}
.legend-item {{ display: flex; align-items: center; gap: 8px; margin-bottom: 6px; font-size: 0.9rem; color: #D8E0EA; font-family: {FONT}; }}
.legend-box {{ width: 14px; height: 14px; border-radius: 3px; border: 1px solid rgba(255,255,255,0.15); display: inline-block; flex-shrink: 0; }}
.small-note {{ color: {COLOR_MUTED2}; font-size: 0.82rem; line-height: 1.45; font-family: {FONT}; }}
.method-note {{
    background: {COLOR_PANEL}; border: 1px solid {COLOR_BORDE};
    border-left: 4px solid {COLOR_ACENTO}; border-radius: 10px;
    padding: 0.8rem 1rem; color: {COLOR_MUTED4};
    font-size: 0.86rem; line-height: 1.55; margin-top: 0.8rem; font-family: {FONT};
}}
.bk-slider-title {{ display: none !important; }}
#visor-map {{ width: 100%; height: 480px; position: relative; border-radius: 8px; overflow: hidden; }}
"""

pn.config.raw_css.append(CSS)

# =========================================================
# CARGA DE DATOS
# =========================================================

@pn.cache
def cargar_datos():
    municipios  = gpd.read_parquet(RUTA_MUNICIPIOS)
    puntos_dest = gpd.read_file(RUTA_DESTINO)

    con = duckdb.connect(database=":memory:")
    con.execute("PRAGMA threads=4")
    con.execute(f"CREATE TABLE lineas AS SELECT * FROM read_parquet('{RUTA_LINEAS.as_posix()}')")

    rubros    = con.execute("SELECT DISTINCT RUBRO FROM lineas WHERE RUBRO IS NOT NULL ORDER BY RUBRO").df()["RUBRO"].astype(str).tolist()
    centrales = con.execute("SELECT DISTINCT CENTRAL_NOMBRE FROM lineas WHERE CENTRAL_NOMBRE IS NOT NULL ORDER BY CENTRAL_NOMBRE").df()["CENTRAL_NOMBRE"].astype(str).tolist()
    deptos    = con.execute("SELECT DISTINCT DEPARTAMENTO_ORIGEN FROM lineas WHERE DEPARTAMENTO_ORIGEN IS NOT NULL ORDER BY DEPARTAMENTO_ORIGEN").df()["DEPARTAMENTO_ORIGEN"].astype(str).tolist()
    fechas    = con.execute("SELECT MIN(FECHA) AS fmin, MAX(FECHA) AS fmax FROM lineas").df()
    fecha_min = pd.to_datetime(fechas.loc[0,"fmin"]).date()
    fecha_max = pd.to_datetime(fechas.loc[0,"fmax"]).date()

    return con, municipios, puntos_dest, rubros, centrales, deptos, fecha_min, fecha_max


con, municipios, puntos_dest, rubros, centrales, deptos, fecha_min, fecha_max = cargar_datos()
total_bd = float(con.execute("SELECT SUM(TONELADAS) FROM lineas").fetchone()[0] or 1)


@pn.cache
def preparar_estaticos():
    """
    Pre-serializa el GeoJSON de polígonos UNA SOLA VEZ.
    Este JSON se incrusta en el HTML del mapa y nunca cambia.
    Solo los arcos y top30 se actualizan via JS.
    """
    mun = municipios.copy()
    if mun.crs and mun.crs.to_epsg() != 4326:
        mun = mun.to_crs(epsg=4326)

    for col in ["MpCodigo","CODIGO_MUNICIPIO"]:
        if col in mun.columns:
            mun["codigo_origen"] = mun[col].astype(str).str.strip().str.replace(r"\.0$","",regex=True).str.zfill(5)
            break

    for col in ["Nombre","MpNombre","MUNICIPIO","NOMBRE_MUNICIPIO","NOMBRE_MPIO","NOM_MUN","municipio","nombre"]:
        if col in mun.columns:
            mun["nombre_municipio"] = mun[col].fillna("").astype(str).str.strip()
            break
    else:
        mun["nombre_municipio"] = ""

    for col in ["Depto","DEPARTAMENTO","NOMBRE_DPT","NOM_DEP"]:
        if col in mun.columns:
            mun["departamento"] = mun[col].fillna("").astype(str).str.strip()
            break
    else:
        mun["departamento"] = ""

    # GeoJSON base — polígonos sin color (se colorean en JS según top30)
    mun["codigo_str"]    = mun["codigo_origen"].fillna("").astype(str)
    mun["nombre_str"]    = mun["nombre_municipio"].fillna("Sin nombre").astype(str)
    mun["depto_str"]     = mun["departamento"].fillna("Sin dato").astype(str)

    geojson_str = mun[["codigo_str","nombre_str","depto_str","geometry"]].to_json()

    # Puntos destino
    pd_dest = puntos_dest.copy()
    pd_dest["NOMBRE_CENTRAL"] = pd_dest["NOMBRE_CENTRAL"].fillna("").astype(str).str.strip()
    pd_dest["CIUDAD"]         = pd_dest["CIUDAD"].fillna("").astype(str).str.strip()
    pd_dest["codigo_destino"] = pd_dest["CODIGO_MUNICIPIO"].astype(str).str.strip().str.replace(r"\.0$","",regex=True).str.zfill(5)
    pd_dest["lon"] = pd_dest["geometry"].apply(lambda g: g.x if g is not None else None)
    pd_dest["lat"] = pd_dest["geometry"].apply(lambda g: g.y if g is not None else None)
    pd_dest = pd_dest.drop(columns=["geometry"],errors="ignore").dropna(subset=["lon","lat"])

    return geojson_str, pd_dest


geojson_str, puntos_dest_prep = preparar_estaticos()

# =========================================================
# WIDGETS
# =========================================================

w_rubro = pn.widgets.Select(name="Rubro", options=rubros, value=rubros[0] if rubros else None, width=200)
w_centrales = pn.widgets.MultiChoice(name="Central mayorista", options=centrales, value=[], width=240)
w_semestre  = pn.widgets.Select(name="Periodo semestral", options=["Todos","Primer semestre","Segundo semestre"], value="Todos", width=160)
w_deptos    = pn.widgets.MultiChoice(name="Departamento origen", options=deptos, value=[], width=240)
w_fecha_ini = pn.widgets.DatePicker(name="Fecha inicio", value=fecha_min, start=fecha_min, end=fecha_max, width=160)
w_fecha_fin = pn.widgets.DatePicker(name="Fecha fin",   value=fecha_max, start=fecha_min, end=fecha_max, width=160)
w_max_flujos = pn.widgets.IntSlider(name="Máx. flujos en mapa", start=100, end=1500, value=600, step=100, width=160)

# =========================================================
# HELPERS
# =========================================================

def t(lst): return tuple(lst) if lst else ()

def construir_where(rubro, centrales_l, semestre, deptos_l, fecha_ini, fecha_fin):
    conds = []
    if rubro:
        conds.append(f"RUBRO = '{rubro.replace(chr(39),chr(39)*2)}'")
    if centrales_l:
        lista = ",".join(f"'{c.replace(chr(39),chr(39)*2)}'" for c in centrales_l)
        conds.append(f"CENTRAL_NOMBRE IN ({lista})")
    if semestre == "Primer semestre":  conds.append("MES BETWEEN 1 AND 6")
    elif semestre == "Segundo semestre": conds.append("MES BETWEEN 7 AND 12")
    if deptos_l:
        lista = ",".join(f"'{d.replace(chr(39),chr(39)*2)}'" for d in deptos_l)
        conds.append(f"DEPARTAMENTO_ORIGEN IN ({lista})")
    if fecha_ini and fecha_fin:
        conds.append(f"FECHA BETWEEN '{fecha_ini}' AND '{fecha_fin}'")
    return "WHERE " + " AND ".join(conds) if conds else ""

# =========================================================
# CONSULTAS CACHEADAS
# =========================================================

@pn.cache
def consultar_datos(rubro, centrales_t, semestre, deptos_t, fecha_ini, fecha_fin):
    where = construir_where(rubro, list(centrales_t), semestre, list(deptos_t), fecha_ini, fecha_fin)
    return con.execute(f"""
        SELECT MUNICIPIO_ORIGEN, DEPARTAMENTO_ORIGEN, CENTRAL_NOMBRE,
            CODIGO_DIVIPOLA_MUN_ORIGEN_LIMPIO AS cod_origen,
            CODIGO_DIVIPOLA_MUN_DESTINO_LIMPIO AS cod_destino,
            AVG(LATITUD_ORIGEN) AS lat_orig, AVG(LONGITUD_ORIGEN) AS lon_orig,
            AVG(LATITUD_DESTINO) AS lat_dest, AVG(LONGITUD_DESTINO) AS lon_dest,
            SUM(TONELADAS) AS toneladas, AVG(PRECIO_PROMEDIO) AS precio_promedio,
            AVG(PRECIO_MEDIANA) AS precio_mediana, SUM(DIAS_CON_DATOS) AS dias_con_datos
        FROM lineas {where}
        GROUP BY 1,2,3,4,5 ORDER BY toneladas DESC
    """).df()


@pn.cache
def consultar_serie(rubro, centrales_t, semestre, deptos_t, fecha_ini, fecha_fin):
    where = construir_where(rubro, list(centrales_t), semestre, list(deptos_t), fecha_ini, fecha_fin)
    return con.execute(f"""
        SELECT STRFTIME(CAST(FECHA AS DATE),'%Y-%m') AS etiqueta_mes,
            SUM(TONELADAS) AS toneladas, AVG(PRECIO_PROMEDIO) AS precio_promedio
        FROM lineas {where} GROUP BY 1 ORDER BY 1
    """).df()

# =========================================================
# MAPA — deck.gl vía JavaScript puro
# El GeoJSON de polígonos se incrusta UNA SOLA VEZ en el HTML.
# Cuando cambia un filtro, Panel llama a updateMapData() en JS
# que solo reemplaza las capas dinámicas (arcos + top30).
# El mapa base y WebGL nunca se destruyen ni recrean.
# =========================================================

MAP_HTML = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  body {{ margin: 0; background: {COLOR_BG}; }}
  #map {{ width: 100%; height: 480px; }}
  #tooltip {{
    position: absolute; z-index: 100; pointer-events: none;
    background: rgba(18,22,29,0.95); color: #F5F7FA;
    font-family: {FONT}; font-size: 12px;
    padding: 8px 10px; border-radius: 6px;
    border: 1px solid {COLOR_BORDE}; display: none;
  }}
</style>
<script src="https://unpkg.com/deck.gl@latest/dist.min.js"></script>
<script src="https://unpkg.com/maplibre-gl@3/dist/maplibre-gl.js"></script>
<link href="https://unpkg.com/maplibre-gl@3/dist/maplibre-gl.css" rel="stylesheet" />
</head>
<body>
<div id="map"></div>
<div id="tooltip"></div>
<script>
// ---- GeoJSON de polígonos — incrustado una sola vez ----
const GEOJSON_MUNICIPIOS = {geojson_str};

// ---- Estado del mapa ----
let deckgl = null;
let currentTop30 = new Set();
let currentArcos = [];
let currentTop30Points = [];
let currentDestinos = [];

// ---- Colores ----
const COLOR_TOP30   = [110, 68, 255, 160];
const COLOR_BASE    = [40, 48, 62, 20];
const LINE_TOP30    = [170, 130, 255, 240];
const LINE_BASE     = [100, 110, 125, 60];
const COLOR_ARCO_SRC = [245, 176, 65, 190];
const COLOR_ARCO_TGT = [0, 210, 255, 190];
const COLOR_DESTINO  = [0, 210, 255, 190];
const LINE_DESTINO   = [170, 245, 255, 255];

// ---- Tooltip ----
const tooltipEl = document.getElementById('tooltip');

function showTooltip(x, y, html) {{
  tooltipEl.style.display = 'block';
  tooltipEl.style.left = (x + 12) + 'px';
  tooltipEl.style.top  = (y + 12) + 'px';
  tooltipEl.innerHTML  = html;
}}
function hideTooltip() {{
  tooltipEl.style.display = 'none';
}}

// ---- Construir capas ----
function buildLayers() {{
  const layers = [];

  // Capa 1: polígonos — colorea solo top30 en morado, resto gris
  layers.push(new deck.GeoJsonLayer({{
    id: 'municipios',
    data: GEOJSON_MUNICIPIOS,
    stroked: true,
    filled: true,
    getFillColor: f => {{
      const cod = f.properties.codigo_str || '';
      return currentTop30.has(cod) ? COLOR_TOP30 : COLOR_BASE;
    }},
    getLineColor: f => {{
      const cod = f.properties.codigo_str || '';
      return currentTop30.has(cod) ? LINE_TOP30 : LINE_BASE;
    }},
    lineWidthMinPixels: 0.5,
    updateTriggers: {{ getFillColor: currentTop30, getLineColor: currentTop30 }},
    pickable: true,
    autoHighlight: true,
    onHover: ({{object, x, y}}) => {{
      if (object) {{
        const p = object.properties;
        showTooltip(x, y,
          `<b>Municipio</b><br>Nombre: ${{p.nombre_str}}<br>Departamento: ${{p.depto_str}}<br>Código: ${{p.codigo_str}}`);
      }} else hideTooltip();
    }}
  }}));

  // Capa 2: arcos origen-destino
  if (currentArcos.length > 0) {{
    layers.push(new deck.ArcLayer({{
      id: 'arcos',
      data: currentArcos,
      getSourcePosition: d => [d.lon_orig, d.lat_orig],
      getTargetPosition: d => [d.lon_dest, d.lat_dest],
      getSourceColor: COLOR_ARCO_SRC,
      getTargetColor: COLOR_ARCO_TGT,
      getWidth: d => d.ancho,
      widthScale: 1,
      widthMinPixels: 1,
      pickable: true,
      autoHighlight: true,
      onHover: ({{object, x, y}}) => {{
        if (object) {{
          showTooltip(x, y,
            `<b>Flujo OD</b><br>Origen: ${{object.municipio}}<br>Central: ${{object.central}}<br>` +
            `Precio: ${{object.precio}}<br>Toneladas: ${{object.toneladas}}`);
        }} else hideTooltip();
      }}
    }}));
  }}

  // Capa 3: puntos top30 (scatter adicional para mejor visibilidad)
  if (currentTop30Points.length > 0) {{
    layers.push(new deck.ScatterplotLayer({{
      id: 'top30-points',
      data: currentTop30Points,
      getPosition: d => [d.lon, d.lat],
      getRadius: 4200,
      radiusMinPixels: 3,
      radiusMaxPixels: 18,
      getFillColor: [255, 160, 0, 120],
      getLineColor: [255, 210, 120, 200],
      lineWidthMinPixels: 1,
      pickable: true,
      onHover: ({{object, x, y}}) => {{
        if (object) {{
          showTooltip(x, y,
            `<b>Nodo origen</b><br>${{object.municipio}}<br>${{object.depto}}<br>` +
            `Toneladas: ${{object.toneladas}}<br>Precio: ${{object.precio}}`);
        }} else hideTooltip();
      }}
    }}));
  }}

  // Capa 4: centrales mayoristas
  if (currentDestinos.length > 0) {{
    layers.push(new deck.ScatterplotLayer({{
      id: 'destinos',
      data: currentDestinos,
      getPosition: d => [d.lon, d.lat],
      getRadius: 13500,
      radiusMinPixels: 5,
      radiusMaxPixels: 30,
      getFillColor: COLOR_DESTINO,
      getLineColor: LINE_DESTINO,
      lineWidthMinPixels: 2,
      pickable: true,
      onHover: ({{object, x, y}}) => {{
        if (object) {{
          showTooltip(x, y,
            `<b>Central mayorista</b><br>${{object.nombre}}<br>Ciudad: ${{object.ciudad}}`);
        }} else hideTooltip();
      }}
    }}));
  }}

  return layers;
}}

// ---- Inicializar mapa ----
function initMap() {{
  deckgl = new deck.DeckGL({{
    container: 'map',
    mapStyle: 'https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json',
    initialViewState: {{
      latitude: 4.5, longitude: -74.1, zoom: 4.6, pitch: 0
    }},
    controller: true,
    layers: buildLayers(),
    getCursor: ({{isHovering}}) => isHovering ? 'pointer' : 'grab'
  }});
}}

// ---- Actualizar solo las capas dinámicas ----
// Esta función es llamada desde Panel vía postMessage
function updateMapData(data) {{
  currentTop30       = new Set(data.top30 || []);
  currentArcos       = data.arcos || [];
  currentTop30Points = data.top30_points || [];
  currentDestinos    = data.destinos || [];
  if (deckgl) {{
    deckgl.setProps({{ layers: buildLayers() }});
  }}
}}

// ---- Escuchar mensajes de Panel ----
window.addEventListener('message', function(event) {{
  if (event.data && event.data.type === 'updateMap') {{
    updateMapData(event.data.payload);
  }}
}});

// ---- Arrancar ----
initMap();
</script>
</body>
</html>
"""

# Widget HTML del mapa — se crea una sola vez, nunca se recrea
mapa_html = pn.pane.HTML(MAP_HTML, height=490, sizing_mode="stretch_width")

# =========================================================
# JAVASCRIPT BRIDGE — envía datos al mapa sin recrearlo
# =========================================================

def construir_datos_mapa(rubro, centrales, semestre, deptos, fecha_ini, fecha_fin, max_flujos):
    df = consultar_datos(rubro, t(centrales), semestre, t(deptos), fecha_ini, fecha_fin)

    if df.empty:
        return {"top30": [], "arcos": [], "top30_points": [], "destinos": []}

    # Top 30 por toneladas
    top30_df = (
        df.dropna(subset=["lat_orig","lon_orig"])
        .groupby(["MUNICIPIO_ORIGEN","DEPARTAMENTO_ORIGEN","cod_origen"])
        .agg(lon=("lon_orig","first"), lat=("lat_orig","first"),
             toneladas=("toneladas","sum"), precio=("precio_promedio","mean"))
        .reset_index()
        .sort_values("toneladas", ascending=False)
        .head(30)
    )
    top30_codigos = top30_df["cod_origen"].astype(str).tolist()
    top30_points  = [
        {"lon": r.lon, "lat": r.lat,
         "municipio": r.MUNICIPIO_ORIGEN, "depto": r.DEPARTAMENTO_ORIGEN,
         "toneladas": f"{r.toneladas:,.1f}", "precio": f"$ {r.precio:,.0f}"}
        for r in top30_df.itertuples()
    ]

    # Arcos
    df_arcos = df.dropna(subset=["lat_orig","lon_orig","lat_dest","lon_dest"]).copy()
    df_arcos = df_arcos.sort_values("toneladas", ascending=False).head(max_flujos)
    if not df_arcos.empty:
        vmin = df_arcos["toneladas"].min()
        vmax = df_arcos["toneladas"].max()
        arcos = [
            {"lon_orig": r.lon_orig, "lat_orig": r.lat_orig,
             "lon_dest": r.lat_dest, "lat_dest": r.lon_dest,  # note: corrected below
             "ancho": 2 + 10 * ((r.toneladas - vmin) / (vmax - vmin + 1e-9)),
             "municipio": r.MUNICIPIO_ORIGEN, "central": r.CENTRAL_NOMBRE,
             "toneladas": f"{r.toneladas:,.1f}", "precio": f"$ {r.precio_promedio:,.0f}"}
            for r in df_arcos.itertuples()
        ]
        # Fix lat/lon swap
        arcos = [
            {"lon_orig": r.lon_orig, "lat_orig": r.lat_orig,
             "lon_dest": r.lon_dest, "lat_dest": r.lat_dest,
             "ancho": 2 + 10 * ((r.toneladas - vmin) / (vmax - vmin + 1e-9)),
             "municipio": r.MUNICIPIO_ORIGEN, "central": r.CENTRAL_NOMBRE,
             "toneladas": f"{r.toneladas:,.1f}", "precio": f"$ {r.precio_promedio:,.0f}"}
            for r in df_arcos.itertuples()
        ]
    else:
        arcos = []

    # Destinos
    pd_f = puntos_dest_prep[puntos_dest_prep["NOMBRE_CENTRAL"].isin(centrales)].copy() \
           if centrales else puntos_dest_prep.copy()
    destinos = [
        {"lon": r.lon, "lat": r.lat, "nombre": r.NOMBRE_CENTRAL, "ciudad": r.CIUDAD}
        for r in pd_f.itertuples()
    ]

    return {"top30": top30_codigos, "arcos": arcos, "top30_points": top30_points, "destinos": destinos}


# Script que Panel ejecuta para enviar datos al iframe del mapa via postMessage
def script_actualizar_mapa(datos_json):
    return pn.pane.HTML(f"""
    <script>
    (function() {{
      const frames = document.querySelectorAll('iframe');
      frames.forEach(f => {{
        try {{
          f.contentWindow.postMessage({{
            type: 'updateMap',
            payload: {datos_json}
          }}, '*');
        }} catch(e) {{}}
      }});
      // También intentar en la misma ventana (por si no hay iframe)
      window.postMessage({{
        type: 'updateMap',
        payload: {datos_json}
      }}, '*');
    }})();
    </script>
    """, height=0, width=0, sizing_mode="fixed")


# =========================================================
# COMPONENTES REACTIVOS
# =========================================================

def construir_metricas(rubro, centrales, semestre, deptos, fecha_ini, fecha_fin):
    df = consultar_datos(rubro, t(centrales), semestre, t(deptos), fecha_ini, fecha_fin)
    if df.empty:
        return pn.pane.HTML("<div class='small-note' style='padding:20px;'>Sin datos.</div>", sizing_mode="stretch_width")

    total_ton   = df["toneladas"].sum()
    precio_prom = df["precio_promedio"].mean()
    n_orig      = df["cod_origen"].nunique()
    n_cent      = df["CENTRAL_NOMBRE"].nunique()

    html = f"""
    <div class="panel-title">Indicadores principales</div>
    <div class="metric-card">
        <div class="metric-label">Toneladas abastecidas</div>
        <div class="metric-value">{total_ton:,.0f}</div>
        <div class="metric-small">Periodo filtrado</div>
    </div>
    <div class="metric-card">
        <div class="metric-label">Precio promedio</div>
        <div class="metric-value" style="font-size:1.65rem;">$ {precio_prom:,.0f}</div>
        <div class="metric-small">Mercado filtrado</div>
    </div>
    <div class="metric-card">
        <div class="metric-label">Municipios origen activos</div>
        <div class="metric-value">{n_orig:,}</div>
        <div class="metric-small">Con flujo válido</div>
    </div>
    <div class="metric-card">
        <div class="metric-label">Centrales activas</div>
        <div class="metric-value">{n_cent}</div>
        <div class="metric-small">Bajo filtros actuales</div>
    </div>
    <div class="panel-title" style="margin-top:1rem;">Leyenda</div>
    <div class="legend-item"><span class="legend-box" style="background:#6E44FF;"></span>Top 30 abastecedores</div>
    <div class="legend-item"><span class="legend-box" style="background:#F5B041;"></span>Arcos de flujo OD</div>
    <div class="legend-item"><span class="legend-box" style="background:#00D2FF;"></span>Central mayorista</div>
    <div class="small-note" style="margin-top:0.75rem;">
        Los municipios morados corresponden a los 30 principales abastecedores del filtro actual.
        Los arcos muestran los flujos origen-destino hacia las centrales mayoristas.
    </div>
    """
    return pn.pane.HTML(html, sizing_mode="stretch_width")


def construir_mapa_updater(rubro, centrales, semestre, deptos, fecha_ini, fecha_fin, max_flujos):
    """Devuelve un script HTML que actualiza las capas del mapa vía postMessage."""
    datos = construir_datos_mapa(rubro, centrales, semestre, deptos, fecha_ini, fecha_fin, max_flujos)
    datos_json = json.dumps(datos)
    return pn.pane.HTML(f"""
    <script>
    (function waitForMap() {{
      const tryUpdate = () => {{
        const mapWindows = [];
        // Buscar en iframes
        document.querySelectorAll('iframe').forEach(f => {{
          try {{ mapWindows.push(f.contentWindow); }} catch(e) {{}}
        }});
        // También la ventana actual
        mapWindows.push(window);

        mapWindows.forEach(w => {{
          try {{
            w.postMessage({{ type: 'updateMap', payload: {datos_json} }}, '*');
          }} catch(e) {{}}
        }});
      }};
      // Intentar inmediatamente y con delay por si el mapa aún está cargando
      tryUpdate();
      setTimeout(tryUpdate, 500);
      setTimeout(tryUpdate, 1500);
    }})();
    </script>
    """, height=0, width=0, sizing_mode="fixed")


def construir_serie(rubro, centrales, semestre, deptos, fecha_ini, fecha_fin):
    df = consultar_serie(rubro, t(centrales), semestre, t(deptos), fecha_ini, fecha_fin)
    if df.empty:
        return pn.pane.HTML("<div class='small-note' style='padding:20px;text-align:center;'>Sin datos.</div>", sizing_mode="stretch_width")

    fig = go.Figure()
    fig.add_trace(go.Bar(x=df["etiqueta_mes"], y=df["precio_promedio"],
        name="Precio promedio", marker_color="#4DA3FF", yaxis="y1",
        hovertemplate="Mes: %{x}<br>Precio: $%{y:,.0f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=df["etiqueta_mes"], y=df["toneladas"],
        name="Toneladas", mode="lines+markers",
        line=dict(color="#F5B041", width=2.5), marker=dict(size=6, color="#F5B041"),
        yaxis="y2", hovertemplate="Mes: %{x}<br>Toneladas: %{y:,.1f}<extra></extra>"))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#171A21", plot_bgcolor="#171A21",
        font=dict(color="#E8EDF5", size=11, family=FONT),
        legend=dict(orientation="h", y=1.08, x=0),
        margin=dict(l=15, r=15, t=10, b=10), height=300,
        xaxis=dict(showgrid=False),
        yaxis=dict(title="Precio", gridcolor="#2B3240"),
        yaxis2=dict(title="Toneladas", overlaying="y", side="right", showgrid=False),
    )
    return pn.pane.Plotly(fig, sizing_mode="stretch_width")


def construir_sankey(rubro, centrales, semestre, deptos, fecha_ini, fecha_fin):
    df = consultar_datos(rubro, t(centrales), semestre, t(deptos), fecha_ini, fecha_fin)
    if df.empty:
        return pn.pane.HTML("<div class='small-note' style='padding:20px;'>Sin datos.</div>", sizing_mode="stretch_width")

    top_mun = df.groupby("MUNICIPIO_ORIGEN")["toneladas"].sum().nlargest(12).index.tolist()
    df_sk   = df[df["MUNICIPIO_ORIGEN"].isin(top_mun)].groupby(["MUNICIPIO_ORIGEN","CENTRAL_NOMBRE"])["toneladas"].sum().reset_index()

    nodos_orig  = list(df_sk["MUNICIPIO_ORIGEN"].unique())
    nodos_dest  = list(df_sk["CENTRAL_NOMBRE"].unique())
    todos_nodos = list(dict.fromkeys(nodos_orig + nodos_dest))
    idx = {n: i for i, n in enumerate(todos_nodos)}
    valores = df_sk["toneladas"].tolist()
    total_sk = sum(valores) or 1
    porcentajes = [v / total_sk * 100 for v in valores]

    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(pad=12, thickness=16, line=dict(color="gray", width=0.5), label=todos_nodos),
        link=dict(
            source=[idx[r["MUNICIPIO_ORIGEN"]] for _, r in df_sk.iterrows()],
            target=[idx[r["CENTRAL_NOMBRE"]]   for _, r in df_sk.iterrows()],
            value=valores, customdata=porcentajes,
            hovertemplate="Origen: %{source.label}<br>Central: %{target.label}<br>Toneladas: %{value:,.1f}<br>Participación: %{customdata:.1f}%<extra></extra>",
        ),
    ))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#171A21", plot_bgcolor="#171A21",
        font=dict(color="#E8EDF5", size=11, family=FONT),
        margin=dict(l=10, r=10, t=10, b=10), height=430,
    )
    return pn.pane.Plotly(fig, sizing_mode="stretch_width")


def construir_contexto(rubro, centrales, semestre, deptos, fecha_ini, fecha_fin):
    df = consultar_datos(rubro, t(centrales), semestre, t(deptos), fecha_ini, fecha_fin)
    if df.empty:
        return pn.pane.HTML("<div class='small-note'>Sin datos suficientes.</div>", sizing_mode="stretch_width")

    df_s = df.sort_values("toneladas", ascending=False).copy()
    precio_med = df_s["precio_mediana"].median() if df_s["precio_mediana"].notna().any() else 0
    df_s["d_precio"] = (precio_med - df_s["precio_mediana"]) / precio_med * 100 if precio_med else 0

    top3 = df_s.head(3)
    mun_str      = ", ".join(top3["MUNICIPIO_ORIGEN"].tolist())
    ventaja_min  = top3["d_precio"].min()
    ventaja_max  = top3["d_precio"].max()
    participacion = (top3["toneladas"].sum() / df_s["toneladas"].sum() * 100) if df_s["toneladas"].sum() > 0 else 0

    periodo_desc  = semestre.lower() if semestre != "Todos" else f"el rango {fecha_ini} a {fecha_fin}"
    centrales_txt = "todas las centrales mayoristas seleccionadas" if not centrales else ", ".join(centrales)

    texto = (
        f"Durante {periodo_desc}, el análisis para <b>{rubro or 'el rubro seleccionado'}</b> hacia "
        f"{centrales_txt} ubica a <b>{mun_str}</b> entre los municipios más eficientes. "
        f"Las ventajas de precio oscilan entre <b>{ventaja_min:.1f}%</b> y <b>{ventaja_max:.1f}%</b>, "
        f"y su participación conjunta representa <b>{participacion:.1f}%</b> del volumen total."
    )
    return pn.pane.HTML(
        f"<div class='panel-title' style='margin-top:0.9rem;'>Contexto interpretativo</div>"
        f"<div class='small-note'>{texto}</div>",
        sizing_mode="stretch_width"
    )


def construir_tabla(rubro, centrales, semestre, deptos, fecha_ini, fecha_fin):
    df = consultar_datos(rubro, t(centrales), semestre, t(deptos), fecha_ini, fecha_fin)
    if df.empty:
        return pn.pane.HTML("<div class='small-note' style='padding:20px;'>Sin datos.</div>", sizing_mode="stretch_width")

    df_s = df.sort_values("toneladas", ascending=False).reset_index(drop=True).copy()
    total_ton    = df_s["toneladas"].sum()
    precio_med   = df_s["precio_mediana"].median() if df_s["precio_mediana"].notna().any() else 0
    ton_max      = df_s["toneladas"].max() or 1
    dias_max     = df_s["dias_con_datos"].max() or 1

    df_s["d_precio"]  = (precio_med - df_s["precio_mediana"]) / precio_med * 100 if precio_med else 0
    df_s["d_volumen"] = df_s["toneladas"] / ton_max * 100
    df_s["d_estab"]   = df_s["dias_con_datos"] / dias_max * 100
    df_s["indice"]    = (0.40 * df_s["d_precio"] + 0.40 * df_s["d_volumen"] + 0.20 * df_s["d_estab"]).round(1)
    df_s["ventaja"]   = df_s["d_precio"].round(1)
    df_s["part_filtro"] = (df_s["toneladas"] / total_ton * 100).round(2)
    df_s["part_total"]  = (df_s["toneladas"] / total_bd * 100).round(2)
    df_s["meses"]       = (df_s["dias_con_datos"] / 30).round(0).astype(int)
    df_s["recursos"]    = df_s["toneladas"] * df_s["precio_promedio"]

    rape_ton = df_s[df_s["DEPARTAMENTO_ORIGEN"].str.upper().isin(DEPTOS_RAPE)]["toneladas"].sum()
    df_s["part_rape"] = df_s.apply(
        lambda r: round(r["toneladas"] / rape_ton * 100, 1)
        if r["DEPARTAMENTO_ORIGEN"].upper() in DEPTOS_RAPE and rape_ton > 0 else 0.0, axis=1
    )

    tabla = df_s[[
        "MUNICIPIO_ORIGEN","DEPARTAMENTO_ORIGEN",
        "precio_promedio","precio_mediana","toneladas","recursos","meses",
        "part_filtro","part_total","part_rape","ventaja","indice",
    ]].copy()
    tabla.insert(0, "Ranking", range(1, len(tabla) + 1))

    tabla["precio_promedio"] = tabla["precio_promedio"].map("$ {:,.0f} COP".format)
    tabla["precio_mediana"]  = tabla["precio_mediana"].map("$ {:,.0f} COP".format)
    tabla["toneladas"]       = tabla["toneladas"].map("{:,.1f}".format)
    tabla["recursos"]        = tabla["recursos"].map("$ {:,.0f} COP".format)
    tabla["part_filtro"]     = tabla["part_filtro"].map("{:.1f}%".format)
    tabla["part_total"]      = tabla["part_total"].map("{:.1f}%".format)
    tabla["part_rape"]       = tabla["part_rape"].map("{:.1f}%".format)
    tabla["ventaja"]         = tabla["ventaja"].map("{:.1f}%".format)
    tabla["indice"]          = tabla["indice"].map("{:.2f}".format)

    tabla.columns = [
        "Ranking","Municipio origen","Departamento origen",
        "Precio promedio","Precio moda","Toneladas acumuladas",
        "Recursos movilizados aprox.","Meses activos",
        "Participación en filtro","Participación total","Participación RAPE",
        "Ventaja precio","Índice de eficiencia",
    ]

    col_widths = {
        "Ranking":75,"Municipio origen":160,"Departamento origen":165,
        "Precio promedio":165,"Precio moda":165,"Toneladas acumuladas":160,
        "Recursos movilizados aprox.":205,"Meses activos":115,
        "Participación en filtro":175,"Participación total":155,
        "Participación RAPE":155,"Ventaja precio":135,"Índice de eficiencia":160,
    }
    return pn.widgets.DataFrame(tabla, sizing_mode="stretch_width", height=420, show_index=False, widths=col_widths, frozen_columns=1)


# =========================================================
# BIND REACTIVO — cada componente independiente
# =========================================================

metricas_r    = pn.bind(construir_metricas,    rubro=w_rubro, centrales=w_centrales, semestre=w_semestre, deptos=w_deptos, fecha_ini=w_fecha_ini, fecha_fin=w_fecha_fin)
mapa_update_r = pn.bind(construir_mapa_updater, rubro=w_rubro, centrales=w_centrales, semestre=w_semestre, deptos=w_deptos, fecha_ini=w_fecha_ini, fecha_fin=w_fecha_fin, max_flujos=w_max_flujos)
serie_r       = pn.bind(construir_serie,       rubro=w_rubro, centrales=w_centrales, semestre=w_semestre, deptos=w_deptos, fecha_ini=w_fecha_ini, fecha_fin=w_fecha_fin)
sankey_r      = pn.bind(construir_sankey,      rubro=w_rubro, centrales=w_centrales, semestre=w_semestre, deptos=w_deptos, fecha_ini=w_fecha_ini, fecha_fin=w_fecha_fin)
contexto_r    = pn.bind(construir_contexto,    rubro=w_rubro, centrales=w_centrales, semestre=w_semestre, deptos=w_deptos, fecha_ini=w_fecha_ini, fecha_fin=w_fecha_fin)
tabla_r       = pn.bind(construir_tabla,       rubro=w_rubro, centrales=w_centrales, semestre=w_semestre, deptos=w_deptos, fecha_ini=w_fecha_ini, fecha_fin=w_fecha_fin)

# =========================================================
# ENCABEZADO
# =========================================================

logo        = pn.pane.Image(str(RUTA_LOGO), height=60, margin=(8, 24, 8, 8))
titulo_html = pn.pane.HTML(f"""
    <div style='padding:8px 0; font-family:{FONT};'>
      <div style='font-size:2rem; font-weight:700; color:{COLOR_TITULO}; letter-spacing:-0.01em;'>
        Visor de precios y abastecimiento agroalimentario
      </div>
      <div style='font-size:0.95rem; color:{COLOR_MUTED}; margin-top:4px;'>
        Lectura territorial de precios, flujos y eficiencia relativa
        de municipios de origen por producto y central mayorista.
      </div>
    </div>
""")
encabezado = pn.Row(logo, titulo_html, styles={"background":"#FFFFFF","padding":"6px 16px","border-bottom":f"2px solid {COLOR_BORDE}"})

# =========================================================
# FILTROS
# =========================================================

def lbl(txt):
    return pn.pane.HTML(f"<div style='font-size:0.72rem;font-weight:700;color:{COLOR_TEXTO};text-transform:uppercase;letter-spacing:0.06em;margin-bottom:3px;font-family:{FONT};'>{txt}</div>")

fila_filtros = pn.Row(
    pn.Column(lbl("Rubro"),               w_rubro,     width=210),
    pn.Column(lbl("Central mayorista"),   w_centrales, width=250),
    pn.Column(lbl("Periodo semestral"),   w_semestre,  width=170),
    pn.Column(lbl("Departamento origen"), w_deptos,    width=250),
    pn.Column(lbl("Periodo"), pn.Row(pn.Column(lbl("Desde"), w_fecha_ini, width=170), pn.Column(lbl("Hasta"), w_fecha_fin, width=170))),
    pn.Column(lbl("Máx. flujos mapa"),    w_max_flujos, width=170),
    styles={"background":COLOR_PANEL,"border":f"1px solid {COLOR_BORDE}","border-radius":"12px","padding":"12px 18px","flex-wrap":"wrap","gap":"12px","align-items":"flex-start"}
)

# =========================================================
# HELPER CARD
# =========================================================

def card(titulo_sec, contenido, **kwargs):
    header = pn.pane.HTML(f"<div class='panel-title'>{titulo_sec}</div>") if titulo_sec else pn.pane.HTML("")
    return pn.Column(header, contenido, styles={"background":COLOR_PANEL,"border":f"1px solid {COLOR_BORDE}","border-radius":"12px","padding":"0.85rem 1rem","box-shadow":"0 2px 10px rgba(0,0,0,0.18)"}, **kwargs)

# =========================================================
# LAYOUT
# =========================================================

col_izq = pn.Column(
    pn.Column(metricas_r, styles={"background":COLOR_PANEL,"border":f"1px solid {COLOR_BORDE}","border-radius":"12px","padding":"0.85rem 1rem","box-shadow":"0 2px 10px rgba(0,0,0,0.18)"}),
    width=285,
)

col_centro = pn.Column(
    card("Mapa de flujos de abastecimiento", pn.Column(mapa_html, mapa_update_r)),
    card("Serie mensual de precio y toneladas", serie_r),
    sizing_mode="stretch_width",
)

col_der = pn.Column(
    pn.Column(
        pn.pane.HTML("<div class='panel-title'>Volumen de abastecimiento a las centrales mayoristas</div>"),
        sankey_r, contexto_r,
        styles={"background":COLOR_PANEL,"border":f"1px solid {COLOR_BORDE}","border-radius":"12px","padding":"0.85rem 1rem","box-shadow":"0 2px 10px rgba(0,0,0,0.18)"},
    ),
    width=440,
)

fila_principal = pn.Row(col_izq, col_centro, col_der, sizing_mode="stretch_width")

seccion_tabla = pn.Column(
    card("Tabla consolidada de análisis", tabla_r),
    pn.pane.HTML(f"""
    <div class="method-note">
        <b>Cómo se calcula el índice de eficiencia:</b><br>
        Combina tres dimensiones normalizadas: ventaja de precio frente al promedio del mercado filtrado,
        volumen acumulado abastecido y número de meses con participación.
        Ponderación: <b>40% precio, 40% volumen y 20% estabilidad operativa</b>.
    </div>
    """, sizing_mode="stretch_width"),
    sizing_mode="stretch_width",
)

pie = pn.pane.HTML(f"""
    <div style='margin-top:0.8rem;padding-top:0.6rem;border-top:1px solid {COLOR_BORDE};color:#8FA0B7;font-size:0.8rem;text-align:center;font-family:{FONT};'>
        Fuente: Sistema de Información de Precios y Abastecimiento del Sector Agropecuario (SIPSA) del DANE, periodo 2020 – 2024.
    </div>
""")

app = pn.Column(
    encabezado,
    pn.Column(fila_filtros, fila_principal, seccion_tabla, pie,
              styles={"background":COLOR_BG,"padding":"10px 16px"}, sizing_mode="stretch_width"),
    sizing_mode="stretch_width",
)

app.servable()
