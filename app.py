import json
from pathlib import Path

import duckdb
import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pydeck as pdk
import streamlit as st

# =========================================================
# CONFIGURACIÓN GENERAL
# =========================================================

st.set_page_config(
    page_title="Visor de precios y abastecimiento agroalimentario",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR            = Path("datos")
RUTA_LINEAS         = BASE_DIR / "lineas_abastecimiento.parquet"
RUTA_MUNICIPIOS     = BASE_DIR / "municipios_ligeros.parquet"
RUTA_LOGO           = BASE_DIR / "MDS-245-ES.jpg"
RUTA_LINEAS_SQL     = RUTA_LINEAS.as_posix()

DEPTOS_RAPE = {
    "BOGOTÁ", "BOGOTÁ, D.C.", "BOGOTA", "BOGOTA D.C.", "BOGOTÁ D.C.",
    "CUNDINAMARCA", "META", "BOYACÁ", "BOYACA", "TOLIMA"
}

MAX_FILAS_TABLA_DEFAULT = 300
MAX_LINEAS_MAPA_DEFAULT = 600
MAX_LINEAS_MAPA_MAX     = 1500

# Keep-alive — evita que Streamlit Cloud duerma el app
st.markdown("""
<script>
(function keepAlive() {
    setInterval(function() {
        fetch(window.location.href, {method:'GET', cache:'no-store'}).catch(function(){});
    }, 45 * 60 * 1000);
})();
</script>
""", unsafe_allow_html=True)

# =========================================================
# ESTILO
# =========================================================

st.markdown("""
<style>
    .stApp { background-color: #0F1116; color: #E8EDF5; }
    [data-testid="stHeader"] { background: rgba(0,0,0,0); }
    .block-container {
        max-width: 100%;
        padding-top: 0.8rem; padding-bottom: 1rem;
        padding-left: 1.1rem; padding-right: 1.1rem;
    }
    h1, h2, h3, h4 { color: #E8EDF5 !important; margin-bottom: 0.2rem; }
    .top-title { font-size: 2rem; font-weight: 700; color: #F2F5FA; margin-bottom: 0.1rem; }
    .top-subtitle { font-size: 0.95rem; color: #AEB9C9; margin-bottom: 0.55rem; }
    .panel {
        background: #171A21; border: 1px solid #2B3240;
        border-radius: 12px; padding: 0.85rem 1rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.18);
    }
    .panel-title { color: #F2F5FA; font-size: 1rem; font-weight: 600; margin-bottom: 0.65rem; }
    .metric-card {
        background: #171A21; border: 1px solid #2B3240;
        border-radius: 12px; padding: 0.8rem 1rem;
        text-align: center; margin-bottom: 0.8rem;
    }
    .metric-label { color: #9EABC0; font-size: 0.82rem; margin-bottom: 0.35rem; }
    .metric-value { color: #FFFFFF; font-size: 2rem; font-weight: 700; line-height: 1.05; }
    .metric-small { color: #C7D0DD; font-size: 0.8rem; margin-top: 0.25rem; }
    .legend-item { display: flex; align-items: center; gap: 8px; margin-bottom: 6px; font-size: 0.9rem; color: #D8E0EA; }
    .legend-box { width: 14px; height: 14px; border-radius: 3px; border: 1px solid rgba(255,255,255,0.15); }
    .small-note { color: #99A7BC; font-size: 0.82rem; line-height: 1.45; }
    .method-note {
        background: #171A21; border: 1px solid #2B3240;
        border-left: 4px solid #4DA3FF; border-radius: 10px;
        padding: 0.8rem 1rem; color: #C7D0DD;
        font-size: 0.86rem; line-height: 1.55; margin-top: 0.8rem;
    }
    .filter-wrap {
        background: #171A21; border: 1px solid #2B3240;
        border-radius: 12px; padding: 0.65rem 0.9rem 0.15rem 0.9rem;
        margin-bottom: 0.85rem;
    }
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div {
        background-color: #12161D !important; border-color: #2B3240 !important;
    }
    div[data-baseweb="tag"] { background-color: #243042 !important; }
    [data-testid="stDateInputField"] { background-color: #12161D !important; }
    [data-testid="stPlotlyChart"], [data-testid="stDeckGlJsonChart"] {
        background: #171A21; border: 1px solid #2B3240;
        border-radius: 12px; padding: 0.45rem;
    }
    .stDataFrame, div[data-testid="stTable"] {
        background: #171A21; border-radius: 12px;
        border: 1px solid #2B3240; padding: 0.25rem;
    }
</style>
""", unsafe_allow_html=True)


# =========================================================
# FUNCIONES AUXILIARES
# =========================================================

def normalizar_codigo_5(valor):
    if pd.isna(valor):
        return np.nan
    s = str(valor).strip().replace("'", "")
    if s.endswith(".0"):
        s = s[:-2]
    if s == "":
        return np.nan
    return s.zfill(5) if s.isdigit() else s


def normalizar_texto(valor):
    if pd.isna(valor):
        return ""
    return str(valor).strip()


def formatear_cop(valor):
    if pd.isna(valor):
        return "Sin dato"
    return f"$ {valor:,.0f}"


def formatear_ton(valor):
    if pd.isna(valor):
        return "Sin dato"
    return f"{valor:,.1f}"


def norm_serie(s):
    if len(s) == 0:
        return s
    s = s.fillna(0)
    if s.max() == s.min():
        return pd.Series(np.ones(len(s)), index=s.index)
    return (s - s.min()) / (s.max() - s.min())


def clasificar_eficiencia(indice):
    if pd.isna(indice):
        return "Sin clasificar"
    if indice >= 0.70:
        return "Alta eficiencia"
    if indice >= 0.45:
        return "Eficiencia media"
    return "Eficiencia baja"


def construir_texto_contexto(df_ef, rubro, destino_label, fecha_ini, fecha_fin, semestre_sel):
    if df_ef.empty:
        return (
            f"No hay suficiente información para destacar municipios más eficientes para "
            f"{rubro.lower()} en el periodo seleccionado."
        )
    top = df_ef.head(3).copy()
    municipios    = ", ".join(top["MUNICIPIO_ORIGEN"].tolist())
    ventaja_min   = top["ventaja_precio_pct"].min()
    ventaja_max   = top["ventaja_precio_pct"].max()
    participacion = top["participacion_total_pct"].sum()

    if semestre_sel != "Todos":
        periodo_desc = semestre_sel.lower()
    else:
        periodo_desc = f"el rango {fecha_ini.strftime('%Y-%m')} a {fecha_fin.strftime('%Y-%m')}"

    todos_negativos = ventaja_max < 0
    todos_positivos = ventaja_min > 0

    if todos_positivos:
        frase_precio = (
            f"Sus precios se ubican entre {ventaja_min:.1f}% y {ventaja_max:.1f}% "
            f"por debajo del promedio del mercado filtrado, lo que representa una ventaja competitiva en precio."
        )
    elif todos_negativos:
        frase_precio = (
            f"En términos de precio, estos municipios tienden a operar por encima del promedio del mercado "
            f"filtrado (entre {abs(ventaja_max):.1f}% y {abs(ventaja_min):.1f}% más costosos), aunque su "
            f"liderazgo en volumen y estabilidad de abastecimiento los posiciona favorablemente en el índice compuesto."
        )
    else:
        frase_precio = (
            f"Su comportamiento de precio es mixto respecto al promedio del mercado filtrado "
            f"(entre {ventaja_min:.1f}% y {ventaja_max:.1f}%), con algunos municipios más competitivos "
            f"en precio y otros que compensan con mayor volumen y estabilidad."
        )

    return (
        f"Durante {periodo_desc}, el análisis para {rubro.lower()} hacia {destino_label} "
        f"ubica a {municipios} entre los municipios con mayor peso relativo en la relación "
        f"precio-volumen-estabilidad. {frase_precio} "
        f"Su participación conjunta representa {participacion:.1f}% del volumen total registrado "
        f"para este producto bajo los filtros temporales activos."
    )


def construir_sankey(sankey_top):
    nodos_origen  = sankey_top["MUNICIPIO_ORIGEN"].astype(str).tolist()
    nodos_destino = sankey_top["CENTRAL_NOMBRE"].astype(str).tolist()
    nodos = list(dict.fromkeys(nodos_origen + nodos_destino))
    idx   = {n: i for i, n in enumerate(nodos)}
    valores      = sankey_top["toneladas_total"].astype(float).tolist()
    total_sankey = sum(valores) if valores else 0
    porcentajes  = [(v / total_sankey) * 100 if total_sankey > 0 else 0 for v in valores]
    fig = go.Figure(data=[go.Sankey(
        arrangement="snap",
        node=dict(pad=12, thickness=16, line=dict(color="gray", width=0.5), label=nodos),
        link=dict(
            source=[idx[s] for s in sankey_top["MUNICIPIO_ORIGEN"].astype(str)],
            target=[idx[t] for t in sankey_top["CENTRAL_NOMBRE"].astype(str)],
            value=valores, customdata=porcentajes,
            hovertemplate=(
                "Origen: %{source.label}<br>Central mayorista: %{target.label}<br>"
                "Toneladas: %{value:,.1f}<br>Participación: %{customdata:.1f}%<extra></extra>"
            )
        )
    )])
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#171A21", plot_bgcolor="#171A21",
        font=dict(color="#E8EDF5", size=11),
        margin=dict(l=10, r=10, t=10, b=10), height=430
    )
    return fig


def obtener_mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except FileNotFoundError:
        return 0.0


def construir_where_sql(rubro, fecha_ini, fecha_fin, semestre_sel, centrales_sel=(), deptos_sel=()):
    clauses = ["RUBRO = ?", "FECHA BETWEEN ? AND ?"]
    params  = [rubro, fecha_ini.isoformat(), fecha_fin.isoformat()]
    if semestre_sel == "Primer semestre":
        clauses.append("MES BETWEEN 1 AND 6")
    elif semestre_sel == "Segundo semestre":
        clauses.append("MES BETWEEN 7 AND 12")
    if centrales_sel:
        clauses.append(f"CENTRAL_NOMBRE IN ({','.join(['?']*len(centrales_sel))})")
        params.extend(list(centrales_sel))
    if deptos_sel:
        clauses.append(f"DEPARTAMENTO_ORIGEN IN ({','.join(['?']*len(deptos_sel))})")
        params.extend(list(deptos_sel))
    return " AND ".join(clauses), params


# =========================================================
# CARGA Y PREPARACIÓN DE CAPAS ESTÁTICAS
# Incluye pre-serialización del GeoJSON base (sin colores)
# =========================================================

# =========================================================
# CARGA DE MUNICIPIOS
# =========================================================

@st.cache_resource(show_spinner=False)
def cargar_y_preparar_capas_estaticas(mtime_municipios):
    municipios = gpd.read_parquet(RUTA_MUNICIPIOS)
    municipios = municipios.copy()

    for col in ["MpCodigo", "CODIGO_MUNICIPIO"]:
        if col in municipios.columns:
            municipios["codigo_origen"] = municipios[col].apply(normalizar_codigo_5)
            break
    else:
        raise ValueError("La capa de municipios no tiene MpCodigo ni CODIGO_MUNICIPIO.")

    for col in ["Nombre","MpNombre","MUNICIPIO","NOMBRE_MUNICIPIO","nombre_municipio"]:
        if col in municipios.columns:
            municipios["nombre_municipio"] = municipios[col].apply(normalizar_texto)
            break
    else:
        municipios["nombre_municipio"] = ""

    for col in ["Depto","DEPARTAMENTO","departamento"]:
        if col in municipios.columns:
            municipios["departamento"] = municipios[col].apply(normalizar_texto)
            break
    else:
        municipios["departamento"] = ""

    codigos_validos = tuple(sorted(
        municipios["codigo_origen"].dropna().astype(str).unique().tolist()
    ))
    return municipios, codigos_validos


# =========================================================
# CONEXIÓN DUCKDB
# =========================================================

@st.cache_resource(show_spinner=False)
def get_duckdb_connection(mtime_lineas, codigos_validos):
    con = duckdb.connect(database=":memory:")
    con.execute("PRAGMA threads=4")
    df_valid = pd.DataFrame({"codigo_origen": list(codigos_validos)})
    con.register("valid_codes_df", df_valid)
    con.execute("CREATE OR REPLACE TEMP TABLE valid_codes AS SELECT * FROM valid_codes_df")
    con.execute(f"""
        CREATE OR REPLACE VIEW lineas AS
        SELECT
            CAST(CAST(FECHA AS DATE) AS DATE)               AS FECHA,
            CAST(MES AS INTEGER)                            AS MES,
            TRIM(CAST(RUBRO AS VARCHAR))                    AS RUBRO,
            TRIM(CAST(CENTRAL_NOMBRE AS VARCHAR))           AS CENTRAL_NOMBRE,
            TRIM(CAST(DEPARTAMENTO_ORIGEN AS VARCHAR))      AS DEPARTAMENTO_ORIGEN,
            TRIM(CAST(MUNICIPIO_ORIGEN AS VARCHAR))         AS MUNICIPIO_ORIGEN,
            LPAD(REGEXP_REPLACE(CAST(CODIGO_DIVIPOLA_MUN_ORIGEN_LIMPIO AS VARCHAR),'\\\\.0$',''),5,'0')  AS codigo_origen,
            LPAD(REGEXP_REPLACE(CAST(CODIGO_DIVIPOLA_MUN_DESTINO_LIMPIO AS VARCHAR),'\\\\.0$',''),5,'0') AS codigo_destino,
            CAST(TONELADAS AS DOUBLE)       AS TONELADAS,
            CAST(PRECIO_PROMEDIO AS DOUBLE) AS PRECIO_PROMEDIO,
            CAST(PRECIO_MEDIANA AS DOUBLE)  AS PRECIO_MEDIANA,
            CAST(DIAS_CON_DATOS AS DOUBLE)  AS DIAS_CON_DATOS,
            CAST(LONGITUD_ORIGEN AS DOUBLE) AS LONGITUD_ORIGEN,
            CAST(LATITUD_ORIGEN AS DOUBLE)  AS LATITUD_ORIGEN,
            CAST(LONGITUD_DESTINO AS DOUBLE) AS LONGITUD_DESTINO,
            CAST(LATITUD_DESTINO AS DOUBLE)  AS LATITUD_DESTINO,
            CASE
                WHEN UPPER(TRIM(CAST(DEPARTAMENTO_ORIGEN AS VARCHAR)))='BOGOTA'      THEN 'BOGOTÁ'
                WHEN UPPER(TRIM(CAST(DEPARTAMENTO_ORIGEN AS VARCHAR)))='BOGOTA D.C.' THEN 'BOGOTÁ, D.C.'
                WHEN UPPER(TRIM(CAST(DEPARTAMENTO_ORIGEN AS VARCHAR)))='BOGOTA, D.C.' THEN 'BOGOTÁ, D.C.'
                WHEN UPPER(TRIM(CAST(DEPARTAMENTO_ORIGEN AS VARCHAR)))='BOGOTÁ D.C.' THEN 'BOGOTÁ, D.C.'
                WHEN UPPER(TRIM(CAST(DEPARTAMENTO_ORIGEN AS VARCHAR)))='BOGOTA DC'  THEN 'BOGOTÁ, D.C.'
                WHEN UPPER(TRIM(CAST(DEPARTAMENTO_ORIGEN AS VARCHAR)))='BOGOTÁ DC'  THEN 'BOGOTÁ, D.C.'
                WHEN UPPER(TRIM(CAST(DEPARTAMENTO_ORIGEN AS VARCHAR)))='BOYACA'     THEN 'BOYACÁ'
                ELSE UPPER(TRIM(CAST(DEPARTAMENTO_ORIGEN AS VARCHAR)))
            END AS DEPARTAMENTO_ORIGEN_NORM,
            CAST(DATE_TRUNC('month', CAST(FECHA AS DATE)) AS DATE) AS periodo_mes,
            STRFTIME(CAST(FECHA AS DATE),'%Y-%m') AS etiqueta_mes,
            CASE WHEN CAST(MES AS INTEGER) BETWEEN 1 AND 6
                 THEN 'Primer semestre' ELSE 'Segundo semestre' END AS semestre
        FROM read_parquet('{RUTA_LINEAS_SQL}')
        WHERE LPAD(REGEXP_REPLACE(CAST(CODIGO_DIVIPOLA_MUN_ORIGEN_LIMPIO AS VARCHAR),'\\\\.0$',''),5,'0')
              IN (SELECT codigo_origen FROM valid_codes)
    """)
    return con


# =========================================================
# CONSULTAS DUCKDB
# =========================================================

@st.cache_data(show_spinner=False)
def consultar_catalogos_lineas(mtime_lineas, codigos_validos):
    con = get_duckdb_connection(mtime_lineas, codigos_validos)
    rubros    = con.execute("SELECT DISTINCT RUBRO FROM lineas WHERE RUBRO IS NOT NULL ORDER BY RUBRO").df()["RUBRO"].astype(str).tolist()
    centrales = con.execute("SELECT DISTINCT CENTRAL_NOMBRE FROM lineas WHERE CENTRAL_NOMBRE IS NOT NULL ORDER BY CENTRAL_NOMBRE").df()["CENTRAL_NOMBRE"].astype(str).tolist()
    deptos    = con.execute("SELECT DISTINCT DEPARTAMENTO_ORIGEN FROM lineas WHERE DEPARTAMENTO_ORIGEN IS NOT NULL ORDER BY DEPARTAMENTO_ORIGEN").df()["DEPARTAMENTO_ORIGEN"].astype(str).tolist()
    rango     = con.execute("SELECT MIN(FECHA) AS fecha_min, MAX(FECHA) AS fecha_max FROM lineas").df()
    fecha_min = pd.to_datetime(rango.loc[0,"fecha_min"]).date()
    fecha_max = pd.to_datetime(rango.loc[0,"fecha_max"]).date()
    return rubros, centrales, deptos, fecha_min, fecha_max


@st.cache_data(show_spinner=False)
def consultar_todo_filtrado(rubro, fecha_ini, fecha_fin, semestre_sel, centrales_sel, deptos_sel, mtime_lineas, codigos_validos):
    con = get_duckdb_connection(mtime_lineas, codigos_validos)
    where_sql, params             = construir_where_sql(rubro, fecha_ini, fecha_fin, semestre_sel, centrales_sel, deptos_sel)
    where_total_sql, params_total = construir_where_sql(rubro, fecha_ini, fecha_fin, semestre_sel, (), ())
    deptos_rape       = list(DEPTOS_RAPE)
    placeholders_rape = ",".join(["?"]*len(deptos_rape))

    query = f"""
        WITH base AS (
            SELECT * FROM lineas WHERE {where_sql}
        ),
        base_total AS (
            SELECT * FROM lineas WHERE {where_total_sql}
        ),
        metricas AS (
            SELECT
                AVG(PRECIO_PROMEDIO)           AS precio_ref,
                SUM(TONELADAS)                 AS volumen_total_filtro,
                COUNT(DISTINCT codigo_origen)  AS municipios_activos,
                COUNT(DISTINCT codigo_destino) AS centrales_activas
            FROM base
        ),
        metricas_total AS (SELECT SUM(TONELADAS) AS volumen_total_total FROM base_total),
        metricas_rape  AS (
            SELECT SUM(TONELADAS) AS volumen_total_rape
            FROM base_total WHERE DEPARTAMENTO_ORIGEN_NORM IN ({placeholders_rape})
        ),
        ranking AS (
            SELECT
                CASE
                    WHEN UPPER(TRIM(MUNICIPIO_ORIGEN)) = 'UNE'     THEN '25845'
                    WHEN UPPER(TRIM(MUNICIPIO_ORIGEN)) = 'FÓMEQUE' THEN '25279'
                    WHEN UPPER(TRIM(MUNICIPIO_ORIGEN)) = 'FOMÉQUE' THEN '25279'
                    WHEN UPPER(TRIM(MUNICIPIO_ORIGEN)) = 'FOMEQUE' THEN '25279'
                    WHEN UPPER(TRIM(MUNICIPIO_ORIGEN)) = 'CERRITO' THEN '68162'
                    ELSE LPAD(REGEXP_REPLACE(CAST(codigo_origen AS VARCHAR), '\.0$', ''), 5, '0')
                END AS codigo_origen,
                MAX(MUNICIPIO_ORIGEN)            AS MUNICIPIO_ORIGEN,
                MAX(DEPARTAMENTO_ORIGEN)         AS DEPARTAMENTO_ORIGEN,
                SUM(TONELADAS)                   AS toneladas_total,
                AVG(PRECIO_PROMEDIO)             AS precio_promedio,
                MODE(PRECIO_PROMEDIO)            AS precio_moda,
                SUM(DIAS_CON_DATOS)              AS dias_con_datos,
                COUNT(DISTINCT periodo_mes)      AS meses_participacion,
                SUM(PRECIO_PROMEDIO * TONELADAS) AS recursos_movilizados_aprox
            FROM base
            GROUP BY 1
        ),
        serie AS (
            SELECT periodo_mes, etiqueta_mes,
                AVG(PRECIO_PROMEDIO) AS precio_promedio,
                SUM(TONELADAS)       AS toneladas_total
            FROM base GROUP BY 1,2 ORDER BY 1
        ),
        flujos AS (
            SELECT
                codigo_origen, MUNICIPIO_ORIGEN, DEPARTAMENTO_ORIGEN,
                CENTRAL_NOMBRE, codigo_destino,
                LONGITUD_ORIGEN, LATITUD_ORIGEN, LONGITUD_DESTINO, LATITUD_DESTINO,
                SUM(TONELADAS)       AS toneladas_total,
                AVG(PRECIO_PROMEDIO) AS precio_promedio
            FROM base
            GROUP BY 1,2,3,4,5,6,7,8,9
            HAVING LONGITUD_ORIGEN IS NOT NULL AND LATITUD_ORIGEN IS NOT NULL
               AND LONGITUD_DESTINO IS NOT NULL AND LATITUD_DESTINO IS NOT NULL
            ORDER BY toneladas_total DESC
        ),
        sankey AS (
            SELECT MUNICIPIO_ORIGEN, CENTRAL_NOMBRE, SUM(TONELADAS) AS toneladas_total
            FROM base GROUP BY 1,2 ORDER BY toneladas_total DESC
        )
        SELECT 'metricas' AS _tabla, TO_JSON(metricas) AS _json FROM metricas
        UNION ALL SELECT 'total',   TO_JSON(metricas_total) FROM metricas_total
        UNION ALL SELECT 'rape',    TO_JSON(metricas_rape)  FROM metricas_rape
        UNION ALL SELECT 'ranking', TO_JSON(ranking)        FROM ranking
        UNION ALL SELECT 'serie',   TO_JSON(serie)          FROM serie
        UNION ALL SELECT 'flujos',  TO_JSON(flujos)         FROM flujos
        UNION ALL SELECT 'sankey',  TO_JSON(sankey)         FROM sankey
    """

    resultado = con.execute(query, params + params_total + deptos_rape).df()

    def extraer(nombre):
        filas = resultado[resultado["_tabla"] == nombre]["_json"].tolist()
        if not filas:
            return pd.DataFrame()
        return pd.DataFrame([json.loads(f) for f in filas])

    return (
        extraer("metricas"), extraer("total"), extraer("rape"),
        extraer("ranking"), extraer("serie"), extraer("flujos"), extraer("sankey")
    )


# =========================================================
# CARGA INICIAL — una sola vez por sesión
# =========================================================

mtime_lineas     = obtener_mtime(RUTA_LINEAS)
mtime_municipios = obtener_mtime(RUTA_MUNICIPIOS)

municipios, codigos_validos = cargar_y_preparar_capas_estaticas(mtime_municipios)

rubros, centrales, deptos, fecha_min_global, fecha_max_global = \
    consultar_catalogos_lineas(mtime_lineas, codigos_validos)


# =========================================================
# ENCABEZADO
# =========================================================

import base64
with open(RUTA_LOGO, "rb") as _f:
    _logo_b64 = base64.b64encode(_f.read()).decode()

st.markdown(f"""
<div style="
    background: #FFFFFF;
    border-bottom: 2px solid #2B3240;
    padding: 10px 20px;
    margin-bottom: 0.85rem;
    margin-left: -1.1rem;
    margin-right: -1.1rem;
    margin-top: -0.8rem;
    display: flex;
    align-items: center;
    gap: 18px;
">
    <img src="data:image/jpeg;base64,{_logo_b64}" style="height:64px; width:auto; flex-shrink:0;" />
    <div>
        <div style="font-size:1.85rem; font-weight:700; color:#111827; letter-spacing:-0.01em; line-height:1.1;">
            Visor de precios y abastecimiento agroalimentario
        </div>
        <div style="font-size:0.92rem; color:#6B7280; margin-top:4px;">
            Lectura territorial de precios, flujos y eficiencia relativa de municipios de origen por producto y central mayorista.
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# =========================================================
# FILTROS
# =========================================================

st.markdown('<div class="filter-wrap">', unsafe_allow_html=True)
f1, f2, f3, f4, f5 = st.columns([1.2, 1.5, 1.1, 1.0, 1.4])

with f1:
    rubro_sel = st.selectbox("Rubro", rubros, index=0 if rubros else None)
with f2:
    centrales_sel = st.multiselect("Central mayorista", options=centrales, default=[])
with f3:
    semestre_sel = st.selectbox("Periodo semestral", ["Todos","Primer semestre","Segundo semestre"], index=0)
with f4:
    deptos_sel = st.multiselect("Departamento origen", deptos, default=[])
with f5:
    rango = st.date_input("Periodo", value=(fecha_min_global, fecha_max_global),
                          min_value=fecha_min_global, max_value=fecha_max_global)

st.markdown("</div>", unsafe_allow_html=True)

max_lineas      = MAX_LINEAS_MAPA_DEFAULT
max_filas_tabla = MAX_FILAS_TABLA_DEFAULT

if isinstance(rango, tuple) and len(rango) == 2:
    fecha_ini, fecha_fin = rango
else:
    fecha_ini, fecha_fin = fecha_min_global, fecha_max_global

centrales_tuple = tuple(centrales_sel)
deptos_tuple    = tuple(deptos_sel)
destino_label   = ", ".join(centrales_sel) if centrales_sel else "todas las centrales mayoristas seleccionadas"


# =========================================================
# CONSULTA PRINCIPAL — 1 solo escaneo DuckDB
# =========================================================

(
    metricas_filtradas_df, volumen_total_df, volumen_rape_df,
    ranking_base, serie_mensual, flujos_mapa, sankey_base
) = consultar_todo_filtrado(
    rubro_sel, fecha_ini, fecha_fin, semestre_sel,
    centrales_tuple, deptos_tuple, mtime_lineas, codigos_validos
)


# =========================================================
# EXTRACCIÓN DE ESCALARES
# =========================================================

def _sf(df, col):
    try:
        v = df.loc[0, col]; return float(v) if pd.notna(v) else 0.0
    except: return 0.0

def _si(df, col):
    try:
        v = df.loc[0, col]; return int(v) if pd.notna(v) else 0
    except: return 0

precio_ref           = _sf(metricas_filtradas_df, "precio_ref")
volumen_total_filtro = _sf(metricas_filtradas_df, "volumen_total_filtro")
municipios_activos   = _si(metricas_filtradas_df, "municipios_activos")
centrales_activas    = _si(metricas_filtradas_df, "centrales_activas")
volumen_total_total  = _sf(volumen_total_df, "volumen_total_total")
volumen_total_rape   = _sf(volumen_rape_df,  "volumen_total_rape")


# =========================================================
# AGREGACIONES DE RANKING E ÍNDICE
# =========================================================

if not ranking_base.empty:
    ranking = ranking_base.copy()
    total_meses = max(serie_mensual["periodo_mes"].nunique(), 1) if not serie_mensual.empty else 1

    ranking["participacion_filtro_pct"] = np.where(volumen_total_filtro > 0, (ranking["toneladas_total"] / volumen_total_filtro) * 100, 0)
    ranking["participacion_total_pct"]  = np.where(volumen_total_total  > 0, (ranking["toneladas_total"] / volumen_total_total)  * 100, 0)
    ranking["participacion_rape_pct"]   = np.where(volumen_total_rape   > 0, (ranking["toneladas_total"] / volumen_total_rape)   * 100, 0)
    ranking["ventaja_precio_pct"]       = np.where(precio_ref > 0, ((precio_ref - ranking["precio_promedio"]) / precio_ref) * 100, 0)
    ranking["frecuencia_relativa"]      = ranking["meses_participacion"] / total_meses

    ranking["score_precio"]    = norm_serie(ranking["ventaja_precio_pct"].clip(lower=0))
    ranking["score_volumen"]   = norm_serie(ranking["toneladas_total"])
    ranking["score_actividad"] = norm_serie(ranking["meses_participacion"])
    ranking["indice_eficiencia"] = (
        ranking["score_precio"] * 0.40 +
        ranking["score_volumen"] * 0.40 +
        ranking["score_actividad"] * 0.20
    )
    ranking["categoria_eficiencia"] = ranking["indice_eficiencia"].apply(clasificar_eficiencia)
    ranking = ranking.sort_values(
        ["indice_eficiencia","toneladas_total","ventaja_precio_pct"], ascending=False
    ).reset_index(drop=True)
    ranking["ranking"] = ranking.index + 1
    municipios_eficientes = ranking.copy()

    # Top 30 por toneladas — solo estos se colorean en morado
    top30_codigos = set(
        ranking.sort_values("toneladas_total", ascending=False)
        .head(30)["codigo_origen"].astype(str).tolist()
    )

    # Flujos — solo top N por toneladas, sin cobertura
    flujos_mapa = flujos_mapa.sort_values("toneladas_total", ascending=False).head(max_lineas).copy() \
                  if not flujos_mapa.empty else flujos_mapa

    if not flujos_mapa.empty:
        vmin = flujos_mapa["toneladas_total"].min()
        vmax = flujos_mapa["toneladas_total"].max()
        flujos_mapa["ancho_linea"] = 2 + 10 * ((flujos_mapa["toneladas_total"] - vmin) / (vmax - vmin + 1e-9))
        flujos_mapa["toneladas_fmt"] = flujos_mapa["toneladas_total"].map(formatear_ton)
        flujos_mapa["precio_fmt"]    = flujos_mapa["precio_promedio"].map(formatear_cop)

    if not sankey_base.empty:
        top_mun_sankey = (
            sankey_base.groupby("MUNICIPIO_ORIGEN", as_index=False)
            .agg(toneladas_total=("toneladas_total","sum"))
            .sort_values("toneladas_total", ascending=False)
            .head(12)["MUNICIPIO_ORIGEN"].tolist()
        )
        sankey_top = sankey_base[sankey_base["MUNICIPIO_ORIGEN"].isin(top_mun_sankey)].copy()
    else:
        sankey_top = pd.DataFrame(columns=["MUNICIPIO_ORIGEN","CENTRAL_NOMBRE","toneladas_total"])
else:
    ranking = pd.DataFrame()
    municipios_eficientes = pd.DataFrame()
    top30_codigos = set()
    flujos_mapa   = pd.DataFrame()
    sankey_top    = pd.DataFrame(columns=["MUNICIPIO_ORIGEN","CENTRAL_NOMBRE","toneladas_total"])


# =========================================================
# PREPARACIÓN DEL MAPA — top30 en morado, base gris estática
# Vectorizado: sin .apply(), O(n) sobre lista Python
# =========================================================

# =========================================================
# PREPARACIÓN DEL MAPA — top30 morado, puntos desde flujos
# =========================================================

top30_list = municipios["codigo_origen"].astype(str).isin(top30_codigos).tolist()
municipios_web = municipios[["nombre_municipio","departamento","codigo_origen","geometry"]].copy()
municipios_web["fill_color"] = [
    [110, 68, 255, 150] if es_top else [40, 48, 62, 18] for es_top in top30_list
]
municipios_web["line_color"] = [
    [170, 130, 255, 240] if es_top else [100, 110, 125, 60] for es_top in top30_list
]
municipios_web["tipo_elemento"] = "Municipio"
municipios_web["detalle_1"]     = "Nombre: "       + municipios_web["nombre_municipio"].fillna("Sin nombre").astype(str)
municipios_web["detalle_2"]     = "Departamento: " + municipios_web["departamento"].fillna("Sin dato").astype(str)
municipios_web["detalle_3"]     = "Código: "       + municipios_web["codigo_origen"].fillna("").astype(str)
municipios_web["detalle_4"]     = ""

geojson_municipios = json.loads(
    municipios_web[["fill_color","line_color","tipo_elemento","detalle_1","detalle_2","detalle_3","detalle_4","geometry"]].to_json()
)

# Tooltip de flujos
if not flujos_mapa.empty:
    flujos_mapa["tipo_elemento"] = "Flujo OD"
    flujos_mapa["detalle_1"]     = "Origen: "            + flujos_mapa["MUNICIPIO_ORIGEN"].fillna("")
    flujos_mapa["detalle_2"]     = "Central mayorista: " + flujos_mapa["CENTRAL_NOMBRE"].fillna("")
    flujos_mapa["detalle_3"]     = "Precio promedio: "   + flujos_mapa["precio_fmt"].fillna("")
    flujos_mapa["detalle_4"]     = "Toneladas: "         + flujos_mapa["toneladas_fmt"].fillna("")

# Puntos de origen desde flujos (naranja)
if not flujos_mapa.empty:
    orig_pts = flujos_mapa.groupby(
        ["MUNICIPIO_ORIGEN","DEPARTAMENTO_ORIGEN"], as_index=False
    ).agg(
        lon=("LONGITUD_ORIGEN","first"),
        lat=("LATITUD_ORIGEN","first"),
        TONELADAS=("toneladas_total","sum"),
        PRECIO_PROMEDIO=("precio_promedio","mean")
    ).dropna(subset=["lon","lat"])
    orig_pts["tipo_elemento"]  = "Municipio de origen"
    orig_pts["detalle_1"]      = "Municipio: "       + orig_pts["MUNICIPIO_ORIGEN"].fillna("")
    orig_pts["detalle_2"]      = "Departamento: "    + orig_pts["DEPARTAMENTO_ORIGEN"].fillna("")
    orig_pts["detalle_3"]      = "Toneladas: "       + orig_pts["TONELADAS"].map(formatear_ton)
    orig_pts["detalle_4"]      = "Precio promedio: " + orig_pts["PRECIO_PROMEDIO"].map(formatear_cop)
else:
    orig_pts = pd.DataFrame(columns=["MUNICIPIO_ORIGEN","lon","lat","tipo_elemento",
                                      "detalle_1","detalle_2","detalle_3","detalle_4"])

# Centrales desde flujos (cyan)
if not flujos_mapa.empty:
    cent_pts = flujos_mapa.groupby("CENTRAL_NOMBRE", as_index=False).agg(
        lon=("LONGITUD_DESTINO","first"), lat=("LATITUD_DESTINO","first")
    ).dropna(subset=["lon","lat"])
    cent_pts["tipo_elemento"] = "Central mayorista"
    cent_pts["detalle_1"]     = "Central: " + cent_pts["CENTRAL_NOMBRE"].fillna("")
    cent_pts["detalle_2"]     = ""
    cent_pts["detalle_3"]     = ""
    cent_pts["detalle_4"]     = ""
else:
    cent_pts = pd.DataFrame(columns=["CENTRAL_NOMBRE","lon","lat","tipo_elemento",
                                      "detalle_1","detalle_2","detalle_3","detalle_4"])


# =========================================================
# FRAGMENTO PRINCIPAL — mapa, métricas, serie, sankey
# =========================================================

@st.fragment
def render_layout_principal(
    volumen_total_filtro, precio_ref, municipios_activos, centrales_activas,
    geojson_municipios, flujos_mapa, orig_pts, cent_pts, serie_mensual,
    sankey_top, rubro_sel, destino_label, fecha_ini, fecha_fin, semestre_sel
):
    left_col, center_col, right_col = st.columns([1.05, 3.8, 1.45], gap="small")

    # ---- Columna izquierda: métricas + leyenda ----
    with left_col:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Indicadores principales</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Toneladas abastecidas</div>
            <div class="metric-value">{volumen_total_filtro:,.0f}</div>
            <div class="metric-small">Periodo filtrado</div>
        </div>
        """, unsafe_allow_html=True)
        if precio_ref and precio_ref > 0:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Precio promedio</div>
                <div class="metric-value" style="font-size:1.65rem;">$ {precio_ref:,.0f}</div>
                <div class="metric-small">Mercado filtrado ($/kg)</div>
            </div>""", unsafe_allow_html=True)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Municipios origen activos</div>
            <div class="metric-value">{municipios_activos}</div>
            <div class="metric-small">Con flujo válido</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Centrales activas</div>
            <div class="metric-value">{centrales_activas}</div>
            <div class="metric-small">Bajo filtros actuales</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Leyenda</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="legend-item"><span class="legend-box" style="background:#6E44FF;"></span>Top 30 abastecedores</div>
        <div class="legend-item"><span class="legend-box" style="background:#F5A020;border-radius:50%;"></span>Municipio de origen activo</div>
        <div class="legend-item"><span class="legend-box" style="background:#F5B041;"></span>Arcos de flujo OD</div>
        <div class="legend-item"><span class="legend-box" style="background:#00D2FF;"></span>Central mayorista</div>
        <div class="small-note" style="margin-top:0.5rem;">
            Los municipios morados son los 30 principales abastecedores.
            Los arcos muestran flujos origen–destino bajo los filtros activos.
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ---- Columna central: mapa + serie ----
    with center_col:
        st.markdown('<div class="panel-title">Mapa de flujos de abastecimiento</div>', unsafe_allow_html=True)

        layers = [pdk.Layer(
            "GeoJsonLayer", data=geojson_municipios,
            stroked=True, filled=True, extruded=False, wireframe=False,
            get_fill_color="properties.fill_color",
            get_line_color="properties.line_color",
            line_width_min_pixels=1.0, pickable=True, auto_highlight=True
        )]
        if not flujos_mapa.empty:
            layers.append(pdk.Layer(
                "ArcLayer", data=flujos_mapa,
                get_source_position=["LONGITUD_ORIGEN","LATITUD_ORIGEN"],
                get_target_position=["LONGITUD_DESTINO","LATITUD_DESTINO"],
                get_source_color=[245,176,65,190], get_target_color=[0,210,255,190],
                get_width="ancho_linea", width_scale=1, width_min_pixels=1,
                pickable=True, auto_highlight=True
            ))
        if not orig_pts.empty:
            layers.append(pdk.Layer(
                "ScatterplotLayer", data=orig_pts,
                get_position="[lon, lat]", get_radius=4200,
                get_fill_color=[245,160,32,180], get_line_color=[255,210,100,220],
                line_width_min_pixels=1, pickable=True, auto_highlight=True
            ))
        if not cent_pts.empty:
            layers.append(pdk.Layer(
                "ScatterplotLayer", data=cent_pts,
                get_position="[lon, lat]", get_radius=13500,
                get_fill_color=[0,210,255,190], get_line_color=[170,245,255,255],
                line_width_min_pixels=2, pickable=True
            ))

        deck = pdk.Deck(
            layers=layers,
            initial_view_state=pdk.ViewState(latitude=4.5, longitude=-74.1, zoom=4.6, pitch=0),
            tooltip={
                "html": "<b>{tipo_elemento}</b><br/>{detalle_1}<br/>{detalle_2}<br/>{detalle_3}<br/>{detalle_4}",
                "style": {"backgroundColor":"rgba(18,22,29,0.95)","color":"#F5F7FA","fontSize":"12px"}
            },
            map_style="dark"
        )
        st.pydeck_chart(deck, use_container_width=True)

        st.markdown('<div class="panel-title" style="margin-top:0.65rem;">Serie mensual de precio y toneladas</div>', unsafe_allow_html=True)
        if not serie_mensual.empty:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=serie_mensual["etiqueta_mes"], y=serie_mensual["precio_promedio"],
                name="Precio promedio", marker_color="#4DA3FF", yaxis="y1",
                hovertemplate="Mes: %{x}<br>Precio: $%{y:,.0f}<extra></extra>"
            ))
            fig.add_trace(go.Scatter(
                x=serie_mensual["etiqueta_mes"], y=serie_mensual["toneladas_total"],
                name="Toneladas", mode="lines+markers",
                line=dict(color="#F5B041", width=2.5), marker=dict(size=6, color="#F5B041"),
                yaxis="y2", hovertemplate="Mes: %{x}<br>Toneladas: %{y:,.1f}<extra></extra>"
            ))
            fig.update_layout(
                template="plotly_dark", paper_bgcolor="#171A21", plot_bgcolor="#171A21",
                margin=dict(l=15, r=15, t=10, b=10), height=300,
                legend=dict(orientation="h", y=1.08, x=0),
                xaxis=dict(showgrid=False),
                yaxis=dict(title="Precio", gridcolor="#2B3240"),
                yaxis2=dict(title="Toneladas", overlaying="y", side="right", showgrid=False)
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No hay datos para la serie mensual con los filtros actuales.")

    # ---- Columna derecha: sankey ----
    with right_col:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown(f'<div class="panel-title">Flujos hacia centrales mayoristas<br><span style="font-weight:400;font-size:0.82rem;color:#AEB9C9;">{rubro_sel}</span></div>', unsafe_allow_html=True)
        if not sankey_top.empty:
            st.plotly_chart(construir_sankey(sankey_top), use_container_width=True)
        else:
            st.info("No hay datos suficientes para mostrar.")
        st.markdown('</div>', unsafe_allow_html=True)


# =========================================================
# FRAGMENTO TABLA — independiente del mapa
# =========================================================

@st.fragment
def render_tabla(ranking, max_filas_tabla):
    st.markdown('<div class="panel-title" style="margin-top:0.8rem;">Tabla consolidada de análisis</div>', unsafe_allow_html=True)
    if not ranking.empty:
        # Columnas ordenables — se conservan como numéricas para el sort
        tabla_num = ranking[[
            "ranking","MUNICIPIO_ORIGEN","DEPARTAMENTO_ORIGEN",
            "precio_promedio","precio_moda","toneladas_total","recursos_movilizados_aprox",
            "meses_participacion","participacion_filtro_pct","participacion_total_pct",
            "participacion_rape_pct","ventaja_precio_pct","indice_eficiencia"
        ]].copy()
        tabla_num = tabla_num.rename(columns={
            "MUNICIPIO_ORIGEN":    "municipio_origen",
            "DEPARTAMENTO_ORIGEN": "departamento_origen",
            "precio_promedio":     "precio_promedio_municipio",
        })
        tabla_num = tabla_num.rename(columns=NOMBRES_COLUMNAS_PRESENTABLES)

        # Selector de ordenamiento
        cols_ordenables = [
            "Ranking", "Precio promedio", "Toneladas acumuladas",
            "Participación en filtro", "Participación total",
            "Ventaja precio", "Índice de eficiencia", "Meses activos"
        ]
        col_sort, col_dir = st.columns([2, 1])
        with col_sort:
            col_orden = st.selectbox(
                "Ordenar por", cols_ordenables,
                index=0, key="tabla_orden_col",
                label_visibility="collapsed"
            )
        with col_dir:
            dir_orden = st.radio(
                "Dirección", ["↓ Mayor a menor", "↑ Menor a mayor"],
                index=0, horizontal=True, key="tabla_orden_dir",
                label_visibility="collapsed"
            )

        ascendente = dir_orden == "↑ Menor a mayor"
        tabla_num = tabla_num.sort_values(col_orden, ascending=ascendente).reset_index(drop=True)
        # Actualizar ranking si el orden cambió
        if col_orden != "Ranking":
            tabla_num["Ranking"] = range(1, len(tabla_num) + 1)

        # Formatear para presentación
        tabla_fmt = tabla_num.copy()
        tabla_fmt["Precio promedio"]            = tabla_fmt["Precio promedio"].map(lambda x: f"$ {x:,.0f} COP")
        tabla_fmt["Precio moda"]                = tabla_fmt["Precio moda"].map(lambda x: f"$ {x:,.0f} COP" if pd.notna(x) else "Sin dato")
        tabla_fmt["Toneladas acumuladas"]        = tabla_fmt["Toneladas acumuladas"].map(lambda x: f"{x:,.1f}")
        tabla_fmt["Recursos movilizados aprox."] = tabla_fmt["Recursos movilizados aprox."].map(lambda x: f"$ {x:,.0f} COP" if pd.notna(x) else "Sin dato")
        tabla_fmt["Participación en filtro"]     = tabla_fmt["Participación en filtro"].map(lambda x: f"{x:.1f}%")
        tabla_fmt["Participación total"]         = tabla_fmt["Participación total"].map(lambda x: f"{x:.1f}%")
        tabla_fmt["Participación RAPE"]          = tabla_fmt["Participación RAPE"].map(lambda x: f"{x:.1f}%")
        tabla_fmt["Ventaja precio"]              = tabla_fmt["Ventaja precio"].map(lambda x: f"{x:.1f}%")
        tabla_fmt["Índice de eficiencia"]        = tabla_fmt["Índice de eficiencia"].map(lambda x: f"{x:.2f}")

        st.dataframe(tabla_fmt.head(max_filas_tabla), use_container_width=True, hide_index=True, height=420)
    else:
        st.info("No hay información disponible para la tabla consolidada.")

    st.markdown("""
    <div class="method-note">
        <b>Cómo se interpreta la ventaja de precio:</b><br>
        La ventaja de precio compara el precio promedio de cada municipio con el promedio general
        del mercado bajo los filtros activos. Un valor <b>positivo</b> indica que el municipio
        abastece a un precio <b>más bajo</b> que el promedio (ventaja competitiva).
        Un valor <b>negativo</b> indica que sus precios son <b>más altos</b> que el promedio,
        aunque puede compensar con mayor volumen o estabilidad de abastecimiento.
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="method-note">
        <b>Cómo se calcula el índice de eficiencia:</b><br>
        El índice compara municipios de origen únicamente dentro del subconjunto filtrado por rubro,
        central mayorista, periodo y demás filtros activos. Combina tres dimensiones normalizadas:
        ventaja de precio frente al promedio del mercado filtrado, volumen acumulado abastecido y número
        de meses con participación. La ponderación usada es 40% precio, 40% volumen y 20% estabilidad operativa.
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div style="margin-top:0.8rem; padding-top:0.6rem; border-top:1px solid #2B3240;
        color:#8FA0B7; font-size:0.8rem; text-align:center;">
        Fuente de información: Sistema de Información de Precios y Abastecimiento del Sector Agropecuario (SIPSA) del DANE, periodo 2020 - 2024.
    </div>
    """, unsafe_allow_html=True)


# =========================================================
# LLAMADA A LOS FRAGMENTOS
# =========================================================

render_layout_principal(
    volumen_total_filtro=volumen_total_filtro,
    precio_ref=precio_ref,
    municipios_activos=municipios_activos,
    centrales_activas=centrales_activas,
    geojson_municipios=geojson_municipios,
    flujos_mapa=flujos_mapa,
    orig_pts=orig_pts,
    cent_pts=cent_pts,
    serie_mensual=serie_mensual,
    sankey_top=sankey_top,
    rubro_sel=rubro_sel,
    destino_label=destino_label,
    fecha_ini=fecha_ini,
    fecha_fin=fecha_fin,
    semestre_sel=semestre_sel,
)

render_tabla(
    ranking=ranking,
    max_filas_tabla=max_filas_tabla,
)
