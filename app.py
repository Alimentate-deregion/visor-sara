import json
from pathlib import Path

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
    initial_sidebar_state="collapsed"
)

BASE_DIR = Path("datos")

RUTA_LINEAS = BASE_DIR / "lineas_abastecimiento.parquet"
RUTA_MUNICIPIOS = BASE_DIR / "municipios_ligeros.parquet"
RUTA_PUNTOS_DESTINO = BASE_DIR / "puntos_destino.geojson"
RUTA_PUNTOS_ORIGEN = BASE_DIR / "puntos_origen.geojson"

DEPTOS_RAPE = {
    "BOGOTÁ", "BOGOTÁ, D.C.", "BOGOTA", "BOGOTA D.C.", "BOGOTÁ D.C.",
    "CUNDINAMARCA", "META", "BOYACÁ", "BOYACA", "TOLIMA"
}

MAX_FILAS_TABLA_DEFAULT = 300
MAX_LINEAS_MAPA_DEFAULT = 600
MAX_LINEAS_MAPA_MAX = 1500


# =========================================================
# ESTILO
# =========================================================

st.markdown("""
<style>
    .stApp {
        background-color: #0F1116;
        color: #E8EDF5;
    }

    [data-testid="stHeader"] {
        background: rgba(0,0,0,0);
    }

    .block-container {
        max-width: 100%;
        padding-top: 0.8rem;
        padding-bottom: 1rem;
        padding-left: 1.1rem;
        padding-right: 1.1rem;
    }

    h1, h2, h3, h4 {
        color: #E8EDF5 !important;
        margin-bottom: 0.2rem;
    }

    .top-title {
        font-size: 2rem;
        font-weight: 700;
        color: #F2F5FA;
        margin-bottom: 0.1rem;
    }

    .top-subtitle {
        font-size: 0.95rem;
        color: #AEB9C9;
        margin-bottom: 0.55rem;
    }

    .panel {
        background: #171A21;
        border: 1px solid #2B3240;
        border-radius: 12px;
        padding: 0.85rem 1rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.18);
    }

    .panel-title {
        color: #F2F5FA;
        font-size: 1rem;
        font-weight: 600;
        margin-bottom: 0.65rem;
    }

    .metric-card {
        background: #171A21;
        border: 1px solid #2B3240;
        border-radius: 12px;
        padding: 0.8rem 1rem;
        text-align: center;
        margin-bottom: 0.8rem;
    }

    .metric-label {
        color: #9EABC0;
        font-size: 0.82rem;
        margin-bottom: 0.35rem;
    }

    .metric-value {
        color: #FFFFFF;
        font-size: 2rem;
        font-weight: 700;
        line-height: 1.05;
    }

    .metric-small {
        color: #C7D0DD;
        font-size: 0.8rem;
        margin-top: 0.25rem;
    }

    .legend-item {
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 6px;
        font-size: 0.9rem;
        color: #D8E0EA;
    }

    .legend-box {
        width: 14px;
        height: 14px;
        border-radius: 3px;
        border: 1px solid rgba(255,255,255,0.15);
    }

    .small-note {
        color: #99A7BC;
        font-size: 0.82rem;
        line-height: 1.45;
    }

    .method-note {
        background: #171A21;
        border: 1px solid #2B3240;
        border-left: 4px solid #4DA3FF;
        border-radius: 10px;
        padding: 0.8rem 1rem;
        color: #C7D0DD;
        font-size: 0.86rem;
        line-height: 1.55;
        margin-top: 0.8rem;
    }

    .filter-wrap {
        background: #171A21;
        border: 1px solid #2B3240;
        border-radius: 12px;
        padding: 0.65rem 0.9rem 0.15rem 0.9rem;
        margin-bottom: 0.85rem;
    }

    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div {
        background-color: #12161D !important;
        border-color: #2B3240 !important;
    }

    div[data-baseweb="tag"] {
        background-color: #243042 !important;
    }

    [data-testid="stDateInputField"] {
        background-color: #12161D !important;
    }

    [data-testid="stPlotlyChart"],
    [data-testid="stDeckGlJsonChart"] {
        background: #171A21;
        border: 1px solid #2B3240;
        border-radius: 12px;
        padding: 0.45rem;
    }

    .stDataFrame, div[data-testid="stTable"] {
        background: #171A21;
        border-radius: 12px;
        border: 1px solid #2B3240;
        padding: 0.25rem;
    }
</style>
""", unsafe_allow_html=True)


# =========================================================
# FUNCIONES AUXILIARES
# =========================================================

def limpiar_codigo(valor):
    if pd.isna(valor):
        return np.nan
    s = str(valor).strip().replace("'", "")
    if s.endswith(".0"):
        s = s[:-2]
    return s.strip()


def normalizar_codigo_5(valor):
    if pd.isna(valor):
        return np.nan
    s = limpiar_codigo(valor)
    if s == "":
        return np.nan
    return s.zfill(5) if s.isdigit() else s


def normalizar_texto(valor):
    if pd.isna(valor):
        return ""
    return str(valor).strip()


def normalizar_depto(valor):
    if pd.isna(valor):
        return ""
    txt = str(valor).strip().upper()
    reemplazos = {
        "BOGOTA": "BOGOTÁ",
        "BOGOTA D.C.": "BOGOTÁ, D.C.",
        "BOGOTA, D.C.": "BOGOTÁ, D.C.",
        "BOGOTÁ D.C.": "BOGOTÁ, D.C.",
        "BOGOTA DC": "BOGOTÁ, D.C.",
        "BOGOTÁ DC": "BOGOTÁ, D.C.",
        "BOYACA": "BOYACÁ"
    }
    return reemplazos.get(txt, txt)


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


def moda_numerica(serie):
    s = pd.to_numeric(serie, errors="coerce").dropna()
    if s.empty:
        return np.nan
    modas = s.mode()
    if modas.empty:
        return np.nan
    return float(modas.iloc[0])


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
    municipios = ", ".join(top["MUNICIPIO_ORIGEN"].tolist())
    ventaja_min = top["ventaja_precio_pct"].min()
    ventaja_max = top["ventaja_precio_pct"].max()
    participacion_total = top["participacion_total_pct"].sum()

    if semestre_sel != "Todos":
        periodo_desc = semestre_sel.lower()
    else:
        periodo_desc = f"el rango {fecha_ini.strftime('%Y-%m')} a {fecha_fin.strftime('%Y-%m')}"

    return (
        f"Durante {periodo_desc}, el análisis para {rubro.lower()} hacia {destino_label} "
        f"ubica a {municipios} entre los municipios más eficientes en la relación precio-volumen-estabilidad. "
        f"En este subconjunto, las ventajas de precio observadas oscilan entre {ventaja_min:.1f}% y "
        f"{ventaja_max:.1f}%, y su participación conjunta representa {participacion_total:.1f}% del volumen "
        f"total registrado para este producto bajo los filtros temporales activos."
    )


def construir_sankey(sankey_top):
    nodos_origen = sankey_top["MUNICIPIO_ORIGEN"].astype(str).tolist()
    nodos_destino = sankey_top["CENTRAL_NOMBRE"].astype(str).tolist()
    nodos = list(dict.fromkeys(nodos_origen + nodos_destino))
    idx = {n: i for i, n in enumerate(nodos)}

    valores = sankey_top["toneladas_total"].astype(float).tolist()
    total_sankey = sum(valores) if len(valores) > 0 else 0
    porcentajes = [(v / total_sankey) * 100 if total_sankey > 0 else 0 for v in valores]

    fig = go.Figure(data=[go.Sankey(
        arrangement="snap",
        node=dict(
            pad=12,
            thickness=16,
            line=dict(color="gray", width=0.5),
            label=nodos
        ),
        link=dict(
            source=[idx[s] for s in sankey_top["MUNICIPIO_ORIGEN"].astype(str)],
            target=[idx[t] for t in sankey_top["CENTRAL_NOMBRE"].astype(str)],
            value=valores,
            customdata=porcentajes,
            hovertemplate=(
                "Origen: %{source.label}<br>"
                "Central mayorista: %{target.label}<br>"
                "Toneladas: %{value:,.1f}<br>"
                "Participación: %{customdata:.1f}%<extra></extra>"
            )
        )
    )])

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#171A21",
        plot_bgcolor="#171A21",
        font=dict(color="#E8EDF5", size=11),
        margin=dict(l=10, r=10, t=10, b=10),
        height=430
    )
    return fig


def obtener_mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except FileNotFoundError:
        return 0.0


# =========================================================
# CARGA DE DATOS
# =========================================================

@st.cache_data(show_spinner=False)
def cargar_datos(mtime_lineas, mtime_municipios, mtime_destino, mtime_origen):
    lineas = pd.read_parquet(RUTA_LINEAS)
    municipios = gpd.read_parquet(RUTA_MUNICIPIOS)
    puntos_destino = gpd.read_file(RUTA_PUNTOS_DESTINO)
    puntos_origen = gpd.read_file(RUTA_PUNTOS_ORIGEN)
    return lineas, municipios, puntos_destino, puntos_origen


lineas, municipios, puntos_destino, puntos_origen = cargar_datos(
    obtener_mtime(RUTA_LINEAS),
    obtener_mtime(RUTA_MUNICIPIOS),
    obtener_mtime(RUTA_PUNTOS_DESTINO),
    obtener_mtime(RUTA_PUNTOS_ORIGEN),
)


# =========================================================
# LIMPIEZA Y PREPARACIÓN
# =========================================================

lineas["FECHA"] = pd.to_datetime(lineas["FECHA"], errors="coerce")
puntos_origen["FECHA"] = pd.to_datetime(puntos_origen["FECHA"], errors="coerce")

for col in ["RUBRO", "DESTINO", "NOMBRE_DESTINO", "DEPARTAMENTO_ORIGEN", "MUNICIPIO_ORIGEN", "CENTRAL_NOMBRE"]:
    if col in lineas.columns:
        lineas[col] = lineas[col].apply(normalizar_texto)

for col in ["RUBRO", "DESTINO", "NOMBRE_DESTINO", "DEPARTAMENTO_ORIGEN", "MUNICIPIO_ORIGEN", "CENTRAL_NOMBRE"]:
    if col in puntos_origen.columns:
        puntos_origen[col] = puntos_origen[col].apply(normalizar_texto)

for col in [
    "TONELADAS", "PRECIO_PROMEDIO", "PRECIO_MEDIANA", "DIAS_CON_DATOS",
    "LATITUD_ORIGEN", "LONGITUD_ORIGEN", "LATITUD_DESTINO", "LONGITUD_DESTINO"
]:
    if col in lineas.columns:
        lineas[col] = pd.to_numeric(lineas[col], errors="coerce")

for col in ["TONELADAS", "PRECIO_PROMEDIO", "PRECIO_MEDIANA", "DIAS_CON_DATOS"]:
    if col in puntos_origen.columns:
        puntos_origen[col] = pd.to_numeric(puntos_origen[col], errors="coerce")

lineas["codigo_origen"] = lineas["CODIGO_DIVIPOLA_MUN_ORIGEN_LIMPIO"].apply(normalizar_codigo_5)
lineas["codigo_destino"] = lineas["CODIGO_DIVIPOLA_MUN_DESTINO_LIMPIO"].apply(normalizar_codigo_5)
lineas["DEPARTAMENTO_ORIGEN_NORM"] = lineas["DEPARTAMENTO_ORIGEN"].apply(normalizar_depto)

if "MpCodigo" in municipios.columns:
    municipios["codigo_origen"] = municipios["MpCodigo"].apply(normalizar_codigo_5)
elif "CODIGO_MUNICIPIO" in municipios.columns:
    municipios["codigo_origen"] = municipios["CODIGO_MUNICIPIO"].apply(normalizar_codigo_5)
else:
    raise ValueError("La capa de municipios no tiene MpCodigo ni CODIGO_MUNICIPIO.")

if "Nombre" in municipios.columns:
    municipios["nombre_municipio"] = municipios["Nombre"].apply(normalizar_texto)
elif "MpNombre" in municipios.columns:
    municipios["nombre_municipio"] = municipios["MpNombre"].apply(normalizar_texto)
elif "MUNICIPIO" in municipios.columns:
    municipios["nombre_municipio"] = municipios["MUNICIPIO"].apply(normalizar_texto)
elif "NOMBRE_MUNICIPIO" in municipios.columns:
    municipios["nombre_municipio"] = municipios["NOMBRE_MUNICIPIO"].apply(normalizar_texto)
elif "NOMBRE_MPIO" in municipios.columns:
    municipios["nombre_municipio"] = municipios["NOMBRE_MPIO"].apply(normalizar_texto)
elif "NOM_MUN" in municipios.columns:
    municipios["nombre_municipio"] = municipios["NOM_MUN"].apply(normalizar_texto)
elif "municipio" in municipios.columns:
    municipios["nombre_municipio"] = municipios["municipio"].apply(normalizar_texto)
elif "nombre" in municipios.columns:
    municipios["nombre_municipio"] = municipios["nombre"].apply(normalizar_texto)
else:
    municipios["nombre_municipio"] = ""

if "Depto" in municipios.columns:
    municipios["departamento"] = municipios["Depto"].apply(normalizar_texto)
elif "DEPARTAMENTO" in municipios.columns:
    municipios["departamento"] = municipios["DEPARTAMENTO"].apply(normalizar_texto)
elif "NOMBRE_DPT" in municipios.columns:
    municipios["departamento"] = municipios["NOMBRE_DPT"].apply(normalizar_texto)
elif "NOM_DEP" in municipios.columns:
    municipios["departamento"] = municipios["NOM_DEP"].apply(normalizar_texto)
else:
    municipios["departamento"] = ""

puntos_destino["codigo_destino"] = puntos_destino["CODIGO_MUNICIPIO"].apply(normalizar_codigo_5)
puntos_destino["NOMBRE_CENTRAL"] = puntos_destino["NOMBRE_CENTRAL"].apply(normalizar_texto)
puntos_destino["CIUDAD"] = puntos_destino["CIUDAD"].apply(normalizar_texto)

puntos_origen["codigo_origen"] = puntos_origen["CODIGO_DIVIPOLA_MUN_ORIGEN_LIMPIO"].apply(normalizar_codigo_5)
puntos_origen["codigo_destino"] = puntos_origen["CODIGO_DIVIPOLA_MUN_DESTINO_LIMPIO"].apply(normalizar_codigo_5)
puntos_origen["DEPARTAMENTO_ORIGEN_NORM"] = puntos_origen["DEPARTAMENTO_ORIGEN"].apply(normalizar_depto)

codigos_validos = set(municipios["codigo_origen"].dropna().unique())

lineas = lineas[lineas["codigo_origen"].isin(codigos_validos)].copy()
puntos_origen = puntos_origen[puntos_origen["codigo_origen"].isin(codigos_validos)].copy()

lineas["periodo_mes"] = lineas["FECHA"].dt.to_period("M").dt.to_timestamp()
lineas["etiqueta_mes"] = lineas["FECHA"].dt.strftime("%Y-%m")
lineas["semestre"] = np.where(lineas["MES"].isin([1, 2, 3, 4, 5, 6]), "Primer semestre", "Segundo semestre")

puntos_origen["periodo_mes"] = puntos_origen["FECHA"].dt.to_period("M").dt.to_timestamp()
puntos_origen["etiqueta_mes"] = puntos_origen["FECHA"].dt.strftime("%Y-%m")
puntos_origen["semestre"] = np.where(
    puntos_origen["MES"].isin([1, 2, 3, 4, 5, 6]),
    "Primer semestre",
    "Segundo semestre"
)


# =========================================================
# ENCABEZADO
# =========================================================

st.markdown('<div class="top-title">Visor de precios y abastecimiento agroalimentario</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="top-subtitle">Lectura territorial de precios, flujos y eficiencia relativa de municipios de origen por producto y central mayorista.</div>',
    unsafe_allow_html=True
)


# =========================================================
# FILTROS SUPERIORES
# =========================================================

st.markdown('<div class="filter-wrap">', unsafe_allow_html=True)

f1, f2, f3, f4, f5, f6, f7 = st.columns([1.2, 1.5, 1.1, 1.0, 1.2, 0.9, 0.9])

rubros = sorted(lineas["RUBRO"].dropna().unique().tolist())
centrales = sorted(lineas["CENTRAL_NOMBRE"].dropna().unique().tolist())
deptos = sorted(lineas["DEPARTAMENTO_ORIGEN"].dropna().unique().tolist())

with f1:
    rubro_sel = st.selectbox("Rubro", rubros, index=0)

with f2:
    centrales_sel = st.multiselect(
        "Central mayorista",
        options=centrales,
        default=[]
    )

with f3:
    semestre_sel = st.selectbox(
        "Periodo semestral",
        ["Todos", "Primer semestre", "Segundo semestre"],
        index=0
    )

with f4:
    deptos_sel = st.multiselect("Departamento origen", deptos, default=[])

with f5:
    fecha_min = lineas["FECHA"].min().date()
    fecha_max = lineas["FECHA"].max().date()
    rango = st.date_input(
        "Periodo",
        value=(fecha_min, fecha_max),
        min_value=fecha_min,
        max_value=fecha_max
    )

with f6:
    max_lineas = st.slider("Top flujos", 100, MAX_LINEAS_MAPA_MAX, MAX_LINEAS_MAPA_DEFAULT, 100)

with f7:
    max_filas_tabla = st.slider("Filas tabla", 50, 1000, MAX_FILAS_TABLA_DEFAULT, 50)

st.markdown("</div>", unsafe_allow_html=True)

if isinstance(rango, tuple) and len(rango) == 2:
    fecha_ini, fecha_fin = rango
else:
    fecha_ini, fecha_fin = fecha_min, fecha_max


# =========================================================
# UNIVERSOS DE CÁLCULO
# =========================================================

df_total = lineas[lineas["RUBRO"] == rubro_sel].copy()
df_total = df_total[(df_total["FECHA"].dt.date >= fecha_ini) & (df_total["FECHA"].dt.date <= fecha_fin)].copy()

if semestre_sel != "Todos":
    df_total = df_total[df_total["semestre"] == semestre_sel].copy()

df_rape_base = df_total[df_total["DEPARTAMENTO_ORIGEN_NORM"].isin(DEPTOS_RAPE)].copy()

df = df_total.copy()

if centrales_sel:
    df = df[df["CENTRAL_NOMBRE"].isin(centrales_sel)].copy()

if deptos_sel:
    df = df[df["DEPARTAMENTO_ORIGEN"].isin(deptos_sel)].copy()

if centrales_sel:
    destino_label = ", ".join(centrales_sel)
else:
    destino_label = "todas las centrales mayoristas seleccionadas"

puntos_origen_f = puntos_origen.copy()
puntos_origen_f = puntos_origen_f[
    (puntos_origen_f["FECHA"].dt.date >= fecha_ini) &
    (puntos_origen_f["FECHA"].dt.date <= fecha_fin)
].copy()

puntos_origen_f = puntos_origen_f[puntos_origen_f["RUBRO"] == rubro_sel].copy()

if semestre_sel != "Todos":
    puntos_origen_f = puntos_origen_f[puntos_origen_f["semestre"] == semestre_sel].copy()

if centrales_sel:
    puntos_origen_f = puntos_origen_f[puntos_origen_f["CENTRAL_NOMBRE"].isin(centrales_sel)].copy()

if deptos_sel:
    puntos_origen_f = puntos_origen_f[puntos_origen_f["DEPARTAMENTO_ORIGEN"].isin(deptos_sel)].copy()

if not puntos_origen_f.empty:
    puntos_origen_agg = (
        puntos_origen_f.groupby(
            ["codigo_origen", "MUNICIPIO_ORIGEN", "DEPARTAMENTO_ORIGEN"],
            as_index=False
        )
        .agg(
            TONELADAS=("TONELADAS", "sum"),
            PRECIO_PROMEDIO=("PRECIO_PROMEDIO", "mean"),
            geometry=("geometry", "first")
        )
    )

    puntos_origen_agg = gpd.GeoDataFrame(
        puntos_origen_agg,
        geometry="geometry",
        crs=puntos_origen_f.crs
    )
else:
    puntos_origen_agg = gpd.GeoDataFrame(
        columns=["codigo_origen", "MUNICIPIO_ORIGEN", "DEPARTAMENTO_ORIGEN", "TONELADAS", "PRECIO_PROMEDIO", "geometry"],
        geometry="geometry",
        crs=puntos_origen.crs
    )


# =========================================================
# AGREGACIONES PRINCIPALES
# =========================================================

if not df.empty:
    precio_ref = df["PRECIO_PROMEDIO"].mean()
    volumen_total_filtro = df["TONELADAS"].sum()
    volumen_total_total = df_total["TONELADAS"].sum()
    volumen_total_rape = df_rape_base["TONELADAS"].sum()

    municipios_activos = df["codigo_origen"].nunique()
    centrales_activas = df["codigo_destino"].nunique()

    ranking = (
        df.groupby(
            ["codigo_origen", "MUNICIPIO_ORIGEN", "DEPARTAMENTO_ORIGEN"],
            as_index=False
        )
        .agg(
            toneladas_total=("TONELADAS", "sum"),
            precio_promedio=("PRECIO_PROMEDIO", "mean"),
            precio_moda=("PRECIO_PROMEDIO", moda_numerica),
            dias_con_datos=("DIAS_CON_DATOS", "sum"),
            meses_participacion=("periodo_mes", "nunique")
        )
    )

    recursos_por_municipio = (
        df.assign(valor_mensual_aprox=df["PRECIO_PROMEDIO"] * df["TONELADAS"])
        .groupby("codigo_origen", as_index=False)
        .agg(recursos_movilizados_aprox=("valor_mensual_aprox", "sum"))
    )

    ranking = ranking.merge(
        recursos_por_municipio,
        on="codigo_origen",
        how="left"
    )

    total_meses = max(df["periodo_mes"].nunique(), 1)

    ranking["participacion_filtro_pct"] = np.where(
        volumen_total_filtro > 0,
        (ranking["toneladas_total"] / volumen_total_filtro) * 100,
        0
    )

    ranking["participacion_total_pct"] = np.where(
        volumen_total_total > 0,
        (ranking["toneladas_total"] / volumen_total_total) * 100,
        0
    )

    ranking["participacion_rape_pct"] = np.where(
        volumen_total_rape > 0,
        (ranking["toneladas_total"] / volumen_total_rape) * 100,
        0
    )

    ranking["ventaja_precio_pct"] = np.where(
        precio_ref > 0,
        ((precio_ref - ranking["precio_promedio"]) / precio_ref) * 100,
        0
    )

    ranking["frecuencia_relativa"] = ranking["meses_participacion"] / total_meses

    ranking["score_precio"] = norm_serie(ranking["ventaja_precio_pct"].clip(lower=0))
    ranking["score_volumen"] = norm_serie(ranking["toneladas_total"])
    ranking["score_actividad"] = norm_serie(ranking["meses_participacion"])

    ranking["indice_eficiencia"] = (
        ranking["score_precio"] * 0.40 +
        ranking["score_volumen"] * 0.40 +
        ranking["score_actividad"] * 0.20
    )

    ranking["categoria_eficiencia"] = ranking["indice_eficiencia"].apply(clasificar_eficiencia)

    ranking = ranking.sort_values(
        ["indice_eficiencia", "toneladas_total", "ventaja_precio_pct"],
        ascending=[False, False, False]
    ).reset_index(drop=True)

    ranking["ranking"] = ranking.index + 1

    municipios_eficientes = ranking.copy()

    top30_codigos = set(
        ranking.sort_values("toneladas_total", ascending=False)
        .head(30)["codigo_origen"]
        .astype(str)
        .tolist()
    )

    serie_mensual = (
        df.groupby("periodo_mes", as_index=False)
        .agg(
            precio_promedio=("PRECIO_PROMEDIO", "mean"),
            toneladas_total=("TONELADAS", "sum")
        )
        .sort_values("periodo_mes")
    )
    serie_mensual["etiqueta_mes"] = serie_mensual["periodo_mes"].dt.strftime("%Y-%m")

    flujos_mapa = (
        df.groupby(
            [
                "codigo_origen", "MUNICIPIO_ORIGEN", "DEPARTAMENTO_ORIGEN",
                "CENTRAL_NOMBRE", "codigo_destino",
                "LONGITUD_ORIGEN", "LATITUD_ORIGEN",
                "LONGITUD_DESTINO", "LATITUD_DESTINO"
            ],
            as_index=False
        )
        .agg(
            toneladas_total=("TONELADAS", "sum"),
            precio_promedio=("PRECIO_PROMEDIO", "mean")
        )
        .sort_values("toneladas_total", ascending=False)
        .head(max_lineas)
        .copy()
    )

    flujos_mapa = flujos_mapa.dropna(
        subset=["LONGITUD_ORIGEN", "LATITUD_ORIGEN", "LONGITUD_DESTINO", "LATITUD_DESTINO"]
    ).copy()

    if not flujos_mapa.empty:
        vmin = flujos_mapa["toneladas_total"].min()
        vmax = flujos_mapa["toneladas_total"].max()

        if vmax > vmin:
            flujos_mapa["ancho_linea"] = 2 + 10 * ((flujos_mapa["toneladas_total"] - vmin) / (vmax - vmin))
        else:
            flujos_mapa["ancho_linea"] = 4

        flujos_mapa["toneladas_fmt"] = flujos_mapa["toneladas_total"].map(formatear_ton)
        flujos_mapa["precio_fmt"] = flujos_mapa["precio_promedio"].map(formatear_cop)

    sankey_base = (
        df.groupby(["MUNICIPIO_ORIGEN", "CENTRAL_NOMBRE"], as_index=False)
        .agg(toneladas_total=("TONELADAS", "sum"))
        .sort_values("toneladas_total", ascending=False)
    )

    top_municipios_sankey = (
        df.groupby(["MUNICIPIO_ORIGEN"], as_index=False)
        .agg(toneladas_total=("TONELADAS", "sum"))
        .sort_values("toneladas_total", ascending=False)
        .head(12)["MUNICIPIO_ORIGEN"]
        .tolist()
    )

    sankey_top = sankey_base[sankey_base["MUNICIPIO_ORIGEN"].isin(top_municipios_sankey)].copy()

else:
    precio_ref = 0
    volumen_total_filtro = 0
    volumen_total_total = 0
    volumen_total_rape = 0
    municipios_activos = 0
    centrales_activas = 0
    ranking = pd.DataFrame()
    municipios_eficientes = pd.DataFrame()
    top30_codigos = set()
    serie_mensual = pd.DataFrame(columns=["periodo_mes", "precio_promedio", "toneladas_total", "etiqueta_mes"])
    flujos_mapa = pd.DataFrame()
    sankey_top = pd.DataFrame(columns=["MUNICIPIO_ORIGEN", "CENTRAL_NOMBRE", "toneladas_total"])


# =========================================================
# MUNICIPIOS PARA MAPA
# =========================================================

municipios_mapa = municipios.copy()

if not ranking.empty:
    muni_merge = ranking[["codigo_origen", "toneladas_total", "categoria_eficiencia", "ranking"]].copy()
else:
    muni_merge = pd.DataFrame(columns=["codigo_origen", "toneladas_total", "categoria_eficiencia", "ranking"])

municipios_mapa = municipios_mapa.merge(muni_merge, on="codigo_origen", how="left")
municipios_mapa["toneladas_total"] = municipios_mapa["toneladas_total"].fillna(0)
municipios_mapa["categoria_eficiencia"] = municipios_mapa["categoria_eficiencia"].fillna("Sin flujo")
municipios_mapa["es_top30"] = municipios_mapa["codigo_origen"].astype(str).isin(top30_codigos)

def fill_color(row):
    if row["es_top30"]:
        return [110, 68, 255, 150]
    if row["categoria_eficiencia"] == "Alta eficiencia":
        return [235, 87, 87, 145]
    if row["categoria_eficiencia"] == "Eficiencia media":
        return [65, 145, 255, 120]
    return [40, 48, 62, 18]

def line_color(row):
    if row["es_top30"]:
        return [170, 130, 255, 240]
    if row["categoria_eficiencia"] == "Alta eficiencia":
        return [255, 120, 120, 220]
    if row["categoria_eficiencia"] == "Eficiencia media":
        return [95, 170, 255, 220]
    return [100, 110, 125, 60]

municipios_mapa["fill_color"] = municipios_mapa.apply(fill_color, axis=1)
municipios_mapa["line_color"] = municipios_mapa.apply(line_color, axis=1)
municipios_mapa["codigo_txt"] = municipios_mapa["codigo_origen"].fillna("")


# =========================================================
# CAMPOS UNIFICADOS PARA TOOLTIP GLOBAL
# =========================================================

municipios_web = municipios_mapa[
    ["nombre_municipio", "departamento", "codigo_txt", "fill_color", "line_color", "geometry"]
].copy()

municipios_web["nombre_municipio"] = municipios_web["nombre_municipio"].fillna("").astype(str).str.strip()
municipios_web["departamento"] = municipios_web["departamento"].fillna("").astype(str).str.strip()
municipios_web["codigo_txt"] = municipios_web["codigo_txt"].fillna("").astype(str).str.strip()

municipios_web["nombre_tooltip"] = np.where(
    municipios_web["nombre_municipio"] != "",
    municipios_web["nombre_municipio"],
    "Sin nombre disponible"
)

municipios_web["departamento_tooltip"] = np.where(
    municipios_web["departamento"] != "",
    municipios_web["departamento"],
    "Sin dato"
)

municipios_web["codigo_tooltip"] = np.where(
    municipios_web["codigo_txt"] != "",
    municipios_web["codigo_txt"],
    "Sin código"
)

municipios_web["tipo_elemento"] = "Municipio"
municipios_web["detalle_1"] = "Nombre: " + municipios_web["nombre_tooltip"]
municipios_web["detalle_2"] = "Departamento: " + municipios_web["departamento_tooltip"]
municipios_web["detalle_3"] = "Código: " + municipios_web["codigo_tooltip"]
municipios_web["detalle_4"] = ""

if not flujos_mapa.empty:
    flujos_mapa["tipo_elemento"] = "Flujo OD"
    flujos_mapa["detalle_1"] = "Origen: " + flujos_mapa["MUNICIPIO_ORIGEN"].fillna("")
    flujos_mapa["detalle_2"] = "Central mayorista: " + flujos_mapa["CENTRAL_NOMBRE"].fillna("")
    flujos_mapa["detalle_3"] = "Precio promedio: " + flujos_mapa["precio_fmt"].fillna("")
    flujos_mapa["detalle_4"] = "Toneladas: " + flujos_mapa["toneladas_fmt"].fillna("")

if not puntos_origen_agg.empty:
    puntos_origen_agg["lon"] = puntos_origen_agg["geometry"].apply(lambda g: g.x if g is not None else np.nan)
    puntos_origen_agg["lat"] = puntos_origen_agg["geometry"].apply(lambda g: g.y if g is not None else np.nan)
    puntos_origen_agg["toneladas_fmt"] = puntos_origen_agg["TONELADAS"].map(formatear_ton)
    puntos_origen_agg["precio_fmt"] = puntos_origen_agg["PRECIO_PROMEDIO"].map(formatear_cop)
    puntos_origen_agg["tipo_elemento"] = "Nodo de origen"
    puntos_origen_agg["detalle_1"] = "Origen: " + puntos_origen_agg["MUNICIPIO_ORIGEN"].fillna("")
    puntos_origen_agg["detalle_2"] = "Departamento: " + puntos_origen_agg["DEPARTAMENTO_ORIGEN"].fillna("")
    puntos_origen_agg["detalle_3"] = "Precio promedio: " + puntos_origen_agg["precio_fmt"].fillna("")
    puntos_origen_agg["detalle_4"] = "Toneladas: " + puntos_origen_agg["toneladas_fmt"].fillna("")

if centrales_sel:
    puntos_destino_f = puntos_destino[puntos_destino["NOMBRE_CENTRAL"].isin(centrales_sel)].copy()
else:
    puntos_destino_f = puntos_destino.copy()

if not puntos_destino_f.empty:
    puntos_destino_f["tipo_elemento"] = "Central mayorista"
    puntos_destino_f["detalle_1"] = "Nombre: " + puntos_destino_f["NOMBRE_CENTRAL"].fillna("")
    puntos_destino_f["detalle_2"] = "Ciudad: " + puntos_destino_f["CIUDAD"].fillna("")
    puntos_destino_f["detalle_3"] = "Código municipio: " + puntos_destino_f["codigo_destino"].fillna("")
    puntos_destino_f["detalle_4"] = ""


# =========================================================
# ENCABEZADOS AMIGABLES
# =========================================================

NOMBRES_COLUMNAS_PRESENTABLES = {
    "ranking": "Ranking",
    "municipio_origen": "Municipio origen",
    "departamento_origen": "Departamento origen",
    "precio_promedio_municipio": "Precio promedio",
    "precio_moda": "Precio moda",
    "toneladas_total": "Toneladas acumuladas",
    "meses_participacion": "Meses activos",
    "participacion_filtro_pct": "Participación en filtro",
    "participacion_total_pct": "Participación total",
    "participacion_rape_pct": "Participación RAPE",
    "ventaja_precio_pct": "Ventaja precio",
    "indice_eficiencia": "Índice de eficiencia",
    "recursos_movilizados_aprox": "Recursos movilizados aprox."
}


# =========================================================
# LAYOUT PRINCIPAL
# =========================================================

left_col, center_col, right_col = st.columns([1.05, 3.8, 1.45], gap="small")


# =========================================================
# COLUMNA IZQUIERDA
# =========================================================

with left_col:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Indicadores principales</div>', unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Toneladas abastecidas</div>
            <div class="metric-value">{volumen_total_filtro:,.0f}</div>
            <div class="metric-small">Periodo filtrado</div>
        </div>

        <div class="metric-card">
            <div class="metric-label">Precio promedio</div>
            <div class="metric-value" style="font-size:1.65rem;">{formatear_cop(precio_ref)}</div>
            <div class="metric-small">Mercado filtrado</div>
        </div>

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
        """,
        unsafe_allow_html=True
    )

    st.markdown('<div class="panel-title">Leyenda</div>', unsafe_allow_html=True)
    st.markdown("""
        <div class="legend-item"><span class="legend-box" style="background:#6E44FF;"></span>Top 30 abastecedores</div>
        <div class="legend-item"><span class="legend-box" style="background:#EB5757;"></span>Alta eficiencia</div>
        <div class="legend-item"><span class="legend-box" style="background:#4191FF;"></span>Eficiencia media</div>
        <div class="legend-item"><span class="legend-box" style="background:#F5B041;"></span>Arcos de flujo OD</div>
        <div class="legend-item"><span class="legend-box" style="background:#00D2FF;"></span>Central mayorista</div>
    """, unsafe_allow_html=True)

    st.markdown(
        """
        <div class="small-note">
            Los municipios morados corresponden a los 30 principales abastecedores del filtro actual.
            Cambian cuando cambian rubro, fechas, semestre, central mayorista o departamento.
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown('</div>', unsafe_allow_html=True)


# =========================================================
# COLUMNA CENTRAL
# =========================================================

with center_col:
    st.markdown('<div class="panel-title">Mapa de flujos de abastecimiento</div>', unsafe_allow_html=True)

    layers = []

    municipios_web_reducido = municipios_web[
        ["fill_color", "line_color", "tipo_elemento", "detalle_1", "detalle_2", "detalle_3", "detalle_4", "geometry"]
    ].copy()

    geojson_municipios = json.loads(municipios_web_reducido.to_json())

    layers.append(
        pdk.Layer(
            "GeoJsonLayer",
            data=geojson_municipios,
            stroked=True,
            filled=True,
            extruded=False,
            wireframe=False,
            get_fill_color="properties.fill_color",
            get_line_color="properties.line_color",
            line_width_min_pixels=1.0,
            pickable=True,
            auto_highlight=True
        )
    )

    if not flujos_mapa.empty:
        layers.append(
            pdk.Layer(
                "ArcLayer",
                data=flujos_mapa,
                get_source_position=["LONGITUD_ORIGEN", "LATITUD_ORIGEN"],
                get_target_position=["LONGITUD_DESTINO", "LATITUD_DESTINO"],
                get_source_color=[245, 176, 65, 190],
                get_target_color=[0, 210, 255, 190],
                get_width="ancho_linea",
                width_scale=1,
                width_min_pixels=1,
                pickable=True,
                auto_highlight=True
            )
        )

    if not puntos_origen_agg.empty:
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                data=puntos_origen_agg,
                get_position="[lon, lat]",
                get_radius=4200,
                get_fill_color=[255, 160, 0, 120],
                get_line_color=[255, 210, 120, 200],
                line_width_min_pixels=1,
                pickable=True
            )
        )

    if not puntos_destino_f.empty:
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                data=puntos_destino_f,
                get_position="[LONGITUD, LATITUD]",
                get_radius=13500,
                get_fill_color=[0, 210, 255, 190],
                get_line_color=[170, 245, 255, 255],
                line_width_min_pixels=2,
                pickable=True
            )
        )

    deck = pdk.Deck(
        layers=layers,
        initial_view_state=pdk.ViewState(
            latitude=4.5,
            longitude=-74.1,
            zoom=4.6,
            pitch=0
        ),
        tooltip={
            "html": """
                <b>{tipo_elemento}</b><br/>
                {detalle_1}<br/>
                {detalle_2}<br/>
                {detalle_3}<br/>
                {detalle_4}
            """,
            "style": {
                "backgroundColor": "rgba(18,22,29,0.95)",
                "color": "#F5F7FA",
                "fontSize": "12px"
            }
        },
        map_style="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json"
    )

    st.pydeck_chart(deck, use_container_width=True)

    st.markdown('<div class="panel-title" style="margin-top:0.65rem;">Serie mensual de precio y toneladas</div>', unsafe_allow_html=True)

    if not serie_mensual.empty:
        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=serie_mensual["etiqueta_mes"],
                y=serie_mensual["precio_promedio"],
                name="Precio promedio",
                marker_color="#4DA3FF",
                yaxis="y1",
                hovertemplate="Mes: %{x}<br>Precio: $%{y:,.0f}<extra></extra>"
            )
        )

        fig.add_trace(
            go.Scatter(
                x=serie_mensual["etiqueta_mes"],
                y=serie_mensual["toneladas_total"],
                name="Toneladas",
                mode="lines+markers",
                line=dict(color="#F5B041", width=2.5),
                marker=dict(size=6, color="#F5B041"),
                yaxis="y2",
                hovertemplate="Mes: %{x}<br>Toneladas: %{y:,.1f}<extra></extra>"
            )
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#171A21",
            plot_bgcolor="#171A21",
            margin=dict(l=15, r=15, t=10, b=10),
            height=300,
            legend=dict(orientation="h", y=1.08, x=0),
            xaxis=dict(showgrid=False),
            yaxis=dict(title="Precio", gridcolor="#2B3240"),
            yaxis2=dict(title="Toneladas", overlaying="y", side="right", showgrid=False)
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No hay datos para la serie mensual con los filtros actuales.")


# =========================================================
# COLUMNA DERECHA
# =========================================================

with right_col:
    st.markdown('<div class="panel">', unsafe_allow_html=True)

    st.markdown('<div class="panel-title">Sankey origen - central mayorista</div>', unsafe_allow_html=True)

    if not sankey_top.empty:
        sankey_fig = construir_sankey(sankey_top)
        st.plotly_chart(sankey_fig, use_container_width=True)
    else:
        st.info("No hay datos suficientes para el Sankey.")

    st.markdown('<div class="panel-title" style="margin-top:0.9rem;">Contexto interpretativo</div>', unsafe_allow_html=True)

    if not municipios_eficientes.empty:
        texto_resumen = construir_texto_contexto(
            municipios_eficientes,
            rubro_sel,
            destino_label,
            fecha_ini,
            fecha_fin,
            semestre_sel
        )
        st.markdown(
            f"<div class='small-note'>{texto_resumen}</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            "<div class='small-note'>No se identificaron municipios con información suficiente para destacar eficiencia relativa en este filtro.</div>",
            unsafe_allow_html=True
        )

    st.markdown('</div>', unsafe_allow_html=True)


# =========================================================
# TABLA CONSOLIDADA A TODO EL ANCHO
# =========================================================

st.markdown('<div class="panel-title" style="margin-top:0.8rem;">Tabla consolidada de análisis</div>', unsafe_allow_html=True)

if not ranking.empty:
    tabla_consolidada = ranking[
        [
            "ranking",
            "MUNICIPIO_ORIGEN",
            "DEPARTAMENTO_ORIGEN",
            "precio_promedio",
            "precio_moda",
            "toneladas_total",
            "recursos_movilizados_aprox",
            "meses_participacion",
            "participacion_filtro_pct",
            "participacion_total_pct",
            "participacion_rape_pct",
            "ventaja_precio_pct",
            "indice_eficiencia"
        ]
    ].copy()

    tabla_consolidada = tabla_consolidada.rename(columns={
        "MUNICIPIO_ORIGEN": "municipio_origen",
        "DEPARTAMENTO_ORIGEN": "departamento_origen",
        "precio_promedio": "precio_promedio_municipio",
        "precio_moda": "precio_moda"
    })

    tabla_consolidada["precio_promedio_municipio"] = tabla_consolidada["precio_promedio_municipio"].map(
        lambda x: f"$ {x:,.0f} COP"
    )
    tabla_consolidada["precio_moda"] = tabla_consolidada["precio_moda"].map(
        lambda x: f"$ {x:,.0f} COP" if pd.notna(x) else "Sin dato"
    )
    tabla_consolidada["toneladas_total"] = tabla_consolidada["toneladas_total"].map(
        lambda x: f"{x:,.1f}"
    )
    tabla_consolidada["recursos_movilizados_aprox"] = tabla_consolidada["recursos_movilizados_aprox"].map(
        lambda x: f"$ {x:,.0f} COP" if pd.notna(x) else "Sin dato"
    )
    tabla_consolidada["participacion_filtro_pct"] = tabla_consolidada["participacion_filtro_pct"].map(
        lambda x: f"{x:.1f}%"
    )
    tabla_consolidada["participacion_total_pct"] = tabla_consolidada["participacion_total_pct"].map(
        lambda x: f"{x:.1f}%"
    )
    tabla_consolidada["participacion_rape_pct"] = tabla_consolidada["participacion_rape_pct"].map(
        lambda x: f"{x:.1f}%"
    )
    tabla_consolidada["ventaja_precio_pct"] = tabla_consolidada["ventaja_precio_pct"].map(
        lambda x: f"{x:.1f}%"
    )
    tabla_consolidada["indice_eficiencia"] = tabla_consolidada["indice_eficiencia"].map(
        lambda x: f"{x:.2f}"
    )

    tabla_consolidada = tabla_consolidada.rename(columns=NOMBRES_COLUMNAS_PRESENTABLES)
    tabla_consolidada = tabla_consolidada.head(max_filas_tabla)

    st.dataframe(
        tabla_consolidada,
        use_container_width=True,
        hide_index=True,
        height=420
    )
else:
    st.info("No hay información disponible para la tabla consolidada.")


# =========================================================
# NOTA METODOLÓGICA A TODO EL ANCHO
# =========================================================

st.markdown(
    """
    <div class="method-note">
        <b>Cómo se calcula el índice de eficiencia:</b><br>
        El índice compara municipios de origen únicamente dentro del subconjunto filtrado por rubro,
        central mayorista, periodo y demás filtros activos. Combina tres dimensiones normalizadas:
        ventaja de precio frente al promedio del mercado filtrado, volumen acumulado abastecido y número
        de meses con participación. La ponderación usada es 40% precio, 40% volumen y 20% estabilidad operativa.
        <br>
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# PIE
# =========================================================

st.markdown(
    """
    <div style="
        margin-top:0.8rem;
        padding-top:0.6rem;
        border-top:1px solid #2B3240;
        color:#8FA0B7;
        font-size:0.8rem;
        text-align:center;
    ">
        Fuente de información: Sistema de Información de Precios y Abastecimiento del Sector Agropecuario (SIPSA) del DANE, periodo 2020 - 2024.
    </div>
    """,
    unsafe_allow_html=True
)