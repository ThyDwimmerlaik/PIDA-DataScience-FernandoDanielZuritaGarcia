"""
utils/pipeline.py
=================
Pipeline de feature engineering para DistribuData — 2PD.

Reproduce fielmente la Sección 3 del PIDA para evitar training-serving skew.

Uso desde la app Streamlit:
    from utils.pipeline import cargar_artefactos, cargar_datos
    from utils.pipeline import construir_grilla_semanal, agregar_features
    from utils.pipeline import preparar_X, predecir_proximas_semanas

Variables de entorno opcionales (solo si usar_mysql=True):
  DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD, DB_CONNECT_TIMEOUT
"""

from __future__ import annotations

import streamlit as st
import json
import logging
import os
import pickle
import time
from datetime import datetime, timezone
from functools import lru_cache
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import pymysql
import pymysql.cursors

# ── Logging ───────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES DEL PIPELINE
# Deben coincidir exactamente con la Sección 3 del notebook.
# ─────────────────────────────────────────────────────────────────────────────

SEED         = 55
MIN_SEMANAS  = 12        # umbral mínimo de semanas de historial por tipo
LAGS         = [1, 2, 4, 8]
VENTANAS     = [4, 8]    # ventanas de rolling stats incluidas en el modelo

# Tipos con historial suficiente (definidos en Sección 3)
TIPOS_VALIDOS: list[str] = [
    "Aluminio",
    "Aluminio Europeo",
    "AluminioCorte",
    "Herraje",
    "Pelicula",
    "Plastico",
    "Vidrio",
]

# Tipos excluidos por historial insuficiente (documentados, no usados)
TIPOS_EXCLUIDOS: list[str] = ["VidrioCorte", "Domo", "Grabado", "Procesos"]

# Categoría de referencia del OHE (drop='first', orden alfabético)
OHE_REFERENCIA = "Aluminio"

# Columnas OHE generadas (6 columnas)
FEATS_OHE: list[str] = [
    "tipo_Aluminio Europeo",
    "tipo_AluminioCorte",
    "tipo_Herraje",
    "tipo_Pelicula",
    "tipo_Plastico",
    "tipo_Vidrio",
]

# Features numéricas (32) — orden exacto del entrenamiento
FEATS_NUM: list[str] = [
    # Temporalidad numérica y cíclica
    'anio', 'mes', 'trimestre', 'semana_anio',
    'semana_sin', 'semana_cos', 'mes_sin', 'mes_cos',
    'es_temporada_alta', 'es_cierre_anio',
    # Lags
    'lag_1', 'lag_2', 'lag_4', 'lag_8',
    # Rolling
    'rolling_mean_4', 'rolling_std_4',
    'rolling_mean_8', 'rolling_std_8',
    # Contexto demanda
    'n_transacciones_lag1',
    'delta_demanda', 'semanas_sin_ventas',
    # Precio / inventario
    'precio_promedio', 'precio_lista_prom', 'descuento_color',
    'existencia_actual', 'existencia_minima',
    'ratio_existencia', 'flag_stockout', 'margen_precio',
    'venta_lag1', 'venta_desc_lag1'
]

FEATS_ALL: list[str] = FEATS_NUM + FEATS_OHE   # 38 features totales

TARGET     = "demanda_semanal"
TARGET_LOG = "demanda_log"

# Columnas de contexto que se rellenan hacia adelante (no se zeroan en grilla)
COLS_FFILL = [
    "precio_promedio", "precio_lista_prom", "descuento_color",
    "existencia_actual", "existencia_minima",
]

# Columnas que son NaN en semanas sin ventas → deben ser 0 (no hay facturación)
COLS_FILL_CERO_GRILLA = ["demanda_semanal", "n_transacciones", "n_clientes",
                          "venta", "venta_desc"]
COLS_FILL_CERO_FEAT   = ["venta", "venta_desc"]   # también en X antes de predict

# Winsorización: caps calculados en Sección 3 y guardados junto al modelo.
# Si no están disponibles, no se aplica winsorización a datos nuevos.
WINSORIZATION_CAPS: dict[str, dict] = {}

# ── Ambiente ──────────────────────────────────────────────────────────────────
secure_env = st.secrets["SECURE_ENV"] == "True"

# ─────────────────────────────────────────────────────────────────────────────
# CACHÉ DE MÓDULO (warm Lambda invocations reutilizan estos objetos)
# ─────────────────────────────────────────────────────────────────────────────

_model_cache: Optional[object]  = None
_ohe_cache:   Optional[object]  = None

# ─────────────────────────────────────────────────────────────────────────────
# 1. CARGA DE ARTEFACTOS (MODELO Y OHE)
# ─────────────────────────────────────────────────────────────────────────────

def cargar_artefactos(
    ruta_modelo: str = "artifacts/modelo_final_xgb_tunned.json",
    ruta_ohe:    str = "artifacts/ohe_tipo_material.pkl",
) -> tuple[object, object]:
    """
    Carga el modelo XGBoost y el OHE desde rutas locales.

    El modelo puede estar en formato nativo XGBoost (.json / .ubj)
    o serializado con joblib (.pkl). Se detecta automáticamente por extensión.

    Los objetos se cachean en módulo: la primera llamada lee el disco,
    las siguientes devuelven los objetos ya cargados.

    Parameters
    ----------
    ruta_modelo : str
        Ruta al modelo XGBoost tunned (relativa al directorio raíz de la app).
    ruta_ohe    : str
        Ruta al OneHotEncoder serializado con joblib.

    Returns
    -------
    model : XGBRegressor listo para .predict()
    ohe   : OneHotEncoder ajustado sobre tipo_material
    """
    global _model_cache, _ohe_cache

    if _model_cache is not None and _ohe_cache is not None:
        return _model_cache, _ohe_cache

    # Modelo
    logger.info("Cargando modelo desde '%s'", ruta_modelo)
    if ruta_modelo.endswith(".json") or ruta_modelo.endswith(".ubj"):
        import xgboost as xgb
        _model_cache = xgb.XGBRegressor()
        _model_cache.load_model(ruta_modelo)
    else:
        _model_cache = joblib.load(ruta_modelo)
    logger.info("Modelo cargado.")

    # OHE
    logger.info("Cargando OHE desde '%s'", ruta_ohe)
    _ohe_cache = joblib.load(ruta_ohe)
    logger.info("OHE cargado.")

    # Caps de winsorización (opcional, mismo directorio que el modelo)
    ruta_caps = os.path.join(os.path.dirname(ruta_modelo), "artifacts/winsorization_caps.pkl")
    if os.path.exists(ruta_caps):
        caps = joblib.load(ruta_caps)
        WINSORIZATION_CAPS.update(caps)
        logger.info("Caps de winsorización cargados.")

    return _model_cache, _ohe_cache


# ─────────────────────────────────────────────────────────────────────────────
# 2. CARGA DE DATOS TRANSACCIONALES
# ─────────────────────────────────────────────────────────────────────────────

_QUERY_TRANSACCIONAL = """
SELECT
    n.NotaId           AS nota_id,
    n.Fecha            AS fecha,
    t.Nombre           AS tipo_material,
    c.Nombre           AS color_material,
    c.Descuento        AS descuento_color,
    n.Precio           AS precio_venta,
    m.PrecioLista      AS precio_lista,
    m.Existencia       AS existencia_actual,
    m.ExistenciaMin    AS existencia_minima,
    n.ClienteId        AS cliente_id,
    n.Cantidad         AS cantidad
FROM  nota     n
JOIN  material m  ON n.MaterialId = m.MaterialId
JOIN  colores  c  ON m.ColorId    = c.ColorId
JOIN  tipo     t  ON m.TipoId     = t.TipoId
WHERE n.Fecha >= %s
ORDER BY n.Fecha ASC
"""


def _conexion_mysql() -> pymysql.connections.Connection:
    """
    Abre una conexión MySQL con las credenciales del entorno.

    Lambda no mantiene conexiones persistentes entre invocaciones;
    se abre una conexión nueva en cada llamada a cargar_datos().
    Para producción de alto tráfico, considera RDS Proxy.
    """
    return pymysql.connect(
        host            = st.secrets["DB_HOST"],
        port            = int(st.secrets["DB_PORT"]),
        database        = st.secrets["DB_NAME"],
        user            = st.secrets["DB_USER"],
        password        = st.secrets["DB_PASSWORD"],
        connect_timeout = int(st.secrets["DB_CONNECT_TIMEOUT"]),
        read_timeout    = 30,
        write_timeout   = 30,
        charset         = "utf8mb3",
        cursorclass     = pymysql.cursors.DictCursor,
        autocommit      = True,
    )


def cargar_datos(
    desde_fecha: str = "2021-01-01",
    usar_mysql:  bool = True,
    ruta_csv:    str  = "artifacts/dataset.csv",
) -> pd.DataFrame:
    """
    Carga los datos transaccionales desde MySQL o desde un CSV local.

    Parameters
    ----------
    desde_fecha : str
        Fecha ISO mínima (YYYY-MM-DD) para filtrar registros MySQL.
        No aplica cuando usar_mysql=False.
    usar_mysql : bool
        True  → consulta MySQL (requiere variables de entorno DB_*).
        False → carga el CSV local indicado en ruta_csv.
    ruta_csv : str
        Ruta al CSV local, relativa al directorio raíz de la app.
        Solo se usa cuando usar_mysql=False.
    """
    if usar_mysql:
        logger.info("Cargando datos desde MySQL desde %s", desde_fecha)
        conn = _conexion_mysql()
        try:
            with conn.cursor() as cur:
                cur.execute(_QUERY_TRANSACCIONAL, (desde_fecha,))
                rows = cur.fetchall()
            df = pd.DataFrame(rows)
            logger.info("MySQL: %d filas cargadas.", len(df))
        finally:
            conn.close()
    else:
        logger.info("Cargando CSV local: '%s'", ruta_csv)
        df = pd.read_csv(ruta_csv)
        logger.info("CSV: %d filas cargadas.", len(df))

    return _normalizar_raw(df)


def _normalizar_raw(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica los mismos tipos y limpiezas de la Sección 3 al dataframe crudo.
    - Convierte fecha a datetime
    - Convierte columnas numéricas
    - Imputa color_material y precio_lista
    - Elimina registros inválidos (cantidad ≤ 0, precio_venta ≤ 0)
    - Filtra a TIPOS_VALIDOS
    """
    df = df.copy()
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    df = df[df["fecha"].notna()].copy()

    cols_num = [
        "precio_venta", "precio_lista", "existencia_actual",
        "existencia_minima", "descuento_color", "cantidad",
    ]
    for col in cols_num:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Imputar color_material (sección 3.1.3)
    df["color_material"] = df["color_material"].fillna("N/A")

    # Imputar precio_lista con mediana por tipo (sección 3.1.3)
    mediana_tipo = df.groupby("tipo_material")["precio_lista"].transform("median")
    df["precio_lista"] = df["precio_lista"].fillna(mediana_tipo)
    df["precio_lista"] = df["precio_lista"].fillna(df["precio_lista"].median())

    # Eliminar registros inválidos (sección 3.1.2)
    df = df[(df["cantidad"] > 0) & (df["precio_venta"] > 0)].copy()

    # Winsorización (sólo si los caps están disponibles)
    for col, caps in WINSORIZATION_CAPS.items():
        if col in df.columns:
            df[col] = df[col].clip(lower=caps["lower"], upper=caps["upper"])

    # Filtrar a tipos válidos
    df = df[df["tipo_material"].isin(TIPOS_VALIDOS)].copy()

    logger.info(
        "Normalización: %d filas válidas, %d tipos.",
        len(df), df["tipo_material"].nunique(),
    )
    return df.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# 3. CONSTRUCCIÓN DE LA GRILLA SEMANAL
# ─────────────────────────────────────────────────────────────────────────────

def construir_grilla_semanal(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega las transacciones al nivel tipo_material × semana ISO y
    expande la grilla para que todas las semanas existan por tipo
    (semanas sin ventas quedan con demanda = 0).

    Replica exactamente la Sección 3.2.1 del notebook.

    Parameters
    ----------
    df : pd.DataFrame resultado de cargar_datos() / _normalizar_raw()

    Returns
    -------
    pd.DataFrame con una fila por (tipo_material, periodo_semana).
    """
    df = df.copy()

    # Variables de tiempo base
    df["periodo_semana"] = df["fecha"].dt.to_period("W").dt.start_time
    df["anio"]           = df["fecha"].dt.year
    df["mes"]            = df["fecha"].dt.month
    df["trimestre"]      = df["fecha"].dt.quarter
    df["semana_anio"]    = df["fecha"].dt.isocalendar().week.astype(int)

    # Variables de venta (Sección 3.2.1)
    df["venta"]      = df["cantidad"] * df["precio_venta"]
    df["venta_desc"] = df["venta"] - (df["venta"] * df["descuento_color"] / 100)

    agg_dict = {
        "demanda_semanal"   : ("cantidad",         "sum"),
        "n_transacciones"   : ("nota_id",          "count"),
        "n_clientes"        : ("cliente_id",       "nunique"),
        "precio_promedio"   : ("precio_venta",     "mean"),
        "precio_lista_prom" : ("precio_lista",     "mean"),
        "descuento_color"   : ("descuento_color",  "first"),
        "existencia_actual" : ("existencia_actual","mean"),
        "existencia_minima" : ("existencia_minima","mean"),
        "anio"              : ("anio",             "first"),
        "mes"               : ("mes",              "first"),
        "trimestre"         : ("trimestre",        "first"),
        "semana_anio"       : ("semana_anio",      "first"),
        "venta"             : ("venta",            "sum"),
        "venta_desc"        : ("venta_desc",       "sum"),
    }

    df_semana = (
        df.groupby(["tipo_material", "periodo_semana"])
        .agg(**agg_dict)
        .reset_index()
        .sort_values(["tipo_material", "periodo_semana"])
        .reset_index(drop=True)
    )

    # Expandir a grilla completa tipo × semana (sin huecos temporales)
    semanas_globales = pd.date_range(
        start=df_semana["periodo_semana"].min(),
        end=df_semana["periodo_semana"].max(),
        freq="W-MON",
    )
    grilla = (
        pd.MultiIndex.from_product(
            [TIPOS_VALIDOS, semanas_globales],
            names=["tipo_material", "periodo_semana"],
        )
        .to_frame(index=False)
    )
    df_semana = (
        grilla
        .merge(df_semana, on=["tipo_material", "periodo_semana"], how="left")
        .sort_values(["tipo_material", "periodo_semana"])
        .reset_index(drop=True)
    )

    # Semanas sin ventas → cero en demanda y conteos
    df_semana[COLS_FILL_CERO_GRILLA] = (
        df_semana[COLS_FILL_CERO_GRILLA].fillna(0)
    )

    # Contexto de precio/inventario: forward-fill dentro de cada tipo
    df_semana[COLS_FFILL] = (
        df_semana
        .groupby("tipo_material", group_keys=False)[COLS_FFILL]
        .transform(lambda s: s.ffill().bfill())
    )

    # Reconstruir variables temporales desde la fecha canónica de la semana
    df_semana["anio"]        = df_semana["periodo_semana"].dt.year
    df_semana["mes"]         = df_semana["periodo_semana"].dt.month
    df_semana["trimestre"]   = df_semana["periodo_semana"].dt.quarter
    df_semana["semana_anio"] = (
        df_semana["periodo_semana"].dt.isocalendar().week.astype(int)
    )

    logger.info(
        "Grilla semanal: %d filas, %d semanas, %d tipos.",
        len(df_semana),
        df_semana["periodo_semana"].nunique(),
        df_semana["tipo_material"].nunique(),
    )
    return df_semana


# ─────────────────────────────────────────────────────────────────────────────
# 4. INGENIERÍA DE CARACTERÍSTICAS
# ─────────────────────────────────────────────────────────────────────────────

def agregar_features(df_semana: pd.DataFrame) -> pd.DataFrame:
    """
    Añade todas las features del modelo sobre la grilla semanal.
    Replica fielmente las secciones 3.2.3 – 3.2.6 del notebook.

    Requiere que df_semana tenga las columnas producidas por
    construir_grilla_semanal().

    Returns
    -------
    pd.DataFrame con todas las columnas de FEATS_NUM más
    'demanda_log' y 'baseline_ma4'.
    """
    df = df_semana.copy()

    # ── 4.1 Temporalidad cíclica (Sección 3.2.3) ─────────────────────────────
    df["semana_sin"]        = np.sin(2 * np.pi * df["semana_anio"] / 52)
    df["semana_cos"]        = np.cos(2 * np.pi * df["semana_anio"] / 52)
    df["mes_sin"]           = np.sin(2 * np.pi * df["mes"] / 12)
    df["mes_cos"]           = np.cos(2 * np.pi * df["mes"] / 12)
    df["es_temporada_alta"] = df["mes"].isin([4, 5, 6, 7, 8]).astype(int)
    df["es_cierre_anio"]    = df["mes"].isin([11, 12, 1]).astype(int)

    # ── 4.2 Lags (Sección 3.2.4) ─────────────────────────────────────────────
    for lag in LAGS:
        df[f"lag_{lag}"] = (
            df.groupby("tipo_material")["demanda_semanal"].shift(lag)
        )

    # ── 4.3 Rolling stats (Sección 3.2.5) ────────────────────────────────────
    # shift(1) previo: la ventana no incluye la semana actual (evita leakage)
    for w in VENTANAS:
        shifted = df.groupby("tipo_material")["demanda_semanal"].shift(1)
        df[f"rolling_mean_{w}"] = (
            shifted
            .groupby(df["tipo_material"])
            .transform(lambda s: s.rolling(w, min_periods=1).mean())
        )
        df[f"rolling_std_{w}"] = (
            shifted
            .groupby(df["tipo_material"])
            .transform(lambda s: s.rolling(w, min_periods=1).std())
            .fillna(0)
        )

    # Baseline: rolling_mean_4 en escala original
    df["baseline_ma4"] = df["rolling_mean_4"]

    # ── 4.4 Variables de contexto (Sección 3.2.6) ────────────────────────────
    df["ratio_existencia"] = (
        df["existencia_actual"] / (df["existencia_minima"] + 1)
    ).round(4)
    df["flag_stockout"] = (
        df["existencia_actual"] < df["existencia_minima"]
    ).astype(int)
    df["margen_precio"] = (
        df["precio_promedio"] - df["precio_lista_prom"]
    ).round(2)
    df["delta_demanda"] = (
        df.groupby("tipo_material")["demanda_semanal"].diff().fillna(0)
    )

    def _racha_cero(s: pd.Series) -> list[int]:
        resultado, contador = [], 0
        for v in s:
            contador = contador + 1 if v == 0 else 0
            resultado.append(contador)
        return resultado

    df["semanas_sin_ventas"] = (
        df.groupby("tipo_material")["demanda_semanal"]
        .transform(_racha_cero)
    )

    # ── 4.5 Transformación logarítmica del target ─────────────────────────────
    df[TARGET_LOG] = np.log1p(df[TARGET])

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 5. CONSTRUCCIÓN DE LA MATRIZ X PARA PREDICCIÓN
# ─────────────────────────────────────────────────────────────────────────────

def preparar_X(
    df_features: pd.DataFrame,
    ohe: object,
    eliminar_lags_nulos: bool = True,
) -> pd.DataFrame:
    """
    Aplica OHE a tipo_material y devuelve la matriz X con las 38 features
    en el orden exacto del entrenamiento.

    Parameters
    ----------
    df_features : salida de agregar_features()
    ohe         : OneHotEncoder ajustado en Sección 3
    eliminar_lags_nulos : bool
        True → elimina filas donde lag_1 o lag_4 son NaN (útil para training).
        False → conserva todas las filas (necesario para predicción iterativa).

    Returns
    -------
    pd.DataFrame con columnas = FEATS_ALL (38 features), índice preservado.
    """
    df = df_features.copy()

    # Eliminar filas con lags nulos (primeras N semanas por tipo)
    if eliminar_lags_nulos:
        df = df.dropna(subset=["lag_1", "lag_4"]).copy()

    # Imputar lag_2 y lag_8 residuales
    df["lag_2"] = df["lag_2"].fillna(df["lag_1"])
    df["lag_8"] = df["lag_8"].fillna(df.get("rolling_mean_8", pd.Series(0, index=df.index)).fillna(0))

    # Imputar venta y venta_desc en semanas sin ventas → 0
    for col in COLS_FILL_CERO_FEAT:
        df[col] = df[col].fillna(0)

    # Imputar std residuales (primera semana de cada tipo tiene std=NaN)
    for w in VENTANAS:
        col_std = f"rolling_std_{w}"
        if col_std in df.columns:
            df[col_std] = df[col_std].fillna(0)

    # OHE de tipo_material
    tipo_encoded = ohe.transform(df[["tipo_material"]])
    df_ohe = pd.DataFrame(
        tipo_encoded,
        columns=FEATS_OHE,
        index=df.index,
    )

    # Construir X en orden exacto del entrenamiento
    X = pd.concat([df[FEATS_NUM], df_ohe], axis=1)

    # Verificación de sanidad: no deben quedar NaN
    nan_totales = X.isnull().sum().sum()
    if nan_totales > 0:
        cols_con_nan = X.columns[X.isnull().any()].tolist()
        logger.warning(
            "preparar_X: %d NaN restantes en columnas: %s",
            nan_totales, cols_con_nan,
        )
        X = X.fillna(0)

    return X


# ─────────────────────────────────────────────────────────────────────────────
# 6. PREDICCIÓN ITERATIVA PARA SEMANAS FUTURAS
# ─────────────────────────────────────────────────────────────────────────────

def _rolling_mean_col(df_tipo: pd.DataFrame, col: str, w: int) -> float:
    """Media de las últimas w semanas de una columna de contexto."""
    if col not in df_tipo.columns:
        return 0.0
    ultimas = df_tipo[col].iloc[-w:]
    return float(ultimas.mean()) if len(ultimas) > 0 else 0.0

def predecir_proximas_semanas(
    model: object,
    ohe: object,
    df_historico: pd.DataFrame,
    n_semanas: int = 4,
) -> pd.DataFrame:
    """
    Genera predicciones para las próximas n_semanas semanas usando
    predicción iterativa: cada semana futura alimenta el lag_1 de la siguiente.

    El histórico debe contener al menos 8 semanas previas (para lag_8)
    por cada tipo_material.

    Parameters
    ----------
    model        : XGBRegressor cargado con cargar_artefactos()
    ohe          : OneHotEncoder cargado con cargar_artefactos()
    df_historico : DataFrame con historial reciente, resultado de agregar_features()
    n_semanas    : número de semanas futuras a predecir (máximo recomendado: 8)

    Returns
    -------
    pd.DataFrame con columnas:
        tipo_material, periodo_semana, pred_demanda, pred_log,
        intervalo_inf, intervalo_sup, flag_stockout
    donde intervalo = predicción ± MAE_MODELO (16.48 unidades).
    """
    MAE_MODELO    = 16.48   # MAE del modelo en test, Sección 4
    LOG_CAP_UPPER = float(np.log1p(df_historico[TARGET].max()) + 2.0)

    df_work = df_historico.copy()
    ultima_semana = df_work["periodo_semana"].max()

    resultados = []

    df_work["tipo_material"] = df_work["tipo_material"].astype(str)

    for paso in range(1, n_semanas + 1):
        semana_pred = ultima_semana + pd.Timedelta(weeks=paso)
        logger.info("Prediciendo semana %d: %s", paso, semana_pred.date())

        filas_nuevas = []
        for tipo in TIPOS_VALIDOS:
            hist_tipo = (
                df_work[df_work["tipo_material"] == tipo]
                .sort_values("periodo_semana")
            )
            if len(hist_tipo) == 0:
                logger.warning("Sin historial para tipo '%s', omitiendo.", tipo)
                continue

            ultima = hist_tipo.iloc[-1]

            # Construir fila de la semana futura
            fila = {
                "tipo_material"   : tipo,
                "periodo_semana"  : semana_pred,
                "anio"            : semana_pred.year,
                "mes"             : semana_pred.month,
                "trimestre"       : (semana_pred.month - 1) // 3 + 1,
                "semana_anio"     : int(semana_pred.isocalendar()[1]),
                # Precio e inventario: último valor conocido
                "precio_promedio" : ultima["precio_promedio"],
                "precio_lista_prom": ultima["precio_lista_prom"],
                "descuento_color" : ultima["descuento_color"],
                "existencia_actual": ultima["existencia_actual"],
                "existencia_minima": ultima["existencia_minima"],
                # Venta: 0 porque aún no ha ocurrido
                "venta"           : 0.0,
                "venta_lag1"      : _rolling_mean_col(hist_tipo, "venta", 4),
                "venta_desc"      : 0.0,
                "venta_desc_lag1" : _rolling_mean_col(hist_tipo, "venta_desc", 4),
                # Conteos: 0 (no hay transacciones futuras conocidas)
                "n_transacciones" :     0.0,
                "n_transacciones_lag1" : _rolling_mean_col(hist_tipo, "n_transacciones", 4),
                "n_clientes"           : _rolling_mean_col(hist_tipo, "n_clientes", 4),
                # Lags desde historial conocido
                "lag_1"           : _get_lag(hist_tipo, 1),
                "lag_2"           : _get_lag(hist_tipo, 2),
                "lag_4"           : _get_lag(hist_tipo, 4),
                "lag_8"           : _get_lag(hist_tipo, 8),
                # Rolling desde historial
                "rolling_mean_4"  : _rolling_mean(hist_tipo, 4),
                "rolling_std_4"   : _rolling_std(hist_tipo, 4),
                "rolling_mean_8"  : _rolling_mean(hist_tipo, 8),
                "rolling_std_8"   : _rolling_std(hist_tipo, 8),
                # Target placeholder (no se usa en predicción)
                TARGET            : 0.0,
                TARGET_LOG        : 0.0,
                "baseline_ma4"    : _rolling_mean(hist_tipo, 4),
            }

            # Features derivadas determinísticas
            sem_num = fila["semana_anio"]
            fila["semana_sin"]        = np.sin(2 * np.pi * sem_num / 52)
            fila["semana_cos"]        = np.cos(2 * np.pi * sem_num / 52)
            fila["mes_sin"]           = np.sin(2 * np.pi * fila["mes"] / 12)
            fila["mes_cos"]           = np.cos(2 * np.pi * fila["mes"] / 12)
            fila["es_temporada_alta"] = int(fila["mes"] in [4, 5, 6, 7, 8])
            fila["es_cierre_anio"]    = int(fila["mes"] in [11, 12, 1])
            fila["ratio_existencia"]  = (
                fila["existencia_actual"] / (fila["existencia_minima"] + 1)
            )
            fila["flag_stockout"] = int(
                fila["existencia_actual"] < fila["existencia_minima"]
            )
            fila["margen_precio"] = (
                fila["precio_promedio"] - fila["precio_lista_prom"]
            )
            fila["delta_demanda"]     = fila["lag_1"] - _get_lag(hist_tipo, 2)
            fila["semanas_sin_ventas"] = _semanas_sin_ventas(hist_tipo)

            filas_nuevas.append(fila)

        if not filas_nuevas:
            logger.error("Sin filas para paso %d. Deteniendo iteración.", paso)
            break

        df_paso = pd.DataFrame(filas_nuevas)

        # Construir X y predecir
        X_paso   = preparar_X(df_paso, ohe, eliminar_lags_nulos=False)
        log_pred = model.predict(X_paso)
        log_pred = np.clip(log_pred, 0, LOG_CAP_UPPER)   # evitar overflow en expm1
        pred_orig = np.clip(np.expm1(log_pred), 0, None)

        df_paso["pred_demanda"]  = pred_orig
        df_paso["pred_log"]      = log_pred
        df_paso["intervalo_inf"] = np.clip(pred_orig - MAE_MODELO, 0, None)
        df_paso["intervalo_sup"] = pred_orig + MAE_MODELO
        df_paso[TARGET]          = pred_orig   # alimenta el lag_1 del siguiente paso
        df_paso[TARGET_LOG]      = log_pred

        resultados.append(
            df_paso[[
                "tipo_material", "periodo_semana",
                "pred_demanda", "pred_log",
                "intervalo_inf", "intervalo_sup",
                "flag_stockout",
            ]]
        )

        # Añadir predicción al historial para calcular lags del siguiente paso
        df_work = pd.concat([df_work, df_paso], ignore_index=True)

    if not resultados:
        return pd.DataFrame()

    df_pred = pd.concat(resultados, ignore_index=True)
    df_pred["pred_demanda"]  = df_pred["pred_demanda"].round(2)
    df_pred["intervalo_inf"] = df_pred["intervalo_inf"].round(2)
    df_pred["intervalo_sup"] = df_pred["intervalo_sup"].round(2)

    logger.info(
        "Predicción completada: %d filas para %d semanas.",
        len(df_pred), n_semanas,
    )
    return df_pred


# ── Helpers para predicción iterativa ────────────────────────────────────────

def _get_lag(df_tipo: pd.DataFrame, n: int) -> float:
    """Devuelve el valor de lag_n desde el historial ordenado."""
    col = f"lag_{n}"
    if col in df_tipo.columns:
        val = df_tipo.iloc[-1].get(col, np.nan)
        if not pd.isna(val):
            return float(val)
    # Fallback: usar la n-ésima fila desde el final
    idx = -n
    if abs(idx) <= len(df_tipo):
        return float(df_tipo[TARGET].iloc[idx])
    return 0.0


def _rolling_mean(df_tipo: pd.DataFrame, w: int) -> float:
    """Media de las últimas w semanas (sin incluir la predicha)."""
    ultimas = df_tipo[TARGET].iloc[-w:] if len(df_tipo) >= w else df_tipo[TARGET]
    return float(ultimas.mean()) if len(ultimas) > 0 else 0.0


def _rolling_std(df_tipo: pd.DataFrame, w: int) -> float:
    """Desv. estándar de las últimas w semanas."""
    ultimas = df_tipo[TARGET].iloc[-w:] if len(df_tipo) >= w else df_tipo[TARGET]
    return float(ultimas.std()) if len(ultimas) > 1 else 0.0


def _semanas_sin_ventas(df_tipo: pd.DataFrame) -> int:
    """Número de semanas consecutivas con demanda = 0 al final del historial."""
    vals = df_tipo[TARGET].values[::-1]
    for i, v in enumerate(vals):
        if v != 0:
            return i
    return len(vals)

