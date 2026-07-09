"""
app.py — DistribuData Dashboard
================================
Aplicación Streamlit de predicción de demanda semanal para 2PD.

Páginas
-------
1. Resumen Ejecutivo   — KPIs globales, serie total y alertas de stockout
2. Predicción por Tipo — detalle semanal por tipo de material con banda ±MAE
3. Explicabilidad      — SHAP values, comparativa de modelos, guía de lectura

Ejecución
--------
    streamlit run app.py

Estructura de archivos requerida
---------------------------------
    app.py
    artifacts/
        modelo_final_xgb.json           model_xgb.save_model('modelo_final_xgb.json')
        modelo_final_xgb.pkl            joblib.dump(model_xgb, ...)
        modelo_final_xgb_tunned.json    model_xgb.save_model('modelo_final_xgb.json')
        modelo_final_xgb_tunned.pkl     joblib.dump(model_xgb, ...)
        ohe_tipo_material.pkl           joblib.dump(ohe, ...)
        df_features.parquet             generado en Sección 3
        predicciones_test.parquet       generado en Sección 4
        metricas_modelos.json           generado en Sección 4
        dataset.csv                     fallback si no hay MySQL
    media/
        fig_4_4_4_shap_dependence.png
        fig_4_4_4_shap_summary.png
        fig_4_4_4_shap_waterfall.png
    venv/                               Ambiente para librerias
    .env                                Variables de entorno
    utils/
        __init__.py
        pipeline.py
    diagnostic.py
    requirements.txt
"""

# ─────────────────────────────────────────────────────────────────────────────
# st.set_page_config DEBE ser la primera llamada Streamlit del archivo
# ─────────────────────────────────────────────────────────────────────────────
import streamlit as st

st.set_page_config(
    page_title="DistribuData — 2PD",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": (
            "**DistribuData** — Sistema de predicción de demanda semanal para 2PD.\n\n"
            "Modelo: XGBoost  |  MAE: 16.48 uds/sem  |  R²: 0.955"
        )
    },
)

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import json
import logging
import os
import sys
from pathlib import Path
from io import StringIO

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Asegurar que utils/ está en el path
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from utils.pipeline import (
    secure_env,
    cargar_artefactos,
    cargar_datos,
    construir_grilla_semanal,
    agregar_features,
    preparar_X,
    predecir_proximas_semanas,
    TIPOS_VALIDOS,
    FEATS_ALL,
    FEATS_OHE,
    TARGET,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES
# ─────────────────────────────────────────────────────────────────────────────
MAE_MODELO   = 16.48
RMSE_MODELO  = 59.56
R2_MODELO    = 0.9547
MAPE_MODELO  = 18.4
MEJORA_RMSE  = 62.8
N_SEMANAS_PRED = 4

# Rutas de artefactos
RUTA_MODELO          = ROOT / "artifacts/modelo_final_xgb_tunned.json"
RUTA_OHE             = ROOT / "artifacts/ohe_tipo_material.pkl"
RUTA_FEATURES        = ROOT / "artifacts/df_features.parquet"
RUTA_PREDICCIONES    = ROOT / "artifacts/predicciones_test.parquet"
RUTA_METRICAS        = ROOT / "artifacts/metricas_modelos.json"
RUTA_CSV             = ROOT / "artifacts/dataset.csv"
RUTA_SHAP_SUMMARY    = ROOT / "media/fig_4_4_4_shap_summary.png"
RUTA_SHAP_DEPENDENCE = ROOT / "media/fig_4_4_4_shap_dependence.png"
RUTA_SHAP_WATERFALL  = ROOT / "media/fig_4_4_4_shap_waterfall.png"
RUTA_CAPS            = ROOT / "artifacts/winsorization_caps.pkl"

# Paleta de colores
C = {
    "navy":      "#1C3A5E",
    "blue":      "#2E86AB",
    "amber":     "#E8A020",
    "green":     "#1A9E5E",
    "red":       "#C0392B",
    "light":     "#E8F4FD",
    "offwhite":  "#F8FAFC",
    "gray":      "#64748B",
    "graylight": "#E2E8F0",
    "pred":      "#E85D04",   # naranja para predicciones futuras
}

# Paleta por tipo de material
COLORES_TIPO = {
    "Herraje":         "#1C3A5E",
    "Aluminio":        "#2E86AB",
    "AluminioCorte":   "#1A9E5E",
    "Vidrio":          "#7C3AED",
    "Pelicula":        "#E8A020",
    "Aluminio Europeo":"#64748B",
    "Plastico":        "#C0392B",
}


# ─────────────────────────────────────────────────────────────────────────────
# CSS PERSONALIZADO
# ─────────────────────────────────────────────────────────────────────────────
def _inyectar_css():
    st.markdown(
        f"""
        <style>
        /* Sidebar */
        [data-testid="stSidebar"] {{
            background-color: {C['navy']};
        }}
        [data-testid="stSidebar"] * {{
            color: white !important;
        }}
        [data-testid="stSidebar"] .stRadio label {{
            color: white !important;
            font-size: 0.95rem;
        }}

        /* Métricas */
        [data-testid="stMetric"] {{
            background-color: white;
            border: 1px solid {C['graylight']};
            border-radius: 10px;
            padding: 16px 20px;
            box-shadow: 0 1px 4px rgba(0,0,0,0.06);
        }}
        [data-testid="stMetricLabel"] {{
            font-size: 0.78rem !important;
            color: {C['gray']} !important;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }}
        [data-testid="stMetricValue"] {{
            font-size: 2rem !important;
            color: {C['navy']} !important;
            font-weight: 700;
        }}

        /* Alerta stockout */
        .stockout-banner {{
            background: #FEF2F2;
            border-left: 5px solid {C['red']};
            border-radius: 6px;
            padding: 12px 16px;
            margin: 6px 0;
            font-size: 0.92rem;
        }}

        /* Badge verde / rojo */
        .badge-ok  {{ background:#D1FAE5; color:#065F46;
                      padding:3px 10px; border-radius:20px;
                      font-size:0.82rem; font-weight:600; }}
        .badge-warn{{ background:#FEE2E2; color:#991B1B;
                      padding:3px 10px; border-radius:20px;
                      font-size:0.82rem; font-weight:600; }}

        /* Separador sección */
        .section-title {{
            font-size: 1.05rem;
            font-weight: 700;
            color: {C['navy']};
            border-bottom: 2px solid {C['graylight']};
            padding-bottom: 6px;
            margin: 24px 0 14px 0;
        }}

        /* Ocultar footer Streamlit */
        footer {{ visibility: hidden; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# CARGA DE ARTEFACTOS (cacheados en memoria)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Cargando modelo XGBoost…")
def _cargar_modelo_y_ohe():
    """Carga modelo y OHE una sola vez. st.cache_resource los mantiene en RAM."""
    if not RUTA_MODELO.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo en '{RUTA_MODELO}'.\n"
            "Ejecuta en Colab:  model_xgb.save_model('modelo_final_xgb.json')"
        )
    if not RUTA_OHE.exists():
        raise FileNotFoundError(
            f"No se encontró el OHE en '{RUTA_OHE}'.\n"
            "Ejecuta en Colab:  joblib.dump(ohe, 'ohe_tipo_material.pkl')"
        )
    model, ohe = cargar_artefactos(
        ruta_modelo=str(RUTA_MODELO),
        ruta_ohe=str(RUTA_OHE),
    )
    return model, ohe


@st.cache_data(show_spinner="Cargando historial…")
def _cargar_historico() -> pd.DataFrame:
    """
    Carga el dataset de features completo.
    Prioriza df_features.parquet (Sección 3).
    Si no existe, reconstruye desde el CSV.
    """
    if RUTA_FEATURES.exists():
        df = pd.read_parquet(RUTA_FEATURES)
        df["periodo_semana"] = pd.to_datetime(df["periodo_semana"])
        return df

    if not RUTA_CSV.exists():
        raise FileNotFoundError(
            f"No se encontró 'df_features.parquet' ni '{RUTA_CSV}'.\n"
            "Descarga 'df_features.parquet' desde Colab o coloca el CSV en el directorio raíz."
        )

    # Reconstruir pipeline completo desde CSV
    df_raw    = cargar_datos(usar_mysql=False, ruta_csv=str(RUTA_CSV))
    df_semana = construir_grilla_semanal(df_raw)
    df_feat   = agregar_features(df_semana)
    df_feat["tipo_material"] = df_feat["tipo_material"].astype(str)
    return df_feat


@st.cache_data(show_spinner="Cargando predicciones del test set…")
def _cargar_predicciones_test() -> pd.DataFrame:
    """Carga predicciones del test set generadas en Sección 4."""
    if not RUTA_PREDICCIONES.exists():
        return pd.DataFrame()
    df = pd.read_parquet(RUTA_PREDICCIONES)
    df["periodo_semana"] = pd.to_datetime(df["periodo_semana"])
    return df


@st.cache_data(ttl=3600, show_spinner="Generando predicciones…")
def _generar_predicciones_futuras(_model, _ohe, df_hist_json: str) -> pd.DataFrame:
    """
    Genera predicciones para las próximas N_SEMANAS_PRED semanas.
    df_hist_json es la versión JSON del historial (hashable para cache).
    """
    df_hist = pd.read_json(StringIO(df_hist_json), orient="split")
    df_hist["periodo_semana"] = pd.to_datetime(df_hist["periodo_semana"])
    return predecir_proximas_semanas(
        model=_model,
        ohe=_ohe,
        df_historico=df_hist,
        n_semanas=N_SEMANAS_PRED,
    )


def _cargar_metricas_modelos() -> dict:
    if not RUTA_METRICAS.exists():
        return {}
    with open(RUTA_METRICAS) as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS DE VISUALIZACIÓN
# ─────────────────────────────────────────────────────────────────────────────

def _layout_plotly(fig: go.Figure, titulo: str = "", alto: int = 380) -> go.Figure:
    """Aplica tema consistente a todas las figuras Plotly."""
    fig.update_layout(
        title=dict(text=titulo, font=dict(size=14, color=C["navy"]), x=0),
        height=alto,
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="system-ui, -apple-system, sans-serif",
                  size=12, color=C["gray"]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="left", x=0, bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=10, r=10, t=50 if titulo else 20, b=10),
        hovermode="x unified",
        xaxis=dict(gridcolor=C["graylight"], zeroline=False, showline=False),
        yaxis=dict(gridcolor=C["graylight"], zeroline=False, showline=False),
    )
    return fig


def _grafica_serie_total(
    df_hist: pd.DataFrame,
    df_pred: pd.DataFrame,
) -> go.Figure:
    """Serie de tiempo de demanda total semanal (todos los tipos) + predicción."""
    hist_total = (
        df_hist.groupby("periodo_semana")[TARGET]
        .sum()
        .reset_index()
        .sort_values("periodo_semana")
    )
    # Últimas 78 semanas para no saturar
    hist_total = hist_total.tail(78)

    fig = go.Figure()

    # Serie histórica
    fig.add_trace(go.Scatter(
        x=hist_total["periodo_semana"],
        y=hist_total[TARGET],
        mode="lines",
        name="Demanda real",
        line=dict(color=C["navy"], width=2),
        hovertemplate="<b>%{x|%d %b %Y}</b><br>Real: %{y:,.0f} uds<extra></extra>",
    ))

    # Media móvil 4 semanas
    hist_total["ma4"] = hist_total[TARGET].rolling(4, min_periods=1).mean()
    fig.add_trace(go.Scatter(
        x=hist_total["periodo_semana"],
        y=hist_total["ma4"],
        mode="lines",
        name="Media móvil 4 sem.",
        line=dict(color=C["gray"], width=1.5, dash="dot"),
        hovertemplate="%{y:,.0f} uds<extra>MA4</extra>",
    ))

    # Predicciones futuras
    if not df_pred.empty:
        pred_total = (
            df_pred.groupby("periodo_semana")["pred_demanda"]
            .sum()
            .reset_index()
            .sort_values("periodo_semana")
        )
        # Punto de conexión (última semana histórica)
        ultimo_hist = hist_total.iloc[[-1]][["periodo_semana", TARGET]] \
                        .rename(columns={TARGET: "pred_demanda"})
        pred_total = pd.concat([ultimo_hist, pred_total], ignore_index=True)

        fig.add_trace(go.Scatter(
            x=pred_total["periodo_semana"],
            y=pred_total["pred_demanda"],
            mode="lines+markers",
            name="Predicción",
            line=dict(color=C["pred"], width=2.5, dash="dash"),
            marker=dict(size=7, color=C["pred"]),
            hovertemplate="<b>%{x|%d %b %Y}</b><br>Pred: %{y:,.0f} uds<extra></extra>",
        ))

        # Banda ±MAE total (MAE × n_tipos)
        n_tipos  = df_pred["tipo_material"].nunique()
        mae_total = MAE_MODELO * n_tipos
        fig.add_trace(go.Scatter(
            x=pd.concat([pred_total["periodo_semana"],
                         pred_total["periodo_semana"].iloc[::-1]]),
            y=pd.concat([pred_total["pred_demanda"] + mae_total,
                         (pred_total["pred_demanda"] - mae_total).clip(0).iloc[::-1]]),
            fill="toself",
            fillcolor="rgba(232,93,4,0.10)",
            line=dict(color="rgba(0,0,0,0)"),
            name="Intervalo ±MAE",
            hoverinfo="skip",
        ))

        # Línea vertical de corte
        ultima_semana = hist_total["periodo_semana"].max()
        fig.add_vline(
            x=ultima_semana.timestamp() * 1000,
            line=dict(color=C["gray"], width=1, dash="dot"),
            annotation_text="Hoy",
            annotation_position="top right",
            annotation_font_color=C["gray"],
        )

    return _layout_plotly(fig, alto=400)


def _grafica_tipo(
    df_hist: pd.DataFrame,
    df_pred: pd.DataFrame,
    tipo: str,
) -> go.Figure:
    """Gráfica de serie + predicción para un tipo de material específico."""
    color = COLORES_TIPO.get(tipo, C["blue"])
    mask_h = df_hist["tipo_material"] == tipo
    hist_t = (
        df_hist[mask_h][["periodo_semana", TARGET, "baseline_ma4"]]
        .sort_values("periodo_semana")
        .tail(78)
    )

    fig = go.Figure()

    # Serie real
    fig.add_trace(go.Scatter(
        x=hist_t["periodo_semana"],
        y=hist_t[TARGET],
        mode="lines",
        name="Demanda real",
        line=dict(color=color, width=2.2),
        hovertemplate="<b>%{x|%d %b %Y}</b><br>Real: %{y:,.1f} uds<extra></extra>",
    ))

    # Baseline MA4
    fig.add_trace(go.Scatter(
        x=hist_t["periodo_semana"],
        y=hist_t["baseline_ma4"],
        mode="lines",
        name="Baseline MA4",
        line=dict(color=C["gray"], width=1.2, dash="dot"),
        hovertemplate="%{y:,.1f} uds<extra>MA4</extra>",
    ))

    # Predicciones
    if not df_pred.empty and tipo in df_pred["tipo_material"].values:
        pred_t = (
            df_pred[df_pred["tipo_material"] == tipo]
            .sort_values("periodo_semana")
        )
        # Conectar con último punto histórico
        ultimo = hist_t.iloc[[-1]][["periodo_semana", TARGET]] \
                   .rename(columns={TARGET: "pred_demanda"})
        ultimo["intervalo_inf"] = ultimo["pred_demanda"]
        ultimo["intervalo_sup"] = ultimo["pred_demanda"]
        pred_t_plot = pd.concat([ultimo, pred_t], ignore_index=True)

        # Banda de confianza
        fig.add_trace(go.Scatter(
            x=pd.concat([pred_t_plot["periodo_semana"],
                         pred_t_plot["periodo_semana"].iloc[::-1]]),
            y=pd.concat([pred_t_plot["intervalo_sup"],
                         pred_t_plot["intervalo_inf"].clip(0).iloc[::-1]]),
            fill="toself",
            fillcolor="rgba(232,93,4,0.12)",
            line=dict(color="rgba(0,0,0,0)"),
            name="Intervalo ±MAE",
            hoverinfo="skip",
        ))

        # Línea de predicción
        fig.add_trace(go.Scatter(
            x=pred_t_plot["periodo_semana"],
            y=pred_t_plot["pred_demanda"],
            mode="lines+markers",
            name="Predicción",
            line=dict(color=C["pred"], width=2.5, dash="dash"),
            marker=dict(size=8, color=C["pred"]),
            hovertemplate=(
                "<b>%{x|%d %b %Y}</b><br>"
                "Pred: %{y:,.1f} uds<extra></extra>"
            ),
        ))

    return _layout_plotly(fig, alto=380)


def _tabla_predicciones(df_pred: pd.DataFrame, tipo: str) -> pd.DataFrame:
    """Formatea la tabla de predicciones de 4 semanas para un tipo."""
    if df_pred.empty or tipo not in df_pred["tipo_material"].values:
        return pd.DataFrame()
    sub = (
        df_pred[df_pred["tipo_material"] == tipo]
        .sort_values("periodo_semana")
        [["periodo_semana", "pred_demanda", "intervalo_inf", "intervalo_sup", "flag_stockout"]]
        .copy()
    )
    sub["Semana"]          = sub["periodo_semana"].dt.strftime("%-d %b %Y")
    sub["Predicción (uds)"]= sub["pred_demanda"].map("{:.1f}".format)
    sub["Rango estimado"]  = (
        sub["intervalo_inf"].map("{:.1f}".format) + " — " +
        sub["intervalo_sup"].map("{:.1f}".format)
    )
    sub["Alerta stock"]    = sub["flag_stockout"].map(
        {1: "🔴 Riesgo", 0: "🟢 OK"}
    )
    return sub[["Semana", "Predicción (uds)", "Rango estimado", "Alerta stock"]] \
             .reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# PÁGINA 1 — RESUMEN EJECUTIVO
# ─────────────────────────────────────────────────────────────────────────────

def pagina_resumen(df_hist: pd.DataFrame, df_pred: pd.DataFrame):
    st.markdown("## 📊 Resumen Ejecutivo")
    st.caption(
        f"Modelo XGBoost — MAE {MAE_MODELO} uds/sem · R² {R2_MODELO} · "
        f"Periodo test: abr 2025 – abr 2026"
    )

    # ── KPIs ──────────────────────────────────────────────────────────────────
    total_pred = df_pred["pred_demanda"].sum() if not df_pred.empty else 0
    alertas    = int(df_pred["flag_stockout"].sum()) if not df_pred.empty else 0
    tipos_pred = df_pred["tipo_material"].nunique() if not df_pred.empty else 0

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric(
            "Predicción próx. 4 semanas",
            f"{total_pred:,.0f} uds",
            help="Suma de demanda predicha para todos los tipos en las próximas 4 semanas.",
        )
    with c2:
        st.metric(
            "Alertas de stockout",
            f"{alertas}",
            delta=f"{alertas} tipo(s) en riesgo" if alertas > 0 else "Sin alertas",
            delta_color="inverse",
            help="Tipos de material con existencia_actual < existencia_minima.",
        )
    with c3:
        st.metric(
            "Mejora sobre baseline",
            f"+{MEJORA_RMSE}%",
            help="Reducción de RMSE del modelo XGBoost vs. media móvil 4 semanas.",
        )
    with c4:
        st.metric(
            "Cobertura de predicción",
            f"{tipos_pred} / {len(TIPOS_VALIDOS)} tipos",
            help="Tipos de material con predicción disponible esta semana.",
        )

    st.markdown("")

    # ── Serie de tiempo total ─────────────────────────────────────────────────
    st.markdown('<p class="section-title">Demanda total semanal</p>',
                unsafe_allow_html=True)
    fig_total = _grafica_serie_total(df_hist, df_pred)
    st.plotly_chart(fig_total, width='stretch')

    # ── Alertas de stockout ──────────────────────────────────────────────────
    st.markdown('<p class="section-title">Alertas de inventario</p>',
                unsafe_allow_html=True)

    if df_pred.empty:
        st.info("Sin predicciones disponibles para mostrar alertas.")
    else:
        alertas_df = (
            df_pred[df_pred["flag_stockout"] == 1]
            [["tipo_material", "periodo_semana", "pred_demanda"]]
            .sort_values(["tipo_material", "periodo_semana"])
            .copy()
        )
        if alertas_df.empty:
            st.success("✅ Sin alertas de stockout en las próximas 4 semanas.")
        else:
            st.error(
                f"⚠️  **{len(alertas_df)} registro(s) con riesgo de quiebre de stock** "
                f"en las próximas {N_SEMANAS_PRED} semanas."
            )
            for _, row in alertas_df.iterrows():
                st.markdown(
                    f'<div class="stockout-banner">'
                    f'🔴 <b>{row["tipo_material"]}</b> — semana del '
                    f'{pd.Timestamp(row["periodo_semana"]).strftime("%-d %b %Y")} — '
                    f'Predicción: <b>{row["pred_demanda"]:.1f} uds</b>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    # ── Descarga ──────────────────────────────────────────────────────────────
    if not df_pred.empty:
        st.markdown("")
        csv_export = df_pred.copy()
        csv_export["periodo_semana"] = csv_export["periodo_semana"].dt.strftime("%Y-%m-%d")
        st.download_button(
            label="⬇️  Descargar predicciones (CSV)",
            data=csv_export.to_csv(index=False).encode("utf-8"),
            file_name="distribudata_predicciones.csv",
            mime="text/csv",
            disabled=not secure_env,
        )


# ─────────────────────────────────────────────────────────────────────────────
# PÁGINA 2 — PREDICCIÓN POR TIPO
# ─────────────────────────────────────────────────────────────────────────────

def pagina_prediccion(df_hist: pd.DataFrame, df_pred: pd.DataFrame):
    st.markdown("## 📦 Predicción por Tipo de Material")

    # ── Selector ──────────────────────────────────────────────────────────────
    col_sel, col_info = st.columns([2, 3])
    with col_sel:
        tipo = st.selectbox(
            "Tipo de material",
            options=TIPOS_VALIDOS,
            index=TIPOS_VALIDOS.index("Herraje"),
            help="Selecciona el tipo de material para ver el detalle de predicción.",
        )

    # ── KPIs del tipo seleccionado ────────────────────────────────────────────
    hist_tipo = df_hist[df_hist["tipo_material"] == tipo].sort_values("periodo_semana")
    media_hist = hist_tipo[TARGET].mean()
    max_hist   = hist_tipo[TARGET].max()
    ultima_sem = hist_tipo.iloc[-1] if len(hist_tipo) > 0 else None

    pred_tipo = (
        df_pred[df_pred["tipo_material"] == tipo].sort_values("periodo_semana")
        if not df_pred.empty else pd.DataFrame()
    )
    proxima_pred = pred_tipo.iloc[0]["pred_demanda"] if not pred_tipo.empty else None
    flag_stock   = (
        int(ultima_sem["flag_stockout"])
        if ultima_sem is not None and "flag_stockout" in ultima_sem else 0
    )

    with col_info:
        estado_badge = (
            '<span class="badge-warn">🔴 Riesgo de quiebre</span>'
            if flag_stock else
            '<span class="badge-ok">🟢 Stock suficiente</span>'
        )
        st.markdown(
            f"**Estado inventario actual:** {estado_badge}",
            unsafe_allow_html=True,
        )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric(
            "Demanda media hist.",
            f"{media_hist:,.1f} uds/sem",
            help="Media semanal histórica del tipo seleccionado.",
        )
    with c2:
        st.metric(
            "Máximo histórico",
            f"{max_hist:,.1f} uds",
            help="Semana con mayor demanda registrada.",
        )
    with c3:
        if proxima_pred is not None:
            delta_vs_media = proxima_pred - media_hist
            st.metric(
                "Predicción próx. semana",
                f"{proxima_pred:,.1f} uds",
                delta=f"{delta_vs_media:+.1f} vs. media",
                help=f"Predicción ±{MAE_MODELO} unidades (MAE del modelo).",
            )
        else:
            st.metric("Predicción próx. semana", "—")
    with c4:
        if ultima_sem is not None:
            exist = ultima_sem.get("existencia_actual", None)
            exist_min = ultima_sem.get("existencia_minima", None)
            if exist is not None and exist_min is not None:
                st.metric(
                    "Existencia actual",
                    f"{exist:,.0f} uds",
                    delta=f"Mín: {exist_min:,.0f}",
                    delta_color="normal" if exist >= exist_min else "inverse",
                    help="Existencia actual vs. existencia mínima definida por 2PD.",
                )
            else:
                st.metric("Existencia actual", "—")
        else:
            st.metric("Existencia actual", "—")

    st.markdown("")

    # ── Gráfica de serie + predicción ─────────────────────────────────────────
    st.markdown(
        f'<p class="section-title">Serie histórica + predicción — {tipo}</p>',
        unsafe_allow_html=True,
    )
    fig_tipo = _grafica_tipo(df_hist, df_pred, tipo)
    st.plotly_chart(fig_tipo, width='stretch')
    st.caption(
        f"La banda sombreada representa el intervalo de confianza ±{MAE_MODELO:.1f} unidades "
        f"(MAE del modelo en datos de prueba)."
    )

    # ── Tabla de predicciones ─────────────────────────────────────────────────
    st.markdown(
        f'<p class="section-title">Predicciones — próximas {N_SEMANAS_PRED} semanas</p>',
        unsafe_allow_html=True,
    )
    df_tabla = _tabla_predicciones(df_pred, tipo)
    if df_tabla.empty:
        st.info("Sin predicciones disponibles para este tipo.")
    else:
        st.dataframe(
            df_tabla,
            width='stretch',
            hide_index=True,
            column_config={
                "Semana":            st.column_config.TextColumn("Semana"),
                "Predicción (uds)":  st.column_config.TextColumn("Predicción (uds)"),
                "Rango estimado":    st.column_config.TextColumn(
                    f"Rango estimado (±{MAE_MODELO:.0f} uds)"
                ),
                "Alerta stock":      st.column_config.TextColumn("Alerta stock"),
            },
        )

    # ── Descarga por tipo ─────────────────────────────────────────────────────
    if not pred_tipo.empty:
        export = pred_tipo.copy()
        export["periodo_semana"] = export["periodo_semana"].dt.strftime("%Y-%m-%d")
        st.download_button(
            label=f"⬇️  Descargar predicciones de {tipo} (CSV)",
            data=export.to_csv(index=False).encode("utf-8"),
            file_name=f"prediccion_{tipo.lower().replace(' ', '_')}.csv",
            mime="text/csv",
            disabled=not secure_env,
        )


# ─────────────────────────────────────────────────────────────────────────────
# PÁGINA 3 — EXPLICABILIDAD
# ─────────────────────────────────────────────────────────────────────────────

def pagina_explicabilidad(model):
    st.markdown("## 🔍 Explicabilidad del Modelo")
    st.caption(
        "Esta sección responde: ¿por qué el modelo predice lo que predice? "
        "Está pensada para que el equipo de compras entienda qué señales usa el modelo."
    )

    # ── Importancia de features ───────────────────────────────────────────────
    st.markdown('<p class="section-title">¿Qué variables usa el modelo?</p>',
                unsafe_allow_html=True)

    try:
        importancias = model.feature_importances_
        fi_df = (
            pd.DataFrame({"feature": FEATS_ALL, "importancia": importancias})
            .sort_values("importancia", ascending=True)
            .tail(15)
        )
        fig_fi = go.Figure(go.Bar(
            x=fi_df["importancia"],
            y=fi_df["feature"],
            orientation="h",
            marker=dict(
                color=fi_df["importancia"],
                colorscale=[[0, C["light"]], [0.5, C["blue"]], [1, C["navy"]]],
                showscale=False,
            ),
            hovertemplate="%{y}: %{x:.4f}<extra></extra>",
        ))
        _layout_plotly(fig_fi, "Top 15 features por importancia (ganancia)", alto=420)
        st.plotly_chart(fig_fi, width='stretch')
    except Exception:
        st.info("No se pudo calcular la importancia de features desde el modelo cargado.")

    # ── Imágenes SHAP (si existen) ───────────────────────────────────────────
    shap_disponible = any([
        RUTA_SHAP_SUMMARY.exists(),
        RUTA_SHAP_DEPENDENCE.exists(),
        RUTA_SHAP_WATERFALL.exists(),
    ])

    if shap_disponible:
        st.markdown('<p class="section-title">Análisis SHAP — impacto de cada variable</p>',
                    unsafe_allow_html=True)
        st.caption(
            "Los SHAP values miden cuánto contribuye cada variable a cada predicción individual. "
            "Un valor positivo empuja la predicción hacia arriba; uno negativo, hacia abajo."
        )
        tabs = []
        tab_labels = []
        if RUTA_SHAP_SUMMARY.exists():
            tab_labels.append("📊 Importancia global")
        if RUTA_SHAP_DEPENDENCE.exists():
            tab_labels.append("📈 Dependencia")
        if RUTA_SHAP_WATERFALL.exists():
            tab_labels.append("🔎 Predicción individual")

        if tab_labels:
            tabs = st.tabs(tab_labels)
            idx = 0
            if RUTA_SHAP_SUMMARY.exists():
                with tabs[idx]:
                    st.image(str(RUTA_SHAP_SUMMARY), width='stretch')
                    st.caption(
                        "Izquierda: importancia media absoluta por feature. "
                        "Derecha: dirección e intensidad de impacto en cada predicción."
                    )
                idx += 1
            if RUTA_SHAP_DEPENDENCE.exists():
                with tabs[idx]:
                    st.image(str(RUTA_SHAP_DEPENDENCE), width='stretch')
                    st.caption(
                        "Relación entre el valor de la feature y su impacto en la predicción. "
                        "El color indica el valor de la feature de interacción automática."
                    )
                idx += 1
            if RUTA_SHAP_WATERFALL.exists():
                with tabs[idx]:
                    st.image(str(RUTA_SHAP_WATERFALL), width='stretch')
                    st.caption(
                        "Descomposición de una predicción individual: cada barra muestra "
                        "cuánto sumó o restó cada feature al valor base del modelo."
                    )
    else:
        st.info(
            "Las imágenes SHAP no están disponibles. "
            "Genera y guarda los plots en la Sección 4 del notebook "
            f"('fig_4_4_4_shap_*.png') y colócalos en '{ROOT}'."
        )

    # ── Guía de las top features en lenguaje de negocio ──────────────────────
    st.markdown('<p class="section-title">Guía de interpretación</p>',
                unsafe_allow_html=True)

    guia = [
        ("🥇", "n_transacciones",
         "Número de pedidos distintos en la semana.",
         "Semanas con muchos pedidos anticipan alta demanda la siguiente semana. "
         "Es la señal más fuerte del modelo."),
        ("🥈", "venta_desc",
         "Facturación total con descuento aplicado.",
         "Captura campañas comerciales y épocas de precio bajo que impulsan compras. "
         "Cuando el descuento sube, la demanda tiende a aumentar la semana siguiente."),
        ("🥉", "venta",
         "Facturación bruta de la semana.",
         "Indicador del volumen monetario vendido. Correlaciona con la cantidad "
         "vendida pero aporta información adicional sobre el precio promedio."),
        ("4️⃣",  "delta_demanda",
         "Variación de demanda semana a semana.",
         "Si la demanda subió fuerte la semana pasada, el modelo aprende que "
         "puede seguir subiendo o que viene un ajuste a la baja."),
        ("5️⃣",  "precio_promedio",
         "Precio unitario promedio de la semana.",
         "Precios altos pueden inhibir demanda en tipos sensibles al precio "
         "como Aluminio y AluminioCorte."),
    ]

    for medal, feature, definicion, interpretacion in guia:
        with st.expander(f"{medal}  **{feature}** — {definicion}"):
            st.write(interpretacion)

    # ── Comparativa de modelos ─────────────────────────────────────────────────
    metricas = _cargar_metricas_modelos()
    if metricas:
        st.markdown('<p class="section-title">Comparativa de modelos evaluados</p>',
                    unsafe_allow_html=True)
        df_met = pd.DataFrame(metricas).T.reset_index().rename(columns={"index": "Modelo"})

        # Columnas numéricas para formatear
        cols_num = ["MAE", "RMSE", "MAPE (%)", "R²", "MAE / media (%)", "Mejora RMSE (%)"]
        cols_num = [c for c in cols_num if c in df_met.columns]

        for col in cols_num:
            df_met[col] = pd.to_numeric(df_met[col], errors="coerce").round(2)

        st.dataframe(
            df_met,
            width='stretch',
            hide_index=True,
            column_config={
                "Modelo":          st.column_config.TextColumn("Modelo"),
                "MAE":             st.column_config.NumberColumn("MAE", format="%.2f"),
                "RMSE":            st.column_config.NumberColumn("RMSE", format="%.2f"),
                "MAPE (%)":        st.column_config.NumberColumn("MAPE (%)", format="%.1f"),
                "R²":              st.column_config.NumberColumn("R²", format="%.4f"),
                "MAE / media (%)": st.column_config.NumberColumn("MAE/media (%)", format="%.1f"),
                "Mejora RMSE (%)": st.column_config.NumberColumn("Mejora RMSE (%)", format="%.1f"),
            },
        )
        st.caption(
            "El modelo seleccionado (XGBoost base) supera al modelo afinado en todas las "
            "métricas de escala original, ya que el tuning optimizó log-RMSE en CV, "
            "no las métricas de negocio."
        )


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

def _sidebar():
    with st.sidebar:
        st.markdown(
            f"""
            <div style="padding: 8px 0 20px 0;">
              <div style="font-size:1.5rem; font-weight:800;
                          letter-spacing:-0.5px; color:white;">
                📦 DistribuData
              </div>
              <div style="font-size:0.78rem; color:#8BA3BE;
                          margin-top:2px;">
                2PD — Distribución de Aluminio
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        pagina = st.radio(
            "Navegación",
            options=[
                "📊  Resumen Ejecutivo",
                "📦  Predicción por Tipo",
                "🔍  Explicabilidad",
            ],
            label_visibility="collapsed",
        )

        st.markdown("---")

        # Opción de recarga
        recargar = st.button(
            "🔄  Actualizar datos",
            help="Fuerza la recarga de datos y predicciones.",
            disabled=not secure_env,
        )
        if recargar:
            st.cache_data.clear()
            st.rerun()

        st.markdown("---")

        # Info del modelo
        st.markdown(
            f"""
            <div style="font-size:0.78rem; color:#8BA3BE; line-height:1.7;">
              <b style="color:white;">Modelo activo</b><br>
              XGBoost (SEED=55)<br>
              MAE: {MAE_MODELO} uds/sem<br>
              R²: {R2_MODELO}<br>
              MAPE: {MAPE_MODELO}%<br><br>
              <b style="color:white;">Tipos modelados</b><br>
              {"<br>".join(TIPOS_VALIDOS)}
            </div>
            """,
            unsafe_allow_html=True,
        )

    return pagina


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    _inyectar_css()

    # ── Sidebar y navegación ─────────────────────────────────────────────────
    pagina = _sidebar()

    # ── Carga de artefactos ──────────────────────────────────────────────────
    try:
        model, ohe = _cargar_modelo_y_ohe()
    except FileNotFoundError as exc:
        logger.error("**Error de configuración:**", exc_info=True)
        st.error(f"**Error de configuración:** {exc}")
        st.stop()

    # ── Carga de datos históricos ────────────────────────────────────────────
    try:
        df_hist = _cargar_historico()
    except FileNotFoundError as exc:
        logger.error("**Error al cargar historial:**", exc_info=True)
        st.error(f"**Error al cargar historial:** {exc}")
        st.stop()
    except Exception as exc:
        logger.error("**Error inesperado al cargar datos:**", exc_info=True)
        st.error(f"**Error inesperado al cargar datos:** {exc}")
        st.stop()

    # ── Generación de predicciones futuras ───────────────────────────────────
    try:
        df_hist_json = df_hist.to_json(orient="split", date_format="iso")
        df_pred = _generar_predicciones_futuras(model, ohe, df_hist_json)
    except Exception as exc:
        logger.error("No se pudieron generar predicciones futuras", exc_info=True)
        st.warning(f"No se pudieron generar predicciones futuras.")
        df_pred = pd.DataFrame()

    # ── Enrutamiento de páginas ──────────────────────────────────────────────
    if "Resumen" in pagina:
        pagina_resumen(df_hist, df_pred)
    elif "Predicción" in pagina:
        pagina_prediccion(df_hist, df_pred)
    elif "Explicabilidad" in pagina:
        pagina_explicabilidad(model)


if __name__ == "__main__":
    main()
