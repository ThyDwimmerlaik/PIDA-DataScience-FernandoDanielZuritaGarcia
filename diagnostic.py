from utils.pipeline import (
    cargar_artefactos, cargar_datos,
    construir_grilla_semanal, agregar_features,
    predecir_proximas_semanas, TIPOS_VALIDOS
)
import pandas as pd

model, ohe = cargar_artefactos(
    ruta_modelo="artifacts/modelo_final_xgb_tunned.json",
    ruta_ohe="artifacts/ohe_tipo_material.pkl"
)

df_raw      = cargar_datos(usar_mysql=False, ruta_csv="artifacts/dataset.csv")
df_semana   = construir_grilla_semanal(df_raw)
df_features = agregar_features(df_semana)

# Verificar que los 7 tipos tienen datos recientes no nulos
resumen = (
    df_features.sort_values("periodo_semana")
    .groupby("tipo_material")
    .tail(4)
    .groupby("tipo_material")[["demanda_semanal","n_transacciones","lag_1"]]
    .mean()
    .round(2)
)
print(resumen)

# Verificar que predecir_proximas_semanas produce 28 filas
df_pred = predecir_proximas_semanas(model, ohe, df_features, n_semanas=4)
print(f"\nFilas de predicción: {len(df_pred)}  (esperado: {len(TIPOS_VALIDOS) * 4})")
print(df_pred.groupby("tipo_material")["pred_demanda"].mean().sort_values(ascending=False))