# DistribuData — 2PD

Predicción semanal de demanda para 2PD, empresa distribuidora de aluminio y herrajes.
Proyecto Integrador — Diplomado en Data Scientist.

## Requisitos
Python 3.10+ y los paquetes en `requirements.txt`.

## Instalación
```bash
python -m venv venv
source venv/bin/activate      # Mac/Linux
pip install -r requirements.txt
```

## Artefactos necesarios
Descarga desde Colab y colocar en `artifacts/`:
- `modelo_final_xgb.json`
- `ohe_tipo_material.pkl`
- `df_features.parquet`
- `predicciones_test.parquet`
- `metricas_modelos.json`
- `dataset.csv`

## Ejecutar
```bash
streamlit run app.py
```
