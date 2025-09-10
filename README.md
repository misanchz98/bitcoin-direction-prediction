                                                                                         
# 📈 Análisis y predicción direccional del precio de *Bitcoin* mediante técnicas de *deep learning*

Este repositorio contiene los notebooks y módulos desarrollados para analizar y comparar el desempeño de modelos de deep learning en la predicción diaria de la dirección del precio de *Bitcoin* (clasificación binaria: 0 = baja, 1 = sube). Se utilizan variables OHLCV, indicadores técnicos y métricas *on-chain*, evaluando cuatro arquitecturas: LSTM, GRU, CNN+LSTM y CNN+GRU.

## Objetivos principales:
- Identificar las variables más relevantes y predictivas.
- Comparar el rendimiento de los modelos en precisión y desempeño económico.
- Comprender la volatilidad del mercado de *Bitcoin* para apoyar decisiones de inversión más fundamentadas.

## 📁 Estructura del repositorio

El repositorio está organizado en las siguientes carpetas y archivos principales:

```plaintext
bitcoin-direction-prediction/
├── 01_data_preparation/
│   └── ...
├── 02_data_analysis/
│   └── ...
├── 03_data_modeling/
│   └── ...
├── .gitignore
└── README.md
```

### 🗂️ Descripción de Carpetas Principales

- **`01_data_preparation`**  
  Carpeta donde se recopilan los datos diarios de *Bitcoin*, se calculan indicadores técnicos y métricas de la *blockchain*, generando finalmente el archivo `btc_historical_data.csv` para la siguiente etapa.

- **`02_data_analysis`**  
  Carpeta donde se realiza el análisis descriptivo y la depuración del conjunto de datos, aplicando las transformaciones necesarias para generar el archivo `btc_historical_data_eda.csv`, listo para la modelización.

- **`03_modeling`**  
  Carpeta donde se ajustan y evalúan los modelos **LSTM**, **GRU**, **CNN+LSTM** y **CNN+GRU**, seleccionando la mejor configuración y generando los resultados de predicción y evaluación para determinar el modelo más eficiente y rentable.
