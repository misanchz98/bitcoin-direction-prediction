                                                                                         
# 📈 Análisis y predicción direccional del precio de *Bitcoin* mediante técnicas de *deep learning*

Este repositorio contiene los notebooks y módulos desarrollados para analizar y comparar el desempeño de modelos de deep learning en la predicción diaria de la dirección del precio de Bitcoin (clasificación binaria: 0 = baja, 1 = sube). Se utilizan variables OHLCV, indicadores técnicos y métricas on-chain, evaluando cuatro arquitecturas: LSTM, GRU, CNN+LSTM y CNN+GRU.

Objetivos principales:

Identificar las variables más relevantes y predictivas.

Comparar el rendimiento de los modelos en precisión y desempeño económico.

Comprender la volatilidad del mercado de Bitcoin para apoyar decisiones de inversión más fundamentadas.

## 📁 Estructura del repositorio

El repositorio está organizado en las siguientes carpetas y archivos principales:

```plaintext
proyecto/
├── src/
│   ├── main.py
│   ├── utils.py
│   └── config.py
├── tests/
│   ├── test_main.py
│   └── test_utils.py
├── docs/
│   └── README.md
├── .gitignore
└── README.md
```