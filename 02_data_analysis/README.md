# 📊 Análisis y Procesado de los Datos

En esta carpeta se realiza el **análisis descriptivo** de los datos generados en la fase anterior y la **preparación** de los mismos para la fase de modelización.

## 📁 Estructura del Repositorio

```plaintext
01_data_analysis/
├── data/
│   └── btc_historical_data_eda.csv
│
├── images/
│     └── ...
│
├── utils/
│     └── FuncionesMineria.py
│
├── 02_data_analysis.ipynb
├── requirements.txt
└── README.md
```

### 🗂️ Descripción de Archivos

#### 📁 `data/`

- **`btc_historical_data_eda.csv`**  
  Generado por el *notebook* `02_data_analysis.ipynb`. Contiene el **dataset final**, listo para su uso en la fase de modelización y evaluación.

- **`FuncionesMineria.py`**  
  Script de Python que incluye todas las funciones utilizadas en `02_data_analysis.ipynb` para facilitar el análisis de los datos.

#### 📁 `images/` 

  Carpeta que almacena todas las imágenes y gráficos generados o utilizados en el *notebook* `02_data_analysis.ipynb`.

#### 📄 Archivos Principales

- **`02_data_analysis.ipynb`**  
  *Jupyter notebook* donde se lleva a cabo el análisis descriptivo de los datos y se aplican las transformaciones necesarias para obtener el *dataset* final.

- **`requirements.txt`**  
  Archivo que lista todas las librerías necesarias.
