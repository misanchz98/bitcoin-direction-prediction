# 📊 Modelización

Esta carpeta contiene el código desarrollado para la fase de **modelado**, cuyo objetivo es identificar la red neuronal más adecuada para la predicción binaria del movimiento del precio del *Bitcoin*.

Las principales tareas abordadas incluyen:

- Optimización de hiperparámetros en modelos de redes neuronales.
- Evaluación mediante métricas clásicas de clasificación y métricas específicas del ámbito financiero.

## 📁 Estructura del Repositorio

La organización del repositorio es la siguiente:

```plaintext
03_modeling/
├── utils/
│   ├── callbacks.py
│   ├── cross_validation.py
│   ├── general.py
│   ├── models.py
│   ├── pipelines.py
│   └── random_search.csv
│
├── 03_modeling.ipynb 
├── requirements.txt
└── README.md
```

### 🗂️ Descripción de Archivos

#### 📁 `utils/`

- **`callbacks.py`**  
  Define *callbacks* personalizados que se integran en el proceso de entrenamiento de redes neuronales para mejorar el control y seguimiento del aprendizaje.

- **`cross_validation.py`**  
  Implementa las clases necesarias para aplicar la técnica *Purged Walk Forward Cross Validation*, especialmente útil en contextos de series temporales financieras.

- **`general.py`**  
  Contiene funciones auxiliares de propósito general utilizadas en distintos módulos del proyecto.

- **`models.py`**  
  Incluye funciones para construir arquitecturas básicas de redes neuronales, adaptadas al problema de predicción binaria.

- **`pipelines.py`**  
  Desarrolla los *pipelines* que integran el ajuste de hiperparámetros y la evaluación de modelos, combinando técnicas de validación y métricas específicas.

- **`random_search.ipynb`**
  Contiene la  la clase `TimeSeriesRandomSearchCV`, diseñada para realizar la optimización de hiperparámetros mediante la técnica de *Random Search*, combinada con el esquema de validación *Purged Walk Forward Cross Validation*.


#### 📄 Archivos Principales

- **`03_modeling.ipynb`**  
  *Notebook* principal donde se ejecutan los *pipelines* definidos y se lleva a cabo el proceso completo de modelización y evaluación.

- **`requirements.txt`**  
  Lista de dependencias necesarias para reproducir el entorno de ejecución del proyecto y garantizar la compatibilidad de los notebooks.

