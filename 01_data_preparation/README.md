# 📊  Preparación de los datos

Esta carpeta contiene todo el código desarrollado para la **preparación de datos**, con el objetivo de utilizarlos posteriormente en análisis y modelización.  
El propósito principal es generar un **dataset** que incluya:

- **Datos históricos de Bitcoin** (variables OHLV).  
- **Indicadores técnicos**: tendencia, *momentum*, volatilidad y volumen.  
- **Métricas on-chain**.  


## 📁 Estructura del Repositorio

El repositorio está organizado de la siguiente manera:

```plaintext
01_data_preparation/
├── data/
│   ├── blockchain/
│   │      ├── bdc_daily_data.csv
│   │      └── bdc_info_data.csv
│   └── btc_historical_data.csv
│
├── 01_data_preparation.ipynb  
├── 01_scraper_blockchain.ipynb
├── requirements.txt
└── README.md
```

### 🗂️ Descripción de Archivos

#### 📁 `data/`

- **`btc_historical_data.csv`**  
  Generado por `01_data_preparation.ipynb`. Contiene el **dataset final** con:
  - Datos históricos de BTC/USD
  - Indicadores técnicos
  - Métricas *on-chain*  
  Este archivo se utiliza como base en la fase de análisis (`02_data_analysis.ipynb`).

#### 📁 `blockchain/`

- **`bdc_daily_data.csv`**  
  Generado por `01_scraper_blockchain.ipynb`. Contiene la información diaria extraída de *Blockchain.com*.

- **`bdc_info_data.csv`**  
  Generado por `01_scraper_blockchain.ipynb`. Incluye la descripción de las variables obtenidas de *Blockchain.com*.


#### 📄 Archivos Principales

- **`01_data_preparation.ipynb`**  
  *Notebook* donde se realiza la creación del *dataset* final (`btc_historical_data.csv`).

- **`01_scraper_blockchain.ipynb`**  
  *Web scraper* encargado de extraer las métricas *on-chain* desde Blockchain.com.

- **`requirements.txt`**  
  Contiene todas las librerías necesarias para ejecutar los *notebooks*.
