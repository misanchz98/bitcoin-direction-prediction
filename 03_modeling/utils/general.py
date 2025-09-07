import numpy as np
import pandas as pd
from typing import Dict, Optional
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)

# =======================================
# CREACIÓN DE SECUENCIAS
# =======================================

def create_windows_multivariate_np(data, target, window_size, horizon=1, shuffle=False):
    """
    Crea secuencias multivariadas para redes neuronales.
    Devuelve X shape (n_samples, window_size, n_features) y y shape (n_samples,).
    """
    if isinstance(data, pd.DataFrame):
        data = data.values
    if isinstance(target, (pd.DataFrame, pd.Series)):
        target = target.values
    
    X, y = [], []
    for i in range(len(data) - window_size - horizon + 1):
        X.append(data[i:i+window_size, :])
        y.append(target[i+window_size+horizon-1])
    
    X, y = np.array(X), np.array(y)
    
    if shuffle:
        idx = np.arange(X.shape[0])
        np.random.shuffle(idx)
        X, y = X[idx], y[idx]
    
    return X, y

# =======================================
# EVALUACIÓN DE MÉTRICAS FINANCIERAS
# =======================================

# ============================================
# 1. Función auxiliar: máximo drawdown
# ============================================
def calculate_max_drawdown(cumulative_returns: np.ndarray) -> float:
    peak = np.maximum.accumulate(cumulative_returns)
    drawdown = (cumulative_returns - peak) / peak
    return float(drawdown.min())

# ============================================
# 2. Evaluar métricas (con strategy)
# ============================================
def evaluate_metrics(y_true: np.ndarray, y_pred: np.ndarray, returns: np.ndarray = None,
                     strategy: str = None) -> dict:
    """
    Evalúa métricas de clasificación y, si se provee returns, métricas SOLO de la estrategia indicada.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    if len(y_true) != len(y_pred):
        raise ValueError("y_true y y_pred deben tener la misma longitud")
    if len(y_true) == 0:
        raise ValueError("Los arrays no pueden estar vacíos")
    
    # Métricas base de clasificación
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred)
    }
    
    # Métricas financieras solo si hay returns y strategy
    if returns is not None and strategy is not None:
        returns = np.array(returns)
        if len(returns) != len(y_pred):
            raise ValueError("returns debe tener la misma longitud que y_pred")
        
        # Estrategias
        if strategy == "longonly":
            strat_returns = np.where(y_pred == 1, returns, 0)
        elif strategy == "longshort":
            strat_returns = np.where(y_pred == 1, returns, -returns)
        elif strategy == "shortonly":
            strat_returns = np.where(y_pred == 0, -returns, 0)
        else:
            raise ValueError(f"Estrategia no soportada: {strategy}")
        
        if len(strat_returns) > 0:
            cum_return = float(np.cumsum(strat_returns)[-1])
            mean_return = float(np.mean(strat_returns))
            std_return = float(np.std(strat_returns))
            sharpe = mean_return / std_return * np.sqrt(252) if std_return > 1e-8 else 0.0
            max_dd = calculate_max_drawdown(np.cumsum(strat_returns))
            win_rate = float(np.sum(strat_returns > 0) / len(strat_returns))
            
            if strategy == "longonly":
                num_trades = int(np.sum(y_pred == 1))
            elif strategy == "shortonly":
                num_trades = int(np.sum(y_pred == 0))
            else:  # longshort
                num_trades = int(len(y_pred))
            
            metrics.update({
                "cum_return": cum_return,
                "mean_return": mean_return,
                "volatility": std_return,
                "sharpe": sharpe,
                "max_drawdown": max_dd,
                "win_rate": win_rate,
                "num_trades": num_trades
            })
        else:
            metrics.update({
                "cum_return": 0.0, "mean_return": 0.0, "volatility": 0.0,
                "sharpe": 0.0, "max_drawdown": 0.0, "win_rate": 0.0, "num_trades": 0
            })
    
    return metrics

# =======================================
# GRAFICAS
# =======================================
# Graficas
def plot_model_comparison(results_df, metric="sharpe", strategy="longonly", figsize=(10, 8)):
    """
    Grafica un único boxplot comparando los modelos seleccionados y agrega
    una tabla resumen con mean ± std para cada modelo.
    """
    # Filtrar por estrategia
    df_strat = results_df[results_df["strategy"] == strategy]
    
    data = []
    labels = []
    summary_rows = []
    
    for model in df_strat["model"].unique():
        model_data = df_strat[df_strat["model"] == model][metric].values
        if len(model_data) > 0:
            data.append(model_data)
            labels.append(model)
            summary_rows.append([model, f"{np.mean(model_data):.4f}", f"{np.std(model_data):.4f}"])
    
    if not data:
        print(f"No hay datos para la estrategia {strategy}")
        return
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.boxplot(data, labels=labels, showmeans=True)
    ax.set_title(f"Comparación de modelos - Estrategia: {strategy} | Métrica: {metric.upper()}")
    ax.set_ylabel(metric.title())
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    
    table = plt.table(
        cellText=summary_rows,
        colLabels=["Modelo", "Mean", "Std"],
        cellLoc="center",
        loc="bottom",
        bbox=[0, -0.3, 1, 0.2]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    
    plt.subplots_adjust(left=0.1, bottom=0.25)
    plt.show()