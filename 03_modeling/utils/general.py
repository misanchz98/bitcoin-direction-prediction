import numpy as np
import pandas as pd
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

def calculate_max_drawdown(cumulative_returns):
    """Calcula el máximo drawdown a partir de cumulative_returns (array)."""
    if len(cumulative_returns) == 0:
        return 0.0
    
    cumulative_returns = np.array(cumulative_returns)
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdown = cumulative_returns - running_max
    return np.min(drawdown) if len(drawdown) > 0 else 0.0

def evaluate_metrics(y_true, y_pred, returns=None):
    """
    Evalúa métricas de clasificación y, si se provee returns, métricas de estrategia.
    Devuelve diccionario con 'base' y por cada estrategia (longshort, longonly, shortonly).
    """
    base_metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0), 
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred)
    }

    results = {"base": base_metrics}

    if returns is not None:
        returns = np.array(returns)
        strategies = {
            "longshort": np.where(y_pred == 1, returns, -returns),
            "longonly":  np.where(y_pred == 1, returns, 0),
            "shortonly": np.where(y_pred == 0, -returns, 0)
        }

        for strat_name, strat_returns in strategies.items():
            strat_metrics = base_metrics.copy()
            
            if len(strat_returns) > 0:
                cum_return = np.cumsum(strat_returns)[-1]
                mean_return = np.mean(strat_returns)
                std_return = np.std(strat_returns)
                
                if std_return > 1e-8:
                    sharpe = mean_return / std_return * np.sqrt(252)
                else:
                    sharpe = 0.0
                
                max_dd = calculate_max_drawdown(np.cumsum(strat_returns))
                win_rate = np.sum(strat_returns > 0) / len(strat_returns) if len(strat_returns) > 0 else 0
                
                strat_metrics.update({
                    "cum_return": cum_return,
                    "mean_return": mean_return,
                    "volatility": std_return,
                    "sharpe": sharpe,
                    "max_drawdown": max_dd,
                    "win_rate": win_rate,
                    "num_trades": np.sum(y_pred != 0) if strat_name != "longshort" else len(y_pred)
                })
            else:
                strat_metrics.update({
                    "cum_return": 0.0, "mean_return": 0.0, "volatility": 0.0,
                    "sharpe": 0.0, "max_drawdown": 0.0, "win_rate": 0.0, "num_trades": 0
                })
            
            results[strat_name] = strat_metrics

    return results