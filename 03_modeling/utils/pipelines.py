import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import logging
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)

from utils.general import (
    create_windows_multivariate_np
)

from utils.random_search import TimeSeriesRandomSearchCV
from utils.cross_validation import PurgedWalkForwardCV


# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

def run_pipeline_random_search(df, target_col='Target', return_col='Return', 
                                 window_size=30, horizon=1, test_size=0.2,
                                 strategy='longonly', scoring='sharpe', n_iter=20):
    """
    Ejecuta el pipeline completo: prepara datos, genera secuencias, corre Random Search y plotea resultados.
    Devuelve la instancia TimeSeriesRandomSearchCV con los resultados en .results_.
    """
    print(f"Configuración del experimento:")
    print(f"  Estrategia: {strategy}")
    print(f"  Métrica: {scoring}")
    print(f"  Iteraciones: {n_iter}")
    
    # Separar train/test (simple split temporal)
    n_test = int(len(df) * test_size)
    df_train = df.iloc[:-n_test]
    
    feature_cols = [c for c in df.columns if c not in [target_col, return_col]]
    X_raw = df_train[feature_cols].values
    y_raw = df_train[target_col].values
    returns_raw = df_train[return_col].values
    
    # Crear secuencias
    X_seq, y_seq = create_windows_multivariate_np(X_raw, y_raw, window_size, horizon)
    returns_seq = returns_raw[window_size + horizon - 1:]
    
    # Índices temporales para PurgedWalkForwardCV
    seq_indices = df_train.index[window_size + horizon - 1:]
    pred_times = pd.Series(seq_indices, index=seq_indices)
    eval_times = pd.Series(seq_indices + pd.Timedelta(hours=horizon), index=seq_indices)
    
    # Grid de hiperparámetros
    param_grid = {
        'learning_rate': [0.0001, 0.0005, 0.001, 0.005],
        'batch_size': [16, 32, 64],
        'epochs': [20, 30, 50],
        'dropout': [0.1, 0.2, 0.3],
        'lstm_units': [32, 64, 128],
        'cnn_filters': [16, 32, 64],
        'kernel_size': [2, 3, 5]
    }
    
    # Instanciar Random Search
    random_search = TimeSeriesRandomSearchCV(
        param_grid=param_grid,
        n_iter=n_iter,
        scoring=scoring,
        strategy=strategy,
        cv_params={
            'n_splits': 5,
            'n_test_splits': 1,
            'min_train_splits': 2
        }
    )
    
    model_types = ['LSTM', 'GRU', 'LSTM_CNN', 'GRU_CNN']
    
    # Ejecutar búsqueda
    random_search.fit(
        X=X_seq,
        y=y_seq,
        pred_times=pred_times,
        eval_times=eval_times,
        returns=returns_seq,
        model_types=model_types
    )
    
    # Plotear resultados
    random_search.plot_results()
    
    return random_search

# ============================================
# Evaluar modelos con Purged Walk-Forward
# ============================================
def run_pipeline_evaluate_models(df, models_dict, target_col="Target", return_col="Return", 
                             window_size=30, horizon=1, strategy="longonly", cv_params=None):
    """
    Evalúa un conjunto de modelos seleccionados usando Purged Walk-Forward CV 
    y la función evaluate_metrics SOLO para la estrategia elegida.
    """
    if cv_params is None:
        cv_params = {
            "n_splits": 5,
            "n_test_splits": 1,
            "min_train_splits": 2
        }
    
    # Features, target y retornos
    feature_cols = [c for c in df.columns if c not in [target_col, return_col]]
    X_raw = df[feature_cols].values
    y_raw = df[target_col].values
    returns_raw = df[return_col].values
    
    # Crear secuencias
    X_seq, y_seq = create_windows_multivariate_np(X_raw, y_raw, window_size, horizon)
    returns_seq = returns_raw[window_size + horizon - 1:]
    
    # Índices temporales
    seq_indices = df.index[window_size + horizon - 1:]
    pred_times = pd.Series(seq_indices, index=seq_indices)
    eval_times = pd.Series(seq_indices + pd.Timedelta(hours=horizon), index=seq_indices)
    
    # CV
    cv = PurgedWalkForwardCV(**cv_params)
    
    results = []
    
    for model_name, model in models_dict.items():
        print(f"\nEvaluando modelo: {model_name}")
        
        for fold, (train_idx, val_idx) in enumerate(cv.split(X=pd.DataFrame(index=seq_indices), 
                                                             y=pd.Series(y_seq, index=seq_indices), 
                                                             pred_times=pred_times, 
                                                             eval_times=eval_times)):
            if len(val_idx) < 20 or len(train_idx) < 50:
                continue
            
            X_val, y_val, returns_val = X_seq[val_idx], y_seq[val_idx], returns_seq[val_idx]
            
            # Escalar con stats del train
            scaler = StandardScaler()
            n_samples, seq_len, n_features = X_seq[train_idx].shape
            X_train_scaled = scaler.fit_transform(X_seq[train_idx].reshape(-1, n_features)).reshape(n_samples, seq_len, n_features)
            X_val_scaled = scaler.transform(X_val.reshape(-1, n_features)).reshape(X_val.shape[0], seq_len, n_features)
            
            # Predecir
            y_pred_proba = model.predict(X_val_scaled, verbose=0).ravel()
            y_pred = (y_pred_proba > 0.5).astype(int)
            
            # Evaluar SOLO la estrategia pedida
            metrics = evaluate_metrics(y_val, y_pred, returns_val, strategy=strategy)
            metrics["model"] = model_name
            metrics["fold"] = fold
            metrics["strategy"] = strategy
            results.append(metrics)
    
    results_df = pd.DataFrame(results)

    # Gráfico
    plot_model_comparison(results_df, metric="sharpe", strategy=strategy)

    return results_df
