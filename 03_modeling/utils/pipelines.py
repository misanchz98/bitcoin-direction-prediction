
import pandas as pd
import numpy as np
from utils.general import create_windows_multivariate_np
from utils.random_search import TimeSeriesRandomSearchCV

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

