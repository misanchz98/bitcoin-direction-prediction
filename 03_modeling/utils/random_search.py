import random
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from utils.models import build_lstm, build_gru, build_lstm_cnn, build_gru_cnn
from utils.callbacks import SharpeCallback
from utils.cross_validation import PurgedWalkForwardCV

class TimeSeriesRandomSearchCV:
    """
    Clase que implementa Random Search combinado con Purged Walk-Forward CV
    Guarda por cada combinación: mean_score, std_score, n_folds, fold_scores y los hyperparams.
    """
    def __init__(self, 
                 param_grid: Dict,
                 cv_params: Dict = None,
                 n_iter: int = 10,
                 scoring: str = 'sharpe',
                 strategy: str = 'longonly',
                 random_state: int = 42):
        
        self.param_grid = param_grid
        self.n_iter = n_iter
        self.scoring = scoring
        self.strategy = strategy
        self.random_state = random_state
        
        self.cv_params = cv_params or {
            'n_splits': 5,
            'n_test_splits': 1,
            'min_train_splits': 2
        }
        
        self.results_ = []
        self.best_params_ = None
        self.best_score_ = -np.inf
        
        random.seed(random_state)
        np.random.seed(random_state)
        tf.random.set_seed(random_state)

    def _generate_param_combinations(self) -> List[Dict]:
        """Genera n_iter combinaciones aleatorias (puede haber repeticiones)."""
        combinations = []
        for _ in range(self.n_iter):
            combination = {}
            for param, values in self.param_grid.items():
                combination[param] = random.choice(values)
            combinations.append(combination)
        return combinations

    def _create_model(self, model_type: str, input_shape: Tuple, params: Dict):
        """Crea modelo según tipo."""
        if model_type == 'LSTM':
            return build_lstm(input_shape, **params)
        elif model_type == 'GRU':
            return build_gru(input_shape, **params)
        elif model_type == 'LSTM_CNN':
            return build_lstm_cnn(input_shape, **params)
        elif model_type == 'GRU_CNN':
            return build_gru_cnn(input_shape, **params)
        else:
            raise ValueError(f"Tipo de modelo no soportado: {model_type}")

    def _calculate_sharpe_ratio(self, y_true, y_pred, returns, strategy=None):
        """
        Calcula Sharpe anualizado para la predicción y_pred y returns.
        (El parámetro y_true está aquí por compatibilidad pero no se usa para Sharpe).
        """
        if strategy is None:
            strategy = self.strategy
        if len(returns) == 0:
            return 0.0
        
        returns = np.array(returns)
        y_pred = np.array(y_pred)
        
        if strategy == 'longonly':
            strategy_returns = np.where(y_pred == 1, returns, 0)
        elif strategy == 'longshort':
            strategy_returns = np.where(y_pred == 1, returns, -returns)
        elif strategy == 'shortonly':
            strategy_returns = np.where(y_pred == 0, -returns, 0)
        else:
            strategy_returns = np.where(y_pred == 1, returns, 0)
        
        if len(strategy_returns) == 0 or strategy_returns.std() <= 1e-8:
            return 0.0
        
        mean_return = strategy_returns.mean()
        std_return = strategy_returns.std()
        return mean_return / std_return * np.sqrt(252)

    def _evaluate_fold(self, y_true, y_pred, returns):
        """Devuelve métricas base y sharpe calculado."""
        base_metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'sharpe': self._calculate_sharpe_ratio(y_true, y_pred, returns)
        }
        return base_metrics

    def get_model_by_id(self, comb_id: str):
        """
        Devuelve el modelo asociado a un comb_id específico.
        Ejemplo: get_model_by_id("LSTM_CNN_comb3")
        """
        for res in self.results_:
            if res.get("comb_id") == comb_id:
                return res["model"]
        raise ValueError(f"No se encontró combinación {comb_id}")


    def fit(self, X, y, pred_times, eval_times, returns, model_types=['LSTM', 'GRU']):
        """
        Ejecuta Random Search con PurgedWalkForwardCV.
        Guarda por combinación: mean_score, std_score, n_folds, fold_scores y los hyperparams.
        """
        
        df_index = pd.DataFrame(index=pred_times.index)
        y_series = pd.Series(y, index=pred_times.index)
        
        param_combinations = self._generate_param_combinations()
        
        print(f"Ejecutando Random Search con {self.n_iter} combinaciones y {len(model_types)} modelos...")
        print(f"Estrategia de trading: {self.strategy}")
        print(f"Métrica de optimización: {self.scoring}")
        
        cv = PurgedWalkForwardCV(**self.cv_params)
        
        total_iterations = len(param_combinations) * len(model_types)
        current_iteration = 0
        
        for param_idx, params in enumerate(param_combinations):
            print(f"\nCombinación {param_idx + 1}/{len(param_combinations)}: {params}")
            
            for model_type in model_types:
                current_iteration += 1
                print(f"  Evaluando {model_type} ({current_iteration}/{total_iterations})")
                
                fold_scores = []
                
                for fold, (train_idx, val_idx) in enumerate(cv.split(X=df_index, y=y_series, pred_times=pred_times, eval_times=eval_times)):
                    
                    if len(val_idx) < 20 or len(train_idx) < 50:
                        # Evitar folds demasiado pequeños
                        continue
                    
                    # Preparar datos del fold
                    X_train, y_train = X[train_idx], y[train_idx]
                    X_val, y_val = X[val_idx], y[val_idx]
                    returns_val = returns[val_idx]
                    
                    # Escalado por feature (fit en train, transform en val)
                    scaler = StandardScaler()
                    n_samples, seq_len, n_features = X_train.shape
                    
                    X_train_scaled = scaler.fit_transform(X_train.reshape(-1, n_features)).reshape(n_samples, seq_len, n_features)
                    X_val_scaled = scaler.transform(X_val.reshape(-1, n_features)).reshape(X_val.shape[0], seq_len, n_features)
                    
                    try:
                        # Crear y entrenar modelo
                        model = self._create_model(model_type, X_train.shape[1:], params)
                        
                        # Callbacks incluyendo el SharpeCallback
                        sharpe_callback = SharpeCallback(X_val_scaled, y_val, returns_val, threshold=0.5, strategy=self.strategy, verbose=0)
                        callbacks = [
                            EarlyStopping(monitor="val_sharpe", patience=7, mode='max', restore_best_weights=False),
                            ReduceLROnPlateau(monitor="val_loss", patience=3, factor=0.5, min_lr=1e-6),
                            sharpe_callback
                        ]
                        
                        model.fit(
                            X_train_scaled, y_train,
                            validation_data=(X_val_scaled, y_val),
                            epochs=params.get('epochs', 50),
                            batch_size=params.get('batch_size', 32),
                            callbacks=callbacks,
                            verbose=0
                        )
                        
                        # Restaurar los mejores pesos según el callback Sharpe
                        sharpe_callback.restore_best_weights()
                        
                        # Predicción y evaluación del fold
                        y_pred_proba = model.predict(X_val_scaled, verbose=0).ravel()
                        y_pred = (y_pred_proba > 0.5).astype(int)
                        
                        fold_metrics = self._evaluate_fold(y_val, y_pred, returns_val)
                        
                        if self.scoring == 'sharpe':
                            # Preferir el best_sharpe encontrado por el callback si existe
                            score = sharpe_callback.best_sharpe if hasattr(sharpe_callback, 'best_sharpe') and sharpe_callback.best_sharpe > -np.inf else fold_metrics.get('sharpe', 0)
                        else:
                            score = fold_metrics.get(self.scoring, 0)
                        
                        fold_scores.append(score)
                        
                    except Exception as e:
                        print(f"    Error en fold {fold}: {str(e)}")
                        continue
                
                # Guardar resultado por combinación si hay folds válidos
                if fold_scores:
                    mean_score = np.mean(fold_scores)
                    std_score = np.std(fold_scores)
                    
                    result = {
                        'model_type': model_type,
                        'comb_id': f"{model_type}_comb{len([r for r in self.results_ if r['model_type']==model_type])+1}",
                        'mean_score': mean_score,
                        'std_score': std_score,
                        'n_folds': len(fold_scores),
                        'strategy': self.strategy,
                        'fold_scores': fold_scores,
                        'model': model,
                        **params
                    }
                    self.results_.append(result)
                    
                    if mean_score > self.best_score_:
                        self.best_score_ = mean_score
                        self.best_params_ = {
                            'model_type': model_type,
                            'strategy': self.strategy,
                            **params
                        }
                    
                    print(f"    Score: {mean_score:.4f} ± {std_score:.4f} (n_folds={len(fold_scores)})")
                else:
                    print(f"    Sin resultados válidos para {model_type}")
        
        print(f"\nMejor configuración encontrada:")
        print(f"Score ({self.scoring}): {self.best_score_:.4f}")
        print(f"Estrategia: {self.strategy}")
        print(f"Parámetros: {self.best_params_}")
        
        return self

    def get_results_df(self):
        """Retorna DataFrame con todos los resultados (cada fila = una combinación)."""
        return pd.DataFrame(self.results_)

    def plot_results(self, figsize=(15, 12)):
        """
        Genera visualizaciones SOLO de las gráficas 3 a 6:
        - Para cada modelo: boxplot donde cada caja = una combinación (usa fold_scores)
        - Debajo de cada boxplot se dibuja una tabla con mean_score y std_score por combinación
        """
        if not self.results_:
            print("No hay resultados para mostrar. Ejecuta fit() primero.")
            return
        
        results_df = self.get_results_df()
        
        # Solo 4 subplots (2 filas x 2 columnas)
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(
            f'Random Search - Estrategia: {getattr(self, "strategy", "N/A").upper()} | Métrica: {self.scoring.upper()}',
            fontsize=16
        )
        
        model_types = ['LSTM', 'GRU', 'LSTM_CNN', 'GRU_CNN']
        positions = [(0,0), (0,1), (1,0), (1,1)]
        
        for model_type, pos in zip(model_types, positions):
            model_data = results_df[results_df['model_type'] == model_type].reset_index(drop=True)
            ax = axes[pos]
            
            if not model_data.empty:
                # Extraer scores por fold de cada combinación
                fold_scores_list = model_data['fold_scores'].tolist()
                labels = [f"comb_{i+1}" for i in range(len(fold_scores_list))]
                
                # Boxplot
                ax.boxplot(fold_scores_list, labels=labels, showmeans=True)
                ax.set_title(f'{model_type} - Boxplots por Combinación ({len(labels)} combos)')
                ax.set_ylabel(f'{self.scoring.title()} (por fold)')
                ax.set_xlabel('')
                ax.tick_params(axis='x')
                
                # ======= TABLA con mean y std de cada combinación ==========
                table_data = []
                for i, row in model_data.iterrows():
                    table_data.append([
                        f"{model_type}_comb{i+1}",
                        f"{row['mean_score']:.4f}",
                        f"{row['std_score']:.4f}"
                    ])
                
                table = ax.table(
                    cellText=table_data,
                    colLabels=['Modelo', 'Mean Score', 'Std Score'],
                    cellLoc='center',
                    loc='bottom',
                    bbox=[0, -0.45, 1, 0.35]  # ajusta la posición y altura
                )
                table.auto_set_font_size(False)
                table.set_fontsize(8)
            else:
                ax.text(
                    0.5, 0.5, f'Sin datos\npara {model_type}',
                    transform=ax.transAxes, ha='center', va='center',
                    fontsize=12, bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8)
                )
                ax.set_title(f'{model_type} - Sin datos')
                ax.set_xticks([])
                ax.set_yticks([])
        
        plt.tight_layout()
        plt.show()
