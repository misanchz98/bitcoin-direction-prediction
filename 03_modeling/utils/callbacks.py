import numpy as np
import tensorflow as tf

# =======================================
# CALLBACK PERSONALIZADO PARA SHARPE
# =======================================

class SharpeCallback(tf.keras.callbacks.Callback):
    """
    Callback que monitorea el ratio de Sharpe durante el entrenamiento,
    guarda los mejores pesos según Sharpe y expone el histórico de Sharpe por época.
    """
    def __init__(self, X_val, y_val, returns_val, threshold=0.5, strategy='longonly', verbose=1):
        super().__init__()
        self.X_val = X_val
        self.y_val = y_val
        self.returns_val = returns_val
        self.threshold = threshold
        self.strategy = strategy
        self.verbose = verbose
        self.best_sharpe = -np.inf
        self.best_weights = None
        self.sharpe_history = []

    def _calculate_sharpe(self, y_pred, returns):
        """Calcula Sharpe anualizado según la estrategia."""
        returns = np.array(returns)
        y_pred = np.array(y_pred)
        
        if self.strategy == 'longonly':
            strategy_returns = np.where(y_pred == 1, returns, 0)
        elif self.strategy == 'longshort':
            strategy_returns = np.where(y_pred == 1, returns, -returns)
        elif self.strategy == 'shortonly':
            strategy_returns = np.where(y_pred == 0, -returns, 0)
        else:
            strategy_returns = np.where(y_pred == 1, returns, 0)
        
        if len(strategy_returns) == 0 or strategy_returns.std() <= 1e-8:
            return 0.0
        
        return strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)

    def on_epoch_end(self, epoch, logs=None):
        # Predicción sobre validación y cálculo de Sharpe
        y_pred_proba = self.model.predict(self.X_val, verbose=0)
        y_pred = (y_pred_proba > self.threshold).astype(int)
        
        sharpe = self._calculate_sharpe(y_pred.flatten(), self.returns_val)
        
        if logs is None:
            logs = {}
        logs["val_sharpe"] = sharpe
        self.sharpe_history.append(sharpe)
        
        if self.verbose > 0:
            print(f" - val_sharpe: {sharpe:.4f}")
        
        if sharpe > self.best_sharpe:
            self.best_sharpe = sharpe
            self.best_weights = self.model.get_weights()

    def restore_best_weights(self):
        """Restaura los mejores pesos guardados (si existen)."""
        if self.best_weights is not None:
            self.model.set_weights(self.best_weights)