# utils/models.py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Conv1D, MaxPooling1D
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

#def build_lstm(input_shape, lstm_units=64, dropout=0.2, learning_rate=0.001, **kwargs):
#    model = Sequential([
#        LSTM(lstm_units, input_shape=input_shape, return_sequences=False),
#        Dropout(dropout),
#        Dense(32, activation="relu"),
#        Dropout(dropout),
#        Dense(1, activation="sigmoid")
#    ])
#    model.compile(
#        optimizer=Adam(learning_rate=learning_rate),
#        loss="binary_crossentropy",
#        metrics=["accuracy"]
#    )
#    return model
def build_lstm(input_shape, lstm_units=64, dropout=0.2, learning_rate=0.001, **kwargs):
    model = Sequential([
        LSTM(lstm_units, 
             input_shape=input_shape, 
             return_sequences=False, 
             activation='tanh', 
             recurrent_activation='sigmoid', 
             use_bias=True),
        Dropout(dropout),
        Dense(32, activation="relu"),
        Dropout(dropout),
        Dense(1, activation="sigmoid")
    ])
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model


def build_gru(input_shape, lstm_units=64, dropout=0.2, learning_rate=0.001, **kwargs):
    model = Sequential([
        GRU(lstm_units, input_shape=input_shape, return_sequences=False),
        Dropout(dropout),
        Dense(32, activation="relu"),
        Dropout(dropout),
        Dense(1, activation="sigmoid")
    ])
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model


#def build_lstm_cnn(input_shape, lstm_units=32, cnn_filters=32, kernel_size=3, dropout=0.2, learning_rate=0.001, **kwargs):
#    model = Sequential([
#        Conv1D(cnn_filters, kernel_size=kernel_size, activation="relu", input_shape=input_shape),
#        MaxPooling1D(pool_size=2),
#        LSTM(lstm_units, return_sequences=False),
#        Dropout(dropout),
#        Dense(16, activation="relu"),
#        Dense(1, activation="sigmoid")
#    ])
#    model.compile(
#        optimizer=Adam(learning_rate=learning_rate),
#        loss="binary_crossentropy",
#        metrics=["accuracy"]
#    )
#    return model
def build_lstm_cnn(input_shape, lstm_units=32, cnn_filters=32, kernel_size=3, dropout=0.2, learning_rate=0.001, **kwargs):
    model = Sequential([
        Conv1D(cnn_filters, kernel_size=kernel_size, activation="relu", input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        LSTM(lstm_units, 
             return_sequences=False, 
             activation='tanh', 
             recurrent_activation='sigmoid', 
             use_bias=True),
        Dropout(dropout),
        Dense(16, activation="relu"),
        Dense(1, activation="sigmoid")
    ])
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model


def build_gru_cnn(input_shape, lstm_units=32, cnn_filters=32, kernel_size=3, dropout=0.2, learning_rate=0.001, **kwargs):
    model = Sequential([
        Conv1D(cnn_filters, kernel_size=kernel_size, activation="relu", input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        GRU(lstm_units, return_sequences=False),
        Dropout(dropout),
        Dense(16, activation="relu"),
        Dense(1, activation="sigmoid")
    ])
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model
