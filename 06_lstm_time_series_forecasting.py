# Experiment 6: Time-series forecasting using an LSTM
# Install: pip install numpy matplotlib scikit-learn tensorflow

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense

tf.keras.utils.set_random_seed(42)
rng = np.random.default_rng(42)

time = np.arange(0, 200, 0.1)
data = np.sin(time) + 0.1 * rng.normal(size=len(time))

scaler = MinMaxScaler()
scaled = scaler.fit_transform(data.reshape(-1, 1))

def create_sequences(values, sequence_length):
    X, y = [], []
    for i in range(len(values) - sequence_length):
        X.append(values[i:i + sequence_length, 0])
        y.append(values[i + sequence_length, 0])
    return np.asarray(X), np.asarray(y)

SEQ_LEN = 20
X, y = create_sequences(scaled, SEQ_LEN)
X = X.reshape((X.shape[0], X.shape[1], 1))

split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

model = Sequential([
    Input(shape=(SEQ_LEN, 1)),
    LSTM(50, activation="tanh"),
    Dense(1),
])
model.compile(optimizer="adam", loss="mse")
history = model.fit(
    X_train, y_train, epochs=10, batch_size=32,
    validation_data=(X_test, y_test), verbose=1
)

predictions = model.predict(X_test, verbose=0)
predictions = scaler.inverse_transform(predictions)
actual = scaler.inverse_transform(y_test.reshape(-1, 1))

plt.figure()
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure()
plt.plot(actual, label="Actual")
plt.plot(predictions, label="Predicted")
plt.title("Actual vs Predicted")
plt.xlabel("Test time step")
plt.ylabel("Value")
plt.legend()
plt.tight_layout()
plt.show()
