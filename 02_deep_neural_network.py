# Experiment 2: Deep Neural Network for Iris classification
# Install: pip install numpy scikit-learn tensorflow

import numpy as np
import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense

tf.keras.utils.set_random_seed(42)

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42, stratify=iris.target
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = Sequential([
    Input(shape=(4,)),
    Dense(16, activation="relu"),
    Dense(12, activation="relu"),
    Dense(8, activation="relu"),
    Dense(3, activation="softmax"),
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()
model.fit(X_train, y_train, epochs=50, batch_size=8, verbose=1)

loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
predictions = np.argmax(model.predict(X_test, verbose=0), axis=1)

print(f"\nTest Accuracy: {accuracy * 100:.2f}%")
print("Predicted Classes:", predictions)
print("Actual Classes:   ", y_test)
