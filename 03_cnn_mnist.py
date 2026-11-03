# Experiment 3: CNN image classification using MNIST
# Install: pip install tensorflow numpy

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout

tf.keras.utils.set_random_seed(42)

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype("float32")[..., np.newaxis] / 255.0
x_test = x_test.astype("float32")[..., np.newaxis] / 255.0

model = Sequential([
    Input(shape=(28, 28, 1)),
    Conv2D(32, (3, 3), activation="relu"),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(10, activation="softmax"),
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.1)

loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
predictions = np.argmax(model.predict(x_test[:10], verbose=0), axis=1)

print(f"\nTest Accuracy: {accuracy * 100:.2f}%")
print("Predicted classes:", predictions)
print("Actual classes:   ", y_test[:10])
