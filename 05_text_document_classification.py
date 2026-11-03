# Experiment 5: Deep learning for textual document classification
# Uses the IMDB movie-review dataset.
# Install: pip install tensorflow numpy

import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Embedding, GlobalAveragePooling1D, Dense

tf.keras.utils.set_random_seed(42)
VOCAB_SIZE = 10000
MAX_LENGTH = 200

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(
    num_words=VOCAB_SIZE
)
x_train = tf.keras.utils.pad_sequences(x_train, maxlen=MAX_LENGTH)
x_test = tf.keras.utils.pad_sequences(x_test, maxlen=MAX_LENGTH)

model = Sequential([
    Input(shape=(MAX_LENGTH,)),
    Embedding(VOCAB_SIZE, 32),
    GlobalAveragePooling1D(),
    Dense(64, activation="relu"),
    Dense(1, activation="sigmoid"),
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(x_train, y_train, epochs=5, batch_size=32, validation_split=0.2)

loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"\nTest Accuracy: {accuracy * 100:.2f}%")

probabilities = model.predict(x_test[:5], verbose=0).flatten()
print("\nReview Classification (first five test reviews):")
for i, probability in enumerate(probabilities, start=1):
    label = "Positive" if probability >= 0.5 else "Negative"
    print(f"Review {i}: {label} (probability: {probability:.4f})")
