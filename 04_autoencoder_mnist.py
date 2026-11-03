# Experiment 4: Autoencoder for MNIST image compression
# Install: pip install tensorflow matplotlib numpy

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Model, layers

tf.keras.utils.set_random_seed(42)

(x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype("float32").reshape(-1, 784) / 255.0
x_test = x_test.astype("float32").reshape(-1, 784) / 255.0

inputs = layers.Input(shape=(784,))
encoded = layers.Dense(32, activation="relu")(inputs)
decoded = layers.Dense(784, activation="sigmoid")(encoded)

autoencoder = Model(inputs, decoded)
encoder = Model(inputs, encoded)

autoencoder.compile(optimizer="adam", loss="binary_crossentropy")
autoencoder.fit(
    x_train, x_train, epochs=5, batch_size=256,
    shuffle=True, validation_data=(x_test, x_test)
)

encoded_images = encoder.predict(x_test[:5], verbose=0)
reconstructed = autoencoder.predict(x_test[:5], verbose=0)

plt.figure(figsize=(10, 4))
for i in range(5):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_test[i].reshape(28, 28), cmap="gray")
    plt.title("Original")
    plt.axis("off")

    plt.subplot(2, 5, i + 6)
    plt.imshow(reconstructed[i].reshape(28, 28), cmap="gray")
    plt.title("Reconstructed")
    plt.axis("off")

plt.tight_layout()
plt.show()

print("Original size:", x_test[:5].shape)
print("Compressed size:", encoded_images.shape)
