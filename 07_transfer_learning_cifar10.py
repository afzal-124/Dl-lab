# Experiment 7: Transfer learning with pre-trained MobileNetV2 on CIFAR-10
# Install: pip install tensorflow numpy matplotlib
# Note: First run downloads CIFAR-10 and ImageNet weights. Training may take time.

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

tf.keras.utils.set_random_seed(42)
IMAGE_SIZE = (96, 96)
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
y_train = y_train.reshape(-1)
y_test = y_test.reshape(-1)

# Resize and preprocess in batches to avoid holding resized copies in memory.
def prepare_image(image, label):
    image = tf.image.resize(tf.cast(image, tf.float32), IMAGE_SIZE)
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    return image, label

train_ds = (
    tf.data.Dataset.from_tensor_slices((x_train, y_train))
    .shuffle(10000, seed=42)
    .map(prepare_image, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)
test_ds = (
    tf.data.Dataset.from_tensor_slices((x_test, y_test))
    .map(prepare_image, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(96, 96, 3), include_top=False, weights="imagenet"
)
base_model.trainable = False

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(96, 96, 3)),
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax"),
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
history = model.fit(train_ds, epochs=5, validation_data=test_ds)

loss, test_accuracy = model.evaluate(test_ds, verbose=0)
class_names = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]
images, labels = next(iter(test_ds))
probabilities = model.predict(images[:1], verbose=0)
predicted = int(np.argmax(probabilities[0]))

print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
print("Actual:", class_names[int(labels[0])])
print("Predicted:", class_names[predicted])

plt.figure(figsize=(7, 4))
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
