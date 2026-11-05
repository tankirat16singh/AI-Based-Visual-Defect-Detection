
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# Load prepared training data
from prepare_data import X_train

# Build CNN Autoencoder

input_img = layers.Input(shape=(128, 128, 3))

# Encoder
x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(input_img)
x = layers.MaxPooling2D((2, 2), padding="same")(x)

x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
x = layers.MaxPooling2D((2, 2), padding="same")(x)

# Decoder
x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
x = layers.UpSampling2D((2, 2))(x)

x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
x = layers.UpSampling2D((2, 2))(x)

output_img = layers.Conv2D(
    3, (3, 3), activation="sigmoid", padding="same"
)(x)

# Create model
autoencoder = models.Model(input_img, output_img)

autoencoder.compile(
    optimizer="adam",
    loss="mean_squared_error"
)

autoencoder.summary()

# Train model
history = autoencoder.fit(
    X_train,
    X_train,
    epochs=20,
    batch_size=16,
    validation_split=0.1,
    shuffle=True
)

# Save trained model
autoencoder.save("hazelnut_autoencoder.keras")

print("Model trained and saved successfully!")
