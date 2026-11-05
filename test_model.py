
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Load trained model
model = tf.keras.models.load_model("hazelnut_autoencoder.keras")

# Function to load image
def load_image(path):
    image = cv2.imread(path)

    if image is None:
        raise FileNotFoundError(f"Image not found: {path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (128, 128))
    image = image.astype("float32") / 255.0

    return np.expand_dims(image, axis=0)

# Image paths
good_path = "data/hazelnut/test/good"
defect_path = "data/hazelnut/test/crack"

good_file = good_path + "/" + sorted(__import__("os").listdir(good_path))[0]
defect_file = defect_path + "/" + sorted(__import__("os").listdir(defect_path))[0]

# Load images
good_image = load_image(good_file)
defect_image = load_image(defect_file)

# Reconstruct images
good_reconstructed = model.predict(good_image)
defect_reconstructed = model.predict(defect_image)

# Calculate reconstruction errors
good_error = np.mean((good_image - good_reconstructed) ** 2)
defect_error = np.mean((defect_image - defect_reconstructed) ** 2)

print("Normal image error:", good_error)
print("Defective image error:", defect_error)

# Display results
plt.figure(figsize=(12, 8))

images = [
    good_image[0],
    good_reconstructed[0],
    defect_image[0],
    defect_reconstructed[0]
]

titles = [
    "Original Normal",
    "Reconstructed Normal",
    "Original Defective",
    "Reconstructed Defective"
]

for i in range(4):
    plt.subplot(2, 2, i + 1)
    plt.imshow(np.clip(images[i], 0, 1))
    plt.title(titles[i])
    plt.axis("off")

plt.tight_layout()
plt.show()
