
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Load trained model
model = tf.keras.models.load_model(
    "hazelnut_autoencoder.keras"
)

# Experimental classification threshold
THRESHOLD = 0.0007553650

def detect_defect(image_path):

    # Load image
    image = cv2.imread(image_path)

    if image is None:
        print("Image not found!")
        return

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (128, 128))
    image = image.astype("float32") / 255.0

    original = np.expand_dims(image, axis=0)

    # Reconstruct image
    reconstructed = model.predict(original, verbose=0)

    # Calculate pixel-wise reconstruction error
    error_map = np.mean(
        (original - reconstructed) ** 2,
        axis=3
    )[0]

    # Calculate overall reconstruction error
    error = np.mean(error_map)

    print("\nReconstruction Error:", error)

    # Classify product
    if error > THRESHOLD:
        print("RESULT: DEFECTIVE PRODUCT")
    else:
        print("RESULT: NORMAL PRODUCT")

    # Create binary defect mask
    pixel_threshold = 0.002

    defect_mask = (
        error_map > pixel_threshold
    ).astype(np.uint8)

    print(
        "Highlighted defect pixels:",
        np.sum(defect_mask)
    )

    # Display original and reconstructed images
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(reconstructed[0])
    plt.title("Reconstructed Image")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(image)
    plt.imshow(error_map, cmap="jet", alpha=0.6)
    plt.title("Defect Heatmap")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    # Display heatmap and binary mask
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(error_map, cmap="jet")
    plt.title("Anomaly Heatmap")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(defect_mask, cmap="gray")
    plt.title("Binary Defect Mask")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


# Test a defective hazelnut
detect_defect("data/hazelnut/test/crack/000.png")
