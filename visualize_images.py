
import os
import cv2
import matplotlib.pyplot as plt

base_path = "data/hazelnut"

# Normal image
good_path = os.path.join(base_path, "train", "good")
good_image_name = os.listdir(good_path)[0]

good_image = cv2.imread(os.path.join(good_path, good_image_name))
good_image = cv2.cvtColor(good_image, cv2.COLOR_BGR2RGB)

# Defective image
defect_path = os.path.join(base_path, "test", "crack")
defect_image_name = os.listdir(defect_path)[0]

defect_image = cv2.imread(os.path.join(defect_path, defect_image_name))
defect_image = cv2.cvtColor(defect_image, cv2.COLOR_BGR2RGB)

# Display images
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(good_image)
plt.title("Normal Hazelnut")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(defect_image)
plt.title("Defective Hazelnut")
plt.axis("off")

plt.tight_layout()
plt.show()
