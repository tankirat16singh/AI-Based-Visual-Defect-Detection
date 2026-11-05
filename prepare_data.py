
import os
import cv2
import numpy as np

dataset_path = "data/hazelnut/train/good"

images = []

for filename in os.listdir(dataset_path):

    image_path = os.path.join(dataset_path, filename)

    image = cv2.imread(image_path)

    if image is None:
        continue

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = cv2.resize(image, (128, 128))

    image = image / 255.0

    images.append(image)

X_train = np.array(images, dtype=np.float32)

print("Dataset prepared successfully!")
print("Training data shape:", X_train.shape)
print("Number of images:", len(X_train))
