
import os

dataset_path = "data/hazelnut"

print("Checking dataset...")

if os.path.exists(dataset_path):
    print("Dataset found!")

    print("Hazelnut folders:")
    print(os.listdir(dataset_path))

else:
    print("Dataset path not found!")
