""" Filter two image from each folder having more than 10."""

import random
import os
import shutil

datasets_path = "../Datasets/images/lfw_funneled/"
test_image_path = "../src/test_dataset/"

if not os.path.exists(test_image_path):
    os.makedirs(test_image_path, exist_ok=True)
    print("Test dataset folder created.")
    print("Creating test dataset...")
else:
    print("Test dataset folder already exists.")
    print("Cleaning test dataset...")
    shutil.rmtree(test_image_path)
    os.makedirs(test_image_path, exist_ok=True)
    print("Test dataset folder removed and recreated.")
    print("Creating test dataset...")
    
counter = 0
for person in os.listdir(datasets_path):
    folder_path = os.path.join(datasets_path, person)
    if os.path.isdir(folder_path):
        photos = [photo for photo in os.listdir(folder_path) if photo.lower().endswith(('.jpg','.png', '.jpeg'))]
        if len(photos) >= 10:
            selected_photos = random.sample(photos, 2)
        else:
            continue

        counter += 1
        os.makedirs(test_image_path, exist_ok=True)
        os.makedirs(os.path.join(test_image_path, person), exist_ok=True)

        for image in selected_photos:
            src = os.path.join(folder_path, image)
            dst = os.path.join(test_image_path, person, image)
            shutil.copy(src, dst)
    else:
        print(f"{folder_path} is not a directory.")
print("{} new folders will be created.".format(counter))