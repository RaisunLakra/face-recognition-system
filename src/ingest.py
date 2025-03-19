""" Filter the folders based on image data should be greater than 10 and less than 20."""

import random
import os
import shutil

datasets_path = "../Datasets/images/lfw_funneled/"
processed_image_path = "../Datasets/processed_images/"

if not os.path.exists(processed_image_path):
    os.makedirs(processed_image_path, exist_ok=True)
    print("Test dataset folder created.")
    print("Creating test dataset...")
else:
    print("Test dataset folder already exists.")
    print("Cleaning test dataset...")
    shutil.rmtree(processed_image_path)
    os.makedirs(processed_image_path, exist_ok=True)
    print("Test dataset folder removed and recreated.")
    print("Creating test dataset...")

counter = 0
for person in os.listdir(datasets_path):
    folder_path = os.path.join(datasets_path, person)
    if os.path.isdir(folder_path):
        photos = [photo for photo in os.listdir(folder_path) if photo.lower().endswith(('.jpg','.png', '.jpeg'))]
        if 10 <= len(photos) <= 20:
            selected_photos = photos
        elif len(photos) > 20:
            selected_photos = random.sample(photos, 20)
        else:
            continue

        counter += 1
        os.makedirs(processed_image_path, exist_ok=True)
        os.makedirs(os.path.join(processed_image_path, person), exist_ok=True)

        for image in selected_photos:
            src = os.path.join(folder_path, image)
            dst = os.path.join(processed_image_path, person, image)
            shutil.copy(src, dst)
    else:
        print(f"{folder_path} is not a directory.")
print("{} new folders will be created.".format(counter))