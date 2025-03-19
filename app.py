# Name: Raisun Lakra
# Description: Develop a face recognition system using opencv.
# Date: 18/03/2025
# Version: 1.0

import cv2
import os
from src.face_detector import detect_live
from src.face_encoder import encode_faces
from src.face_recognizer import recognize_faces
from src.utils import load_encodings

DATASET_PATH = "Datasets/processed_images/"
ENCODINGS_FILE = "face_encodings.pkl"
TEST_DATASET_PATH = "src/test_dataset/"

if not os.path.exists(ENCODINGS_FILE):
    if not os.path.exists(DATASET_PATH) or not os.listdir(DATASET_PATH):
        print("Error: Dataset directory is empty or does not exist.")
        exit(1)
    print("Encoding faces...")
    encode_faces(DATASET_PATH, ENCODINGS_FILE)

def main():
    print("Welcome to Face Recognition App!")
    print("1. Live Face Detection")
    print("2. Recognize Faces from Test Image")
    print("3. Exit")
    choice = input("Choose an option (1 or 2): ")

    if choice == "1":
        detect_live()
    elif choice == "2":
        # Load encodings
        try:
            data = load_encodings(ENCODINGS_FILE)
        except FileNotFoundError:
            print("Encoding file not found. Run the encoding process first.")
            return

        # List test images
        test_images = []
        for folder in os.listdir(TEST_DATASET_PATH):
            folder_path = os.path.join(TEST_DATASET_PATH, folder)
            if os.path.isdir(folder_path):
                for image in os.listdir(folder_path):
                    if image.lower().endswith(('.jpg','.png', '.jpeg')):
                        test_images.append(os.path.join(folder, image))

        if not test_images:
            print("No test images found.")
            return

        print("Available test images:")
        for i, image in enumerate(test_images):
            print(f"{i + 1}. {image}")

        # Let user select an image
        selected_index = int(input("Select an image by number: ")) - 1
        selected_image = test_images[selected_index]
        image_path = os.path.join(TEST_DATASET_PATH, selected_image)

        # Read and recognize faces
        frame = cv2.imread(image_path)
        if frame is not None:
            recognized_frame = recognize_faces(frame, data)
            cv2.imshow("Recognized Faces", recognized_frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("Failed to load image.")
    
    elif choice == "3":
        print("Exiting...")
        exit(0)
    else:
        print("Invalid choice. Exiting.")

if __name__ == "__main__":
    # Encode faces if not already done
    if not os.path.exists(ENCODINGS_FILE):
        print("Encoding faces...")
        encode_faces(DATASET_PATH, ENCODINGS_FILE)

    main()