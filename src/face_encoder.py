import os
import cv2
import face_recognition
import pickle

def encode_image(image_path):
    image = cv2.imread(image_path)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect face locations
    face_locations = face_recognition.face_locations(rgb_image)

    # Get face encodings
    encodings = face_recognition.face_encodings(rgb_image, face_locations)

    return encodings

def encode_faces(dataset_path, encodings_file):
    known_encodings = []
    known_names = []

    for person_name in os.listdir(dataset_path):
        person_dir = os.path.join(dataset_path, person_name)

        if not os.path.isdir(person_dir):
            continue

        for image_name in os.listdir(person_dir):
            image_path = os.path.join(person_dir, image_name)
            print(f"Processing {image_path}")

            image = cv2.imread(image_path)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Detect face locations
            face_locations = face_recognition.face_locations(rgb_image)

            # Get face encodings
            encodings = face_recognition.face_encodings(rgb_image, face_locations)

            for encoding in encodings:
                known_encodings.append(encoding)
                known_names.append(person_name)

    # Save encodings
    data = {"encodings": known_encodings, "names": known_names}
    with open(encodings_file, "wb") as f:
        pickle.dump(data, f)

    print("Encoding complete and saved.")