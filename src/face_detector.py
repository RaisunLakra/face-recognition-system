import cv2
import face_recognition
import numpy as np
import os

from src.utils import load_encodings

class FaceDetector:
    def __init__(self, faceCascadePath):
        """Initialize Haar Cascade for face detection."""
        if not os.path.exists(faceCascadePath):
            raise FileNotFoundError(f"Cascade file not found: {faceCascadePath}")

        self.faceCascade = cv2.CascadeClassifier(faceCascadePath)

        # Check if the cascade is loaded properly
        if self.faceCascade.empty():
            raise ValueError(f"Error loading cascade file: {faceCascadePath}")

    def detect(self, image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)):
        """Detect faces and return bounding box coordinates."""
        rects = self.faceCascade.detectMultiScale(image, scaleFactor=scaleFactor, minNeighbors=minNeighbors, minSize=minSize)
        return rects

def detect_live(encodings_file="face_encodings.pkl"):
    """Perform live face recognition using Haar Cascades and face_recognition."""
    try:
        data = load_encodings(encodings_file)
    except FileNotFoundError as e:
        print("Error while loading encoded file.")
        print(e)
        return

    # Start video capture
    video_cap = cv2.VideoCapture(0)

    # Ensure camera is accessible
    if not video_cap.isOpened():
        print("Error: Could not open camera.")
        return

    # Load Haar Cascade
    frontal_face_Cascade_Path = "src/haarcascade_frontalface_default.xml"
    try:
        fd = FaceDetector(frontal_face_Cascade_Path)
    except (FileNotFoundError, ValueError) as e:
        print(e)
        return

    while True:
        ret, frame = video_cap.read()
        if not ret:
            print("Failed to capture image")
            break

        # Resize the frame for processing speed
        frame_resized = cv2.resize(frame, (640, 480))

        # Convert to grayscale for face detection
        gray_image = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)

        # Detect faces using Haar Cascade
        faces = fd.detect(gray_image)

        # Convert frame to RGB for face recognition
        rgb_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

        for (x, y, w, h) in faces:
            # Crop the detected face from the frame
            face_image = rgb_frame[y:y+h, x:x+w]

            # Encode the detected face
            # face_encodings = face_recognition.face_encodings(rgb_frame, [(y, x + w, y + h, x)])
            face_encodings = face_recognition.face_encodings(rgb_frame, [(y, x + w, y + h, x)])

            if face_encodings:
                face_encoding = face_encodings[0]

                # Compare detected face encoding with stored encodings
                face_distances = face_recognition.face_distance(data["encodings"], face_encoding)
                best_match_index = np.argmin(face_distances)

                # Set a strict threshold (0.4) for recognition
                threshold = 0.4
                if face_distances[best_match_index] < threshold:
                    name = data["names"][best_match_index]
                    confidence = 1 - face_distances[best_match_index]
                else:
                    name = "Unknown"
                    confidence = 0

                # Draw rectangle and label with confidence
                label = f"{name} ({confidence:.2f})"
                cv2.rectangle(frame_resized, (x, y), (x + w, y + h), (127, 255, 0), 2)
                cv2.putText(frame_resized, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (127, 255, 0), 2)

        # Show the video feed
        cv2.imshow("Live Video - Face Recognition", frame_resized)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord("q") or cv2.getWindowProperty("Live Video - Face Recognition", cv2.WND_PROP_VISIBLE) < 1:
            break

    # Release resources
    video_cap.release()
    cv2.destroyAllWindows()
