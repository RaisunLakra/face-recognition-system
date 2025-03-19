import cv2
import face_recognition
import numpy as np

window_name = "Live Video - Face Recognition"

def recognize_faces(frame, data):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect face and encode them
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        face_distances = face_recognition.face_distance(data["encodings"], face_encoding)
        best_match_index = np.argmin(face_distances)

        # Set threshold = 0.4 for strict threshold for unknown faces
        threshold = 0.4
        if face_distances[best_match_index] < threshold:
            name = data["names"][best_match_index]
            confidence = 1 - face_distances[best_match_index]
        else:
            name = "Unknown"
            confidence = 0

        # Draw rectangle and label them with confidence
        label = f"{name} ({confidence:.2f})"
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Show image with face recognition
    while True:
        frame_resized = cv2.resize(frame, (int(frame.shape[1] * 1.5), int(frame.shape[0] * 1.5)))
        # cv2.imshow("Recognized Faces", frame)
        cv2.imshow("Recognized Faces", frame_resized)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
            print("Window closed")
            break
        # if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 0:
        #     print("Press q to close window.")
        # if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
        #     break

    cv2.destroyAllWindows()
