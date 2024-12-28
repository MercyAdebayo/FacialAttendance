import cv2
import face_recognition
import numpy as np
import os
from datetime import datetime

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EMBEDDINGS_DIR = os.path.join(BASE_DIR, "embeddings")

def load_known_embeddings():
    known_encodings = []
    known_ids = []

    # Loop through all .npy files in the embeddings folder
    for file_name in os.listdir(EMBEDDINGS_DIR):
        if file_name.endswith(".npy"):
            file_path = os.path.join(EMBEDDINGS_DIR, file_name)
            encoding = np.load(file_path)
            student_id = os.path.splitext(file_name)[0]  # e.g., "student_001"
            known_encodings.append(encoding)
            known_ids.append(student_id)

    return known_encodings, known_ids

def main():
    # Load all known embeddings
    known_encodings, known_ids = load_known_embeddings()

    if not known_encodings:
        print("No enrolled students found. Please run enroll.py first.")
        return

    # Open the default webcam
    cap = cv2.VideoCapture(0)
    print("Press 'Q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame. Exiting...")
            break

        # Convert frame to RGB (face_recognition expects RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect face locations
        face_locations = face_recognition.face_locations(rgb_frame)
        # Encode faces
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for face_encoding, face_location in zip(face_encodings, face_locations):
            # Compare face_encoding with known_encodings
            distances = face_recognition.face_distance(known_encodings, face_encoding)
            best_match_index = np.argmin(distances)
            match_threshold = 0.6  # Tweak this for your environment

            if distances[best_match_index] < match_threshold:
                student_id = known_ids[best_match_index]
                # Mark attendance (for this demo, just print to console)
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"[{current_time}] Recognized: {student_id}")

                # Draw a rectangle and label on the frame
                top, right, bottom, left = face_location
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, student_id, (left, top - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

        # Display the result
        cv2.imshow("Real-Time Recognition", frame)

        # Press Q to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
