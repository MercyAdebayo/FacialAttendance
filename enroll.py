import cv2
import numpy as np
import os
import time
import face_recognition

# Request student for their student number to enroll
student_number = input("Enter the student number: ")


# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EMBEDDINGS_DIR = os.path.join(BASE_DIR, "embeddings")
DATA_DIR = os.path.join(BASE_DIR, "data")
ENROLLED_FACES_DIR = os.path.join(DATA_DIR, "enrolled faces")

def main():
    # Create directories if they don't exist
    if not os.path.exists(EMBEDDINGS_DIR):
        os.makedirs(EMBEDDINGS_DIR)
    if not os.path.exists(ENROLLED_FACES_DIR):
        os.makedirs(ENROLLED_FACES_DIR)

    # start video capture from webcam
    cap = cv2.VideoCapture(0)
    print("Press 'S' to take a snapshot and enroll this student.")
    print("Press 'Q' to quit without enrolling.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image. Exiting...")
            break

        cv2.imshow("Enrolling.. (Press S to save)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s') or key == ord('S'):
            image_path = os.path.join(ENROLLED_FACES_DIR, student_number + ".jpg")
            cv2.imwrite(image_path, frame)
            print("Image saved successfully.")

            # Detect face in the image
            face_locations = face_recognition.face_locations(frame)

            if len(face_locations) == 0:
                print("No face detected in the image. Please try again.")
                os.remove(image_path)
                continue

            # Get the face embeddings
            # Assume only one face for enrollment
            face_encoding = face_recognition.face_encodings(frame, [face_locations[0]])[0]

            # Save the embedding as a .npy file
            embedding_path = os.path.join(EMBEDDINGS_DIR, f"{student_number}.npy")
            np.save(embedding_path, face_encoding)
            print(f"Saved embedding to {embedding_path}")

            break
        elif key == ord('q'):
            print("Enrollment cancelled.")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
