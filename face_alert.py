#!/usr/bin/env python3
import cv2
import pickle
import face_recognition
import numpy as np

# Load known faces
with open("known_faces.pkl", "rb") as f:
    known_faces = pickle.load(f)  # {"Alice": [enc1, …], ...}

# Tolerance for face matching: lower = stricter (default ~0.6)
TOLERANCE = 0.5

# Start webcam
video = cv2.VideoCapture(0)
if not video.isOpened():
    raise RuntimeError("Unable to access webcam")

print("🎥  Starting video feed. Press 'q' to quit.")

while True:
    ret, frame = video.read()
    if not ret:
        break

    # Resize frame for faster detection
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Detect face locations in the small frame
    face_locs_small = face_recognition.face_locations(rgb_small)

    # Scale face locations back to full size
    face_locs = [(top*4, right*4, bottom*4, left*4) for top, right, bottom, left in face_locs_small]

    # Convert original frame to RGB for encoding
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Get encodings from original frame using scaled-up locations
    face_encs = face_recognition.face_encodings(rgb_frame, face_locs)

    for (top, right, bottom, left), face_encoding in zip(face_locs, face_encs):
        matches = []
        for name, enc_list in known_faces.items():
            results = face_recognition.compare_faces(enc_list, face_encoding, TOLERANCE)
            if any(results):
                matches.append(name)

        name = matches[0] if matches else "Unknown"

        # Draw rectangle and name
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

    cv2.imshow("Face Alert", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
