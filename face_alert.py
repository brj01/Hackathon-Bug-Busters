#!/usr/bin/env python3
import cv2
import pickle
import face_recognition
import numpy as np

# Load known faces
with open("known_faces.pkl", "rb") as f:
    known_faces = pickle.load(f)  # {"Alice": [enc1, …], ...}

TOLERANCE = 0.5  # Lower = stricter match
FRAME_SKIP = 10   # Process every Nth frame

video = cv2.VideoCapture(0)
if not video.isOpened():
    raise RuntimeError("Unable to access webcam")

print("🎥  Starting video feed. Press 'q' to quit.")
frame_count = 0
last_face_locs = []
last_names = []

while True:
    ret, frame = video.read()
    if not ret:
        break

    frame_count += 1

    if frame_count % FRAME_SKIP == 0:
        # Resize and process current frame
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Detect faces and encode
        face_locs_small = face_recognition.face_locations(rgb_small)
        face_encs = face_recognition.face_encodings(rgb_small, face_locs_small)

        # Scale face locations back to full size
        last_face_locs = [(top*4, right*4, bottom*4, left*4) for top, right, bottom, left in face_locs_small]
        last_names = []

        for face_encoding in face_encs:
            best_match = ("Unknown", 1.0)
            for name, enc_list in known_faces.items():
                distances = face_recognition.face_distance(enc_list, face_encoding)
                if len(distances) > 0:
                    min_dist = np.min(distances)
                    if min_dist < TOLERANCE and min_dist < best_match[1]:
                        best_match = (name, min_dist)
            last_names.append(best_match[0])

    # Draw latest results regardless of frame skip
    for (top, right, bottom, left), name in zip(last_face_locs, last_names):
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

    cv2.imshow("Face Alert", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
