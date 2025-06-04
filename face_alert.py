#!/usr/bin/env python3
import cv2
import pickle
import face_recognition
import numpy as np
import time
#!/usr/bin/env python3
import os
import pickle
import face_recognition

ENCODINGS_PATH = "known_faces.pkl"

def add_person_to_encodings(name, image_paths):
    # Load existing encodings if file exists
    if os.path.exists(ENCODINGS_PATH):
        with open(ENCODINGS_PATH, "rb") as f:
            known_encodings = pickle.load(f)
    else:
        known_encodings = {}

    # Ensure list exists for the person
    if name not in known_encodings:
        known_encodings[name] = []

    added_count = 0

    for img_path in image_paths:
        if not os.path.isfile(img_path):
            print(f"❌ File not found: {img_path}")
            continue

        image = face_recognition.load_image_file(img_path)
        face_locations = face_recognition.face_locations(image, model="cnn")

        if len(face_locations) != 1:
            print(f"⚠️  Skipping {img_path}: found {len(face_locations)} faces")
            continue

        encoding = face_recognition.face_encodings(image, face_locations)[0]
        known_encodings[name].append(encoding)
        added_count += 1
        print(f"✅ Added encoding from {img_path}")

    if added_count > 0:
        with open(ENCODINGS_PATH, "wb") as f:
            pickle.dump(known_encodings, f)
        print(f"✨ Saved updated encodings with {added_count} new image(s) for '{name}'")
    else:
        print(f"⚠️ No valid images were added for '{name}'")

# Example usage:
if __name__ == "__main__":
    add_person_to_encodings("Alice", ["path/to/alice1.jpg", "path/to/alice2.jpg"])

# Load known faces
with open("known_faces.pkl", "rb") as f:
    known_faces = pickle.load(f)  # {"Alice": [enc1, …], ...}

TOLERANCE = 0.5
FRAME_SKIP = 10
DETECTION_THRESHOLD = 5
COOLDOWN_SECONDS = 300  # 5 minutes

# Track detections and cooldowns
detection_counts = {}  # {"Alice": count}
last_detection_times = {}  # {"Alice": timestamp}

# Open webcam
video = cv2.VideoCapture(0)
if not video.isOpened():
    raise RuntimeError("Unable to access webcam")

print("🎥 Starting face detection. Press 'q' to quit.")
frame_count = 0

while True:
    ret, frame = video.read()
    if not ret:
        break

    frame_count += 1

    if frame_count % FRAME_SKIP == 0:
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locs = face_recognition.face_locations(rgb_small)
        face_encs = face_recognition.face_encodings(rgb_small, face_locs)

        for face_encoding in face_encs:
            best_match = ("Unknown", 1.0)
            for name, enc_list in known_faces.items():
                distances = face_recognition.face_distance(enc_list, face_encoding)
                if len(distances) > 0:
                    min_dist = np.min(distances)
                    if min_dist < TOLERANCE and min_dist < best_match[1]:
                        best_match = (name, min_dist)

            name = best_match[0]
            if name != "Unknown":
                now = time.time()
                if name not in detection_counts:
                    detection_counts[name] = 0
                if name not in last_detection_times:
                    last_detection_times[name] = 0

                # Check cooldown
                if now - last_detection_times[name] > COOLDOWN_SECONDS:
                    detection_counts[name] += 1
                    if detection_counts[name] >= DETECTION_THRESHOLD:
                        print(f"✅ {name} detected")
                        last_detection_times[name] = now
                        detection_counts[name] = 0  # reset count
            # You can optionally track unknowns or ignore them

    # Just show camera feed without any drawing
    cv2.imshow("Face Alert", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
