#!/usr/bin/env python3
import os
import pickle
import face_recognition
import matplotlib.pyplot as plt

KNOWN_FACES_DIR = "known_faces"
ENCODINGS_PATH = "known_faces.pkl"

def encode_faces():
    known_encodings = {}  # { "Alice": [enc1, enc2], ... }

    for person in os.listdir(KNOWN_FACES_DIR):
        person_dir = os.path.join(KNOWN_FACES_DIR, person)
        if not os.path.isdir(person_dir):
            continue

        encodings = []
        for img_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_name)
            image = face_recognition.load_image_file(img_path)

            # Use CNN model for detection (more accurate)
            face_locations = face_recognition.face_locations(image, model="cnn")

            if len(face_locations) != 1:
                print(f"⚠️  Skipping {img_path}: found {len(face_locations)} faces")
                continue

            print(f"✅  Found face at locations: {face_locations} in {img_path}")

            encoding = face_recognition.face_encodings(image, face_locations)[0]
            encodings.append(encoding)

            # Optional: visualize detection - comment out if you don't want GUI popups
            # import matplotlib.pyplot as plt
            # plt.imshow(image)
            # for (top, right, bottom, left) in face_locations:
            #     plt.gca().add_patch(plt.Rectangle((left, top), right-left, bottom-top,
            #                                       edgecolor='red', facecolor='none', linewidth=2))
            # plt.title(f"Detected face in {img_name}")
            # plt.show()

        if encodings:
            known_encodings[person] = encodings
            print(f"✅  Encoded {len(encodings)} images for {person}")

    with open(ENCODINGS_PATH, "wb") as f:
        pickle.dump(known_encodings, f)
    print(f"✨  Saved encodings to {ENCODINGS_PATH}")

if __name__ == "__main__":
    encode_faces()
