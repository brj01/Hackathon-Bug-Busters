#!/usr/bin/env python3
import os
import pickle
import time

import cv2
import face_recognition
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Default constants (you can override any of these when constructing FaceRecognizer)
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_TOLERANCE = 0.5
DEFAULT_FRAME_SKIP = 10
DEFAULT_DETECTION_THRESHOLD = 5
DEFAULT_COOLDOWN_SECONDS = 300  # 5 minutes
# ──────────────────────────────────────────────────────────────────────────────


class FaceRecognizer:
    """
    Holds:
      - known_faces: { person_name: [encoding, ...], ... }
      - detection_counts: { person_name: int }
      - last_detection_times: { person_name: float }
      - parameters: tolerance, frame_skip, detection_threshold, cooldown_seconds

    Usage:
        # 1. Instantiate once, pointing at your pickle path:
        recognizer = FaceRecognizer(
            encodings_path="/path/to/known_faces.pkl",
            tolerance=0.4,              # optional
            frame_skip=10,              # optional
            detection_threshold=5,      # optional
            cooldown_seconds=300        # optional
        )

        # 2. When you get a new camera frame (as a BGR numpy array):
        newly_reported = recognizer.process_frame(frame_bgr)

        #    newly_reported is a list of names who just crossed the threshold in this frame.
        #    Because FaceRecognizer keeps its own counts+cooldowns, you never reload the pickle
        #    or re‐pass those dicts.
    """

    def __init__(
        self,
        encodings_path: str,
        tolerance: float = DEFAULT_TOLERANCE,
        frame_skip: int = DEFAULT_FRAME_SKIP,
        detection_threshold: int = DEFAULT_DETECTION_THRESHOLD,
        cooldown_seconds: float = DEFAULT_COOLDOWN_SECONDS,
    ):
        self.encodings_path = encodings_path
        self.tolerance = tolerance
        self.frame_skip = frame_skip
        self.detection_threshold = detection_threshold
        self.cooldown_seconds = cooldown_seconds

        # Load the pickle exactly once at __init__ time:
        self.known_faces: dict[str, list[np.ndarray]] = self._load_known_faces()

        # Internal state for detection across frames:
        self.detection_counts: dict[str, int] = {}
        self.last_detection_times: dict[str, float] = {}

        # Frame counter to enforce “process every Nth frame”:
        self._frame_counter = 0

    def _load_known_faces(self) -> dict[str, list[np.ndarray]]:
        """
        Load { name: [enc1, enc2, …], … } from encodings_path.
        If file doesn’t exist, returns {}.
        """
        if os.path.exists(self.encodings_path):
            with open(self.encodings_path, "rb") as f:
                return pickle.load(f)
        return {}

    def add_person(self, name: str, image_paths: list[str]) -> int:
        """
        Exactly like the old add_person_to_encodings, but using this instance’s encodings_path.
        Returns the number of newly added encodings.
        """
        if os.path.exists(self.encodings_path):
            with open(self.encodings_path, "rb") as f:
                known_encodings = pickle.load(f)
        else:
            known_encodings = {}

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
            with open(self.encodings_path, "wb") as f:
                pickle.dump(known_encodings, f)
            print(f"✨ Saved updated encodings with {added_count} new image(s) for '{name}'")
            # Also update our in‐memory copy of known_faces, so future detection sees new encodings:
            self.known_faces = known_encodings
        else:
            print(f"⚠️ No valid images were added for '{name}'")

        return added_count

    def _detect_in_single_frame(self, frame_bgr: np.ndarray) -> list[str]:
        """
        Exactly the core detection logic (no frame_skip logic here).
        Returns a list of names who “just crossed” the threshold in this single frame.
        """

        # 1) Resize → RGB → face locations + encodings
        small_frame = cv2.resize(frame_bgr, (0, 0), fx=0.25, fy=0.25)
        rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        face_locs = face_recognition.face_locations(rgb_small)
        face_encs = face_recognition.face_encodings(rgb_small, face_locs)

        now = time.time()
        newly_reported: list[str] = []

        for face_encoding in face_encs:
            best_match = ("Unknown", 1.0)
            for name, enc_list in self.known_faces.items():
                distances = face_recognition.face_distance(enc_list, face_encoding)
                if len(distances) > 0:
                    min_dist = np.min(distances)
                    if min_dist < self.tolerance and min_dist < best_match[1]:
                        best_match = (name, min_dist)

            name = best_match[0]
            if name != "Unknown":
                # Initialize state if first time seeing this person
                if name not in self.detection_counts:
                    self.detection_counts[name] = 0
                if name not in self.last_detection_times:
                    self.last_detection_times[name] = 0.0

                # If cooldown has elapsed, increment count
                if now - self.last_detection_times[name] > self.cooldown_seconds:
                    self.detection_counts[name] += 1

                    # If we’ve reached the threshold, “report” now:
                    if self.detection_counts[name] >= self.detection_threshold:
                        newly_reported.append(name)
                        self.last_detection_times[name] = now
                        self.detection_counts[name] = 0  # reset

        return newly_reported

    def process_frame(self, frame_bgr: np.ndarray) -> list[str]:
        """
        Call this for every new BGR frame you obtain from the camera.
        Internally, it will only “run detection” every self.frame_skip frames.
        Returns a list of names who have just reached the threshold
        on this particular frame. (Empty list if no one crossed it now.)

        Example:
            recognizer = FaceRecognizer("/…/known_faces.pkl")
            …
            # In your camera callback (receiving a BGR numpy array):
            new_names = recognizer.process_frame(frame_bgr)
            # new_names might be ["Alice"] or [] or ["Bob", "Charlie"], etc.

        This method is stateful:
         - It increments its own frame counter.
         - When frame_counter % frame_skip == 0, it runs the core
           detection logic (_detect_in_single_frame).
         - Otherwise, it immediately returns [].
        """
        self._frame_counter += 1
        if (self._frame_counter % self.frame_skip) != 0:
            return []

        # Every Nth frame, do the actual face recognition check:
        return self._detect_in_single_frame(frame_bgr)


# ──────────────────────────────────────────────────────────────────────────────
# Optional: allow standalone desktop testing with a live webcam loop
# ──────────────────────────────────────────────────────────────────────────────

def run_live_detection(
    encodings_path: str = "known_faces.pkl",
    tolerance: float = DEFAULT_TOLERANCE,
    frame_skip: int = DEFAULT_FRAME_SKIP,
    detection_threshold: int = DEFAULT_DETECTION_THRESHOLD,
    cooldown_seconds: float = DEFAULT_COOLDOWN_SECONDS
):
    """
    Convenience function if you want to run this module on your desktop
    exactly as before. It will:
      1) Instantiate FaceRecognizer(...)
      2) Open cv2.VideoCapture(0) and loop until you press 'q'
      3) On every Nth frame, call process_frame() and print each name detected
    """
    recognizer = FaceRecognizer(
        encodings_path=encodings_path,
        tolerance=tolerance,
        frame_skip=frame_skip,
        detection_threshold=detection_threshold,
        cooldown_seconds=cooldown_seconds
    )

    video = cv2.VideoCapture(0)
    if not video.isOpened():
        raise RuntimeError("Unable to access webcam")

    print("🎥 Starting face detection. Press 'q' to quit.")
    while True:
        ret, frame = video.read()
        if not ret:
            break

        newly = recognizer.process_frame(frame)
        for name in newly:
            print(f"✅ {name} detected")

        cv2.imshow("Face Alert", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()


# Only run the demo-loop if you directly execute python face_module.py
if __name__ == "__main__":
    # Example: add a person “Alice” (change these paths to real image files)
    FaceRecognizer("known_faces.pkl").add_person(
        "Alice", ["path/to/alice1.jpg", "path/to/alice2.jpg"]
    )
    # Then start the live loop (press 'q' to quit):
    run_live_detection("known_faces.pkl")
