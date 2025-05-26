import os
import cv2
import argparse

def main():
    parser = argparse.ArgumentParser(
        description="Capture images for a new known face."
    )
    parser.add_argument(
        "--name", "-n", required=True,
        help="Name of the person (will create known_faces/<name>/)"
    )
    parser.add_argument(
        "--output", "-o", default="known_faces",
        help="Base output directory"
    )
    args = parser.parse_args()

    person_dir = os.path.join(args.output, args.name)
    os.makedirs(person_dir, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open webcam")
        return

    print("🎥 Webcam opened. Press SPACE to capture, ESC to quit.")
    count = len(os.listdir(person_dir))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.putText(frame,
                    f"{args.name}: {count} images",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0,255,0), 2)
        cv2.imshow("Capture Known Faces", frame)

        key = cv2.waitKey(1)
        # SPACE = 32, ESC = 27
        if key == 32:
            img_path = os.path.join(person_dir,
                                    f"{args.name}_{count+1}.jpg")
            cv2.imwrite(img_path, frame)
            print(f"🖼️  Saved {img_path}")
            count += 1
        elif key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("✅ Done.")

if __name__ == "__main__":
    main()

