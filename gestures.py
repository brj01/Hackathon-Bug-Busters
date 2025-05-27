import cv2
import mediapipe as mp
import pyautogui

# Initialize MediaPipe and webcam
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

def get_hand_direction(landmarks):
    # Compare y-coordinates of wrist (0) and middle finger tip (12)
    wrist_y = landmarks[0].y
    middle_tip_y = landmarks[12].y
    if wrist_y > middle_tip_y:
        return "up"
    else:
        return "down"

while True:
    success, img = cap.read()
    if not success:
        break

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            direction = get_hand_direction(hand_landmarks.landmark)
            if direction == "up":
                pyautogui.scroll(20)  # Scroll up
            elif direction == "down":
                pyautogui.scroll(-20)  # Scroll down

    cv2.imshow("Hand Scroll", img)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()