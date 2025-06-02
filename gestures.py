import cv2
import numpy as np
import pyautogui
import time
import sys

try:
    import mediapipe as mp

    MP_AVAILABLE = True
except ImportError:
    MP_AVAILABLE = False
    print("MediaPipe not installed. Install it for gesture detection.", file=sys.stderr)


class GestureDetector:
    def __init__(self, min_det_conf=0.7):
        if not MP_AVAILABLE:
            raise ImportError("MediaPipe is required for gesture detection.")

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=min_det_conf,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils

        # Gesture thresholds (optimized values)
        self.pinch_threshold = 0.04
        self.click_threshold = 0.04
        self.scroll_threshold = 0.05
        self.fist_threshold = 0.07

        # Smoothing parameters
        self.smoothing_factor = 0.5
        self.prev_x, self.prev_y = 0.5, 0.5

    def detect_gestures(self, frame):
        """Enhanced gesture detection with smoothing and improved logic"""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        if not results.multi_hand_landmarks:
            return False, 0, 0, 'no_hand', False, False, False, False, False

        hand_landmarks = results.multi_hand_landmarks[0]
        landmarks = hand_landmarks.landmark

        # Get required landmarks
        idx_tip = landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        thumb_tip = landmarks[self.mp_hands.HandLandmark.THUMB_TIP]
        mid_tip = landmarks[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        ring_tip = landmarks[self.mp_hands.HandLandmark.RING_FINGER_TIP]
        pinky_tip = landmarks[self.mp_hands.HandLandmark.PINKY_TIP]

        # Apply smoothing to coordinates
        x_index = self.smoothing_factor * idx_tip.x + (1 - self.smoothing_factor) * self.prev_x
        y_index = self.smoothing_factor * idx_tip.y + (1 - self.smoothing_factor) * self.prev_y
        self.prev_x, self.prev_y = x_index, y_index

        # Calculate distances
        pinch_dist = np.hypot(idx_tip.x - thumb_tip.x, idx_tip.y - thumb_tip.y)
        click_dist = np.hypot(thumb_tip.x - mid_tip.x, thumb_tip.y - mid_tip.y)
        scroll_down_dist = np.hypot(thumb_tip.x - ring_tip.x, thumb_tip.y - ring_tip.y)
        scroll_up_dist = np.hypot(thumb_tip.x - pinky_tip.x, thumb_tip.y - pinky_tip.y)

        # Determine gestures
        pinch_state = 'full_pinch' if pinch_dist < self.pinch_threshold else 'half_pinch'
        double_click = click_dist < self.click_threshold
        scroll_down = scroll_down_dist < self.scroll_threshold
        scroll_up = scroll_up_dist < self.scroll_threshold

        # Improved fist detection
        mcp_points = [
            self.mp_hands.HandLandmark.INDEX_FINGER_MCP,
            self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP,
            self.mp_hands.HandLandmark.RING_FINGER_MCP,
            self.mp_hands.HandLandmark.PINKY_MCP
        ]

        fist_detected = all(
            landmarks[tip].y > landmarks[mcp].y
            for tip, mcp in zip(
                [self.mp_hands.HandLandmark.INDEX_FINGER_TIP,
                 self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                 self.mp_hands.HandLandmark.RING_FINGER_TIP,
                 self.mp_hands.HandLandmark.PINKY_TIP],
                mcp_points
            )
        )

        return True, x_index, y_index, pinch_state, double_click, scroll_down, scroll_up, False, fist_detected


class GestureController:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.detector = GestureDetector()
        self.screen_w, self.screen_h = pyautogui.size()

        # State management
        self.is_dragging = False
        self.scroll_cooldown = 0.15
        self.last_scroll_time = 0
        self.scroll_amount = 80
        self.click_cooldown = 0.5
        self.last_click_time = 0

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            current_time = time.time()

            # Process gestures
            (success, x_index, y_index, pinch_state, double_click,
             scroll_down, scroll_up, _, fist) = self.detector.detect_gestures(frame)

            mouse_x = int(x_index * self.screen_w)
            mouse_y = int(y_index * self.screen_h)

            self.handle_movement(success, mouse_x, mouse_y, pinch_state)
            self.handle_gestures(success, current_time, mouse_x, mouse_y,
                                 pinch_state, double_click, scroll_down, scroll_up, fist)

            self.display_feedback(frame, pinch_state, double_click, scroll_down, scroll_up, fist)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def handle_movement(self, success, x, y, pinch_state):
        """Handle cursor movement and dragging"""
        if success and pinch_state in ['half_pinch', 'full_pinch']:
            pyautogui.moveTo(x, y, duration=0)

            if pinch_state == 'full_pinch' and not self.is_dragging:
                pyautogui.mouseDown()
                self.is_dragging = True
            elif pinch_state == 'half_pinch' and self.is_dragging:
                pyautogui.mouseUp()
                self.is_dragging = False
        elif self.is_dragging:
            pyautogui.mouseUp()
            self.is_dragging = False

    def handle_gestures(self, success, current_time, x, y, pinch_state,
                        double_click, scroll_down, scroll_up, fist):
        """Handle all gesture actions with cooldowns"""
        if not success or pinch_state != 'half_pinch':
            return

        # Double click with cooldown
        if double_click and (current_time - self.last_click_time) > self.click_cooldown:
            pyautogui.doubleClick(x, y)
            self.last_click_time = current_time

        # Scroll with cooldown
        if (current_time - self.last_scroll_time) > self.scroll_cooldown:
            if scroll_down:
                pyautogui.scroll(-self.scroll_amount, x=x, y=y)
                self.last_scroll_time = current_time
            elif scroll_up:
                pyautogui.scroll(self.scroll_amount, x=x, y=y)
                self.last_scroll_time = current_time

        # Fist as right-click
        if fist and (current_time - self.last_click_time) > self.click_cooldown:
            pyautogui.rightClick(x, y)
            self.last_click_time = current_time

    def display_feedback(self, frame, pinch_state, double_click, scroll_down, scroll_up, fist):
        """Visual feedback with optimized drawing"""
        text_lines = [
            f"Pinch: {pinch_state}",
            f"Double Click: {double_click}",
            f"Scroll Down: {scroll_down}",
            f"Scroll Up: {scroll_up}",
            f"Fist: {fist}"
        ]

        for i, line in enumerate(text_lines):
            cv2.putText(frame, line, (10, 30 + i * 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Gesture Control", frame)


if __name__ == "__main__":
    try:
        controller = GestureController()
        controller.run()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)