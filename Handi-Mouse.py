import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import math

# Setup
pyautogui.FAILSAFE = False
SCREEN_W, SCREEN_H = pyautogui.size()

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# Parameters
SMOOTHING = 0.75
CLICK_DIST_THRESHOLD = 40
SCROLL_SENSITIVITY = 3
ZOOM_SENSITIVITY = 2
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# States
prev_x, prev_y = 0, 0
right_click_state = False
left_click_state = False
scroll_mode = False
scroll_ref_y = None
select_mode = False
copy_state = False
paste_state = False
zoom_mode = False
zoom_ref_y = None

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

def distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

with mp_hands.Hands(max_num_hands=2,
                    min_detection_confidence=0.6,
                    min_tracking_confidence=0.5) as hands:
    print("[INFO] Running... Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        h, w, _ = frame.shape
        info_texts = []

        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                handedness = results.multi_handedness[hand_idx].classification[0].label
                lm = [(int(pt.x * w), int(pt.y * h)) for pt in hand_landmarks.landmark]

                index_tip = lm[8]
                thumb_tip = lm[4]
                middle_tip = lm[12]
                ring_tip = lm[16]
                pinky_tip = lm[20]

                if handedness == "Right":
                    # === Right Hand Features ===
                    # Cursor movement
                    x = np.interp(index_tip[0], (0, w), (0, SCREEN_W))
                    y = np.interp(index_tip[1], (0, h), (0, SCREEN_H))

                    curr_x = prev_x + (x - prev_x) * (1 - SMOOTHING)
                    curr_y = prev_y + (y - prev_y) * (1 - SMOOTHING)

                    pyautogui.moveTo(curr_x, curr_y, _pause=False)
                    prev_x, prev_y = curr_x, curr_y
                    info_texts.append("MOVE (Right Hand)")

                    # Right Click (Index + Thumb pinch)
                    if distance(index_tip, thumb_tip) < CLICK_DIST_THRESHOLD and not right_click_state:
                        pyautogui.click(button="right", _pause=False)
                        info_texts.append("RIGHT CLICK")
                        right_click_state = True
                    elif distance(index_tip, thumb_tip) >= CLICK_DIST_THRESHOLD:
                        right_click_state = False

                    # Scroll
                    if distance(thumb_tip, middle_tip) < CLICK_DIST_THRESHOLD:
                        if not scroll_mode:
                            scroll_mode = True
                            scroll_ref_y = middle_tip[1]
                            info_texts.append("SCROLL MODE ON")
                        else:
                            dy = scroll_ref_y - middle_tip[1]
                            if abs(dy) > 5:
                                scroll_amt = int(dy / 5 * SCROLL_SENSITIVITY)
                                pyautogui.scroll(scroll_amt, _pause=False)
                                info_texts.append(f"SCROLL {scroll_amt}")
                                scroll_ref_y = middle_tip[1]
                    else:
                        if scroll_mode:
                            info_texts.append("SCROLL MODE OFF")
                        scroll_mode = False
                        scroll_ref_y = None

                    # Select
                    if distance(thumb_tip, ring_tip) < CLICK_DIST_THRESHOLD:
                        if not select_mode:
                            select_mode = True
                            pyautogui.mouseDown(_pause=False)
                            info_texts.append("SELECT START")
                        else:
                            info_texts.append("SELECTING...")
                    else:
                        if select_mode:
                            pyautogui.mouseUp(_pause=False)
                            select_mode = False
                            info_texts.append("SELECT END")

                    # Copy
                    if distance(thumb_tip, pinky_tip) < CLICK_DIST_THRESHOLD and not copy_state:
                        pyautogui.hotkey("ctrl", "c", _pause=False)
                        info_texts.append("COPY (Ctrl+C)")
                        copy_state = True
                    elif distance(thumb_tip, pinky_tip) >= CLICK_DIST_THRESHOLD:
                        copy_state = False

                    # Paste (Thumb + Index + Middle)
                    if (distance(thumb_tip, index_tip) < CLICK_DIST_THRESHOLD and
                        distance(thumb_tip, middle_tip) < CLICK_DIST_THRESHOLD and
                        distance(index_tip, middle_tip) < CLICK_DIST_THRESHOLD):
                        if not paste_state:
                            pyautogui.hotkey("ctrl", "v", _pause=False)
                            info_texts.append("PASTE (Ctrl+V)")
                            paste_state = True
                    else:
                        paste_state = False

                elif handedness == "Left":
                    # === Left Hand Features ===
                    # Left Click (Index + Thumb pinch)
                    if distance(index_tip, thumb_tip) < CLICK_DIST_THRESHOLD and not left_click_state:
                        pyautogui.click(button="left", _pause=False)
                        info_texts.append("LEFT CLICK")
                        left_click_state = True
                    elif distance(index_tip, thumb_tip) >= CLICK_DIST_THRESHOLD:
                        left_click_state = False

                    # Zoom (Thumb + Middle pinch)
                    if distance(thumb_tip, middle_tip) < CLICK_DIST_THRESHOLD:
                        if not zoom_mode:
                            zoom_mode = True
                            zoom_ref_y = middle_tip[1]
                            info_texts.append("ZOOM MODE ON (Left Hand)")
                        else:
                            dy = zoom_ref_y - middle_tip[1]
                            if abs(dy) > 5:
                                if dy > 0:
                                    pyautogui.hotkey("ctrl", "+", _pause=False)  # Zoom In
                                    info_texts.append("ZOOM IN")
                                else:
                                    pyautogui.hotkey("ctrl", "-", _pause=False)  # Zoom Out
                                    info_texts.append("ZOOM OUT")
                                zoom_ref_y = middle_tip[1]
                    else:
                        if zoom_mode:
                            info_texts.append("ZOOM MODE OFF")
                        zoom_mode = False
                        zoom_ref_y = None

                # Draw landmarks
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Overlay text
        for i, text in enumerate(info_texts):
            cv2.putText(frame, text, (10, 30 + i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Finger Cursor - Right: Mouse | Left: Click+Zoom", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
