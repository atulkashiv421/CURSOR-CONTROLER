import cv2
import mediapipe as mp
import pyautogui
import math
import time

# ========== Setup ==========

cap = cv2.VideoCapture(0)
hands = mp.solutions.hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
draw = mp.solutions.drawing_utils

screen_w, screen_h = pyautogui.size()
last_click = 0
dragging = False

# Thresholds
click_threshold = 40  # pinch distance for click
drag_threshold = 50   # distance for drag

# Smoothing
prev_x, prev_y = 0, 0
smooth_factor = 0.2  # smaller = smoother but slower

# ========== Main Loop ==========
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:
            draw.draw_landmarks(frame, hand)

            # Landmarks: index tip & thumb tip
            x_index = int(hand.landmark[8].x * screen_w)
            y_index = int(hand.landmark[8].y * screen_h)
            x_thumb = int(hand.landmark[4].x * screen_w)
            y_thumb = int(hand.landmark[4].y * screen_h)

            # Smooth cursor movement
            curr_x = prev_x + (x_index - prev_x) * smooth_factor
            curr_y = prev_y + (y_index - prev_y) * smooth_factor
            pyautogui.moveTo(curr_x, curr_y)
            prev_x, prev_y = curr_x, curr_y

            # Distance between thumb and index
            distance = math.hypot(x_thumb - x_index, y_thumb - y_index)

            # Pinch Click
            if distance < click_threshold and time.time() - last_click > 0.3:
                pyautogui.click()
                last_click = time.time()
                dragging = False  # stop dragging after click
                cv2.putText(frame, "Click!", (50,100),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 3)

            # Drag & Drop
            elif distance < drag_threshold and not dragging:
                pyautogui.mouseDown()
                dragging = True
                cv2.putText(frame, "Drag Start", (50,150),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 3)

            elif distance >= drag_threshold and dragging:
                pyautogui.mouseUp()
                dragging = False
                cv2.putText(frame, "Drag End", (50,150),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 3)

    # Display
    cv2.imshow("Hand Mouse Control", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
