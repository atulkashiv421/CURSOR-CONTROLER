import cv2
import mediapipe as mp
import pyautogui
import time
import pyttsx3   # 🔊 Text-to-Speech

# Setup
cap = cv2.VideoCapture(0)
hands = mp.solutions.hands.Hands(min_detection_confidence=0.7,
                                 min_tracking_confidence=0.7)
draw = mp.solutions.drawing_utils

# Face Detection (Haar Cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# TTS Engine
engine = pyttsx3.init()
engine.setProperty("rate", 160)   # speed
engine.setProperty("volume", 1.0) # volume

last_action = 0  # delay to avoid double scroll
last_speak = 0   # delay for speaking again

# Screen size
screen_w, screen_h = pyautogui.size()

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # ---------- FACE DETECTION ----------
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) > 0:
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        # Speak only once every 5 sec
        if time.time() - last_speak > 5:
            engine.say("Hi Sir, how can I help you?")
            engine.runAndWait()
            last_speak = time.time()

    # ---------- HAND DETECTION ----------
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:
            draw.draw_landmarks(frame, hand)

            # Index finger tip and base
            index_tip = hand.landmark[8]  # (x,y)
            index_base_y = hand.landmark[6].y

            # Middle finger tip
            middle_tip_y = hand.landmark[12].y

            # Convert normalized coords to screen coords
            cursor_x = int(index_tip.x * screen_w)
            cursor_y = int(index_tip.y * screen_h)

            # Move cursor
            pyautogui.moveTo(cursor_x, cursor_y)

            # Gesture 1: Only Index finger up → Next Reel
            if index_tip.y < index_base_y and middle_tip_y > index_base_y:
                if time.time() - last_action > 1:
                    pyautogui.press("down")   # ✅ Next reel (YouTube)
                    last_action = time.time()
                    cv2.putText(frame, "Next Reel!", (50, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

            # Gesture 2: Index + Middle finger up → Previous Reel
            if index_tip.y < index_base_y and middle_tip_y < index_base_y:
                if time.time() - last_action > 1:
                    pyautogui.press("up")     # ✅ Previous reel (YouTube)
                    last_action = time.time()
                    cv2.putText(frame, "Previous Reel!", (50, 150),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

    cv2.imshow("AI Assistant Controller", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
