
# import cv2 as cv
# import mediapipe as mp
# import util
# import pyautogui
# import time

# # Screen size
# #screen_width, screen_height = pyautogui.size()

# # MediaPipe hands
# mpHands = mp.solutions.hands  # imports hands module from mediapipe
# hands = mpHands.Hands(
#     static_image_mode=False,
#     model_complexity=1,
#     min_detection_confidence=0.7,
#     min_tracking_confidence=0.7,
#     max_num_hands=1
# )
# draw = mp.solutions.drawing_utils

# # Track previous direction
# prev_direction = None
# jumped = False
# frame_count = 0

# def detect_gesture(frame, landmark_list, raw_landmarks, processed):
#     global jumped, prev_direction

#     if len(landmark_list) >= 21:
#         index_up = raw_landmarks[8].y < raw_landmarks[6].y
#         middle_up = raw_landmarks[12].y < raw_landmarks[10].y
#         ring_down = raw_landmarks[16].y > raw_landmarks[14].y
#         pinky_down = raw_landmarks[20].y > raw_landmarks[18].y

#         two_fingers_up = index_up and middle_up and ring_down and pinky_down  #for right
#         one_finger_up = index_up and not middle_up and ring_down and pinky_down #for left

#         tips = [raw_landmarks[8], raw_landmarks[12], raw_landmarks[16], raw_landmarks[20]]
#         pips = [raw_landmarks[6], raw_landmarks[10], raw_landmarks[14], raw_landmarks[18]]
#         fist = all(tip.y > pip.y for tip, pip in zip(tips, pips))   # for jump

#         if two_fingers_up and prev_direction != 'right':
#             pyautogui.press('right')
#             print("Right")
#             prev_direction = 'right'

#         elif one_finger_up and prev_direction != 'left':
#             pyautogui.press('left')
#             print("Left")
#             prev_direction = 'left'

#         elif not one_finger_up and not two_fingers_up:
#             prev_direction = None

#         # wrist = raw_landmarks[0]
#         # x = wrist.x
#         # # print(f"Wrist X: {x:.2f}")
#         # if x < 0.5 and prev_direction != 'left':
#         #     pyautogui.press('left')
#         #     print("← Moved LEFT by hand position")
#         #     prev_direction = 'left'

#         # elif x > 0.6 and prev_direction != 'right':
#         #     pyautogui.press('right')
#         #     print("→ Moved RIGHT by hand position")
#         #     prev_direction = 'right'

#         # elif 0.5 <= x <= 0.6:
#         #     prev_direction = None


#         if fist:
#             if not jumped:
#                 pyautogui.press('up')
#                 print("Jump")
#                 jumped = True
#         else:
#             jumped = False

# def main():
#     global frame_count
#     cap = cv.VideoCapture(0)
#     cap.set(cv.CAP_PROP_FRAME_WIDTH, 320)
#     cap.set(cv.CAP_PROP_FRAME_HEIGHT, 240)

#     try:
#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 break

#             frame = cv.flip(frame, 1)
#             frameRGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
#             processed = hands.process(frameRGB)

#             landmark_list = []
#             raw_landmarks = []
#             if processed.multi_hand_landmarks:
#                 hand_landmarks = processed.multi_hand_landmarks[0]
#                 draw.draw_landmarks(frame, hand_landmarks, mpHands.HAND_CONNECTIONS)

#                 for lm in hand_landmarks.landmark:
#                     landmark_list.append((lm.x, lm.y))
#                     raw_landmarks.append(lm)

#                 detect_gesture(frame, landmark_list, raw_landmarks, processed)

#             frame_count += 1
#             cv.imshow('Hand Control', frame)

#             if cv.waitKey(1) & 0xFF == ord('q'):
#                 break

#             time.sleep(0.01)

#     finally:
#         cap.release()
#         cv.destroyAllWindows()

# if __name__ == '__main__':
#     main()



import cv2
import pyautogui
from ultralytics import YOLO
import torch

# Load trained YOLOv8 model
model = YOLO(r'C:\Users\kriti\Downloads\Hand Detection.v12i.yolov8\runs\detect\train\weights\best.pt')

# Custom class names (must match your training data)
class_names = ['left', 'right', 'jump']

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

prev_direction = None
jumped = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    results = model(frame)

    boxes = results[0].boxes.data.cpu().numpy()  # [x1, y1, x2, y2, conf, cls]
    detected_classes = set()

    for box in boxes:
        x1, y1, x2, y2, conf, cls = box
        label = class_names[int(cls)]
        detected_classes.add(label)

        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Trigger actions
    if 'left' in detected_classes and prev_direction != 'left':
        pyautogui.press('left')
        prev_direction = 'left'
        print("Left")

    elif 'right' in detected_classes and prev_direction != 'right':
        pyautogui.press('right')
        prev_direction = 'right'
        print("Right")

    elif 'jump' in detected_classes and not jumped:
        pyautogui.press('up')
        jumped = True
        print("Jump")

    elif not detected_classes:
        prev_direction = None
        jumped = False

    cv2.imshow('YOLO Gesture Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
