import cv2
import torch
from ultralytics import YOLO

# Load the trained YOLOv8 model (directly from local path)
model = YOLO(r'C:\Users\Isha Gupta\Desktop\Python\runs\detect\train4\weights\best.pt')

# Start the webcam feed (0 is the default camera)
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Resize frame to match YOLOv8 input size (640x640)
    frame_resized = cv2.resize(frame, (640, 640))

    # Convert frame to RGB
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

    # Perform inference (hand gesture detection) with a lower confidence threshold
    results = model(frame_rgb, conf=0.05)  # Lowered confidence threshold

    # Access the first result (assuming it's a list of detections)
    result = results[0]

    # Display detection details
    for detection in result.boxes.data:  # Accessing each detected box
        x1, y1, x2, y2, confidence, class_id = detection
        print(f"Class: {class_id}, Confidence: {confidence}")

    # Plot results on the frame (bounding boxes, labels, etc.)
    frame_with_boxes = result.plot()

    # Convert the frame back to BGR for OpenCV compatibility
    frame_bgr = cv2.cvtColor(frame_with_boxes, cv2.COLOR_RGB2BGR)

    # Display the frame with detections
    cv2.imshow("Webcam Hand Gesture Detection", frame_bgr)

    # If you press 'q' on your keyboard, the program will stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()
