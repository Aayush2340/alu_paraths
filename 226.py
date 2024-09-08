import cv2
import torch

# Load the trained YOLO model (e.g., YOLOv5)
model = torch.hub.load('ultralytics/yolov5', 'custom', path='path_to_your_trained_model.pt')  # Replace with your model path

# Set up the webcam or video capture
cap = cv2.VideoCapture(0)  # Use 0 for webcam

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO inference on the frame
    results = model(frame)

    # Extract bounding boxes, confidence scores, and class labels
    detections = results.xyxy[0].cpu().numpy()  # xyxy format: [x1, y1, x2, y2, confidence, class]

    for detection in detections:
        x1, y1, x2, y2, confidence, class_id = detection
        if confidence > 0.5:  # Filter by confidence
            # Get the label of the detected hand sign
            label = model.names[int(class_id)]

            # Draw the bounding box and label on the frame
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} ({confidence:.2f})", (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Display the frame with detections
    cv2.imshow('Hand Sign Detection', frame)

    if cv2.waitKey(5) & 0xFF == 27:  # Press ESC to exit
        break

# Release the capture and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
