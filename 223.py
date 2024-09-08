import cv2
import mediapipe as mp

# Initialize MediaPipe Hands and Drawing utilities
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Set up the webcam or video capture
cap = cv2.VideoCapture(0)  # Use 0 for webcam

# Dictionary to store the previous positions of landmarks
previous_positions = {}

# Initialize the MediaPipe Hands
with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the BGR image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the image and find hands
        results = hands.process(image)

        # Convert the image color back to BGR for OpenCV
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Get image dimensions
        h, w, _ = image.shape
        cx, cy = w // 2, h // 2

        # Draw coordinate axes (center of the screen)
        cv2.line(image, (cx, 0), (cx, h), (0, 255, 0), 2)  # Vertical line
        cv2.line(image, (0, cy), (w, cy), (0, 255, 0), 2)  # Horizontal line

        # Process each hand and its landmarks
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Process each landmark
                for idx, landmark in enumerate(hand_landmarks.landmark):
                    # Get the coordinates of the landmark in pixels
                    cx_lm, cy_lm = int(landmark.x * w), int(landmark.y * h)

                    # Create a key for the current landmark
                    key = f"Landmark_{idx}"

                    # Check if the position has changed
                    if key in previous_positions:
                        prev_cx, prev_cy = previous_positions[key]
                        if prev_cx != cx_lm or prev_cy != cy_lm:
                            print(f"{idx}. [{cx_lm}, {cy_lm}]")
                    else:
                        print(f"{idx}. [{cx_lm}, {cy_lm}]")

                    # Update the stored position
                    previous_positions[key] = (cx_lm, cy_lm)

        # Display the image with landmarks
        cv2.imshow('Hand Tracking', image)

        if cv2.waitKey(5) & 0xFF == 27:  # Press ESC to exit
            break

# Release the capture and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
