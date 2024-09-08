import cv2
import mediapipe as mp

# Initialize MediaPipe Hands and Drawing utilities
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Set up the webcam or video capture
cap = cv2.VideoCapture(0)  # Use 0 for webcam

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

        # Draw the hand landmarks and determine their positions relative to the axes
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Determine the position of each landmark
                for idx, landmark in enumerate(hand_landmarks.landmark):
                    # Get the coordinates of the landmark in pixels
                    cx_lm, cy_lm = int(landmark.x * w), int(landmark.y * h)

                    # Determine the quadrant
                    if cx_lm < cx and cy_lm < cy:
                        quadrant = "Top-Left"
                    elif cx_lm > cx and cy_lm < cy:
                        quadrant = "Top-Right"
                    elif cx_lm < cx and cy_lm > cy:
                        quadrant = "Bottom-Left"
                    else:
                        quadrant = "Bottom-Right"

                    # Calculate the relative position
                    relative_x = cx_lm - cx
                    relative_y = cy_lm - cy

                    # Display the index, position information, and relative position
                    text = f"{idx}({cx_lm},{cy_lm}) {quadrant} Rel({relative_x},{relative_y})"
                    font_scale = 0.4  # Smaller font scale for better visibility
                    thickness = 1  # Thinner text
                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
                    text_x = cx_lm - text_size[0] // 2  # Center text horizontally
                    text_y = cy_lm - text_size[1] // 2  # Adjust text vertically

                    # Adjust the text color to contrast with the background
                    text_color = (0, 255, 0)  # Green color

                    cv2.putText(image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness, cv2.LINE_AA)

        # Display the image with landmarks and quadrant info
        cv2.imshow('Hand Tracking with Quadrants', image)

        if cv2.waitKey(5) & 0xFF == 27:  # Press ESC to exit
            break

# Release the capture and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
