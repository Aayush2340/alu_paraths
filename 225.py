import cv2
import mediapipe as mp
import math

# Initialize MediaPipe Hands and Drawing utilities
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def calculate_landmark_positions(landmarks, h, w):
    """Calculate the pixel coordinates of landmarks"""
    return [(int(landmark.x * w), int(landmark.y * h)) for landmark in landmarks]

def distance(point1, point2):
    """Calculate Euclidean distance between two points"""
    return math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)

def is_fist(landmarks_positions):
    """Check if the hand is making a fist"""
    return all(
        landmarks_positions[tip][1] > landmarks_positions[pip][1]
        for tip, pip in [(8, 6), (12, 10), (16, 14), (20, 18)]
    )

def is_open_palm(landmarks_positions):
    """Check if the hand is open with fingers extended"""
    return all(
        landmarks_positions[tip][1] < landmarks_positions[pip][1]
        for tip, pip in [(8, 6), (12, 10), (16, 14), (20, 18)]
    )

def is_thumb_up(landmarks_positions):
    """Check if the thumb is up and other fingers are not fully extended"""
    thumb_up = landmarks_positions[4][1] < landmarks_positions[3][1] < landmarks_positions[2][1]
    other_fingers_not_up = all(
        landmarks_positions[tip][1] > landmarks_positions[pip][1]
        for tip, pip in [(8, 6), (12, 10), (16, 14), (20, 18)]
    )
    return thumb_up and other_fingers_not_up

def is_pointing_upwards(landmarks_positions):
    """Check if the index finger is pointing upwards"""
    index_up = landmarks_positions[8][1] < landmarks_positions[6][1] < landmarks_positions[5][1]
    thumb_neutral_or_down = landmarks_positions[4][1] >= landmarks_positions[2][1]
    other_fingers_not_up = all(
        landmarks_positions[tip][1] > landmarks_positions[pip][1]
        for tip, pip in [(12, 10), (16, 14), (20, 18)]
    )
    return index_up and thumb_neutral_or_down and other_fingers_not_up

def is_peace_sign(landmarks_positions):
    """Check if the hand is making a peace (V) sign"""
    return (
        landmarks_positions[8][1] < landmarks_positions[6][1] and
        landmarks_positions[12][1] < landmarks_positions[10][1] and
        landmarks_positions[16][1] > landmarks_positions[14][1] and
        landmarks_positions[20][1] > landmarks_positions[18][1]
    )

def is_ok_sign(landmarks_positions):
    """Check if the hand is making an OK sign"""
    thumb_index_distance = distance(landmarks_positions[4], landmarks_positions[8])
    middle_ring_distance = distance(landmarks_positions[12], landmarks_positions[16])
    return thumb_index_distance < 30 and middle_ring_distance > 50

def detect_hand_sign(landmarks_positions):
    """Determine the hand sign based on landmark positions"""
    if is_fist(landmarks_positions):
        return "Fist"
    elif is_open_palm(landmarks_positions):
        return "Open Palm"
    elif is_thumb_up(landmarks_positions):
        return "Thumbs Up"
    elif is_pointing_upwards(landmarks_positions):
        return "Pointing Upwards"
    elif is_peace_sign(landmarks_positions):
        return "Peace"
    elif is_ok_sign(landmarks_positions):
        return "OK"
    else:
        return "Unknown"

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
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Get handedness (right or left hand)
                hand_label = results.multi_handedness[idx].classification[0].label
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Calculate landmark positions in pixels
                landmarks_positions = calculate_landmark_positions(hand_landmarks.landmark, h, w)

                # Detect the hand sign based on landmark positions
                hand_sign = detect_hand_sign(landmarks_positions)
                print(f"{hand_label} Hand: {hand_sign}")

                # Display the detected hand sign and hand label on the screen
                cv2.putText(image, f"{hand_label} Hand: {hand_sign}", (10, 30 + idx*30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        # Display the image with landmarks and quadrant info
        cv2.imshow('ISL Hand Sign Detection', image)

        if cv2.waitKey(5) & 0xFF == 27:  # Press ESC to exit
            break

# Release the capture and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
