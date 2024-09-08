import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer
from collections import deque

# Initialize MediaPipe Hands model with higher confidence thresholds
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=2,
                       min_detection_confidence=0.7,  # Increased detection confidence
                       min_tracking_confidence=0.7)  # Increased tracking confidence
mp_drawing = mp.solutions.drawing_utils

# Load the trained model
model = tf.keras.models.load_model('isl_model.h5')

# Initialize label encoder (assuming you're working with 26 alphabets)
encoder = LabelBinarizer()
encoder.fit(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))

# Define a list to store previous landmarks for smoothing
landmark_history = []
smoothing_frames = 5  # Maximum number of frames to average over

# Define a queue to track the last few predictions for smoothing similar gestures
prediction_queue = deque(maxlen=5)  # Store the last 5 predictions


# Function to smooth landmarks over multiple frames
def smooth_landmarks(landmarks):
    global landmark_history
    landmark_history.append(landmarks)

    if len(landmark_history) > smoothing_frames:
        landmark_history.pop(0)

    avg_landmarks = np.mean(landmark_history, axis=0)
    return avg_landmarks


# Function to get the most common prediction from the queue
def get_stable_prediction(queue):
    if len(queue) == queue.maxlen:
        return max(set(queue), key=queue.count)  # Return the most common prediction
    return None


# Function to predict the alphabet based on the extracted landmarks
def predict_alphabet(landmarks):
    landmarks_flattened = landmarks.flatten().reshape(1, 84)  # Reshape to (1, 84)
    prediction = model.predict(landmarks_flattened)

    # Get the predicted label
    predicted_label = encoder.inverse_transform(prediction)[0]

    # Add confidence check for similar gestures
    if predicted_label in ['M', 'N']:
        # Set a higher confidence threshold for 'M' and 'N'
        confidence = np.max(prediction)
        if confidence < 0.75:  # Adjust this threshold as needed
            return None  # Ignore low-confidence predictions for M and N

    return predicted_label


# Function to extract hand landmarks for real-time video feed
def get_hand_landmarks(image):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_image)

    landmarks_list = []

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = []
            for lm in hand_landmarks.landmark:
                h, w, _ = image.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                landmarks.append((cx, cy))
            landmarks_list.append(landmarks)

    # Ensure both hands are represented (pad with zeros if only one hand is detected)
    if len(landmarks_list) == 1:
        landmarks_list.append([(0, 0)] * 21)  # Pad with zeros for the second hand

    # Ensure that the landmarks array has exactly 2 hands with 21 landmarks each
    if len(landmarks_list) == 2:
        return np.array(landmarks_list)  # Shape (2, 21, 2)
    else:
        return np.zeros((2, 21, 2))  # Return a zero array if no hands are detected


# Function to check if the detected hand is valid
# Function to check if the detected hand is valid
def is_valid_hand(landmarks, frame):
    h, w, _ = frame.shape

    # Compute bounding box of the hand
    x_values = [lm[0] for lm in landmarks]
    y_values = [lm[1] for lm in landmarks]

    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)  # Corrected here

    # Check if the hand size is reasonable
    box_width = x_max - x_min
    box_height = y_max - y_min

    # Filter out hands that are too small or too large
    if box_width < 0.05 * w or box_height < 0.05 * h or box_width > 0.9 * w or box_height > 0.9 * h:
        return False

    return True


# Function to check if the detected hand is valid
def is_valid_hand(landmarks, frame):
    h, w, _ = frame.shape

    # Compute bounding box of the hand
    x_values = [lm[0] for lm in landmarks]
    y_values = [lm[1] for lm in landmarks]

    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)  # Corrected here

    # Check if the hand size is reasonable
    box_width = x_max - x_min
    box_height = y_max - y_min

    # Filter out hands that are too small or too large
    if box_width < 0.05 * w or box_height < 0.05 * h or box_width > 0.9 * w or box_height > 0.9 * h:
        return False

    return True


# Start the webcam feed
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Extract hand landmarks
    landmarks = get_hand_landmarks(frame)

    if np.any(landmarks):  # Check if any landmarks are detected
        smoothed_landmarks = smooth_landmarks(landmarks)

        if is_valid_hand(smoothed_landmarks[0], frame):  # Check if the first hand is valid
            predicted_alphabet = predict_alphabet(smoothed_landmarks)

            if predicted_alphabet:  # Add only valid predictions to the queue
                prediction_queue.append(predicted_alphabet)

            # Get a stable prediction by looking at the recent history of predictions
            stable_prediction = get_stable_prediction(prediction_queue)

            if stable_prediction:
                # Display the stable predicted alphabet on the frame
                cv2.putText(frame, f'Prediction: {stable_prediction}', (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the frame
    cv2.imshow('Real-time Hand Gesture Recognition', frame)

    if cv2.waitKey(5) & 0xFF == 27:  # Press 'Esc' to exit
        break

cap.release()
cv2.destroyAllWindows()
