import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('hand_alphabet_cnn_model_3d_fixed.h5')

# Load the label classes (alphabet labels)
label_classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J','K', 'L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']  # Adjust according to your model's classes

# Initialize MediaPipe Hands model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Function to extract landmarks in the same format as for training (21 landmarks, 3 coordinates)
def extract_landmarks(image, results):
    landmarks_list = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = []
            for lm in hand_landmarks.landmark:
                h, w, _ = image.shape
                cx, cy, cz = lm.x * w, lm.y * h, lm.z  # x, y, z (z is relative depth)
                landmarks.append((cx, cy, cz))  # Append 3D landmarks
            landmarks_list.append(landmarks)

    # Zero padding if one hand is missing
    while len(landmarks_list) < 2:
        landmarks_list.append([(0, 0, 0)] * 21)  # Padding for the second hand if not detected

    return np.array(landmarks_list)

# Start the webcam video capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a natural view
    frame = cv2.flip(frame, 1)

    # Convert the frame to RGB as MediaPipe requires RGB input
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame for hand landmarks
    results = hands.process(rgb_frame)

    # Draw hand landmarks on the frame
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # If hand landmarks are detected, extract and preprocess them
    if results.multi_hand_landmarks:
        landmarks = extract_landmarks(frame, results)
        landmarks = landmarks.reshape(-1, 2, 21, 3)  # Reshape for model input

        # Normalize landmarks as done during training
        landmarks = landmarks / np.amax(landmarks)

        # Predict the alphabet using the model
        prediction = model.predict(landmarks)
        predicted_class = np.argmax(prediction, axis=1)[0]

        # Get the corresponding alphabet
        predicted_alphabet = label_classes[predicted_class]

        # Display the predicted alphabet on the video feed
        cv2.putText(frame, f'Prediction: {predicted_alphabet}', (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the frame
    cv2.imshow('Hand Tracking - Alphabet Recognition', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()

# Release resources
hands.close()
