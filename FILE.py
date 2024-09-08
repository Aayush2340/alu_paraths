import cv2
import mediapipe as mp
import numpy as np
import os

# Initialize MediaPipe Hand Model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Create folder to store data
DATA_PATH = 'isl_data'
if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)


# Function to extract hand landmarks for both hands
def get_hand_landmarks(image):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_image)

    landmarks_list = []  # To store landmarks for both hands
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            # Extract landmark coordinates
            landmarks = []
            for id, lm in enumerate(hand_landmarks.landmark):
                h, w, c = image.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                landmarks.append((cx, cy))
            landmarks_list.append(landmarks)
    return landmarks_list


# Function to collect data for each alphabet
def collect_data_for_alphabet(alphabet, sample_count=100):
    cap = cv2.VideoCapture(0)
    collected_samples = 0

    print(f"Collecting data for alphabet '{alphabet}'...")

    while collected_samples < sample_count:
        ret, frame = cap.read()
        if not ret:
            break

        landmarks_list = get_hand_landmarks(frame)  # Get landmarks for both hands
        if landmarks_list:
            # Save landmarks for both hands (if both are detected)
            filename = os.path.join(DATA_PATH, f'{alphabet}_{collected_samples}.npy')
            np.save(filename, landmarks_list)  # Save both hands' landmarks
            collected_samples += 1
            print(f"Collected sample {collected_samples} for alphabet '{alphabet}'")

        # Display the frame
        cv2.putText(frame, f'Collecting for: {alphabet}, Samples: {collected_samples}/{sample_count}',
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('Collecting ISL Data', frame)

        if cv2.waitKey(5) & 0xFF == 27:  # Press 'Esc' to stop early
            break

    cap.release()
    cv2.destroyAllWindows()


# Function to collect data for all alphabets
def collect_data_for_all_alphabets():
    alphabets = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")  # ISL alphabets
    for alphabet in alphabets:
        collect_data_for_alphabet(alphabet, sample_count=100)
        proceed = input(f"Press Enter to continue to next alphabet or type 'exit' to stop: ")
        if proceed.lower() == 'exit':
            break


# Start collecting data for all alphabets
collect_data_for_all_alphabets()
